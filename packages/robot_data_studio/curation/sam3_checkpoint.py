from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator


_PART_RE = re.compile(
    r"^(?P<major>\d{2})(?:\.(?P<minor>\d{2})\.resume(?P<incomplete>\.partial)?|(?P<kind>\.part|\.partial))$"
)


@dataclass(frozen=True)
class LegacyPart:
    path: Path
    start: int
    length: int

    @property
    def stop(self) -> int:
        return self.start + self.length

    def validate(self, *, expected_bytes: int) -> LegacyPart:
        if self.start < 0 or self.length < 1 or self.stop > expected_bytes:
            raise ValueError(f"legacy part outside checkpoint: {self.path}")
        if self.path.stat().st_size != self.length:
            raise ValueError(f"legacy part size changed during audit: {self.path}")
        return self


def _nearest_aligned(value: int, alignment: int) -> int:
    if alignment < 1:
        raise ValueError("snapshot alignment must be positive")
    return ((value + alignment // 2) // alignment) * alignment


def audit_legacy_parts(
    parts_dir: str | Path,
    *,
    expected_bytes: int,
    major_chunks: int = 8,
    snapshot_alignment: int = 8 * 1024 * 1024,
) -> list[LegacyPart]:
    """Reconstruct ranges made by the two interrupted legacy downloaders.

    ``NN.partial`` is a growing prefix of major chunk NN. The competing resume
    downloader split the remainder after an older aligned prefix snapshot into
    four ceil-sized ``NN.MM.resume[.partial]`` ranges.
    """

    root = Path(parts_dir)
    if expected_bytes < 1 or expected_bytes % major_chunks:
        raise ValueError("checkpoint must divide exactly into major chunks")
    major_size = expected_bytes // major_chunks
    parsed: list[tuple[Path, re.Match[str]]] = []
    for path in sorted(root.iterdir()):
        if path.is_file() and (match := _PART_RE.match(path.name)):
            parsed.append((path, match))

    resume_unit: dict[int, int] = {}
    for path, match in parsed:
        minor = match.group("minor")
        if minor is None or match.group("incomplete"):
            continue
        index = int(minor)
        size = path.stat().st_size
        if index < 3:
            resume_unit[int(match.group("major"))] = max(
                size, resume_unit.get(int(match.group("major")), 0)
            )

    result: list[LegacyPart] = []
    for path, match in parsed:
        major = int(match.group("major"))
        if major >= major_chunks:
            raise ValueError(f"legacy major part index out of range: {path.name}")
        major_start = major * major_size
        minor_text = match.group("minor")
        size = path.stat().st_size
        if minor_text is None:
            if size > major_size:
                raise ValueError(f"legacy major part is oversized: {path.name}")
            part = LegacyPart(path=path, start=major_start, length=size)
        else:
            if major not in resume_unit:
                raise ValueError(f"cannot infer resume snapshot for major part {major:02d}")
            unit = resume_unit[major]
            snapshot = _nearest_aligned(major_size - 4 * unit, snapshot_alignment)
            if snapshot < 0 or snapshot >= major_size:
                raise ValueError(f"invalid resume snapshot for major part {major:02d}")
            minor = int(minor_text)
            expected_length = min(unit, major_size - snapshot - minor * unit)
            if minor > 3 or size > expected_length:
                raise ValueError(f"legacy resume part is oversized: {path.name}")
            part = LegacyPart(
                path=path,
                start=major_start + snapshot + minor * unit,
                length=size,
            )
        result.append(part.validate(expected_bytes=expected_bytes))
    return result


def merge_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, stop in sorted(intervals):
        if start < 0 or stop <= start:
            raise ValueError("invalid byte interval")
        if not merged or start > merged[-1][1]:
            merged.append([start, stop])
        else:
            merged[-1][1] = max(merged[-1][1], stop)
    return [(start, stop) for start, stop in merged]


def merged_coverage(parts: Iterable[LegacyPart]) -> int:
    return sum(stop - start for start, stop in merge_intervals((p.start, p.stop) for p in parts))


def missing_intervals(completed: Iterable[tuple[int, int]], expected_bytes: int) -> list[tuple[int, int]]:
    cursor = 0
    missing: list[tuple[int, int]] = []
    for start, stop in merge_intervals(completed):
        if start > cursor:
            missing.append((cursor, start))
        cursor = max(cursor, stop)
    if cursor < expected_bytes:
        missing.append((cursor, expected_bytes))
    return missing


def subtract_intervals(start: int, stop: int, completed: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    remaining = [(start, stop)]
    for covered_start, covered_stop in merge_intervals(completed):
        next_remaining: list[tuple[int, int]] = []
        for left, right in remaining:
            if covered_stop <= left or covered_start >= right:
                next_remaining.append((left, right))
            else:
                if left < covered_start:
                    next_remaining.append((left, covered_start))
                if covered_stop < right:
                    next_remaining.append((covered_stop, right))
        remaining = next_remaining
    return remaining


class CheckpointLock:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._stream = None

    def __enter__(self) -> CheckpointLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("a+")
        try:
            fcntl.flock(self._stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self._stream.close()
            self._stream = None
            raise RuntimeError("another checkpoint downloader holds the cache lock") from error
        self._stream.seek(0)
        self._stream.truncate()
        self._stream.write(f"pid={os.getpid()}\n")
        self._stream.flush()
        return self

    def __exit__(self, *_: object) -> None:
        if self._stream is not None:
            fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
            self._stream.close()
            self._stream = None


RangeReader = Callable[[int, int], bytes]


def verify_part_samples(
    part: LegacyPart,
    *,
    read_remote: RangeReader,
    sample_bytes: int = 64 * 1024,
) -> bool:
    sample = min(sample_bytes, part.length)
    offsets = {0, max(0, (part.length - sample) // 2), part.length - sample}
    with part.path.open("rb") as stream:
        for local_offset in sorted(offsets):
            stream.seek(local_offset)
            local = stream.read(sample)
            remote = read_remote(part.start + local_offset, part.start + local_offset + sample)
            if local != remote:
                return False
    return True


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path, block_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def _copy_slice(source: Path, target_fd: int, *, source_offset: int, target_offset: int, length: int) -> None:
    with source.open("rb") as stream:
        stream.seek(source_offset)
        remaining = length
        position = target_offset
        while remaining:
            block = stream.read(min(8 * 1024 * 1024, remaining))
            if not block:
                raise OSError(f"unexpected EOF while importing {source}")
            os.pwrite(target_fd, block, position)
            position += len(block)
            remaining -= len(block)


def _chunks(start: int, stop: int, size: int) -> Iterator[tuple[int, int]]:
    cursor = start
    while cursor < stop:
        next_cursor = min(stop, cursor + size)
        yield cursor, next_cursor
        cursor = next_cursor


def resume_checkpoint(
    *,
    parts_dir: str | Path,
    cache_dir: str | Path,
    expected_bytes: int,
    expected_sha256: str,
    read_remote: RangeReader,
    remote_chunk_bytes: int = 64 * 1024 * 1024,
) -> dict:
    """Import verified legacy ranges once, download holes, and atomically publish."""

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    final = cache / "model.safetensors"
    incomplete = cache / "model.safetensors.incomplete"
    state_path = cache / "download-state.json"
    with CheckpointLock(cache / ".download.lock"):
        if final.is_file():
            digest = _sha256(final)
            if final.stat().st_size != expected_bytes or digest != expected_sha256:
                raise ValueError("persistent SAM3 checkpoint has invalid size or SHA256")
            return {"status": "complete", "bytes": expected_bytes, "sha256": digest, "cache_hit": True}

        state = {"completed": [], "imported": [], "downloaded_bytes": 0}
        if state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
        completed = [tuple(map(int, item)) for item in state.get("completed", [])]
        if not incomplete.exists():
            with incomplete.open("wb") as stream:
                stream.truncate(expected_bytes)
        elif incomplete.stat().st_size != expected_bytes:
            raise ValueError("persistent incomplete checkpoint has unexpected size")

        candidates = audit_legacy_parts(parts_dir, expected_bytes=expected_bytes)
        verified: list[LegacyPart] = []
        for part in candidates:
            if verify_part_samples(part, read_remote=read_remote):
                verified.append(part)

        started = time.monotonic()
        with incomplete.open("r+b", buffering=0) as target:
            fd = target.fileno()
            for part in sorted(verified, key=lambda item: (-item.length, item.start, item.path.name)):
                for start, stop in subtract_intervals(part.start, part.stop, completed):
                    _copy_slice(
                        part.path,
                        fd,
                        source_offset=start - part.start,
                        target_offset=start,
                        length=stop - start,
                    )
                    completed = merge_intervals([*completed, (start, stop)])
                    state.update(
                        {
                            "schema": "robot-data-studio.sam3-download.v1",
                            "expected_bytes": expected_bytes,
                            "expected_sha256": expected_sha256,
                            "completed": completed,
                            "imported": sorted({*state.get("imported", []), part.path.name}),
                        }
                    )
                    _write_json(state_path, state)
            os.fsync(fd)

            for hole_start, hole_stop in missing_intervals(completed, expected_bytes):
                for start, stop in _chunks(hole_start, hole_stop, remote_chunk_bytes):
                    block = read_remote(start, stop)
                    if len(block) != stop - start:
                        raise OSError("remote Range returned an unexpected byte count")
                    os.pwrite(fd, block, start)
                    os.fsync(fd)
                    completed = merge_intervals([*completed, (start, stop)])
                    state.update(
                        {
                            "schema": "robot-data-studio.sam3-download.v1",
                            "expected_bytes": expected_bytes,
                            "expected_sha256": expected_sha256,
                            "completed": completed,
                            "downloaded_bytes": int(state.get("downloaded_bytes", 0)) + len(block),
                        }
                    )
                    _write_json(state_path, state)

        digest = _sha256(incomplete)
        if digest != expected_sha256:
            raise ValueError("assembled SAM3 checkpoint SHA256 mismatch; incomplete file retained")
        incomplete.replace(final)
        state.update(
            {
                "status": "complete",
                "completed": [[0, expected_bytes]],
                "elapsed_seconds": time.monotonic() - started,
                "sha256": digest,
            }
        )
        _write_json(state_path, state)
        return {
            "status": "complete",
            "bytes": expected_bytes,
            "sha256": digest,
            "cache_hit": False,
            "downloaded_bytes": state.get("downloaded_bytes", 0),
            "imported_part_count": len(state.get("imported", [])),
        }

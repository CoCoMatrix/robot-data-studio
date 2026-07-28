from __future__ import annotations

from pathlib import Path

import pytest

from robot_data_studio.curation.sam3_checkpoint import (
    CheckpointLock,
    LegacyPart,
    audit_legacy_parts,
    merged_coverage,
)


def test_legacy_competing_part_ranges_are_reconstructed_without_double_counting(
    tmp_path: Path,
) -> None:
    # Two 64-byte major chunks. Chunk 0 has a growing 16-byte prefix plus
    # four resume ranges made from an earlier 8-byte prefix snapshot.
    parts = tmp_path / ".parts"
    parts.mkdir()
    (parts / "00.partial").write_bytes(b"a" * 16)
    for index in range(4):
        suffix = "resume" if index < 2 else "resume.partial"
        size = 14 if index < 2 else 7
        (parts / f"00.{index:02d}.{suffix}").write_bytes(bytes([index]) * size)
    (parts / "01.part").write_bytes(b"z" * 64)

    audited = audit_legacy_parts(
        parts,
        expected_bytes=128,
        major_chunks=2,
        snapshot_alignment=8,
    )
    by_name = {part.path.name: part for part in audited}

    assert by_name["00.partial"].start == 0
    assert by_name["00.00.resume"].start == 8
    assert by_name["00.01.resume"].start == 22
    assert by_name["00.02.resume.partial"].start == 36
    assert by_name["00.03.resume.partial"].start == 50
    assert by_name["01.part"].start == 64
    assert merged_coverage(audited) == 114


def test_checkpoint_lock_refuses_a_second_downloader(tmp_path: Path) -> None:
    lock_path = tmp_path / "checkpoint.lock"
    with CheckpointLock(lock_path):
        with pytest.raises(RuntimeError, match="another checkpoint downloader"):
            with CheckpointLock(lock_path):
                pass


def test_legacy_part_rejects_out_of_bounds_intervals(tmp_path: Path) -> None:
    part = tmp_path / "bad.part"
    part.write_bytes(b"bad")
    with pytest.raises(ValueError, match="outside checkpoint"):
        LegacyPart(path=part, start=9, length=3).validate(expected_bytes=10)

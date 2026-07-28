#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import httpx

from robot_data_studio.curation.sam3_checkpoint import (
    CheckpointLock,
    audit_legacy_parts,
    merged_coverage,
    resume_checkpoint,
    verify_part_samples,
)


REPOSITORY = "facebook/sam3"
REVISION = "3c879f39826c"
EXPECTED_BYTES = 3_439_938_512
# Content SHA from the official Git-LFS pointer. The resolve ETag is a Xet CAS
# transport identifier and is deliberately not used as a file digest.
EXPECTED_SHA256 = "6d06f0a5f84e435071fe6603e61d0b4cc7b40e0d39d487cfd4d67d8cc11cc14a"
URL = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/model.safetensors"


class OfficialRangeReader:
    def __init__(self) -> None:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is unset")
        self._authorization = f"Bearer {token}"
        self._client = httpx.Client(trust_env=False, follow_redirects=True, timeout=180.0)

    def close(self) -> None:
        self._client.close()

    def __call__(self, start: int, stop: int) -> bytes:
        if start < 0 or stop <= start or stop > EXPECTED_BYTES:
            raise ValueError("invalid official Range request")
        response = self._client.get(
            URL,
            headers={
                "Authorization": self._authorization,
                "Range": f"bytes={start}-{stop - 1}",
            },
        )
        expected_range = f"bytes {start}-{stop - 1}/{EXPECTED_BYTES}"
        if response.status_code != 206 or response.headers.get("content-range") != expected_range:
            raise OSError(
                f"official Range response mismatch: status={response.status_code} "
                f"content_range={response.headers.get('content-range')}"
            )
        if len(response.content) != stop - start:
            raise OSError("official Range response byte count mismatch")
        return response.content


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--remote-chunk-mib", type=int, default=64)
    args = parser.parse_args()
    if args.remote_chunk_mib < 8 or args.remote_chunk_mib > 256:
        raise ValueError("remote chunk size must be between 8 and 256 MiB")

    cache_dir = Path(args.cache_dir)
    reader = OfficialRangeReader()
    try:
        if args.audit_only:
            with CheckpointLock(cache_dir / ".download.lock"):
                parts = audit_legacy_parts(args.parts_dir, expected_bytes=EXPECTED_BYTES)
                rows = []
                verified = []
                for part in parts:
                    reusable = verify_part_samples(part, read_remote=reader)
                    if reusable:
                        verified.append(part)
                    rows.append(
                        {
                            "name": part.path.name,
                            "start": part.start,
                            "stop": part.stop,
                            "bytes": part.length,
                            "sample_verified": reusable,
                        }
                    )
                report = {
                    "schema": "robot-data-studio.sam3-parts-audit.v1",
                    "repository": REPOSITORY,
                    "revision": REVISION,
                    "expected_bytes": EXPECTED_BYTES,
                    "expected_sha256": EXPECTED_SHA256,
                    "candidate_count": len(parts),
                    "candidate_unique_bytes": merged_coverage(parts),
                    "verified_count": len(verified),
                    "verified_unique_bytes": merged_coverage(verified),
                    "parts": rows,
                }
                _write_json(cache_dir / "parts-audit.json", report)
                print(
                    json.dumps(
                        {key: report[key] for key in (
                            "candidate_count",
                            "candidate_unique_bytes",
                            "verified_count",
                            "verified_unique_bytes",
                        )},
                        sort_keys=True,
                    )
                )
        else:
            report = resume_checkpoint(
                parts_dir=args.parts_dir,
                cache_dir=cache_dir,
                expected_bytes=EXPECTED_BYTES,
                expected_sha256=EXPECTED_SHA256,
                read_remote=reader,
                remote_chunk_bytes=args.remote_chunk_mib * 1024 * 1024,
            )
            print(json.dumps(report, sort_keys=True))
    finally:
        reader.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

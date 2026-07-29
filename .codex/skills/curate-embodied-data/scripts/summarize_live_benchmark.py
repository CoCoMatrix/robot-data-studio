#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


CASES = ("human-cold", "human-warm", "robot-cold", "robot-warm")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--audit-seconds", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit_seconds = {}
    for value in args.audit_seconds:
        name, seconds = value.split("=", 1)
        if name not in CASES or float(seconds) < 0:
            raise ValueError(f"invalid audit timing: {value}")
        audit_seconds[name] = float(seconds)

    cases = {}
    for name in CASES:
        root = args.base / name
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        audit = json.loads((root / "audit.json").read_text(encoding="utf-8"))
        timing = manifest["timing"]
        cases[name] = {
            "episodes": manifest["processed_episode_count"],
            "video_seconds": timing["video_seconds"],
            "api_calls": manifest["c1_api_calls"],
            "c1_cache_hits": manifest["c1_cache_hits"],
            "sam3_cache_hits": manifest["sam3_cache_hits"],
            "sam3_evaluated_masks": manifest["sam3_evaluated_masks"],
            "stages_seconds": {
                "video_decode": timing["video_decode_seconds"],
                "rule_checks": timing["rule_checks_seconds"],
                "c1_qwen": timing["c1_qwen_seconds"],
                "c3": timing["c3_seconds"],
                "sam3": timing["sam3_seconds"],
                "sidecar_write": timing["sidecar_write_seconds"],
                "audit": audit_seconds.get(name),
                "total_before_audit": timing["total_seconds"],
            },
            "episodes_per_hour": timing["episodes_per_hour"],
            "video_hours_per_hour": timing["video_hours_per_hour"],
            "extrapolation": {
                "wall_hours_per_1000_episodes": 1000.0 / timing["episodes_per_hour"],
                "wall_hours_per_100_video_hours": 100.0 / timing["video_hours_per_hour"],
            },
            "audit_status": audit["status"],
            "audit_errors": audit["errors"],
            "forbidden_sensitive_fields": audit["forbidden_sensitive_fields"],
        }

    aggregates = {}
    for mode in ("cold", "warm"):
        selected = [cases[f"human-{mode}"], cases[f"robot-{mode}"]]
        episodes = sum(item["episodes"] for item in selected)
        video_seconds = sum(item["video_seconds"] for item in selected)
        parallel_wall = max(item["stages_seconds"]["total_before_audit"] for item in selected)
        sequential_wall = sum(item["stages_seconds"]["total_before_audit"] for item in selected)
        aggregates[mode] = {
            "two_gpu_parallel": {
                "wall_seconds": parallel_wall,
                "episodes_per_hour": episodes * 3600.0 / parallel_wall,
                "video_hours_per_hour": video_seconds / parallel_wall,
            },
            "single_worker_sequential": {
                "wall_seconds": sequential_wall,
                "episodes_per_hour": episodes * 3600.0 / sequential_wall,
                "video_hours_per_hour": video_seconds / sequential_wall,
            },
            "api_calls": sum(item["api_calls"] for item in selected),
        }

    checkpoint_state = json.loads(
        (args.checkpoint_dir / "download-state.json").read_text(encoding="utf-8")
    )
    checkpoint_bytes = (args.checkpoint_dir / "model.safetensors").stat().st_size
    downloaded = int(checkpoint_state.get("downloaded_bytes", 0))
    report = {
        "schema": "robot-data-studio.live-quality-benchmark.v1",
        "scope": {
            "human_episodes": ["1006561", "1006564", "1006570"],
            "robot_episodes": ["0", "1", "2"],
            "heldout_opened": False,
            "full_scan_run": False,
        },
        "one_time": {
            "checkpoint_bytes": checkpoint_bytes,
            "checkpoint_sha256": checkpoint_state.get("sha256"),
            "recovery_remote_bytes": downloaded,
            "reused_verified_bytes": checkpoint_bytes - downloaded,
            "synthetic_first_load_forward_seconds": 32.299,
            "note": "remote transfer wall time was not retained across the initial Xet-ETag/LFS-SHA fail-closed retry",
        },
        "cases": cases,
        "aggregates": aggregates,
        "extrapolation_contract": (
            "Parametric per-1000-episode and per-100-video-hour projections only; "
            "the unauthorized full corpus was not enumerated."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"status": "passed", "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

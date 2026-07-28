#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CORE_IDS = ("S1", "S2", "S3", "S4", "S5", "C1", "C2", "C3")
FORBIDDEN_KEYS = {"api_key", "authorization", "secret", "access_token", "hf_token"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _forbidden_key_paths(value: Any, prefix: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}"
            if str(key).casefold() in FORBIDDEN_KEYS:
                found.append(child)
            found.extend(_forbidden_key_paths(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_forbidden_key_paths(item, f"{prefix}[{index}]"))
    return found


def audit(root: Path, expected_episodes: list[str], *, require_c2_real: bool = False) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = [str(item["episode_id"]) for item in manifest["episodes"]]
    errors: list[str] = []
    if actual != expected_episodes:
        errors.append(f"episode mismatch: expected {expected_episodes}, got {actual}")
    if manifest.get("processed_episode_count") != len(expected_episodes):
        errors.append("processed episode count mismatch")
    if manifest.get("official_loader_reload") != "verified":
        errors.append("official loader reload is not verified")

    episode_rows: list[dict[str, Any]] = []
    all_documents: list[Any] = [manifest]
    for entry in manifest["episodes"]:
        path = root / entry["sidecar"]
        if _sha256(path) != entry["sidecar_sha256"]:
            errors.append(f"sidecar hash mismatch: {entry['episode_id']}")
        sidecar = json.loads(path.read_text(encoding="utf-8"))
        all_documents.append(sidecar)
        core = [item for item in sidecar["metric_evidence"] if item["group"] == "qwen_core"]
        by_id = {item["metric_id"]: item for item in core}
        if len(core) != len(CORE_IDS) or set(by_id) != set(CORE_IDS):
            errors.append(f"core contract mismatch: {entry['episode_id']}")
            continue
        evaluated = sum(by_id[metric_id]["status"] == "evaluated" for metric_id in CORE_IDS)
        if abs(float(sidecar["coverage"]) - evaluated / len(CORE_IDS)) > 1e-12:
            errors.append(f"coverage mismatch: {entry['episode_id']}")
        if by_id["C1"]["status"] != "evaluated" or by_id["C1"]["evidence"].get("provider_mode") != "real":
            errors.append(f"C1 is not real evaluated: {entry['episode_id']}")
        if by_id["C3"]["status"] != "evaluated" or by_id["C3"]["evidence"].get("provider_mode") != "deterministic_real_frames":
            errors.append(f"C3 is not real-frame evaluated: {entry['episode_id']}")
        for metric_id in ("S1", "S3", "S4", "S5"):
            if by_id[metric_id]["status"] != "not_evaluated":
                errors.append(f"{metric_id} must remain N/A: {entry['episode_id']}")
        c2 = by_id["C2"]
        if require_c2_real:
            if c2["status"] != "evaluated" or c2["evidence"].get("provider_mode") != "real_local":
                errors.append(f"C2 is not real local evaluated: {entry['episode_id']}")
            checkpoint_loaded = c2["evidence"].get("checkpoint_loaded") is True or (
                sidecar.get("live_quality_fast", {})
                .get("sam3_checkpoint", {})
                .get("checkpoint_loaded")
                is True
            )
            if not checkpoint_loaded:
                errors.append(f"C2 checkpoint load is not proven: {entry['episode_id']}")
        elif c2["status"] not in {"evaluated", "not_evaluated"}:
            errors.append(f"C2 status is invalid: {entry['episode_id']}")
        mock_flag = c2["evidence"].get("mock_counted_as_real")
        if mock_flag is True or (mock_flag is None and c2["evidence"].get("provider_mode") != "real_local"):
            errors.append(f"C2 mock status is ambiguous: {entry['episode_id']}")
        episode_rows.append(
            {
                "episode_id": str(entry["episode_id"]),
                "coverage": sidecar["coverage"],
                "qwen_core_score": sidecar["qwen_core_score"],
                "final_decision": sidecar["final_decision"],
                "c1_score": by_id["C1"]["score"],
                "c1_state": by_id["C1"]["evidence"].get("overall_state"),
                "c3_score": by_id["C3"]["score"],
                "c3_issue_count": by_id["C3"]["evidence"].get("issue_count"),
            }
        )

    non_json = [str(path.relative_to(root)) for path in root.rglob("*") if path.is_file() and path.suffix != ".json"]
    if non_json:
        errors.append(f"non-JSON live artifacts found: {non_json}")
    forbidden = [path for document in all_documents for path in _forbidden_key_paths(document)]
    if forbidden:
        errors.append(f"forbidden sensitive fields found: {forbidden}")
    return {
        "schema": "robot-data-studio.live-quality-audit.v1",
        "status": "passed" if not errors else "failed",
        "manifest_sha256": _sha256(manifest_path),
        "episode_count": len(episode_rows),
        "episodes": episode_rows,
        "sam3": manifest.get("sam3"),
        "non_json_files": non_json,
        "forbidden_sensitive_fields": forbidden,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--episode", action="append", required=True)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--require-c2-real", action="store_true")
    args = parser.parse_args()
    result = audit(
        args.root.resolve(),
        [str(value) for value in args.episode],
        require_c2_real=args.require_c2_real,
    )
    if args.write:
        output = args.root.resolve() / "audit.json"
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

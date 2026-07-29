#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DENIED = ("val", "validation", "test", "eval")
MEDIA_SUFFIXES = {".mp4", ".avi", ".mov", ".png", ".jpg", ".jpeg"}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(values: list[Any] | None) -> bool:
    return values is None or all(math.isfinite(float(value)) for value in values)


def _denied_uri(uri: str) -> bool:
    return any(token in part.casefold() for part in Path(uri).parts for token in DENIED)


def audit(root: Path) -> dict[str, Any]:
    promotion = _read(root / "lerobot_v3" / "promotion_manifest.json")
    final = _read(root / "lerobot_v3" / "final_accepted_manifest.json")
    expected = {row["episode_id"]: row for row in promotion["episodes"]}
    episodes = []
    errors = []
    for episode_id, metadata in expected.items():
        path = root / "lerobot_v3" / metadata["data_path"]
        count = 0
        previous_timestamp = None
        observation_joint_valid = 0
        action_joint_valid = 0
        eef_valid = 0
        eef_target_valid = 0
        camera_valid = 0
        media_reference_count = 0
        names = None
        with path.open(encoding="utf-8") as stream:
            header = json.loads(next(stream))
            if header.get("format") != "lerobot_v3_dual_action_reference_v0.2":
                errors.append(f"{episode_id}: unexpected format")
            for line in stream:
                row = json.loads(line)
                count += 1
                timestamp = float(row["timestamp"])
                if previous_timestamp is not None and timestamp <= previous_timestamp:
                    errors.append(f"{episode_id}: non-monotonic timestamp at {count - 1}")
                previous_timestamp = timestamp
                observation = row["observation"]
                action = row["action"]
                joint = observation["joint_position"]
                target = action["joint_target"]
                current_names = (joint["names"], target["names"])
                if names is None:
                    names = current_names
                elif names != current_names:
                    errors.append(f"{episode_id}: joint name order changed")
                if not _finite(joint["values"]) or not _finite(target["values"]):
                    errors.append(f"{episode_id}: non-finite joint value")
                observation_joint_valid += int(joint["valid"])
                action_joint_valid += int(target["valid"])
                for key, counter in (
                    ("eef_pose", "eef"),
                    ("camera_pose", "camera"),
                ):
                    modality = observation[key]
                    for pose in modality["poses"].values():
                        if pose["valid"] and not _finite(pose["matrix"]):
                            errors.append(f"{episode_id}: non-finite {counter} pose")
                        if not pose["valid"] and pose["matrix"] is not None:
                            errors.append(f"{episode_id}: invalid {counter} pose carries values")
                        if pose["valid"]:
                            if counter == "eef":
                                eef_valid += 1
                            else:
                                camera_valid += 1
                for pose in action["eef_pose_target"]["poses"].values():
                    if pose["valid"]:
                        eef_target_valid += 1
                    elif pose["matrix"] is not None:
                        errors.append(f"{episode_id}: invalid EEF target carries values")
                if not row.get("language") and not row.get("language_missing"):
                    errors.append(f"{episode_id}: language is not explicitly represented")
                for reference in observation["media_references"].values():
                    media_reference_count += 1
                    uri = reference["uri"]
                    if _denied_uri(uri):
                        errors.append(f"{episode_id}: denied media URI")
                    if not Path(uri).is_file():
                        errors.append(f"{episode_id}: missing referenced media")
                    if reference["frame_index"] != row["source_frame_index"]:
                        errors.append(f"{episode_id}: media/source frame mismatch")
                    if reference.get("episode_time") != row["timestamp"]:
                        errors.append(f"{episode_id}: episode_time is not row timestamp")
                    expected_file_time = (
                        float(header["provenance"].get("from_timestamp", 0.0))
                        + float(reference["episode_time"])
                    )
                    if abs(float(reference.get("file_time")) - expected_file_time) > 1e-9:
                        errors.append(f"{episode_id}: file_time mapping mismatch")
        if count != metadata["frame_count"]:
            errors.append(f"{episode_id}: promotion frame count mismatch")
        episodes.append(
            {
                "episode_id": episode_id,
                "frame_count": count,
                "observation_joint_valid_frames": observation_joint_valid,
                "action_joint_valid_frames": action_joint_valid,
                "eef_valid_pose_count": eef_valid,
                "eef_target_valid_pose_count": eef_target_valid,
                "camera_valid_pose_count": camera_valid,
                "media_reference_count": media_reference_count,
                "round_trip": metadata.get("round_trip"),
                "round_trip_hashes": metadata.get("round_trip_hashes"),
                "jsonl_bytes": path.stat().st_size,
            }
        )
    copied_media = [
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.suffix.casefold() in MEDIA_SUFFIXES
    ]
    if copied_media:
        errors.append("output contains copied media")
    accepted_ids = [row["episode_id"] for row in final["episodes"]]
    if any(episode_id not in expected for episode_id in accepted_ids):
        errors.append("final manifest contains an unpromoted episode")
    return {
        "schema": "robot-data-studio.dual-action-round-trip-audit.v1",
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "episode_count": len(episodes),
        "accepted_episode_count": len(accepted_ids),
        "copied_media_files": copied_media,
        "episodes": episodes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    report = audit(args.root)
    if args.write:
        (args.root / "audit_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())

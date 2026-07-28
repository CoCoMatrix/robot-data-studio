#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
from lerobot.datasets.lerobot_dataset import LeRobotDataset


REQUIRED_FEATURES = {
    "observation.state",
    "observation.state.validity",
    "action",
    "action.validity",
    "observation.eef_pose_absolute",
    "action.eef_pose_absolute",
    "observation.eef_pose_absolute.validity",
    "action.eef_pose_absolute.validity",
    "observation.camera0_pose_absolute",
    "observation.camera0_pose_absolute.validity",
    "source.timestamp",
    "source.frame_index",
    "source.validity",
    "source.quality_hand",
    "source.quality_hand.validity",
    "observation.images.camera0",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def values(value):
    return value if isinstance(value, list) else [value]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    info = json.loads((root / "meta" / "info.json").read_text())
    missing = sorted(REQUIRED_FEATURES - set(info["features"]))
    if missing:
        raise ValueError(f"official features missing: {missing}")
    dataset = LeRobotDataset(
        args.repo_id, root=root, download_videos=False, video_backend="pyav"
    )

    episode_rows = []
    for path in sorted((root / "meta" / "episodes").rglob("*.parquet")):
        for row in pq.read_table(path).to_pylist():
            episode_rows.append(
                {
                    key: row[key]
                    for key in (
                        "episode_index",
                        "length",
                        "dataset_from_index",
                        "dataset_to_index",
                        "videos/observation.images.camera0/chunk_index",
                        "videos/observation.images.camera0/file_index",
                        "videos/observation.images.camera0/from_timestamp",
                        "videos/observation.images.camera0/to_timestamp",
                    )
                }
            )
    episode_rows.sort(key=lambda row: row["episode_index"])
    if len(episode_rows) != args.expected_episodes:
        raise ValueError("episode metadata count mismatch")
    if sum(row["length"] for row in episode_rows) != len(dataset):
        raise ValueError("episode lengths do not sum to official loader length")

    episode_counts: Counter[int] = Counter()
    validity_true = Counter()
    source_timestamp_max_error = 0.0
    previous_global_index = -1
    columns = [
        "episode_index", "frame_index", "index", "timestamp",
        "source.timestamp", "source.frame_index",
        "observation.state.validity", "action.validity",
        "observation.eef_pose_absolute.validity",
        "action.eef_pose_absolute.validity",
        "observation.camera0_pose_absolute.validity",
        "source.quality_hand.validity",
    ]
    for path in sorted((root / "data").rglob("*.parquet")):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=256, columns=columns):
            for row in batch.to_pylist():
                episode = int(row["episode_index"])
                frame = int(row["frame_index"])
                index = int(row["index"])
                if index != previous_global_index + 1:
                    raise ValueError("global indexes are not contiguous")
                previous_global_index = index
                episode_counts[episode] += 1
                expected_time = frame / info["fps"]
                if abs(float(row["timestamp"]) - expected_time) > 1e-4:
                    raise ValueError("official timestamp is not frame_index/fps")
                source_timestamp_max_error = max(
                    source_timestamp_max_error,
                    abs(float(values(row["source.timestamp"])[0]) - expected_time),
                )
                for key in columns[6:]:
                    validity_true[key] += sum(bool(value) for value in values(row[key]))
    if dict(episode_counts) != {
        int(row["episode_index"]): int(row["length"]) for row in episode_rows
    }:
        raise ValueError("Parquet episode boundary mismatch")

    video_audit = []
    pts_by_file: dict[tuple[int, int], list[float]] = {}
    for path in sorted((root / "videos" / "observation.images.camera0").rglob("*.mp4")):
        chunk = int(path.parent.name.removeprefix("chunk-"))
        file_index = int(path.stem.removeprefix("file-"))
        container = av.open(str(path))
        stream = container.streams.video[0]
        pts = [float(frame.pts * frame.time_base) for frame in container.decode(stream) if frame.pts is not None]
        width, height = stream.codec_context.width, stream.codec_context.height
        container.close()
        if any(now <= previous for previous, now in zip(pts, pts[1:])):
            raise ValueError("official video PTS is not strictly monotonic")
        pts_by_file[(chunk, file_index)] = pts
        video_audit.append(
            {
                "path": str(path.relative_to(root)),
                "frames": len(pts),
                "width": width,
                "height": height,
                "first_pts": pts[0],
                "last_pts": pts[-1],
                "sha256": sha256(path),
            }
        )
    for row in episode_rows:
        key = (
            row["videos/observation.images.camera0/chunk_index"],
            row["videos/observation.images.camera0/file_index"],
        )
        pts = pts_by_file[key]
        start, stop = row["dataset_from_index"], row["dataset_to_index"]
        expected_start = row["videos/observation.images.camera0/from_timestamp"]
        expected_stop = row["videos/observation.images.camera0/to_timestamp"]
        if abs(pts[start] - expected_start) > 1e-6:
            raise ValueError("episode video from_timestamp does not match decoded PTS")
        if abs(pts[stop - 1] - (expected_stop - 1 / info["fps"])) > 1e-6:
            raise ValueError("episode video to_timestamp does not match decoded PTS")

    files = sorted(path for path in root.rglob("*") if path.is_file())
    report = {
        "schema": "robot-data-studio.official-v3-audit.v1",
        "status": "passed",
        "repo_id": args.repo_id,
        "official_codebase_version": info["codebase_version"],
        "loader_module": LeRobotDataset.__module__,
        "episode_count": len(episode_rows),
        "frame_count": len(dataset),
        "fps": info["fps"],
        "features": {
            name: {
                "dtype": feature["dtype"],
                "shape": feature["shape"],
                "name_count": len(feature.get("names") or []),
            }
            for name, feature in info["features"].items()
            if name in REQUIRED_FEATURES
        },
        "episodes": episode_rows,
        "validity_true_counts": dict(validity_true),
        "source_timestamp_max_error_seconds": source_timestamp_max_error,
        "videos": video_audit,
        "artifact_file_count": len(files),
        "artifact_manifest_sha256": hashlib.sha256(
            json.dumps(
                [(str(path.relative_to(root)), sha256(path)) for path in files],
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

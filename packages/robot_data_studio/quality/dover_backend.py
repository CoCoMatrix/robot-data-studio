from __future__ import annotations

import json
import hashlib
import math
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from robot_data_studio.quality.filter_models import FilterVisualQualityConfig


@dataclass(frozen=True)
class VideoSample:
    episode_index: int
    camera: str
    video_path: Path
    start_seconds: float
    duration_seconds: float


@dataclass(frozen=True)
class VideoTechnicalScore:
    episode_index: int
    camera: str
    technical_score: float | None
    error_code: str | None


def checkpoint_file_info(config: FilterVisualQualityConfig) -> dict[str, object | None]:
    path = Path(config.checkpoint_path).expanduser() if config.checkpoint_path else None
    if path is None or not path.is_file():
        return {"path": str(path) if path is not None else None, "size": None, "mtime_ns": None}
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def visual_quality_fingerprint(config: FilterVisualQualityConfig) -> str:
    payload = {
        "config": config.model_dump(mode="json"),
        "checkpoint": checkpoint_file_info(config),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_dover_batch(
    samples: list[VideoSample],
    config: FilterVisualQualityConfig,
) -> list[VideoTechnicalScore]:
    if not samples:
        return []
    python_path = Path(config.dover_python_path or "")
    checkpoint_path = Path(config.checkpoint_path or "")
    if (
        config.backend == "rules"
        or not python_path.is_file()
        or not checkpoint_path.is_file()
    ):
        return [_unavailable(sample, "model_unavailable") for sample in samples]

    runner = Path(__file__).resolve().parents[3] / "scripts" / "dover_technical_runner.py"
    with tempfile.TemporaryDirectory(prefix="rds-dover-") as temporary:
        temporary_path = Path(temporary)
        manifest = temporary_path / "manifest.json"
        output = temporary_path / "scores.json"
        manifest.write_text(
            json.dumps(
                [
                    {
                        **asdict(sample),
                        "video_path": str(sample.video_path),
                    }
                    for sample in samples
                ]
            ),
            encoding="utf-8",
        )
        command = [
            str(python_path),
            str(runner),
            "--manifest",
            str(manifest),
            "--output",
            str(output),
            "--checkpoint",
            str(checkpoint_path),
            "--batch-size",
            str(config.batch_size),
            "--device",
            config.model_device,
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            payload = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, subprocess.CalledProcessError, json.JSONDecodeError):
            return [_unavailable(sample, "runner_failed") for sample in samples]

    by_key = {
        (int(item["episode_index"]), str(item["camera"])): item
        for item in payload
        if isinstance(item, dict) and "episode_index" in item and "camera" in item
    }
    results = []
    for sample in samples:
        item = by_key.get((sample.episode_index, sample.camera))
        if item is None or item.get("technical_score") is None:
            results.append(_unavailable(sample, str((item or {}).get("error_code") or "runner_failed")))
            continue
        try:
            score = float(item["technical_score"])
        except (TypeError, ValueError, OverflowError):
            results.append(_unavailable(sample, "invalid_score"))
            continue
        if not math.isfinite(score):
            results.append(_unavailable(sample, "invalid_score"))
            continue
        score = max(0.0, min(1.0, score))
        results.append(VideoTechnicalScore(sample.episode_index, sample.camera, score, None))
    return results


def camera_percentiles(
    scores: list[VideoTechnicalScore],
    *,
    min_cohort_size: int,
) -> dict[tuple[int, str], tuple[float | None, int]]:
    by_camera: dict[str, list[VideoTechnicalScore]] = {}
    for score in scores:
        if score.technical_score is not None:
            by_camera.setdefault(score.camera, []).append(score)

    ranked: dict[tuple[int, str], tuple[float | None, int]] = {}
    for camera, camera_scores in by_camera.items():
        cohort_size = len(camera_scores)
        ordered_values = sorted(float(item.technical_score) for item in camera_scores)
        for item in camera_scores:
            value = float(item.technical_score)
            positions = [index + 1 for index, candidate in enumerate(ordered_values) if candidate == value]
            percentile = sum(positions) / len(positions) / cohort_size
            ranked[(item.episode_index, camera)] = (
                percentile if cohort_size >= min_cohort_size else None,
                cohort_size,
            )
    return ranked


def _unavailable(sample: VideoSample, code: str) -> VideoTechnicalScore:
    return VideoTechnicalScore(sample.episode_index, sample.camera, None, code)

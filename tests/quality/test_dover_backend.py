from __future__ import annotations

import json
from pathlib import Path

from robot_data_studio.quality.dover_backend import (
    VideoSample,
    VideoTechnicalScore,
    camera_percentiles,
    run_dover_batch,
    visual_quality_fingerprint,
)
from robot_data_studio.quality.filter_models import FilterVisualQualityConfig


def test_runner_is_invoked_once_for_a_batch(tmp_path: Path, monkeypatch) -> None:
    python = tmp_path / "python"
    checkpoint = tmp_path / "mobile.pth"
    python.write_text("")
    checkpoint.write_text("")
    calls: list[list[str]] = []

    def fake_run(command, *, check, capture_output, text):
        calls.append(command)
        manifest = Path(command[command.index("--manifest") + 1])
        output = Path(command[command.index("--output") + 1])
        items = json.loads(manifest.read_text())
        output.write_text(json.dumps([
            {"episode_index": item["episode_index"], "camera": item["camera"],
             "technical_score": 0.75, "error_code": None}
            for item in items
        ]))

        class Completed:
            stderr = ""

        return Completed()

    monkeypatch.setattr("robot_data_studio.quality.dover_backend.subprocess.run", fake_run)
    config = FilterVisualQualityConfig(
        backend="dover_mobile",
        dover_python_path=str(python),
        checkpoint_path=str(checkpoint),
    )
    samples = [VideoSample(0, "cam_high", Path("video.mp4"), 1.0, 2.0)]

    results = run_dover_batch(samples, config)

    assert len(calls) == 1
    assert results == [VideoTechnicalScore(0, "cam_high", 0.75, None)]


def test_missing_runner_configuration_falls_back_without_subprocess() -> None:
    samples = [VideoSample(0, "cam_high", Path("video.mp4"), 0.0, 2.0)]

    results = run_dover_batch(samples, FilterVisualQualityConfig())

    assert results == [VideoTechnicalScore(0, "cam_high", None, "model_unavailable")]


def test_malformed_runner_score_degrades_only_that_camera(tmp_path: Path, monkeypatch) -> None:
    python = tmp_path / "python"
    checkpoint = tmp_path / "mobile.pth"
    python.write_text("")
    checkpoint.write_text("")

    def fake_run(command, **_kwargs):
        output = Path(command[command.index("--output") + 1])
        output.write_text(json.dumps([
            {"episode_index": 0, "camera": "bad", "technical_score": "not-a-score"},
            {"episode_index": 0, "camera": "good", "technical_score": 0.8},
        ]))

    monkeypatch.setattr("robot_data_studio.quality.dover_backend.subprocess.run", fake_run)
    config = FilterVisualQualityConfig(
        backend="dover_mobile",
        dover_python_path=str(python),
        checkpoint_path=str(checkpoint),
    )

    results = run_dover_batch(
        [
            VideoSample(0, "bad", Path("bad.mp4"), 0.0, 1.0),
            VideoSample(0, "good", Path("good.mp4"), 0.0, 1.0),
        ],
        config,
    )

    assert results == [
        VideoTechnicalScore(0, "bad", None, "invalid_score"),
        VideoTechnicalScore(0, "good", 0.8, None),
    ]


def test_camera_percentiles_are_isolated_and_average_ties() -> None:
    scores = [
        VideoTechnicalScore(0, "cam_high", 0.1, None),
        VideoTechnicalScore(1, "cam_high", 0.1, None),
        VideoTechnicalScore(2, "cam_high", 0.9, None),
        VideoTechnicalScore(0, "cam_wrist", 0.2, None),
    ]

    ranked = camera_percentiles(scores, min_cohort_size=3)

    assert ranked[(0, "cam_high")] == (0.5, 3)
    assert ranked[(1, "cam_high")] == (0.5, 3)
    assert ranked[(2, "cam_high")] == (1.0, 3)
    assert ranked[(0, "cam_wrist")] == (None, 1)


def test_visual_quality_fingerprint_tracks_config_and_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "dover.pth"
    checkpoint.write_bytes(b"one")
    config = FilterVisualQualityConfig(checkpoint_path=str(checkpoint))

    first = visual_quality_fingerprint(config)
    checkpoint.write_bytes(b"different-size")
    second = visual_quality_fingerprint(config)
    third = visual_quality_fingerprint(config.model_copy(update={"sample_fps": 3.0}))

    assert first != second
    assert second != third

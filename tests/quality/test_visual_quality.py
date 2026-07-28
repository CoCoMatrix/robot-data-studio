from __future__ import annotations

import numpy as np
import pytest

from robot_data_studio.quality.filter_models import FilterVisualQualityConfig
from robot_data_studio.quality.visual_quality import (
    VisualQualityRow,
    aggregate_visual_quality_incidents,
    analyze_sampled_frames,
    robot_motion_by_sample,
    sample_video_frames,
)


def _flat(value: int) -> np.ndarray:
    return np.full((24, 32, 3), value, dtype=np.uint8)


def _checkerboard() -> np.ndarray:
    pattern = (np.indices((24, 32)).sum(axis=0) % 2 * 255).astype(np.uint8)
    return np.repeat(pattern[:, :, None], 3, axis=2)


def test_visual_quality_parallel_decode_config_defaults_and_bounds() -> None:
    config = FilterVisualQualityConfig()
    assert config.max_parallel_video_decodes == 4
    assert config.backend == "auto"
    assert config.batch_size == 8
    assert config.review_percentile == 0.05
    with pytest.raises(ValueError):
        FilterVisualQualityConfig(max_parallel_video_decodes=0)
    with pytest.raises(ValueError):
        FilterVisualQualityConfig(max_parallel_video_decodes=9)


def test_clear_high_contrast_frames_pass_visual_quality() -> None:
    result = analyze_sampled_frames(
        [_checkerboard(), np.roll(_checkerboard(), 1, axis=1)],
        FilterVisualQualityConfig(),
        has_motion=True,
    )

    assert result.issue_count == 0
    assert result.score == 1
    assert result.rows == []


def test_only_physical_black_and_white_failures_are_flagged() -> None:
    result = analyze_sampled_frames(
        [_flat(128), _flat(3), _flat(252), _flat(40)],
        FilterVisualQualityConfig(),
        has_motion=False,
    )

    issues = [row.issue for row in result.rows]

    assert issues == ["black_screen", "white_screen"]
    assert result.score == 1


def test_global_bright_washout_is_diagnostic_only() -> None:
    frame = np.full((24, 32, 3), 165, dtype=np.uint8)
    frame[:12] = 210

    result = analyze_sampled_frames(
        [frame],
        FilterVisualQualityConfig(),
        has_motion=False,
    )

    assert result.rows == []
    assert result.series["brightness"]


def test_global_dark_exposure_is_diagnostic_only() -> None:
    frame = np.full((24, 32, 3), 70, dtype=np.uint8)

    result = analyze_sampled_frames(
        [frame],
        FilterVisualQualityConfig(),
        has_motion=False,
    )

    assert result.rows == []
    assert result.series["brightness"]


def test_repeated_frames_with_motion_are_flagged_as_freeze() -> None:
    frame = _checkerboard()

    result = analyze_sampled_frames(
        [frame.copy() for _ in range(5)],
        FilterVisualQualityConfig(freeze_min_run=3),
        has_motion=True,
        frame_indexes=[0, 25, 50, 75, 100],
        timestamps=[0.0, 0.5, 1.0, 1.5, 2.0],
    )

    freeze = next(row for row in result.rows if row.issue == "freeze")
    assert freeze.frame == 0
    assert freeze.end_frame == 100
    assert freeze.timestamp == 0.0
    assert freeze.end_timestamp == 2.0
    assert result.critical_issue


def test_repeated_frames_without_motion_do_not_trigger_freeze() -> None:
    frame = _checkerboard()

    result = analyze_sampled_frames(
        [frame.copy() for _ in range(5)],
        FilterVisualQualityConfig(freeze_min_run=3),
        has_motion=False,
    )

    assert not any(row.issue == "freeze" for row in result.rows)


def test_continuous_black_screen_becomes_hard_failure_after_two_seconds() -> None:
    result = analyze_sampled_frames(
        [_flat(0) for _ in range(4)],
        FilterVisualQualityConfig(),
        has_motion=False,
        timestamps=[0.0, 0.5, 1.0, 1.5],
        frame_indexes=[0, 10, 20, 30],
    )
    incident = next(row for row in result.rows if row.issue == "black_screen")
    assert (incident.frame, incident.end_frame) == (0, 30)
    assert incident.severity == "critical"
    assert incident.hard_failure is True
    assert result.critical_issue is True
    assert result.score == 0


def test_short_white_screen_is_warning_only() -> None:
    result = analyze_sampled_frames(
        [_flat(255), _flat(255), _checkerboard()],
        FilterVisualQualityConfig(),
        has_motion=False,
        timestamps=[0.0, 0.5, 1.0],
    )
    incident = next(row for row in result.rows if row.issue == "white_screen")
    assert incident.severity == "warning"
    assert incident.hard_failure is False
    assert result.critical_issue is False
    assert result.score == 1


def test_freeze_requires_robot_motion_in_the_same_time_window() -> None:
    frame = _checkerboard()
    result = analyze_sampled_frames(
        [frame.copy() for _ in range(6)],
        FilterVisualQualityConfig(freeze_duration_seconds=2.0),
        has_motion=False,
        robot_motion=[0.0, 0.0, 0.0, 0.01, 0.01, 0.01],
        timestamps=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
    )
    assert not any(row.issue == "freeze" and row.hard_failure for row in result.rows)


def test_blur_and_low_contrast_are_diagnostics_not_issues() -> None:
    result = analyze_sampled_frames(
        [_flat(128)],
        FilterVisualQualityConfig(),
        has_motion=False,
        camera="cam_high",
        timestamps=[0.0],
    )
    assert result.rows == []
    assert result.score == 1
    assert result.metrics["cam_high"][0].sharpness == 0


def test_sampled_rows_and_metrics_use_episode_frame_indexes() -> None:
    result = analyze_sampled_frames(
        [_flat(128), _flat(128)],
        FilterVisualQualityConfig(
            dark_mean_threshold=0,
            dark_global_mean_threshold=0,
            dark_global_p75_threshold=0,
            bright_mean_threshold=255,
            bright_global_mean_threshold=255,
            bright_global_p75_threshold=255,
            low_contrast_std_threshold=0,
        ),
        has_motion=False,
        camera="cam_high",
        frame_indexes=[25, 50],
        timestamps=[0.5, 1.0],
    )

    assert result.rows == []
    assert [metric.frame for metric in result.metrics["cam_high"]] == [25, 50]
    assert [metric.timestamp for metric in result.metrics["cam_high"]] == [0.5, 1.0]


def test_contiguous_rows_are_grouped_with_start_worst_and_end_evidence() -> None:
    rows = [
        VisualQualityRow(0, 0.0, "cam_high", "blur", 12.0, 18.0, metric_value=12.0),
        VisualQualityRow(25, 0.5, "cam_high", "blur", 4.0, 18.0, metric_value=4.0),
        VisualQualityRow(50, 1.0, "cam_high", "blur", 10.0, 18.0, metric_value=10.0),
        VisualQualityRow(100, 2.0, "cam_high", "blur", 8.0, 18.0, metric_value=8.0),
        VisualQualityRow(25, 0.5, "cam_wrist", "blur", 5.0, 18.0, metric_value=5.0),
    ]

    incidents = aggregate_visual_quality_incidents(rows, sample_interval_seconds=0.5)

    assert len(incidents) == 3
    first = incidents[0]
    assert first.camera == "cam_high"
    assert first.issue == "blur"
    assert (first.start_frame, first.end_frame) == (0, 50)
    assert first.sample_count == 3
    assert first.worst_value == 4.0
    assert [frame.frame for frame in first.representative_frames] == [0, 25, 50]


def test_incident_preserves_hard_failure_severity() -> None:
    rows = [VisualQualityRow(0, 0.0, "cam_high", "black_screen", 1.0, 0.8,
        end_frame=40, end_timestamp=2.0, severity="critical", hard_failure=True)]
    incident = aggregate_visual_quality_incidents(rows, sample_interval_seconds=0.5)[0]
    assert incident.severity == "critical"
    assert incident.hard_failure is True


def test_video_sampling_is_limited_to_episode_time_bounds(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "shared.mp4"
    video.write_bytes(b"video")
    captured: list[list[str]] = []

    class Stream:
        def __init__(self):
            self.remaining = bytes(160 * 120 * 3)

        def read(self, size):
            chunk, self.remaining = self.remaining[:size], self.remaining[size:]
            return chunk

    class Process:
        stdout = Stream()

        @staticmethod
        def wait():
            return 0

    def fake_popen(command, **_kwargs):
        captured.append(command)
        return Process()

    monkeypatch.setattr("robot_data_studio.quality.visual_quality._ffmpeg_path", lambda: "ffmpeg")
    monkeypatch.setattr("robot_data_studio.quality.visual_quality.subprocess.Popen", fake_popen)

    sample_video_frames(
        video,
        FilterVisualQualityConfig(max_frames_per_video=1),
        start_seconds=12.5,
        duration_seconds=2.25,
    )

    command = captured[0]
    assert command[command.index("-ss") + 1] == "12.5"
    assert command[command.index("-t") + 1] == "2.25"
    assert command.index("-ss") < command.index("-i")


def test_video_sampling_does_not_truncate_the_episode() -> None:
    import inspect
    source = inspect.getsource(sample_video_frames)
    assert '"-frames:v"' not in source
    assert "max_frames_per_video" not in source


def test_robot_motion_is_aligned_to_each_visual_sample_window() -> None:
    states = np.asarray([[0.0], [0.0], [0.0], [1.0], [1.0]])
    actions = np.zeros_like(states)
    motion = robot_motion_by_sample(
        states,
        actions,
        robot_timestamps=[0.0, 0.5, 1.0, 1.5, 2.0],
        sample_timestamps=[0.0, 1.0, 2.0],
    )
    assert motion == [0.0, 0.0, 1.0]

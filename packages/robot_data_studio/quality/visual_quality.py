from __future__ import annotations

import math
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from robot_data_studio.quality.filter_models import FilterVisualQualityConfig, VisualCameraScore


@dataclass(frozen=True)
class VisualQualityRow:
    frame: int | None
    timestamp: float | None
    camera: str
    issue: str
    value: float | str
    threshold: float | str
    end_frame: int | None = None
    end_timestamp: float | None = None
    metric_value: float | None = None
    severity: str = "warning"
    hard_failure: bool = False


@dataclass(frozen=True)
class VisualQualityMetric:
    frame: int
    timestamp: float
    sharpness: float | None
    brightness: float | None
    contrast: float | None


@dataclass(frozen=True)
class VisualQualityEvidenceFrame:
    frame: int
    timestamp: float


@dataclass(frozen=True)
class VisualQualityIncident:
    camera: str
    issue: str
    start_frame: int
    end_frame: int
    start_timestamp: float
    end_timestamp: float
    sample_count: int
    worst_value: float | str
    threshold: float | str
    representative_frames: list[VisualQualityEvidenceFrame]
    severity: str = "warning"
    hard_failure: bool = False


@dataclass
class VisualQualityResult:
    frame_count: int
    issue_count: int
    score: float
    rows: list[VisualQualityRow] = field(default_factory=list)
    series: dict[str, list[float]] = field(default_factory=dict)
    metrics: dict[str, list[VisualQualityMetric]] = field(default_factory=dict)
    issue_counts: dict[str, int] = field(default_factory=dict)
    critical_issue: bool = False
    camera_count: int = 0
    technical_score: float | None = None
    percentile: float | None = None
    camera_scores: list[VisualCameraScore] = field(default_factory=list)
    hard_failure_reasons: list[str] = field(default_factory=list)
    requested_backend: str = "auto"
    effective_backend: str = "rules"
    model_status: str = "unavailable"


def analyze_sampled_frames(
    frames: list[np.ndarray],
    config: FilterVisualQualityConfig,
    *,
    has_motion: bool,
    robot_motion: list[float] | None = None,
    camera: str = "",
    timestamps: list[float] | None = None,
    frame_indexes: list[int] | None = None,
) -> VisualQualityResult:
    rows: list[VisualQualityRow] = []
    sharpness: list[float] = []
    brightness: list[float] = []
    contrast: list[float] = []
    metrics: list[VisualQualityMetric] = []
    frame_count = len(frames)

    for index, frame in enumerate(frames):
        gray = _grayscale(frame)
        timestamp = timestamps[index] if timestamps and index < len(timestamps) else None
        frame_index = (
            frame_indexes[index]
            if frame_indexes and index < len(frame_indexes)
            else index
        )
        if gray is None:
            rows.append(
                VisualQualityRow(
                    frame_index,
                    timestamp,
                    camera,
                    "bad_frame",
                    "invalid",
                    "rgb uint8",
                    severity="critical",
                    hard_failure=True,
                )
            )
            if timestamp is not None:
                metrics.append(
                    VisualQualityMetric(frame_index, timestamp, None, None, None)
                )
            continue
        laplacian = _laplacian_variance(gray)
        mean = float(np.mean(gray))
        std = float(np.std(gray))
        sharpness.append(laplacian)
        brightness.append(mean)
        contrast.append(std)
        if timestamp is not None:
            metrics.append(
                VisualQualityMetric(
                    frame=frame_index,
                    timestamp=timestamp,
                    sharpness=round(laplacian, 4),
                    brightness=round(mean, 4),
                    contrast=round(std, 4),
                )
            )
    resolved_timestamps = timestamps or [index / config.sample_fps for index in range(frame_count)]
    resolved_frames = frame_indexes or list(range(frame_count))
    rows.extend(
        _screen_rows(frames, config, camera, resolved_timestamps, resolved_frames)
    )

    freeze_rows = _freeze_rows(
        frames,
        config,
        has_motion,
        camera,
        timestamps,
        frame_indexes,
        robot_motion,
    )
    rows.extend(freeze_rows)
    issue_counts = _issue_counts(rows)
    issue_count = len(rows)
    hard_failure = any(row.hard_failure for row in rows)
    score = 0.0 if hard_failure else 1.0
    return VisualQualityResult(
        frame_count=frame_count,
        issue_count=issue_count,
        score=score,
        rows=rows,
        series={
            "sharpness": _sample_series(sharpness),
            "brightness": _sample_series(brightness),
            "contrast": _sample_series(contrast),
        },
        metrics={camera: metrics} if camera else {},
        issue_counts=issue_counts,
        critical_issue=hard_failure,
    )


def sample_video_frames(
    video_path: Path,
    config: FilterVisualQualityConfig,
    *,
    start_seconds: float = 0.0,
    duration_seconds: float | None = None,
) -> list[np.ndarray]:
    ffmpeg = _ffmpeg_path()
    if not ffmpeg:
        return []
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    if start_seconds > 0:
        command.extend(["-ss", str(start_seconds)])
    else:
        command.extend(["-ss", "0.0"])
    command.extend(["-i", str(video_path)])
    if duration_seconds is not None:
        command.extend(["-t", str(duration_seconds)])
    command.extend(
        [
            "-vf",
            f"fps={config.sample_fps},scale={config.sample_width}:{config.sample_height}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
    )
    frame_size = config.sample_width * config.sample_height * 3
    if frame_size <= 0:
        return []
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return []
    frames: list[np.ndarray] = []
    assert process.stdout is not None
    while True:
        output = process.stdout.read(frame_size)
        if len(output) < frame_size:
            break
        frame = np.frombuffer(output, dtype=np.uint8)
        frames.append(frame.reshape((config.sample_height, config.sample_width, 3)).copy())
    return frames if process.wait() == 0 else []


def extract_video_frame_jpeg(video_path: Path, timestamp: float, width: int) -> bytes:
    ffmpeg = _ffmpeg_path()
    if not ffmpeg:
        raise RuntimeError("FFmpeg is not available")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-ss",
        str(max(0.0, timestamp)),
        "-frames:v",
        "1",
        "-vf",
        f"scale=min({width}\\,iw):-2",
        "-q:v",
        "2",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "pipe:1",
    ]
    try:
        output = subprocess.run(command, check=True, capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("Video frame could not be decoded") from error
    if not output:
        raise RuntimeError("Video frame could not be decoded")
    return output


def has_signal_motion(states: np.ndarray, actions: np.ndarray) -> bool:
    motion_values = []
    for values in (states, actions):
        if values.ndim != 2 or values.shape[0] < 2 or not np.all(np.isfinite(values)):
            continue
        diffs = np.diff(values, axis=0)
        if diffs.size:
            motion_values.append(float(np.nanmax(np.abs(diffs))))
    return bool(motion_values and max(motion_values) > 1e-6)


def robot_motion_by_sample(
    states: np.ndarray,
    actions: np.ndarray,
    *,
    robot_timestamps: list[float],
    sample_timestamps: list[float],
) -> list[float]:
    """Return maximum normalized state/action delta in each visual sample window."""
    if not sample_timestamps:
        return []
    result = [0.0]
    timestamp_array = np.asarray(robot_timestamps, dtype=np.float64)
    for previous, current in zip(sample_timestamps[:-1], sample_timestamps[1:], strict=True):
        indices = np.flatnonzero((timestamp_array >= previous) & (timestamp_array <= current))
        maximum = 0.0
        for values in (states, actions):
            if values.ndim != 2 or len(indices) < 2 or values.shape[0] != len(timestamp_array):
                continue
            window = values[indices]
            if np.all(np.isfinite(window)):
                maximum = max(maximum, float(np.max(np.abs(np.diff(window, axis=0)))))
        result.append(maximum)
    return result


def _grayscale(frame: np.ndarray) -> np.ndarray | None:
    if frame.ndim != 3 or frame.shape[2] != 3:
        return None
    if not np.issubdtype(frame.dtype, np.integer) and not np.issubdtype(frame.dtype, np.floating):
        return None
    values = frame.astype(np.float64)
    if not np.all(np.isfinite(values)):
        return None
    return values[:, :, 0] * 0.299 + values[:, :, 1] * 0.587 + values[:, :, 2] * 0.114


def _laplacian_variance(gray: np.ndarray) -> float:
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    center = gray[1:-1, 1:-1] * -4.0
    laplacian = center + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    return float(np.var(laplacian)) if laplacian.size else 0.0


def _freeze_rows(
    frames: list[np.ndarray],
    config: FilterVisualQualityConfig,
    has_motion: bool,
    camera: str,
    timestamps: list[float] | None,
    frame_indexes: list[int] | None,
    robot_motion: list[float] | None,
) -> list[VisualQualityRow]:
    if len(frames) < 2:
        return []
    timestamps = timestamps or [index / config.sample_fps for index in range(len(frames))]
    frame_indexes = frame_indexes or list(range(len(frames)))
    if robot_motion is None:
        robot_motion = [
            config.robot_motion_threshold * 2 if has_motion else 0.0
            for _ in frames
        ]
    active: list[bool] = [False]
    for index in range(1, len(frames)):
        mse = _frame_mse(frames[index - 1], frames[index])
        motion = robot_motion[index] if index < len(robot_motion) else 0.0
        active.append(
            mse is not None
            and mse <= config.freeze_mse_threshold
            and motion >= config.robot_motion_threshold
        )
    rows: list[VisualQualityRow] = []
    for start, end in _true_runs(active):
        start_frame_index = max(0, start - 1)
        duration = timestamps[end] - timestamps[start_frame_index]
        hard_failure = duration + 1e-9 >= config.freeze_duration_seconds
        rows.append(
            VisualQualityRow(
                frame_indexes[start_frame_index],
                timestamps[start_frame_index],
                camera,
                "freeze",
                f"{duration:.3f}s",
                config.freeze_duration_seconds,
                end_frame=frame_indexes[end],
                end_timestamp=timestamps[end],
                metric_value=float(end - start + 1),
                severity="critical" if hard_failure else "warning",
                hard_failure=hard_failure,
            )
        )
    return rows


def _screen_rows(
    frames: list[np.ndarray],
    config: FilterVisualQualityConfig,
    camera: str,
    timestamps: list[float],
    frame_indexes: list[int],
) -> list[VisualQualityRow]:
    classifications: dict[str, list[bool]] = {"black_screen": [], "white_screen": []}
    ratios: dict[str, list[float]] = {"black_screen": [], "white_screen": []}
    for frame in frames:
        gray = _grayscale(frame)
        if gray is None:
            black_ratio = white_ratio = 0.0
        else:
            black_ratio = float(np.mean(gray < config.black_pixel_threshold))
            white_ratio = float(np.mean(gray > config.white_pixel_threshold))
        ratios["black_screen"].append(black_ratio)
        ratios["white_screen"].append(white_ratio)
        classifications["black_screen"].append(black_ratio > config.saturated_area_threshold)
        classifications["white_screen"].append(white_ratio > config.saturated_area_threshold)

    interval = 1.0 / config.sample_fps
    rows: list[VisualQualityRow] = []
    for issue, flags in classifications.items():
        for start, end in _true_runs(flags):
            duration = timestamps[end] - timestamps[start] + interval
            hard_failure = duration + 1e-9 >= config.hard_failure_duration_seconds
            worst_ratio = max(ratios[issue][start : end + 1])
            rows.append(
                VisualQualityRow(
                    frame_indexes[start],
                    timestamps[start],
                    camera,
                    issue,
                    round(worst_ratio, 4),
                    config.saturated_area_threshold,
                    end_frame=frame_indexes[end],
                    end_timestamp=timestamps[end],
                    metric_value=worst_ratio,
                    severity="critical" if hard_failure else "warning",
                    hard_failure=hard_failure,
                )
            )
    return rows


def _true_runs(flags: list[bool]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, flag in enumerate(flags):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(flags) - 1))
    return runs


def _freeze_row(
    start: int,
    end: int,
    run_length: int,
    config: FilterVisualQualityConfig,
    camera: str,
    timestamps: list[float] | None,
    frame_indexes: list[int] | None,
) -> VisualQualityRow:
    timestamp = timestamps[start] if timestamps and start < len(timestamps) else None
    end_timestamp = timestamps[end] if timestamps and end < len(timestamps) else timestamp
    frame = frame_indexes[start] if frame_indexes and start < len(frame_indexes) else start
    end_frame = frame_indexes[end] if frame_indexes and end < len(frame_indexes) else end
    return VisualQualityRow(
        frame,
        timestamp,
        camera,
        "freeze",
        f"{frame}-{end_frame} ({run_length} sampled frames)",
        config.freeze_mse_threshold,
        end_frame=end_frame,
        end_timestamp=end_timestamp,
        metric_value=float(run_length),
    )


def aggregate_visual_quality_incidents(
    rows: list[VisualQualityRow],
    *,
    sample_interval_seconds: float,
) -> list[VisualQualityIncident]:
    eligible = [
        row
        for row in rows
        if row.frame is not None and row.timestamp is not None and row.issue != "video_missing"
    ]
    eligible.sort(key=lambda row: (row.camera, row.issue, row.timestamp or 0.0))
    grouped: list[list[VisualQualityRow]] = []
    for row in eligible:
        if not grouped:
            grouped.append([row])
            continue
        previous = grouped[-1][-1]
        previous_end = previous.end_timestamp or previous.timestamp or 0.0
        is_contiguous = (
            previous.camera == row.camera
            and previous.issue == row.issue
            and (row.timestamp or 0.0)
            <= previous_end + sample_interval_seconds * 1.5 + 1e-9
        )
        if is_contiguous:
            grouped[-1].append(row)
        else:
            grouped.append([row])

    incidents = [_incident_from_rows(group) for group in grouped]
    return sorted(incidents, key=lambda incident: (incident.start_timestamp, incident.camera))


def _incident_from_rows(rows: list[VisualQualityRow]) -> VisualQualityIncident:
    first = rows[0]
    last = rows[-1]
    start_frame = int(first.frame or 0)
    end_frame = int(last.end_frame if last.end_frame is not None else last.frame or start_frame)
    start_timestamp = float(first.timestamp or 0.0)
    end_timestamp = float(
        last.end_timestamp if last.end_timestamp is not None else last.timestamp or start_timestamp
    )
    worst = _worst_row(rows)
    if first.issue == "freeze":
        middle_frame = round((start_frame + end_frame) / 2)
        middle_timestamp = (start_timestamp + end_timestamp) / 2
        representatives = [
            VisualQualityEvidenceFrame(start_frame, start_timestamp),
            VisualQualityEvidenceFrame(middle_frame, middle_timestamp),
            VisualQualityEvidenceFrame(end_frame, end_timestamp),
        ]
    else:
        representatives = [
            VisualQualityEvidenceFrame(start_frame, start_timestamp),
            VisualQualityEvidenceFrame(
                int(worst.frame or start_frame),
                float(worst.timestamp or start_timestamp),
            ),
            VisualQualityEvidenceFrame(end_frame, end_timestamp),
        ]
    representatives = list(
        {
            (item.frame, item.timestamp): item
            for item in representatives
        }.values()
    )
    return VisualQualityIncident(
        camera=first.camera,
        issue=first.issue,
        start_frame=start_frame,
        end_frame=end_frame,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        sample_count=(
            sum(max(1, int(row.metric_value or 1)) for row in rows)
            if first.issue == "freeze"
            else len(rows)
        ),
        worst_value=worst.value,
        threshold=worst.threshold,
        representative_frames=representatives,
        severity="critical" if any(row.hard_failure for row in rows) else "warning",
        hard_failure=any(row.hard_failure for row in rows),
    )


def _worst_row(rows: list[VisualQualityRow]) -> VisualQualityRow:
    numeric = [row for row in rows if row.metric_value is not None]
    if not numeric:
        return rows[0]
    if rows[0].issue in {"bright", "freeze"}:
        return max(numeric, key=lambda row: float(row.metric_value or 0.0))
    return min(numeric, key=lambda row: float(row.metric_value or 0.0))


def _frame_mse(first: np.ndarray, second: np.ndarray) -> float | None:
    if first.shape != second.shape:
        return None
    diff = first.astype(np.float64) - second.astype(np.float64)
    if not np.all(np.isfinite(diff)):
        return None
    return float(np.mean(diff * diff))


def _is_dark_frame(mean: float, p75: float, config: FilterVisualQualityConfig) -> bool:
    return mean < config.dark_mean_threshold or (
        mean <= config.dark_global_mean_threshold and p75 <= config.dark_global_p75_threshold
    )


def _is_bright_frame(mean: float, p75: float, config: FilterVisualQualityConfig) -> bool:
    return mean > config.bright_mean_threshold or (
        mean >= config.bright_global_mean_threshold and p75 >= config.bright_global_p75_threshold
    )


def _exposure_value(mean: float, p75: float) -> str:
    return f"mean={mean:.4f}, p75={p75:.4f}"


def _exposure_threshold(
    absolute_mean_threshold: float,
    global_mean_threshold: float,
    global_p75_threshold: float,
) -> str:
    return (
        f"mean={absolute_mean_threshold:g} "
        f"or global_mean={global_mean_threshold:g}, p75={global_p75_threshold:g}"
    )


def _issue_counts(rows: list[VisualQualityRow]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.issue] = counts.get(row.issue, 0) + 1
    return counts


def _sample_series(values: list[float], limit: int = 120) -> list[float]:
    if len(values) <= limit:
        return [round(value, 4) for value in values]
    step = len(values) / limit
    sampled = [values[min(math.floor(index * step), len(values) - 1)] for index in range(limit)]
    return [round(value, 4) for value in sampled]


def _ffmpeg_path() -> str | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    try:
        import imageio_ffmpeg  # type: ignore[import-not-found]
    except ImportError:
        return None
    return str(imageio_ffmpeg.get_ffmpeg_exe())

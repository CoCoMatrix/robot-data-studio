from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence

from .models import FrameSample, ResamplePoint


class EpisodeBoundaryError(ValueError):
    pass


class PtsToleranceError(ValueError):
    pass


def iter_episode_chunks(
    rows: Iterable[FrameSample],
    *,
    chunk_size: int,
) -> Iterator[list[FrameSample]]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    current_episode: str | None = None
    current: list[FrameSample] = []
    completed: set[str] = set()
    for row in rows:
        if current_episode is None:
            current_episode = row.episode_id
        elif row.episode_id != current_episode:
            if current:
                yield current
            completed.add(current_episode)
            if row.episode_id in completed:
                raise EpisodeBoundaryError("episode rows must be contiguous")
            current_episode = row.episode_id
            current = []
        current.append(row)
        if len(current) == chunk_size:
            yield current
            current = []
    if current:
        yield current


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        return 0.0
    return sum(abs(float(a) - float(b)) for a, b in zip(left, right, strict=True))


def _progress_axis(rows: list[FrameSample], mode: str, event_timestamps: Sequence[float]) -> list[float]:
    if len(rows) <= 1:
        return [0.0] * len(rows)
    if mode == "fixed_time":
        start = rows[0].timestamp
        span = rows[-1].timestamp - start
        if span <= 0:
            return [index / (len(rows) - 1) for index in range(len(rows))]
        return [(row.timestamp - start) / span for row in rows]
    if mode != "event_or_motion":
        raise ValueError(f"Unknown progress mode: {mode}")
    cumulative = [0.0]
    event_set = set(float(value) for value in event_timestamps)
    for previous, row in zip(rows, rows[1:], strict=False):
        motion = 0.0
        if previous.valid and row.valid:
            motion = _distance(previous.state, row.state) + _distance(previous.action, row.action)
        event_bonus = 1.0 if row.timestamp in event_set else 0.0
        cumulative.append(cumulative[-1] + motion + event_bonus)
    total = cumulative[-1]
    if total <= 0:
        return _progress_axis(rows, "fixed_time", ())
    return [value / total for value in cumulative]


def progress_resample(
    rows: Sequence[FrameSample],
    *,
    from_timestamp: float,
    target_points: int = 64,
    mode: str = "fixed_time",
    decoded_pts_tolerance: float = 0.05,
    event_timestamps: Sequence[float] = (),
) -> list[ResamplePoint]:
    if target_points < 1:
        raise ValueError("target_points must be positive")
    ordered = list(rows)
    if not ordered:
        return []
    episode_ids = {row.episode_id for row in ordered}
    if len(episode_ids) != 1:
        raise EpisodeBoundaryError("resampling input must contain exactly one episode")
    for row in ordered:
        file_time = from_timestamp + row.timestamp
        if row.decoded_pts is not None and abs(row.decoded_pts - file_time) > decoded_pts_tolerance:
            raise PtsToleranceError(
                f"decoded PTS {row.decoded_pts} exceeds tolerance for file_time {file_time}"
            )
    axis = _progress_axis(ordered, mode, event_timestamps)
    targets = [0.0] if target_points == 1 else [i / (target_points - 1) for i in range(target_points)]
    episode_id = ordered[0].episode_id
    points: list[ResamplePoint] = []
    for target_index, target in enumerate(targets):
        row_index = min(range(len(ordered)), key=lambda index: (abs(axis[index] - target), index))
        row = ordered[row_index]
        points.append(
            ResamplePoint(
                target_index=target_index,
                progress=target,
                episode_id=episode_id,
                episode_time=row.timestamp,
                file_time=from_timestamp + row.timestamp,
                original_frame_index=row.frame_index,
                original_timestamp=row.timestamp,
                valid=row.valid,
            )
        )
    return points

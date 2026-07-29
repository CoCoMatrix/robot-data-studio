from __future__ import annotations

import math
from collections.abc import Sequence

from .models import EpisodeRecord, Finding, FrameSample
from .profile import SignalCheckProfile, SourceProfile


def _motion(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        return math.inf
    return sum(abs(float(a) - float(b)) for a, b in zip(left, right, strict=True))


def check_t0(episode: EpisodeRecord) -> list[Finding]:
    findings: list[Finding] = []
    if episode.split != "train":
        findings.append(Finding(code="split_not_train", outcome="hard_reject", tier="T0"))
    if not episode.frames:
        findings.append(Finding(code="empty_episode", outcome="hard_reject", tier="T0"))
        return findings
    state_dims = {len(frame.state) for frame in episode.frames}
    action_dims = {len(frame.action) for frame in episode.frames}
    if len(state_dims) != 1 or len(action_dims) != 1:
        findings.append(Finding(code="dimension_mismatch", outcome="hard_reject", tier="T0"))
    finite = all(
        math.isfinite(value)
        for frame in episode.frames
        for value in [frame.timestamp, *frame.state, *frame.action]
    )
    if not finite:
        findings.append(Finding(code="non_finite", outcome="hard_reject", tier="T0"))
    return findings


def check_boundaries(
    episode: EpisodeRecord,
    config: SignalCheckProfile,
) -> list[Finding]:
    if not episode.frames:
        return []
    findings = []
    if episode.frames[0].timestamp > config.boundary_tolerance_seconds:
        findings.append(Finding(code="episode_start_truncated", outcome="review"))
    if episode.expected_duration is not None:
        end_delta = abs(episode.expected_duration - episode.frames[-1].timestamp)
        if end_delta > config.boundary_tolerance_seconds:
            findings.append(Finding(code="episode_end_truncated", outcome="review", value=end_delta))
    return findings


def check_time_signal(
    frames: Sequence[FrameSample],
    source: SourceProfile,
    config: SignalCheckProfile,
) -> list[Finding]:
    if len(frames) < 2:
        return []
    expected = 1.0 / source.fps
    tolerance = expected * config.timestamp_jitter_ratio
    findings = []
    for previous, frame in zip(frames, frames[1:], strict=False):
        delta = frame.timestamp - previous.timestamp
        if delta <= 0:
            findings.append(Finding(code="timestamp_non_monotonic", outcome="hard_reject"))
            break
        if abs(delta - expected) > tolerance:
            findings.append(Finding(code="timestamp_jitter", outcome="review", value=delta))
            break
    return findings


def check_idle_stuck(
    frames: Sequence[FrameSample],
    config: SignalCheckProfile,
) -> list[Finding]:
    eligible = []
    for previous, frame in zip(frames, frames[1:], strict=False):
        if previous.contact_hold or frame.contact_hold or not previous.valid or not frame.valid:
            continue
        motion = _motion(previous.state, frame.state) + _motion(previous.action, frame.action)
        eligible.append((frame.frame_index, motion <= config.idle_motion_threshold))
    if not eligible:
        return []
    idle_ratio = sum(is_idle for _, is_idle in eligible) / len(eligible)
    if idle_ratio <= config.max_idle_ratio:
        return []
    idle_frames = [frame_index for frame_index, is_idle in eligible if is_idle]
    return [
        Finding(
            code="idle_ratio",
            outcome="window_mask",
            start_frame=min(idle_frames),
            end_frame=max(idle_frames),
            value=idle_ratio,
            threshold=config.max_idle_ratio,
        )
    ]


def check_action_artifacts(
    frames: Sequence[FrameSample],
    source: SourceProfile,
    config: SignalCheckProfile,
) -> list[Finding]:
    valid = [frame for frame in frames if frame.valid]
    if not valid or all(not frame.action for frame in valid):
        return []
    findings: list[Finding] = []
    run_start = 0
    for index in range(1, len(valid) + 1):
        run_ended = index == len(valid) or valid[index].action != valid[run_start].action
        if not run_ended:
            continue
        if index - run_start >= config.action_repeat_frames:
            findings.append(
                Finding(
                    code="action_long_repeat",
                    outcome="window_mask",
                    start_frame=valid[run_start].frame_index,
                    end_frame=valid[index - 1].frame_index,
                )
            )
        run_start = index
    unique_ratio = len({tuple(frame.action) for frame in valid}) / len(valid)
    if unique_ratio <= config.quantization_unique_ratio:
        findings.append(
            Finding(
                code="action_quantization",
                outcome="review",
                value=unique_ratio,
                threshold=config.quantization_unique_ratio,
            )
        )
    for channel in source.channels:
        values = [frame.action[channel.action_index] for frame in valid if len(frame.action) > channel.action_index]
        saturated = any(
            (channel.action_min is not None and value <= channel.action_min + config.saturation_tolerance)
            or (channel.action_max is not None and value >= channel.action_max - config.saturation_tolerance)
            for value in values
        )
        if saturated:
            findings.append(
                Finding(code="action_saturation", outcome="review", channel=channel.name)
            )
    return findings


def _correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 3 or len(left) != len(right):
        return None
    if not all(math.isfinite(value) for value in (*left, *right)):
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum((a - left_mean) ** 2 for a in left))
    right_norm = math.sqrt(sum((b - right_mean) ** 2 for b in right))
    if left_norm == 0 or right_norm == 0:
        return None
    return numerator / (left_norm * right_norm)


def _best_alignment(
    state: Sequence[float],
    action: Sequence[float],
    *,
    expected_lag: int,
    max_lag: int,
) -> tuple[int, float] | None:
    candidates: list[tuple[int, float]] = []
    for lag in range(0, max_lag + 1):
        state_slice = state[lag:] if lag else state
        action_slice = action[:-lag] if lag else action
        correlation = _correlation(state_slice, action_slice)
        if correlation is not None:
            candidates.append((lag, correlation))
    if not candidates:
        return None
    strongest = max(abs(correlation) for _, correlation in candidates)
    numerically_tied = [
        item for item in candidates if strongest - abs(item[1]) <= 1e-9
    ]
    return min(numerically_tied, key=lambda item: (abs(item[0] - expected_lag), item[0]))


def check_state_action_alignment(
    frames: Sequence[FrameSample],
    source: SourceProfile,
) -> list[Finding]:
    valid = [frame for frame in frames if frame.valid]
    findings: list[Finding] = []
    max_lag = max(abs(source.latency_frames) + source.latency_tolerance_frames + 2, 2)
    for channel in source.channels:
        if any(
            len(frame.state) <= channel.state_index or len(frame.action) <= channel.action_index
            for frame in valid
        ):
            continue
        state = [frame.state[channel.state_index] for frame in valid]
        action = [frame.action[channel.action_index] for frame in valid]
        alignment = _best_alignment(
            state,
            action,
            expected_lag=source.latency_frames,
            max_lag=max_lag,
        )
        if alignment is None:
            continue
        lag, correlation = alignment
        if abs(lag - source.latency_frames) > source.latency_tolerance_frames:
            findings.append(
                Finding(
                    code="state_action_latency_mismatch",
                    outcome="review",
                    channel=channel.name,
                    value=float(lag),
                    threshold=float(source.latency_tolerance_frames),
                )
            )
        if correlation < -0.5 and not channel.allow_sign_flip:
            findings.append(
                Finding(
                    code="state_action_sign_mismatch",
                    outcome="review",
                    channel=channel.name,
                    value=correlation,
                )
            )
    return findings


def run_cheap_checks(
    episode: EpisodeRecord,
    source: SourceProfile,
    config: SignalCheckProfile,
) -> list[Finding]:
    findings = check_t0(episode)
    findings.extend(check_boundaries(episode, config))
    findings.extend(check_time_signal(episode.frames, source, config))
    findings.extend(check_idle_stuck(episode.frames, config))
    findings.extend(check_action_artifacts(episode.frames, source, config))
    findings.extend(check_state_action_alignment(episode.frames, source))
    return findings

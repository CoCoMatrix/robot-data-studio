from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from robot_data_studio.lerobot.reader import LeRobotDatasetReader
from robot_data_studio.quality.kinematics import (
    JointFieldMapping,
    KinematicChainProfile,
    KinematicProfile,
    PinocchioKinematics,
    PoseFieldMapping,
    SE3ResidualResult,
)


ArrayLike = np.ndarray | list[list[float]]


@dataclass(frozen=True)
class SuddenChangeConfig:
    median_windows: tuple[int, ...] = (5, 7)
    savgol_window: int = 9
    savgol_polyorder: int = 2
    residual_scale: float = 8.0
    delta_scale: float = 8.0
    acceleration_scale: float = 8.0
    jerk_scale: float = 8.0
    residual_min_threshold: float = 1e-9
    delta_min_threshold: float = 1e-9
    acceleration_min_threshold: float = 1e-9
    jerk_min_threshold: float = 1e-9
    event_merge_gap: int = 2


@dataclass(frozen=True)
class StateActionAlignmentConfig:
    max_lag: int = 8
    directional_agreement_threshold: float = 0.65
    actions_are_delta: bool = False
    ignored_indices: list[int] = field(default_factory=list)
    smooth_window: int = 7


@dataclass(frozen=True)
class ExtremeValueConfig:
    alpha: float = 0.5
    lower_quantile: float = 0.01
    upper_quantile: float = 0.99
    gripper_indices: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class KinematicsConfig:
    urdf_path: Path | None = None
    end_effector_link: str | None = None
    joint_names: list[str] = field(default_factory=list)
    joint_state_indices: list[int] = field(default_factory=list)
    eef_position_indices: list[int] = field(default_factory=list)
    position_tolerance: float = 0.05
    resolve_tcp_offset: bool = True
    tcp_offset: tuple[float, float, float] | None = None
    orientation_tolerance: float = 0.1
    profile: KinematicProfile | None = None


@dataclass(frozen=True)
class OrientationAlignmentConfig:
    rotation_correction: np.ndarray | None = None
    position_slice: slice | None = None
    orientation_slice: slice | None = None
    orientation_representation: Literal["rotation_6d", "rotation_matrix", "none"] = "none"


@dataclass(frozen=True)
class QwenFilterConfig:
    gripper_indices: list[int] = field(default_factory=list)
    sudden_change: SuddenChangeConfig = field(default_factory=SuddenChangeConfig)
    state_action_alignment: StateActionAlignmentConfig = field(default_factory=StateActionAlignmentConfig)
    extreme_values: ExtremeValueConfig = field(default_factory=ExtremeValueConfig)
    kinematics: KinematicsConfig = field(default_factory=KinematicsConfig)
    orientation_alignment: OrientationAlignmentConfig = field(default_factory=OrientationAlignmentConfig)


@dataclass(frozen=True)
class SuddenChangeDimensionScore:
    dimension: int
    residual_threshold: float
    delta_threshold: float
    acceleration_threshold: float
    jerk_threshold: float
    max_residual: float
    max_delta: float
    max_acceleration: float
    max_jerk: float


@dataclass(frozen=True)
class SuddenChangeEvent:
    start_frame: int
    end_frame: int
    peak_frame: int
    dimension: int
    source_metric: Literal["delta", "acceleration", "jerk"]
    severity_ratio: float
    residual_ratio: float
    metric_ratio: float


@dataclass(frozen=True)
class SuddenChangeResult:
    frame_mask: np.ndarray
    flagged_frames: list[int]
    dimension_scores: list[SuddenChangeDimensionScore]
    events: list[SuddenChangeEvent] = field(default_factory=list)


@dataclass(frozen=True)
class StateActionDimensionResult:
    dimension: int
    lag: int
    directional_agreement: float
    correlation: float
    flagged: bool


@dataclass(frozen=True)
class StateActionAlignmentResult:
    dimension_results: list[StateActionDimensionResult]
    flagged_dimensions: list[int]
    episode_flagged: bool


@dataclass(frozen=True)
class ExtremeValueDimensionResult:
    key: str
    dimension: int
    low: float
    high: float
    flagged_count: int
    exempt: bool = False


@dataclass(frozen=True)
class ExtremeValueResult:
    frame_mask: np.ndarray
    flagged_frames: list[int]
    dimension_results: dict[str, list[ExtremeValueDimensionResult]]


@dataclass(frozen=True)
class KinematicConsistencyResult:
    loaded_urdf: Path | None
    predicted_positions: np.ndarray
    corrected_positions: np.ndarray
    position_errors: np.ndarray
    max_position_error: float
    mean_position_error: float
    tcp_offset: np.ndarray
    flagged_frames: list[int]
    skipped_reason: str | None = None


@dataclass(frozen=True)
class EpisodeFilterResult:
    episode_index: int
    total_frames: int
    s1_flagged_frames: list[int]
    s2_flagged_dimensions: list[int]
    s3_flagged_frames: list[int]
    s4_flagged_frames: list[int]
    skipped_stages: dict[str, str]


@dataclass(frozen=True)
class QwenFilterSummary:
    dataset_path: str
    total_episodes: int
    total_frames: int
    stage_counts: dict[str, int]
    episode_results: list[EpisodeFilterResult]


def _as_2d_float(values: ArrayLike) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {array.shape}")
    return array


def _ensure_odd(value: int) -> int:
    value = max(3, int(value))
    return value if value % 2 == 1 else value + 1


def _median_filter_1d(values: np.ndarray, window: int) -> np.ndarray:
    window = _ensure_odd(window)
    half = window // 2
    padded = np.pad(values, (half, half), mode="edge")
    return np.asarray([np.median(padded[index : index + window]) for index in range(len(values))])


def _savgol_coefficients(window: int, polyorder: int) -> np.ndarray:
    window = _ensure_odd(window)
    half = window // 2
    x = np.arange(-half, half + 1, dtype=np.float64)
    design = np.vander(x, N=polyorder + 1, increasing=True)
    return np.linalg.pinv(design)[0]


def _savgol_smooth_1d(values: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    window = _ensure_odd(window)
    if len(values) < window:
        window = _ensure_odd(max(3, len(values) if len(values) % 2 else len(values) - 1))
    if window < 3 or len(values) < 3:
        return values.copy()
    polyorder = min(polyorder, window - 1)
    half = window // 2
    padded = np.pad(values, (half, half), mode="edge")
    coeffs = _savgol_coefficients(window, polyorder)
    return np.asarray(
        [float(np.dot(coeffs, padded[index : index + window])) for index in range(len(values))]
    )


def _smooth(values: np.ndarray, median_windows: tuple[int, ...] = (5,), savgol_window: int = 7) -> np.ndarray:
    smoothed = values.astype(np.float64, copy=True)
    for dim in range(smoothed.shape[1]):
        column = smoothed[:, dim]
        for window in median_windows:
            column = _median_filter_1d(column, window)
        column = _savgol_smooth_1d(column, savgol_window, 2)
        smoothed[:, dim] = column
    return smoothed


def _robust_threshold(values: np.ndarray, scale: float, minimum: float) -> float:
    absolute = np.abs(values[np.isfinite(values)])
    if absolute.size == 0:
        return minimum
    center = float(np.median(absolute))
    mad = float(np.median(np.abs(absolute - center)))
    return max(minimum, center + scale * 1.4826 * mad)


def _first_difference(values: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values)
    if len(values) >= 2:
        out[1:] = values[1:] - values[:-1]
    return out


def _second_difference(values: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values)
    if len(values) >= 3:
        out[1:-1] = values[2:] - 2 * values[1:-1] + values[:-2]
    return out


def _third_difference(values: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values)
    if len(values) >= 4:
        out[2:-1] = values[3:] - 3 * values[2:-1] + 3 * values[1:-2] - values[:-3]
    return out


def _safe_ratio(values: np.ndarray, threshold: float) -> np.ndarray:
    denominator = max(float(threshold), 1e-12)
    return values / denominator


def _merge_event_frames(
    frames: np.ndarray,
    gap: int,
) -> list[tuple[int, int]]:
    if frames.size == 0:
        return []
    segments = []
    start = int(frames[0])
    previous = int(frames[0])
    for frame in frames[1:]:
        current = int(frame)
        if current - previous <= gap + 1:
            previous = current
            continue
        segments.append((start, previous))
        start = current
        previous = current
    segments.append((start, previous))
    return segments


def detect_sudden_changes(values: ArrayLike, config: SuddenChangeConfig | None = None) -> SuddenChangeResult:
    config = config or SuddenChangeConfig()
    raw = _as_2d_float(values)
    smoothed = _smooth(raw, config.median_windows, config.savgol_window)
    residual = np.abs(raw - smoothed)
    delta = np.abs(_first_difference(raw))
    acceleration = np.abs(_second_difference(raw))
    jerk = np.abs(_third_difference(raw))
    frame_mask = np.zeros(raw.shape[0], dtype=bool)
    dimension_scores = []
    events: list[SuddenChangeEvent] = []
    for dim in range(raw.shape[1]):
        residual_threshold = _robust_threshold(
            residual[:, dim], config.residual_scale, config.residual_min_threshold
        )
        delta_threshold = _robust_threshold(delta[:, dim], config.delta_scale, config.delta_min_threshold)
        acceleration_threshold = _robust_threshold(
            acceleration[:, dim], config.acceleration_scale, config.acceleration_min_threshold
        )
        jerk_threshold = _robust_threshold(jerk[:, dim], config.jerk_scale, config.jerk_min_threshold)
        delta_ratio = _safe_ratio(delta[:, dim], delta_threshold)
        acceleration_ratio = _safe_ratio(acceleration[:, dim], acceleration_threshold)
        jerk_ratio = _safe_ratio(jerk[:, dim], jerk_threshold)
        residual_ratio = _safe_ratio(residual[:, dim], residual_threshold)
        dim_mask = (residual[:, dim] > residual_threshold) & (
            (delta[:, dim] > delta_threshold)
            | (acceleration[:, dim] > acceleration_threshold)
            | (jerk[:, dim] > jerk_threshold)
        )
        frame_mask |= dim_mask
        for start_frame, end_frame in _merge_event_frames(
            np.flatnonzero(dim_mask).astype(int),
            config.event_merge_gap,
        ):
            window = slice(start_frame, end_frame + 1)
            metric_ratios = {
                "delta": delta_ratio[window],
                "acceleration": acceleration_ratio[window],
                "jerk": jerk_ratio[window],
            }
            source_metric = max(
                metric_ratios,
                key=lambda name: float(np.max(metric_ratios[name])) if metric_ratios[name].size else 0.0,
            )
            local_metric_ratios = metric_ratios[source_metric]
            local_peak_offset = int(np.argmax(local_metric_ratios)) if local_metric_ratios.size else 0
            peak_frame = start_frame + local_peak_offset
            metric_ratio = float(local_metric_ratios[local_peak_offset]) if local_metric_ratios.size else 0.0
            peak_residual_ratio = float(residual_ratio[peak_frame]) if len(residual_ratio) else 0.0
            events.append(
                SuddenChangeEvent(
                    start_frame=start_frame,
                    end_frame=end_frame,
                    peak_frame=peak_frame,
                    dimension=dim,
                    source_metric=source_metric,  # type: ignore[arg-type]
                    severity_ratio=max(peak_residual_ratio, metric_ratio),
                    residual_ratio=peak_residual_ratio,
                    metric_ratio=metric_ratio,
                )
            )
        dimension_scores.append(
            SuddenChangeDimensionScore(
                dimension=dim,
                residual_threshold=residual_threshold,
                delta_threshold=delta_threshold,
                acceleration_threshold=acceleration_threshold,
                jerk_threshold=jerk_threshold,
                max_residual=float(np.max(residual[:, dim])),
                max_delta=float(np.max(delta[:, dim])),
                max_acceleration=float(np.max(acceleration[:, dim])),
                max_jerk=float(np.max(jerk[:, dim])),
            )
        )
    return SuddenChangeResult(
        frame_mask=frame_mask,
        flagged_frames=np.flatnonzero(frame_mask).astype(int).tolist(),
        dimension_scores=dimension_scores,
        events=events,
    )


def _lagged_pair(action: np.ndarray, state: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    if lag > 0:
        return action[:-lag], state[lag:]
    if lag < 0:
        return action[-lag:], state[:lag]
    return action, state


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or len(right) < 2:
        return 0.0
    left = left - np.mean(left)
    right = right - np.mean(right)
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(left, right) / denom)


def _best_lag(action: np.ndarray, state: np.ndarray, max_lag: int) -> tuple[int, float]:
    candidates = []
    max_lag = min(max_lag, max(0, len(action) - 2))
    for lag in range(-max_lag, max_lag + 1):
        left, right = _lagged_pair(action, state, lag)
        candidates.append((lag, _correlation(left, right)))
    return max(candidates, key=lambda item: (item[1], -abs(item[0])))


def _directional_agreement(action: np.ndarray, state: np.ndarray, lag: int) -> float:
    left, right = _lagged_pair(action, state, lag)
    if len(left) < 2:
        return 0.0
    left_diff = np.diff(left)
    right_diff = np.diff(right)
    left_sign = np.sign(left_diff)
    right_sign = np.sign(right_diff)
    active = (np.abs(left_diff) > 1e-12) | (np.abs(right_diff) > 1e-12)
    if not np.any(active):
        return 1.0
    return float(np.mean(left_sign[active] == right_sign[active]))


def detect_state_action_trend_alignment(
    state: ArrayLike,
    action: ArrayLike,
    config: StateActionAlignmentConfig | None = None,
) -> StateActionAlignmentResult:
    config = config or StateActionAlignmentConfig()
    state_array = _as_2d_float(state)
    action_array = _as_2d_float(action)
    if config.actions_are_delta:
        action_array = np.cumsum(action_array, axis=0)
    dims = min(state_array.shape[1], action_array.shape[1])
    state_smoothed = _smooth(state_array[:, :dims], (config.smooth_window,), config.smooth_window)
    action_smoothed = _smooth(action_array[:, :dims], (config.smooth_window,), config.smooth_window)
    ignored = set(config.ignored_indices)
    dimension_results = []
    flagged_dimensions = []
    for dim in range(dims):
        lag, corr = _best_lag(action_smoothed[:, dim], state_smoothed[:, dim], config.max_lag)
        da = _directional_agreement(action_smoothed[:, dim], state_smoothed[:, dim], lag)
        flagged = dim not in ignored and (lag < 0 or da < config.directional_agreement_threshold)
        if flagged:
            flagged_dimensions.append(dim)
        dimension_results.append(
            StateActionDimensionResult(
                dimension=dim,
                lag=lag,
                directional_agreement=da,
                correlation=corr,
                flagged=flagged,
            )
        )
    return StateActionAlignmentResult(
        dimension_results=dimension_results,
        flagged_dimensions=flagged_dimensions,
        episode_flagged=bool(flagged_dimensions),
    )


def detect_extreme_values(
    values_by_key: dict[str, ArrayLike],
    config: ExtremeValueConfig | None = None,
    reference_values_by_key: dict[str, ArrayLike] | None = None,
    reference_bounds_by_key: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> ExtremeValueResult:
    config = config or ExtremeValueConfig()
    first = next(iter(values_by_key.values()), [])
    frame_count = _as_2d_float(first).shape[0] if len(first) else 0
    frame_mask = np.zeros(frame_count, dtype=bool)
    dimension_results: dict[str, list[ExtremeValueDimensionResult]] = {}
    gripper_indices = set(config.gripper_indices)
    for key, values in values_by_key.items():
        array = _as_2d_float(values)
        reference = _as_2d_float(reference_values_by_key.get(key, values) if reference_values_by_key else values)
        reference_bounds = reference_bounds_by_key.get(key) if reference_bounds_by_key else None
        key_results = []
        for dim in range(array.shape[1]):
            if reference_bounds is not None:
                q_low = float(reference_bounds[0][dim])
                q_high = float(reference_bounds[1][dim])
            else:
                q_low = float(np.quantile(reference[:, dim], config.lower_quantile))
                q_high = float(np.quantile(reference[:, dim], config.upper_quantile))
            width = q_high - q_low
            low = q_low - config.alpha * width
            high = q_high + config.alpha * width
            exempt = dim in gripper_indices
            dim_mask = (array[:, dim] < low) | (array[:, dim] > high)
            if not exempt:
                frame_mask |= dim_mask
            key_results.append(
                ExtremeValueDimensionResult(
                    key=key,
                    dimension=dim,
                    low=low,
                    high=high,
                    flagged_count=int(np.sum(dim_mask)) if not exempt else 0,
                    exempt=exempt,
                )
            )
        dimension_results[key] = key_results
    return ExtremeValueResult(
        frame_mask=frame_mask,
        flagged_frames=np.flatnonzero(frame_mask).astype(int).tolist(),
        dimension_results=dimension_results,
    )


def rotation_matrix_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.asarray([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.asarray([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.asarray([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def _fk_positions(joint_states: np.ndarray, config: KinematicsConfig) -> np.ndarray:
    if config.urdf_path is None or config.end_effector_link is None:
        raise ValueError("URDF path and end-effector link are required for kinematic consistency")
    joints = _as_2d_float(joint_states)
    if joints.shape[1] != len(config.joint_names):
        raise ValueError("joint state width must match the configured URDF joint names")
    dummy_pose = np.zeros((len(joints), 7), dtype=np.float64)
    dummy_pose[:, -1] = 1.0
    states = np.concatenate((joints, dummy_pose), axis=1)
    pose_start = joints.shape[1]
    chain = KinematicChainProfile(
        id="legacy_qwen_s4",
        end_effector_frame=config.end_effector_link,
        joints=[
            JointFieldMapping(urdf_joint=name, state_index=index)
            for index, name in enumerate(config.joint_names)
        ],
        logged_pose=PoseFieldMapping(
            position_indices=(pose_start, pose_start + 1, pose_start + 2),
            orientation_indices=(
                pose_start + 3,
                pose_start + 4,
                pose_start + 5,
                pose_start + 6,
            ),
            orientation_representation="quaternion_xyzw",
        ),
    )
    return PinocchioKinematics(config.urdf_path).predict_chain(states, chain)[:, :3, 3]


def evaluate_kinematic_profile(
    states: ArrayLike,
    profile: KinematicProfile,
) -> dict[str, SE3ResidualResult]:
    """Evaluate Qwen S4 with the exact SE(3) engine used by Data Studio."""
    return PinocchioKinematics(profile.urdf_path).evaluate(_as_2d_float(states), profile)


def evaluate_kinematic_consistency(
    joint_states: ArrayLike,
    logged_eef_positions: ArrayLike,
    config: KinematicsConfig | None = None,
) -> KinematicConsistencyResult:
    config = config or KinematicsConfig()
    if config.urdf_path is None or config.end_effector_link is None:
        empty = np.empty((0, 3), dtype=np.float64)
        return KinematicConsistencyResult(
            loaded_urdf=None,
            predicted_positions=empty,
            corrected_positions=empty,
            position_errors=np.asarray([], dtype=np.float64),
            max_position_error=0.0,
            mean_position_error=0.0,
            tcp_offset=np.zeros(3, dtype=np.float64),
            flagged_frames=[],
            skipped_reason="URDF path or end-effector link was not provided.",
        )
    joints = _as_2d_float(joint_states)
    logged = _as_2d_float(logged_eef_positions)
    predicted = _fk_positions(joints, config)
    offset = np.asarray(config.tcp_offset or (0.0, 0.0, 0.0), dtype=np.float64)
    if config.resolve_tcp_offset and logged.size:
        offset = np.median(logged[:, :3] - predicted, axis=0)
    corrected = predicted + offset
    errors = np.linalg.norm(logged[:, :3] - corrected, axis=1)
    flagged = np.flatnonzero(errors > config.position_tolerance).astype(int).tolist()
    return KinematicConsistencyResult(
        loaded_urdf=Path(config.urdf_path),
        predicted_positions=predicted,
        corrected_positions=corrected,
        position_errors=errors,
        max_position_error=float(np.max(errors)) if len(errors) else 0.0,
        mean_position_error=float(np.mean(errors)) if len(errors) else 0.0,
        tcp_offset=offset,
        flagged_frames=flagged,
    )


def apply_orientation_alignment(values: ArrayLike, config: OrientationAlignmentConfig) -> np.ndarray:
    array = _as_2d_float(values).copy()
    correction = np.asarray(config.rotation_correction if config.rotation_correction is not None else np.eye(3))
    if config.position_slice is not None:
        array[:, config.position_slice] = array[:, config.position_slice] @ correction.T
    if config.orientation_slice is not None and config.orientation_representation != "none":
        orientation = array[:, config.orientation_slice]
        if config.orientation_representation == "rotation_6d":
            if orientation.shape[1] != 6:
                raise ValueError("rotation_6d orientation slice must have 6 dimensions")
            first = orientation[:, 0:3] @ correction.T
            second = orientation[:, 3:6] @ correction.T
            array[:, config.orientation_slice] = np.concatenate([first, second], axis=1)
        elif config.orientation_representation == "rotation_matrix":
            if orientation.shape[1] != 9:
                raise ValueError("rotation_matrix orientation slice must have 9 dimensions")
            matrices = orientation.reshape(-1, 3, 3)
            aligned = np.asarray([correction @ matrix for matrix in matrices])
            array[:, config.orientation_slice] = aligned.reshape(-1, 9)
    return array


class QwenManipDataFilter:
    def __init__(self, config: QwenFilterConfig | None = None) -> None:
        self.config = config or QwenFilterConfig()

    def evaluate_lerobot_dataset(self, root: str | Path, max_episodes: int | None = None) -> QwenFilterSummary:
        reader = LeRobotDatasetReader(root)
        episodes = reader.list_episodes()
        if max_episodes is not None:
            episodes = episodes[:max_episodes]
        episode_arrays = []
        for episode in episodes:
            frames = reader.read_episode_frames(episode.episode_index)
            states = np.asarray([frame.observation_state for frame in frames], dtype=np.float64)
            actions = np.asarray([frame.action for frame in frames], dtype=np.float64)
            episode_arrays.append((episode.episode_index, states, actions))
        reference = {
            "state": np.vstack([item[1] for item in episode_arrays]) if episode_arrays else np.empty((0, 0)),
            "action": np.vstack([item[2] for item in episode_arrays]) if episode_arrays else np.empty((0, 0)),
        }
        results = []
        stage_counts = {
            "s1_sudden_change": 0,
            "s2_state_action_alignment": 0,
            "s3_extreme_value": 0,
            "s4_kinematic_consistency": 0,
            "s5_orientation_alignment": 0,
        }
        for episode_index, states, actions in episode_arrays:
            s1_state = detect_sudden_changes(states, self.config.sudden_change)
            s1_action = detect_sudden_changes(actions, self.config.sudden_change)
            s1_frames = sorted(set(s1_state.flagged_frames) | set(s1_action.flagged_frames))
            s2 = detect_state_action_trend_alignment(states, actions, self.config.state_action_alignment)
            s3 = detect_extreme_values(
                {"state": states, "action": actions},
                self.config.extreme_values,
                reference_values_by_key=reference,
            )
            skipped = {}
            s4_frames: list[int] = []
            if self.config.kinematics.profile is not None:
                profile_results = evaluate_kinematic_profile(
                    states,
                    self.config.kinematics.profile,
                )
                flagged = np.zeros(len(states), dtype=bool)
                for result in profile_results.values():
                    flagged |= result.position_errors > self.config.kinematics.position_tolerance
                    flagged |= (
                        result.orientation_errors
                        > self.config.kinematics.orientation_tolerance
                    )
                s4_frames = np.flatnonzero(flagged).astype(int).tolist()
            elif self.config.kinematics.urdf_path is None:
                skipped["s4_kinematic_consistency"] = "URDF path was not provided."
            elif not self.config.kinematics.end_effector_link:
                skipped["s4_kinematic_consistency"] = "End-effector link was not provided."
            elif not self.config.kinematics.joint_names:
                skipped["s4_kinematic_consistency"] = "URDF joint names were not provided."
            elif not self.config.kinematics.eef_position_indices:
                skipped["s4_kinematic_consistency"] = "State indices for logged EEF position were not provided."
            else:
                joint_indices = self.config.kinematics.joint_state_indices or list(
                    range(len(self.config.kinematics.joint_names))
                )
                joint_states = states[:, joint_indices]
                logged_eef = states[:, self.config.kinematics.eef_position_indices]
                s4 = evaluate_kinematic_consistency(joint_states, logged_eef, self.config.kinematics)
                s4_frames = s4.flagged_frames
            if self.config.orientation_alignment.rotation_correction is None:
                skipped["s5_orientation_alignment"] = "No rotation correction was provided."
            stage_counts["s1_sudden_change"] += len(s1_frames)
            stage_counts["s2_state_action_alignment"] += int(s2.episode_flagged)
            stage_counts["s3_extreme_value"] += len(s3.flagged_frames)
            stage_counts["s4_kinematic_consistency"] += len(s4_frames)
            results.append(
                EpisodeFilterResult(
                    episode_index=episode_index,
                    total_frames=int(states.shape[0]),
                    s1_flagged_frames=s1_frames,
                    s2_flagged_dimensions=s2.flagged_dimensions,
                    s3_flagged_frames=s3.flagged_frames,
                    s4_flagged_frames=s4_frames,
                    skipped_stages=skipped,
                )
            )
        return QwenFilterSummary(
            dataset_path=str(Path(root).expanduser().resolve()),
            total_episodes=len(results),
            total_frames=sum(result.total_frames for result in results),
            stage_counts=stage_counts,
            episode_results=results,
        )

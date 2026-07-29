from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from robot_data_studio.quality.contracts import FrameInterval, MetricEvidence


class HumanExtensionConfig(BaseModel):
    confidence_threshold: float = 0.5
    short_hole_max_frames: int = Field(default=2, ge=0)
    bone_length_cv_review: float = Field(default=0.1, gt=0)
    derivative_outlier_scale: float = Field(default=8.0, gt=0)
    calibration_image_size: tuple[int, int] | None = None
    encoded_video_size: tuple[int, int] | None = None
    stored_video_size: tuple[int, int] | None = None
    display_matrix: tuple[int, int, int, int, int, int, int, int, int] | None = None
    video_binding_verified: bool = False


class ConfidenceTrackSummary(BaseModel):
    valid_mask: list[bool]
    valid_spans: list[tuple[int, int]]
    short_holes: list[tuple[int, int]]
    long_invalid_spans: list[tuple[int, int]]
    valid_ratio: float = Field(ge=0.0, le=1.0)


@dataclass(frozen=True)
class ProjectionResult:
    uv: np.ndarray
    depth: np.ndarray
    positive_depth: np.ndarray
    in_fov: np.ndarray


class HumanExtensionResult(BaseModel):
    metrics: list[MetricEvidence]
    per_hand_valid_masks: dict[str, list[bool]]
    bimanual_valid_mask: list[bool]
    either_hand_valid_mask: list[bool]
    confidence_tracks: dict[str, ConfidenceTrackSummary]
    evidence: dict[str, Any] = Field(default_factory=dict)

    @property
    def metrics_by_id(self) -> dict[str, MetricEvidence]:
        return {metric.metric_id: metric for metric in self.metrics}


class HandChunkValidity(BaseModel):
    source_indices: list[int]
    per_hand: dict[str, list[bool]]


def build_chunk_hand_validity(
    validity_by_hand: dict[str, list[bool] | np.ndarray],
    *,
    t0: int,
    horizon: int,
    stride: int,
) -> HandChunkValidity:
    """Apply the action-chunk tail-mask rule independently for each hand."""

    if t0 < 0 or horizon < 1 or stride < 1:
        raise ValueError("t0, horizon and stride must be positive/bounded")
    lengths = {len(np.asarray(values).reshape(-1)) for values in validity_by_hand.values()}
    if not validity_by_hand or len(lengths) != 1:
        raise ValueError("per-hand validity tracks must have one shared episode length")
    episode_length = lengths.pop()
    indexes = [t0 + offset * stride for offset in range(horizon)]
    if indexes[-1] >= episode_length:
        raise ValueError("requested hand chunk would cross episode boundary")
    output: dict[str, list[bool]] = {}
    for hand, raw in validity_by_hand.items():
        source = np.asarray(raw, dtype=np.bool_).reshape(-1)
        alive = True
        chunk_mask: list[bool] = []
        for index in indexes:
            alive = alive and bool(source[index])
            chunk_mask.append(alive)
        output[hand] = chunk_mask
    return HandChunkValidity(source_indices=indexes, per_hand=output)


def _mask_overlap(left: np.ndarray, right: np.ndarray) -> bool:
    if left.shape != right.shape:
        raise ValueError("hand/object masks must share image shape")
    return bool(np.any(left & right))


def evaluate_hand_object_interaction(
    *,
    frame_indexes: list[int],
    hand_masks: dict[str, list[np.ndarray | None]],
    object_masks: list[np.ndarray | None],
    hand_positions: dict[str, np.ndarray],
    object_positions: np.ndarray,
    structured_task: dict[str, Any],
    c1_manipulation_candidate: bool,
    mask_cache_keys: list[str],
) -> MetricEvidence:
    """Build provisional event-level Human interaction evidence from cached inputs."""

    if not c1_manipulation_candidate:
        return _not_evaluated(
            "H9_hand_object_interaction",
            "Task-conditioned hand-object interaction",
            "expensive interaction stage runs only for an eligible C1 manipulation candidate",
            {"eligible_candidate": False, "hard_reject_allowed": False},
        )
    count = len(frame_indexes)
    if count == 0 or len(object_masks) != count:
        raise ValueError("interaction evidence lengths differ")
    object_track = np.asarray(object_positions, dtype=np.float64)
    if object_track.shape != (count, 3):
        raise ValueError("object_positions must have shape (frames, 3)")
    active_hands = structured_task.get("active_hands") or sorted(hand_masks)
    usable_hands = [
        hand
        for hand in active_hands
        if hand in hand_masks
        and hand in hand_positions
        and len(hand_masks[hand]) == count
        and np.asarray(hand_positions[hand]).shape == (count, 3)
    ]
    if not usable_hands or not any(mask is not None for mask in object_masks):
        return _not_evaluated(
            "H9_hand_object_interaction",
            "Task-conditioned hand-object interaction",
            "cached hand/object segmentation or tracks are unavailable",
            {
                "eligible_candidate": True,
                "hard_reject_allowed": False,
                "reused_mask_cache_keys": list(mask_cache_keys),
            },
        )

    states: list[str] = []
    per_frame: list[dict[str, Any]] = []
    previous_contact = False
    for index, frame_index in enumerate(frame_indexes):
        object_mask = object_masks[index]
        candidates: list[tuple[str, np.ndarray, float]] = []
        if object_mask is not None:
            object_mask = np.asarray(object_mask, dtype=bool)
            for hand in usable_hands:
                raw_hand_mask = hand_masks[hand][index]
                if raw_hand_mask is None:
                    continue
                hand_mask = np.asarray(raw_hand_mask, dtype=bool)
                distance = float(
                    np.linalg.norm(np.asarray(hand_positions[hand])[index] - object_track[index])
                )
                candidates.append((hand, hand_mask, distance))
        if not candidates or object_mask is None:
            state = "uncertain_or_occluded"
            contact = False
            distance = None
        else:
            hand, selected_mask, distance = min(candidates, key=lambda item: item[2])
            contact = _mask_overlap(selected_mask, object_mask)
            if contact:
                moving_together = False
                if index > 0:
                    hand_delta = np.asarray(hand_positions[hand])[index] - np.asarray(
                        hand_positions[hand]
                    )[index - 1]
                    object_delta = object_track[index] - object_track[index - 1]
                    norms = np.linalg.norm(hand_delta) * np.linalg.norm(object_delta)
                    moving_together = bool(
                        norms > 1e-9 and float(np.dot(hand_delta, object_delta) / norms) > 0.5
                    )
                state = "manipulate_or_carry" if moving_together else "contact_or_grasp"
            elif previous_contact:
                state = "release"
            else:
                distances = [
                    float(np.linalg.norm(np.asarray(hand_positions[hand])[j] - object_track[j]))
                    for j in range(count)
                ]
                approaching = (
                    index > 0 and distances[index] < distances[index - 1]
                ) or (
                    index + 1 < count and distances[index + 1] < distances[index]
                )
                state = "approach" if approaching else "uncertain_or_occluded"
        states.append(state)
        per_frame.append(
            {
                "frame_index": int(frame_index),
                "state": state,
                "contact": contact,
                "distance": distance,
            }
        )
        previous_contact = contact
    known = sum(state != "uncertain_or_occluded" for state in states)
    return MetricEvidence(
        metric_id="H9_hand_object_interaction",
        label="Task-conditioned hand-object interaction",
        group="human_extension",
        status="evaluated",
        score=known / count,
        fidelity="provisional",
        reason="event-level geometry/motion evidence; uncalibrated and review-only",
        evidence={
            "schema": "human-hand-object-interaction-v1",
            "states": states,
            "frame_evidence": per_frame,
            "structured_task": structured_task,
            "eligible_candidate": True,
            "hard_reject_allowed": False,
            "reused_mask_cache_keys": list(mask_cache_keys),
        },
    )


def evaluate_task_conditioned_hand_availability(
    *,
    per_hand_validity: dict[str, list[bool] | np.ndarray],
    active_hands: list[str],
    event_windows: list[tuple[str, int, int]],
) -> MetricEvidence:
    """Score only task-active hands in interaction phases; never reject an episode."""

    lengths = {len(np.asarray(mask).reshape(-1)) for mask in per_hand_validity.values()}
    if not per_hand_validity or len(lengths) != 1:
        raise ValueError("per-hand validity tracks must share one episode length")
    frame_count = lengths.pop()
    unknown = set(active_hands) - set(per_hand_validity)
    if unknown:
        raise ValueError(f"task-active hand tracks are missing: {sorted(unknown)}")
    relevant_states = {
        "contact",
        "grasp",
        "contact_or_grasp",
        "manipulate",
        "carry",
        "manipulate_or_carry",
        "release",
    }
    relevant = [window for window in event_windows if window[0] in relevant_states]
    if not active_hands or not relevant:
        return _not_evaluated(
            "H7_bimanual_validity",
            "Task-conditioned per-hand validity",
            "task-active hands or interaction event windows are not defined",
            {
                "inactive_hand_absence_penalized": False,
                "whole_episode_reject_allowed": False,
            },
        )
    observations: list[bool] = []
    invalid_intervals: list[FrameInterval] = []
    for state, start, end in relevant:
        if start < 0 or end < start or end >= frame_count:
            raise ValueError("interaction event window is outside the episode")
        for hand in active_hands:
            track = np.asarray(per_hand_validity[hand], dtype=np.bool_).reshape(-1)
            window = track[start : end + 1]
            observations.extend(window.tolist())
            for invalid_start, invalid_end in _spans(window, False):
                invalid_intervals.append(
                    FrameInterval(
                        start_frame=start + invalid_start,
                        end_frame=start + invalid_end,
                        reason=f"task-active {hand} invalid during {state}",
                    )
                )
    score = float(np.mean(observations)) if observations else 0.0
    return MetricEvidence(
        metric_id="H7_bimanual_validity",
        label="Task-conditioned per-hand validity",
        group="human_extension",
        status="evaluated",
        score=score,
        reason="only task-active hand availability in interaction phases is scored",
        abnormal_intervals=invalid_intervals,
        evidence={
            "active_hands": active_hands,
            "event_windows": [list(window) for window in relevant],
            "inactive_hand_absence_penalized": False,
            "review_recommended": score < 1.0,
            "whole_episode_reject_allowed": False,
        },
    )


def _spans(mask: np.ndarray, value: bool) -> list[tuple[int, int]]:
    indexes = np.flatnonzero(mask == value)
    if indexes.size == 0:
        return []
    output: list[tuple[int, int]] = []
    start = previous = int(indexes[0])
    for raw_index in indexes[1:]:
        index = int(raw_index)
        if index == previous + 1:
            previous = index
            continue
        output.append((start, previous))
        start = previous = index
    output.append((start, previous))
    return output


def confidence_track_summary(
    confidence: np.ndarray,
    *,
    threshold: float,
    short_hole_max_frames: int,
) -> ConfidenceTrackSummary:
    values = np.asarray(confidence, dtype=np.float64).reshape(-1)
    valid = np.isfinite(values) & (values >= threshold)
    invalid_spans = _spans(valid, False)
    short = [span for span in invalid_spans if span[1] - span[0] + 1 <= short_hole_max_frames]
    long = [span for span in invalid_spans if span not in short]
    return ConfidenceTrackSummary(
        valid_mask=valid.tolist(),
        valid_spans=_spans(valid, True),
        short_holes=short,
        long_invalid_spans=long,
        valid_ratio=float(valid.mean()) if valid.size else 0.0,
    )


def camera_relative_transforms(
    world_from_camera: np.ndarray,
    world_from_joint: np.ndarray,
) -> np.ndarray:
    camera = np.asarray(world_from_camera, dtype=np.float64)
    joint = np.asarray(world_from_joint, dtype=np.float64)
    if camera.shape != joint.shape or camera.ndim != 3 or camera.shape[1:] != (4, 4):
        raise ValueError("camera and joint transforms must both have shape (frames, 4, 4)")
    return np.linalg.inv(camera) @ joint


def project_camera_points(
    points_camera: np.ndarray,
    intrinsic: np.ndarray,
    *,
    image_size: tuple[int, int],
) -> ProjectionResult:
    points = np.asarray(points_camera, dtype=np.float64)
    matrix = np.asarray(intrinsic, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_camera must have shape (N, 3)")
    if matrix.shape != (3, 3):
        raise ValueError("intrinsic must have shape (3, 3)")
    depth = points[:, 2]
    positive = np.isfinite(depth) & (depth > 0)
    homogeneous = (matrix @ points.T).T
    uv = np.full((len(points), 2), np.nan, dtype=np.float64)
    uv[positive] = homogeneous[positive, :2] / homogeneous[positive, 2:3]
    width, height = image_size
    finite_uv = np.all(np.isfinite(uv), axis=1)
    in_fov = (
        positive
        & finite_uv
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
    )
    return ProjectionResult(uv=uv, depth=depth, positive_depth=positive, in_fov=in_fov)


def _not_evaluated(metric_id: str, label: str, reason: str, evidence: dict[str, Any] | None = None) -> MetricEvidence:
    return MetricEvidence(
        metric_id=metric_id,
        label=label,
        group="human_extension",
        status="not_evaluated",
        reason=reason,
        evidence=evidence or {},
    )


def _safe_transform_track(values: np.ndarray, frames: int) -> np.ndarray | None:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (frames, 4, 4):
        return None
    return array


def _rotation_vectors(rotations: np.ndarray) -> np.ndarray:
    output = np.zeros((len(rotations), 3), dtype=np.float64)
    for index, rotation in enumerate(rotations):
        cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
        angle = math.acos(cosine)
        skew = np.asarray(
            [
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ]
        )
        if angle <= 1e-9:
            output[index] = 0.5 * skew
        else:
            output[index] = angle * skew / max(2.0 * math.sin(angle), 1e-9)
    return output


def _robust_derivative_score(
    values: np.ndarray,
    fps: float,
    scale: float,
) -> tuple[float, dict[str, float]] | None:
    if len(values) < 5:
        return None
    summaries: dict[str, float] = {}
    all_outliers: list[np.ndarray] = []
    current = values
    for name in ("velocity", "acceleration", "jerk"):
        current = np.diff(current, axis=0) * fps
        magnitudes = np.linalg.norm(current, axis=1)
        if magnitudes.size == 0:
            return None
        center = float(np.median(magnitudes))
        mad = float(np.median(np.abs(magnitudes - center)))
        threshold = center + scale * 1.4826 * mad
        if threshold <= 1e-12:
            threshold = 1e-12
        outliers = magnitudes > threshold
        all_outliers.append(outliers)
        summaries[f"max_{name}"] = float(np.max(magnitudes))
        summaries[f"{name}_threshold"] = threshold
    outlier_count = sum(int(np.count_nonzero(mask)) for mask in all_outliers)
    sample_count = sum(int(mask.size) for mask in all_outliers)
    return 1.0 - outlier_count / max(sample_count, 1), summaries


def _bone_metric(
    transforms: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    config: HumanExtensionConfig,
) -> MetricEvidence:
    pairs: list[tuple[str, str]] = []
    for side in ("left", "right"):
        hand = f"{side}Hand"
        forearm = f"{side}Forearm"
        if hand in transforms and forearm in transforms:
            pairs.append((forearm, hand))
        for finger in ("Thumb", "IndexFinger", "MiddleFinger", "RingFinger", "LittleFinger"):
            chain = [hand]
            chain.extend(
                f"{side}{finger}{suffix}"
                for suffix in ("Metacarpal", "Knuckle", "IntermediateBase", "IntermediateTip", "Tip")
                if f"{side}{finger}{suffix}" in transforms
            )
            pairs.extend(
                (parent, child)
                for parent, child in zip(chain, chain[1:], strict=False)
                if parent in transforms and child in transforms
            )
    if not pairs:
        return _not_evaluated("H2_bone_topology", "Bone-length / wrist topology", "no trusted joint topology pairs")
    cvs: dict[str, float] = {}
    for parent, child in pairs:
        valid = masks[parent] & masks[child]
        lengths = np.linalg.norm(
            transforms[parent][valid, :3, 3] - transforms[child][valid, :3, 3], axis=1
        )
        if len(lengths) < 2 or float(np.mean(lengths)) <= 1e-12:
            continue
        cvs[f"{parent}->{child}"] = float(np.std(lengths) / np.mean(lengths))
    if not cvs:
        return _not_evaluated("H2_bone_topology", "Bone-length / wrist topology", "insufficient valid bone samples")
    worst = max(cvs.values())
    score = max(0.0, 1.0 - worst / config.bone_length_cv_review)
    return MetricEvidence(
        metric_id="H2_bone_topology",
        label="Bone-length / wrist topology",
        group="human_extension",
        status="evaluated",
        score=score,
        reason="world-space arm/hand/finger-chain length stability from available named tracks",
        evidence={"coefficient_of_variation": cvs, "review_threshold": config.bone_length_cv_review},
    )


def _identity_metric(transforms: dict[str, np.ndarray], masks: dict[str, np.ndarray]) -> MetricEvidence:
    required = {"leftHand", "rightHand", "leftForearm", "rightForearm"}
    if not required <= transforms.keys():
        return _not_evaluated("H3_left_right_identity", "Left/right identity continuity", "hand/forearm tracks unavailable")
    valid = np.logical_and.reduce([masks[name] for name in required])
    if not np.any(valid):
        return _not_evaluated("H3_left_right_identity", "Left/right identity continuity", "no jointly valid frames")
    positions = {name: transforms[name][valid, :3, 3] for name in required}
    own = np.linalg.norm(positions["leftHand"] - positions["leftForearm"], axis=1)
    own += np.linalg.norm(positions["rightHand"] - positions["rightForearm"], axis=1)
    swapped = np.linalg.norm(positions["leftHand"] - positions["rightForearm"], axis=1)
    swapped += np.linalg.norm(positions["rightHand"] - positions["leftForearm"], axis=1)
    consistent = own <= swapped
    return MetricEvidence(
        metric_id="H3_left_right_identity",
        label="Left/right identity continuity",
        group="human_extension",
        status="evaluated",
        score=float(np.mean(consistent)),
        reason="same-side hand/forearm topology compared with swapped assignment",
        evidence={"evaluated_frames": int(len(consistent)), "swap_suspect_frames": int(np.count_nonzero(~consistent))},
    )


def _text_event_metric(raw_text: str, events: list[dict[str, Any]], duration: float) -> MetricEvidence:
    text_present = bool(raw_text.strip())
    valid_events = 0
    invalid_events = 0
    for event in events:
        try:
            start = float(event["start_time"])
            end = float(event["end_time"])
        except (KeyError, TypeError, ValueError):
            invalid_events += 1
            continue
        if math.isfinite(start) and math.isfinite(end) and 0 <= start <= end <= duration + 1e-6:
            valid_events += 1
        else:
            invalid_events += 1
    checks = [text_present]
    if events:
        checks.append(invalid_events == 0 and valid_events > 0)
    score = sum(checks) / len(checks)
    return MetricEvidence(
        metric_id="H8_text_event_consistency",
        label="Text / event structural consistency",
        group="human_extension",
        status="evaluated",
        score=score,
        reason="cheap structural check only; no video semantic claim",
        evidence={"raw_text_present": text_present, "valid_events": valid_events, "invalid_events": invalid_events},
    )


def analyze_human_arrays(
    *,
    transforms: dict[str, np.ndarray],
    confidences: dict[str, np.ndarray],
    intrinsic: np.ndarray,
    fps: float,
    config: HumanExtensionConfig,
    calibration_image_size: tuple[int, int] | None,
    stored_video_size: tuple[int, int] | None,
    video_frame_count: int | None,
    expected_frame_count: int,
    raw_text: str,
    events: list[dict[str, Any]],
    quality_hand: np.ndarray | None = None,
    video_binding_verified: bool = False,
) -> HumanExtensionResult:
    if expected_frame_count < 1 or fps <= 0:
        raise ValueError("expected_frame_count and fps must be positive")
    safe_transforms = {
        name: track
        for name, values in transforms.items()
        if (track := _safe_transform_track(values, expected_frame_count)) is not None
    }
    confidence_summaries = {
        name: confidence_track_summary(
            values,
            threshold=config.confidence_threshold,
            short_hole_max_frames=config.short_hole_max_frames,
        )
        for name, values in confidences.items()
        if np.asarray(values).reshape(-1).shape[0] == expected_frame_count
    }
    transform_finite = {
        name: np.all(np.isfinite(track), axis=(1, 2))
        for name, track in safe_transforms.items()
    }
    masks = {
        name: np.asarray(summary.valid_mask, dtype=bool) & transform_finite[name]
        for name, summary in confidence_summaries.items()
        if name in safe_transforms
    }
    for name in safe_transforms:
        masks.setdefault(name, transform_finite[name])

    left = masks.get("leftHand", np.zeros(expected_frame_count, dtype=bool))
    right = masks.get("rightHand", np.zeros(expected_frame_count, dtype=bool))
    bimanual = left & right
    either = left | right
    long_intervals = [
        FrameInterval(start_frame=start, end_frame=end, reason=f"{name} confidence invalid")
        for name in ("leftHand", "rightHand")
        for start, end in confidence_summaries.get(
            name,
            ConfidenceTrackSummary(valid_mask=[], valid_spans=[], short_holes=[], long_invalid_spans=[], valid_ratio=0),
        ).long_invalid_spans
    ]
    hand_ratios = [float(mask.mean()) for mask in (left, right)]
    metrics: list[MetricEvidence] = [
        MetricEvidence(
            metric_id="H1_tracking_confidence",
            label="Tracking confidence / valid spans",
            group="human_extension",
            status="evaluated",
            score=float(np.mean(hand_ratios)),
            reason="per-hand confidence masks retained without interpolation",
            abnormal_intervals=long_intervals,
            evidence={
                "short_hole_max_frames": config.short_hole_max_frames,
                "confidence_threshold": config.confidence_threshold,
                "left_valid_ratio": hand_ratios[0],
                "right_valid_ratio": hand_ratios[1],
            },
        )
    ]
    metrics.append(_bone_metric(safe_transforms, masks, config))
    metrics.append(_identity_metric(safe_transforms, masks))

    derivative_scores = []
    derivative_evidence: dict[str, Any] = {}
    if "camera" in safe_transforms:
        for hand_name, mask in (("leftHand", left), ("rightHand", right)):
            if hand_name not in safe_transforms:
                continue
            mask = mask & masks["camera"]
            relative = camera_relative_transforms(safe_transforms["camera"], safe_transforms[hand_name])
            hand_segments = []
            for start, end in _spans(mask, True):
                segment = relative[start : end + 1]
                features = np.concatenate(
                    [segment[:, :3, 3], _rotation_vectors(segment[:, :3, :3])], axis=1
                )
                derivative = _robust_derivative_score(
                    features,
                    fps,
                    config.derivative_outlier_scale,
                )
                if derivative is not None:
                    hand_segments.append((start, end, derivative))
            if hand_segments:
                derivative_scores.extend(item[2][0] for item in hand_segments)
                derivative_evidence[hand_name] = {
                    "segments": [
                        {"start_frame": start, "end_frame": end, **summary}
                        for start, end, (_score, summary) in hand_segments
                    ],
                    "invalid_gaps_bridged": False,
                }
    if derivative_scores:
        metrics.append(
            MetricEvidence(
                metric_id="H4_se3_dynamics",
                label="Camera-relative SE(3) velocity / acceleration / jerk",
                group="human_extension",
                status="evaluated",
                score=float(np.mean(derivative_scores)),
                reason="robust camera-relative translational/rotational SE(3) derivative summary",
                evidence=derivative_evidence,
            )
        )
    else:
        metrics.append(_not_evaluated("H4_se3_dynamics", "Camera-relative SE(3) velocity / acceleration / jerk", "insufficient valid camera-relative tracks"))

    if quality_hand is None:
        metrics.append(_not_evaluated("H5_quality_hand_crosscheck", "quality_hand cross-check", "quality_hand absent"))
    else:
        quality = np.asarray(quality_hand, dtype=np.float64)
        values = quality.reshape(-1)
        per_hand: dict[str, dict[str, float]] = {}
        agreements: list[float] = []
        if quality.shape == (expected_frame_count, 2):
            for column, (hand_name, hand_mask) in enumerate((("left", left), ("right", right))):
                finite = np.isfinite(quality[:, column])
                if not np.any(finite):
                    continue
                finite_values = quality[finite, column]
                complete = finite_values == 1.0
                valid = hand_mask[finite]
                agreement = float(np.mean(complete == valid))
                agreements.append(agreement)
                per_hand[hand_name] = {
                    "complete_visible_ratio": float(np.mean(complete)),
                    "tracking_valid_ratio": float(np.mean(valid)),
                    "agreement": agreement,
                }
        if agreements:
            metrics.append(MetricEvidence(
                metric_id="H5_quality_hand_crosscheck",
                label="quality_hand cross-check",
                group="human_extension",
                status="evaluated",
                score=float(np.mean(agreements)),
                weight=0.0,
                reason="source auxiliary audit only: quality_hand=1 means hand completely visible",
                evidence={
                    "value_counts": {
                        str(value): int(np.count_nonzero(values == value)) for value in np.unique(values)
                    },
                    "validity_ratio": float(either.mean()),
                    "per_hand": per_hand,
                    "cross_supplier_hard_gate": False,
                },
            ))
        else:
            metrics.append(_not_evaluated(
                "H5_quality_hand_crosscheck", "quality_hand cross-check",
                "quality_hand shape/finite values invalid",
            ))

    projection_evidence: dict[str, Any] = {
        "calibration_image_size": (
            list(calibration_image_size) if calibration_image_size is not None else None
        ),
        "stored_video_size": list(stored_video_size) if stored_video_size else None,
        "video_frame_count": video_frame_count,
        "expected_frame_count": expected_frame_count,
        "frame_count_matches": video_frame_count == expected_frame_count if video_frame_count is not None else None,
    }
    fov_values: list[bool] = []
    if calibration_image_size is not None and "camera" in safe_transforms:
        for hand_name, mask in (("leftHand", left), ("rightHand", right)):
            if hand_name not in safe_transforms or not np.any(mask):
                continue
            mask = mask & masks["camera"]
            relative = camera_relative_transforms(safe_transforms["camera"], safe_transforms[hand_name])
            projected = project_camera_points(
                relative[mask, :3, 3], intrinsic, image_size=calibration_image_size
            )
            fov_values.extend(projected.in_fov.tolist())
    projection_evidence["calibration_space_in_fov_ratio"] = (
        float(np.mean(fov_values)) if fov_values else None
    )
    binding_ok = (
        video_binding_verified
        and calibration_image_size is not None
        and stored_video_size == calibration_image_size
        and video_frame_count == expected_frame_count
    )
    if fov_values:
        metrics.append(
            MetricEvidence(
                metric_id="H6_camera_projection",
                label="Camera-relative FOV / projection",
                group="human_extension",
                status="evaluated",
                score=float(np.mean(fov_values)),
                reason="camera-relative projection evaluated in the calibration canvas",
                evidence=projection_evidence,
            )
        )
    else:
        metrics.append(
            _not_evaluated(
                "H6_camera_projection",
                "Camera-relative FOV / projection",
                "camera-relative transform, calibration image size, or points unavailable",
                projection_evidence,
            )
        )
    if binding_ok:
        metrics.append(
            MetricEvidence(
                metric_id="H6b_stored_video_projection",
                label="Stored-video hand projection binding",
                group="human_extension",
                status="evaluated",
                score=1.0,
                reason="calibration-to-stored-video binding verified",
                evidence=projection_evidence,
            )
        )
    else:
        metrics.append(
            _not_evaluated(
                "H6b_stored_video_projection",
                "Stored-video hand projection binding",
                "calibration-to-stored-video resize/crop/rotation binding is not verified",
                projection_evidence,
            )
        )

    metrics.append(
        MetricEvidence(
            metric_id="H7_bimanual_validity",
            label="Bimanual / per-hand validity",
            group="human_extension",
            status="evaluated",
            score=float(either.mean()),
            reason="either-hand score with independent left/right and bimanual masks retained",
            evidence={
                "left_valid_ratio": float(left.mean()),
                "right_valid_ratio": float(right.mean()),
                "bimanual_valid_ratio": float(bimanual.mean()),
                "either_hand_valid_ratio": float(either.mean()),
            },
        )
    )
    duration = max(0.0, (expected_frame_count - 1) / fps)
    metrics.append(_text_event_metric(raw_text, events, duration))
    return HumanExtensionResult(
        metrics=metrics,
        per_hand_valid_masks={"left": left.tolist(), "right": right.tolist()},
        bimanual_valid_mask=bimanual.tolist(),
        either_hand_valid_mask=either.tolist(),
        confidence_tracks=confidence_summaries,
        evidence={
            "no_interpolation": True,
            "camera_motion_cancelled": "T_camera_joint=inv(T_world_camera)@T_world_joint",
        },
    )

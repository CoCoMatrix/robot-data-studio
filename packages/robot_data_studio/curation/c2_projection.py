from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from robot_data_studio.quality.contracts import (
    QWEN_CORE_DEFAULT_WEIGHTS,
    QWEN_CORE_LABELS,
    MetricEvidence,
    evaluate_c2_video_state_consistency,
)

from .sam3 import Sam3Result


def matrix_to_xyz_rot6d(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    return np.concatenate([value[:3, 3], value[:2, :3].reshape(-1)])


def xyz_rot6d_to_matrix(value: Sequence[float]) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64).reshape(9)
    first = pose[3:6]
    second = pose[6:9]
    first_norm = np.linalg.norm(first)
    if first_norm <= 1e-12:
        raise ValueError("Rot6D first row is degenerate")
    first = first / first_norm
    second = second - np.dot(first, second) * first
    second_norm = np.linalg.norm(second)
    if second_norm <= 1e-12:
        raise ValueError("Rot6D second row is degenerate")
    second = second / second_norm
    third = np.cross(first, second)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.stack([first, second, third], axis=0)
    matrix[:3, 3] = pose[:3]
    return matrix


def project_eef_support_mask(
    *,
    eef_pose_absolute: Sequence[float],
    eef_validity: Sequence[bool],
    camera_pose_absolute: Sequence[float],
    camera_valid: bool,
    intrinsic: np.ndarray,
    image_shape: tuple[int, int],
    support_radius_pixels: int = 24,
) -> tuple[np.ndarray, dict]:
    """Project left/right absolute EEF origins as explicit support discs.

    This is intentionally narrower than a mesh silhouette: Robot origins are
    URDF-FK-derived while Human origins are measured hand poses. The scope is
    retained in evidence so it cannot be mistaken for full-link rendering.
    """

    height, width = image_shape
    mask = np.zeros((height, width), dtype=bool)
    evidence = {
        "projection_scope": "absolute EEF support discs",
        "support_radius_pixels": int(support_radius_pixels),
        "projected_hands": [],
        "out_of_fov_hands": [],
    }
    if not camera_valid:
        evidence["reason"] = "camera pose invalid"
        return mask, evidence
    poses = np.asarray(eef_pose_absolute, dtype=np.float64).reshape(2, 9)
    validity = np.asarray(eef_validity, dtype=bool).reshape(2)
    intrinsic = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
    camera_from_absolute = np.linalg.inv(xyz_rot6d_to_matrix(camera_pose_absolute))
    yy, xx = np.ogrid[:height, :width]
    for name, pose, valid in zip(("left", "right"), poses, validity, strict=True):
        if not valid or not np.all(np.isfinite(pose)):
            continue
        absolute_point = np.asarray([pose[0], pose[1], pose[2], 1.0])
        camera_point = camera_from_absolute @ absolute_point
        if camera_point[2] <= 1e-8:
            evidence["out_of_fov_hands"].append(name)
            continue
        pixel = intrinsic @ camera_point[:3]
        u, v = float(pixel[0] / pixel[2]), float(pixel[1] / pixel[2])
        if not (0 <= u < width and 0 <= v < height):
            evidence["out_of_fov_hands"].append(name)
            continue
        disc = (xx - u) ** 2 + (yy - v) ** 2 <= support_radius_pixels**2
        mask |= disc
        evidence["projected_hands"].append(name)
    evidence["projected_pixel_count"] = int(np.count_nonzero(mask))
    return mask, evidence


def evaluate_c2_keyframes(
    *,
    projected_masks: Sequence[np.ndarray],
    projection_evidence: Sequence[dict],
    sam_results: Sequence[Sam3Result],
    frame_indexes: Sequence[int],
    projection_kind: str,
    iou_threshold: float = 0.5,
) -> MetricEvidence:
    historical_group = "human_extension" if projection_kind.startswith("human") else "robot_extension"
    if not (
        len(projected_masks)
        == len(projection_evidence)
        == len(sam_results)
        == len(frame_indexes)
    ):
        raise ValueError("C2 keyframe evidence lengths differ")
    frames: list[dict] = []
    scores: list[float] = []
    for projected, projection, sam, frame_index in zip(
        projected_masks,
        projection_evidence,
        sam_results,
        frame_indexes,
        strict=True,
    ):
        compared = evaluate_c2_video_state_consistency(
            projected_robot_mask=projected,
            observed_robot_mask=sam.mask if sam.status == "evaluated" else None,
            iou_threshold=iou_threshold,
            missing_reason=sam.reason,
        )
        row = {
            "frame_index": int(frame_index),
            "status": compared.status,
            "sam3_confidence": sam.confidence,
            "sam3_cache_hit": sam.cache_hit,
            "sam3_cache_key": sam.cache_key,
            "projection": projection,
        }
        if compared.status == "evaluated" and compared.score is not None:
            row.update(compared.evidence)
            scores.append(float(compared.score))
        else:
            row["reason"] = compared.reason
        frames.append(row)
    if not scores:
        return MetricEvidence(
            metric_id="historical_support_disc_projection",
            label="Historical EEF support-disc projection",
            group=historical_group,
            status="not_evaluated",
            fidelity="historical",
            weight=0.0,
            reason="no keyframe had both a valid projection and real SAM3 mask",
            evidence={
                "provider": "facebook/sam3",
                "provider_mode": "real_failed_closed",
                "projection_kind": projection_kind,
                "frame_evidence": frames,
            },
        )
    return MetricEvidence(
        metric_id="historical_support_disc_projection",
        label="Historical EEF support-disc projection",
        group=historical_group,
        status="evaluated",
        score=float(np.mean(scores)),
        fidelity="historical",
        weight=0.0,
        reason="historical provisional support-disc evidence; not current Robot C2",
        evidence={
            "provider": "facebook/sam3",
            "provider_mode": "real_local",
            "projection_kind": projection_kind,
            "projection_scope": "support discs, not a full URDF mesh silhouette",
            "threshold": iou_threshold,
            "evaluated_frame_count": len(scores),
            "frame_evidence": frames,
        },
    )

from __future__ import annotations

from typing import Any, Literal, Protocol

import numpy as np
from pydantic import BaseModel, Field, model_validator


CURRENT_QWEN_CONTRACT_VERSION = "qwen-cleaning-v2"
HISTORICAL_QWEN_CORE_IDS = ("S1", "S2", "S3", "S4", "S5", "C1", "C2", "C3")
CORE_CONTRACT_IDS = ("S1", "S2", "S3", "C1", "C2", "C3")
RETIRED_QWEN_CORE_IDS = ("S4", "S5")
HUMAN_EXTENSION_IDS = (
    "H1_tracking_confidence",
    "H2_bone_topology",
    "H3_left_right_identity",
    "H4_se3_dynamics",
    "H5_quality_hand_crosscheck",
    "H6_camera_projection",
    "H6b_stored_video_projection",
    "H7_bimanual_validity",
    "H8_text_event_consistency",
    "H9_hand_object_interaction",
)

QWEN_CHECK_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "S1": ("state_action_rows",),
    "S2": ("state_action_rows",),
    "S3": ("state_action_rows", "train_reference_distribution"),
    "C1": ("task_structure", "sparse_video_decode"),
    "C2": ("sparse_video_decode", "robot_masks", "robot_full_mesh_projection"),
    "C3": ("sparse_video_decode",),
}

HUMAN_CHECK_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "H1_tracking_confidence": ("hand_tracking",),
    "H2_bone_topology": ("hand_tracking", "pose_tracks"),
    "H3_left_right_identity": ("hand_tracking", "pose_tracks"),
    "H4_se3_dynamics": ("pose_tracks", "camera_pose"),
    "H5_quality_hand_crosscheck": ("hand_tracking", "quality_hand"),
    "H6_camera_projection": ("pose_tracks", "camera_pose", "camera_calibration"),
    "H6b_stored_video_projection": (
        "pose_tracks", "camera_pose", "camera_calibration", "display_transform"
    ),
    "H7_bimanual_validity": ("hand_tracking",),
    "H8_text_event_consistency": ("task_structure",),
    "H9_hand_object_interaction": (
        "task_structure", "hand_tracking", "pose_tracks", "hand_object_masks"
    ),
}

# Existing scorer defaults are preserved. C2 had no legacy weight and therefore uses
# the neutral 1.0 default until a reviewed configuration says otherwise.
QWEN_CORE_DEFAULT_WEIGHTS: dict[str, float] = {
    "S1": 1.5,
    "S2": 1.5,
    "S3": 2.0,
    "S4": 2.0,
    "S5": 1.0,
    "C1": 2.0,
    "C2": 1.0,
    "C3": 1.5,
}

QWEN_CORE_LABELS: dict[str, str] = {
    "S1": "Sudden change",
    "S2": "State-action trend alignment",
    "S3": "Extreme value",
    "S4": "FK consistency",
    "S5": "Base-frame / EEF orientation alignment",
    "C1": "Instruction consistency",
    "C2": "Video-state consistency",
    "C3": "Video quality",
}


class FrameInterval(BaseModel):
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    reason: str = ""

    @model_validator(mode="after")
    def ordered(self) -> FrameInterval:
        if self.end_frame < self.start_frame:
            raise ValueError("end_frame must be greater than or equal to start_frame")
        return self


MetricGroup = Literal[
    "qwen_core",
    "human_extension",
    "robot_extension",
    "safety_gate",
]


class MetricEvidence(BaseModel):
    metric_id: str
    label: str
    group: MetricGroup
    status: Literal["evaluated", "not_evaluated", "disabled", "retired"]
    enabled: bool = True
    contract_state: Literal["active", "retired"] = "active"
    fidelity: Literal["standard", "provisional", "calibrated", "historical"] = "standard"
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    weight: float = Field(default=1.0, ge=0.0)
    reason: str
    abnormal_intervals: list[FrameInterval] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def score_matches_status(self) -> MetricEvidence:
        if self.status == "evaluated" and self.score is None:
            raise ValueError("evaluated metrics require a score")
        if self.status != "evaluated" and self.score is not None:
            raise ValueError("unevaluated/disabled/retired metrics cannot have a score")
        if self.status in {"disabled", "retired"} and self.enabled:
            raise ValueError("disabled/retired metrics cannot be enabled")
        if self.status == "retired" and self.contract_state != "retired":
            raise ValueError("retired status requires retired contract_state")
        return self


class ScorecardDisposition(BaseModel):
    hard_reject: bool = False
    review: bool = False
    window_masks: list[FrameInterval] = Field(default_factory=list)
    soft_weight: float = Field(default=1.0, ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def deduplicate_shared_findings(self) -> ScorecardDisposition:
        self.reasons = list(dict.fromkeys(self.reasons))
        unique: dict[tuple[int, int, str], FrameInterval] = {}
        for interval in self.window_masks:
            unique.setdefault(
                (interval.start_frame, interval.end_frame, interval.reason), interval
            )
        self.window_masks = list(unique.values())
        return self


class QualityScorecard(BaseModel):
    contract_version: str = CURRENT_QWEN_CONTRACT_VERSION
    qwen_core_score: float | None = Field(default=None, ge=0.0, le=1.0)
    extension_score: float | None = Field(default=None, ge=0.0, le=1.0)
    coverage: float = Field(ge=0.0, le=1.0)
    qwen_core_coverage: float = Field(ge=0.0, le=1.0)
    extension_coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    selected_checks: list[str] = Field(default_factory=list)
    selected_denominator: int = Field(default=0, ge=0)
    evaluated_count: int = Field(default=0, ge=0)
    final_decision: Literal["pass", "review", "mask", "reject"]
    metrics: list[MetricEvidence]
    disposition: ScorecardDisposition = Field(default_factory=ScorecardDisposition)


def _weighted_mean(metrics: list[MetricEvidence]) -> float | None:
    evaluated = [
        metric
        for metric in metrics
        if (
            metric.enabled
            and metric.contract_state == "active"
            and metric.status == "evaluated"
            and metric.score is not None
            and metric.weight > 0
        )
    ]
    weight = sum(metric.weight for metric in evaluated)
    if weight <= 0:
        return None
    return sum(float(metric.score) * metric.weight for metric in evaluated) / weight


def build_quality_scorecard(
    metrics: list[MetricEvidence],
    *,
    disposition: ScorecardDisposition | None = None,
    pass_threshold: float = 0.8,
    review_threshold: float = 0.6,
) -> QualityScorecard:
    disposition = disposition or ScorecardDisposition()
    normalized_metrics: list[MetricEvidence] = []
    retired_seen: set[str] = set()
    for metric in metrics:
        if metric.group == "qwen_core" and metric.metric_id in RETIRED_QWEN_CORE_IDS:
            if metric.metric_id in retired_seen:
                continue
            retired_seen.add(metric.metric_id)
            replacement = retired_metric(metric.metric_id)
            replacement.evidence = {
                "historical_status": metric.status,
                "historical_score": metric.score,
                "historical_reason": metric.reason,
                "historical_evidence": metric.evidence,
            }
            normalized_metrics.append(replacement)
        else:
            normalized_metrics.append(metric)
    metrics = normalized_metrics
    core = [metric for metric in metrics if metric.group == "qwen_core"]
    extension = [
        metric
        for metric in metrics
        if metric.group in {"human_extension", "robot_extension"}
    ]
    evaluated_core_ids = {
        metric.metric_id
        for metric in core
        if (
            metric.metric_id in CORE_CONTRACT_IDS
            and metric.enabled
            and metric.contract_state == "active"
            and metric.status == "evaluated"
        )
    }
    selected_core_ids = [
        metric.metric_id
        for metric in core
        if (
            metric.metric_id in CORE_CONTRACT_IDS
            and metric.enabled
            and metric.contract_state == "active"
            and metric.status not in {"disabled", "retired"}
        )
    ]
    selected_core_ids = list(dict.fromkeys(selected_core_ids))
    selected_extension_ids = list(
        dict.fromkeys(
            metric.metric_id
            for metric in extension
            if metric.metric_id in HUMAN_EXTENSION_IDS
            and metric.enabled
            and metric.contract_state == "active"
            and metric.status not in {"disabled", "retired"}
        )
    )
    evaluated_extension_ids = {
        metric.metric_id
        for metric in extension
        if metric.metric_id in HUMAN_EXTENSION_IDS
        and metric.enabled
        and metric.contract_state == "active"
        and metric.status == "evaluated"
    }
    selected_ids = [*selected_core_ids, *selected_extension_ids]
    evaluated_count = len(evaluated_core_ids) + len(evaluated_extension_ids)
    coverage = evaluated_count / len(selected_ids) if selected_ids else 0.0
    qwen_core_coverage = (
        len(evaluated_core_ids) / len(selected_core_ids) if selected_core_ids else 0.0
    )
    extension_coverage = (
        len(evaluated_extension_ids) / len(selected_extension_ids)
        if selected_extension_ids
        else None
    )
    core_score = _weighted_mean(core)
    decisive_core_score = _weighted_mean(
        [metric for metric in core if metric.fidelity != "provisional"]
    )
    has_provisional_core = any(
        metric.enabled
        and metric.contract_state == "active"
        and metric.status == "evaluated"
        and metric.fidelity == "provisional"
        for metric in core
    )
    extension_score = _weighted_mean(extension)

    if disposition.hard_reject:
        decision = "reject"
    elif disposition.window_masks:
        decision = "mask"
    elif disposition.review or coverage < 1.0 or core_score is None:
        decision = "review"
    elif core_score >= pass_threshold:
        decision = "pass"
    elif core_score >= review_threshold:
        decision = "review"
    elif has_provisional_core and (
        decisive_core_score is None or decisive_core_score >= review_threshold
    ):
        decision = "review"
    else:
        decision = "reject"
    return QualityScorecard(
        qwen_core_score=core_score,
        extension_score=extension_score,
        coverage=coverage,
        qwen_core_coverage=qwen_core_coverage,
        extension_coverage=extension_coverage,
        selected_checks=selected_ids,
        selected_denominator=len(selected_ids),
        evaluated_count=evaluated_count,
        final_decision=decision,
        metrics=metrics,
        disposition=disposition,
    )


def disabled_metric(metric_id: str, *, group: MetricGroup = "qwen_core") -> MetricEvidence:
    return MetricEvidence(
        metric_id=metric_id,
        label=QWEN_CORE_LABELS.get(metric_id, metric_id),
        group=group,
        status="disabled",
        enabled=False,
        reason="disabled by the selected quality profile",
        weight=QWEN_CORE_DEFAULT_WEIGHTS.get(metric_id, 1.0),
    )


def retired_metric(metric_id: str) -> MetricEvidence:
    if metric_id not in RETIRED_QWEN_CORE_IDS:
        raise ValueError(f"{metric_id} is not retired in {CURRENT_QWEN_CONTRACT_VERSION}")
    reasons = {
        "S4": "retired: independent logged-EEF/FK reference is unavailable by incoming-collection design",
        "S5": "retired: directional convention is enforced during pose conversion/chunk construction",
    }
    return MetricEvidence(
        metric_id=metric_id,
        label=QWEN_CORE_LABELS[metric_id],
        group="qwen_core",
        status="retired",
        enabled=False,
        contract_state="retired",
        fidelity="historical",
        reason=reasons[metric_id],
        weight=QWEN_CORE_DEFAULT_WEIGHTS[metric_id],
    )


def build_c1_instruction_evidence(
    *,
    raw_text: str,
    structured_task: dict[str, Any],
    events: dict[str, int],
    approximate_task_success_score: float | None = None,
) -> MetricEvidence:
    evidence = {
        "provider": "legacy_task_success" if approximate_task_success_score is not None else "disabled",
        "approximate": approximate_task_success_score is not None,
        "raw_text": raw_text,
        "structured_task": structured_task,
        "events": events,
    }
    if approximate_task_success_score is None:
        return MetricEvidence(
            metric_id="C1",
            label=QWEN_CORE_LABELS["C1"],
            group="qwen_core",
            status="not_evaluated",
            weight=QWEN_CORE_DEFAULT_WEIGHTS["C1"],
            reason="structured C1 provider disabled; legacy task_success is only an approximation",
            evidence=evidence,
        )
    return MetricEvidence(
        metric_id="C1",
        label=QWEN_CORE_LABELS["C1"],
        group="qwen_core",
        status="evaluated",
        score=max(0.0, min(1.0, float(approximate_task_success_score))),
        weight=QWEN_CORE_DEFAULT_WEIGHTS["C1"],
        reason="legacy task_success approximation; structured C1 provider did not run",
        evidence=evidence,
    )


class RobotMaskProvider(Protocol):
    """Optional segmentation evidence provider; no implementation or model is bundled."""

    def mask(self, image_reference: str) -> np.ndarray: ...


class UrdfProjectionProvider(Protocol):
    """Optional trusted-calibration URDF projection provider."""

    def projected_robot_mask(self, frame_index: int) -> np.ndarray: ...


def evaluate_c2_video_state_consistency(
    *,
    projected_robot_mask: np.ndarray | None,
    observed_robot_mask: np.ndarray | None,
    iou_threshold: float = 0.5,
    missing_reason: str = "trusted projection or robot-mask provider unavailable",
) -> MetricEvidence:
    if projected_robot_mask is None or observed_robot_mask is None:
        return MetricEvidence(
            metric_id="C2",
            label=QWEN_CORE_LABELS["C2"],
            group="qwen_core",
            status="not_evaluated",
            weight=QWEN_CORE_DEFAULT_WEIGHTS["C2"],
            reason=missing_reason,
            evidence={
                "projection_available": projected_robot_mask is not None,
                "robot_mask_available": observed_robot_mask is not None,
            },
        )
    projected = np.asarray(projected_robot_mask, dtype=bool)
    observed = np.asarray(observed_robot_mask, dtype=bool)
    if projected.shape != observed.shape:
        raise ValueError("projected and observed masks must have the same shape")
    union = int(np.count_nonzero(projected | observed))
    if union == 0:
        return MetricEvidence(
            metric_id="C2",
            label=QWEN_CORE_LABELS["C2"],
            group="qwen_core",
            status="not_evaluated",
            weight=QWEN_CORE_DEFAULT_WEIGHTS["C2"],
            reason="both masks are empty; no robot evidence to compare",
            evidence={"projection_available": True, "robot_mask_available": True},
        )
    intersection = int(np.count_nonzero(projected & observed))
    iou = intersection / union
    return MetricEvidence(
        metric_id="C2",
        label=QWEN_CORE_LABELS["C2"],
        group="qwen_core",
        status="evaluated",
        score=iou,
        weight=QWEN_CORE_DEFAULT_WEIGHTS["C2"],
        reason="projected-robot / observed-robot mask IoU evidence",
        evidence={
            "iou": iou,
            "threshold": iou_threshold,
            "consistent": iou >= iou_threshold,
            "intersection_pixels": intersection,
            "union_pixels": union,
        },
    )

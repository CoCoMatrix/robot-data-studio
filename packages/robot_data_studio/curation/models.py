from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic import model_validator

from robot_data_studio.quality.contracts import MetricEvidence


class RawNumericFeature(BaseModel):
    values: list[float] | None = None
    valid: bool = False
    valid_mask: list[bool] = Field(default_factory=list)
    names: list[str] = Field(default_factory=list)
    units: list[str] = Field(default_factory=list)
    action_type: str | None = None
    missing_reason: str | None = None

    @model_validator(mode="after")
    def validate_numeric_feature(self) -> "RawNumericFeature":
        if self.valid and self.values is None:
            raise ValueError("valid numeric feature requires values")
        if not self.valid and self.values is not None:
            raise ValueError("invalid numeric feature must not carry values")
        if self.values is not None:
            if len(self.valid_mask) != len(self.values):
                raise ValueError("numeric valid_mask length mismatch")
            if self.names and len(self.names) != len(self.values):
                raise ValueError("numeric names length mismatch")
            if self.units and len(self.units) != len(self.values):
                raise ValueError("numeric units length mismatch")
        elif any(self.valid_mask):
            raise ValueError("missing numeric feature mask must be false")
        return self

    @classmethod
    def missing(
        cls,
        reason: str,
        *,
        names: list[str] | None = None,
        units: list[str] | None = None,
    ) -> "RawNumericFeature":
        feature_names = list(names or [])
        return cls(
            valid=False,
            valid_mask=[False] * len(feature_names),
            names=feature_names,
            units=list(units or []),
            missing_reason=reason,
        )


class RawPose(BaseModel):
    matrix: list[float] | None = None
    valid: bool = False
    reference_frame: str
    child_frame: str
    transform_direction: str
    translation_unit: Literal["m"] = "m"
    rotation_convention: Literal["matrix_3x3_row_major"] = "matrix_3x3_row_major"
    missing_reason: str | None = None

    @model_validator(mode="after")
    def validate_pose(self) -> "RawPose":
        if self.valid and self.matrix is None:
            raise ValueError("valid pose requires matrix")
        if not self.valid and self.matrix is not None:
            raise ValueError("invalid pose must not carry matrix")
        if self.matrix is not None and len(self.matrix) != 16:
            raise ValueError("pose matrix must have 16 values")
        return self


class RawDualActionFrame(BaseModel):
    observation_joint_position: RawNumericFeature
    observation_eef_pose: dict[str, RawPose] = Field(default_factory=dict)
    action_joint_target: RawNumericFeature
    action_eef_pose_target: dict[str, RawPose] = Field(default_factory=dict)
    camera_pose: dict[str, RawPose] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)


class FrameSample(BaseModel):
    episode_id: str
    frame_index: int
    timestamp: float
    decoded_pts: float | None = None
    valid: bool = True
    state: list[float] = Field(default_factory=list)
    action: list[float] = Field(default_factory=list)
    contact_hold: bool = False
    video_uri: str | None = None
    dual_action: RawDualActionFrame | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResamplePoint(BaseModel):
    target_index: int
    progress: float = Field(ge=0.0, le=1.0)
    episode_id: str
    episode_time: float
    file_time: float
    original_frame_index: int
    original_timestamp: float
    valid: bool


class NormalizedTaskText(BaseModel):
    raw_text: str
    verb: str
    object: str
    target: str | None = None
    modifiers: list[str] = Field(default_factory=list)


EvidenceState = Literal["consistent", "review", "conflict"]


class EpisodeRecord(BaseModel):
    episode_id: str
    source_id: str
    task_id: str
    split: str
    from_timestamp: float = 0.0
    expected_duration: float | None = None
    frames: list[FrameSample] = Field(default_factory=list)
    event_frames: dict[str, int] = Field(default_factory=dict)
    lineage: dict[str, Any] = Field(default_factory=dict)
    raw_task_text: str | None = None
    extension_metrics: list[MetricEvidence] = Field(default_factory=list)


FindingOutcome = Literal["hard_reject", "review", "window_mask", "soft_weight"]


class Finding(BaseModel):
    code: str
    outcome: FindingOutcome
    tier: Literal["T0", "T1", "T2", "T3", "T4"] = "T1"
    message: str = ""
    start_frame: int | None = None
    end_frame: int | None = None
    channel: str | None = None
    value: float | None = None
    threshold: float | None = None
    weight_multiplier: float = Field(default=1.0, ge=0.0, le=1.0)


class WindowMask(BaseModel):
    start_frame: int
    end_frame: int
    reason: str


class Disposition(BaseModel):
    hard_reject: bool = False
    hard_reasons: list[str] = Field(default_factory=list)
    review: bool = False
    review_reasons: list[str] = Field(default_factory=list)
    window_masks: list[WindowMask] = Field(default_factory=list)
    soft_weight: float = Field(default=1.0, ge=0.0, le=1.0)


class EvidenceFrame(BaseModel):
    label: Literal["start", "contact", "end"]
    frame_index: int
    timestamp: float
    uri: str | None = None


class VlmResult(BaseModel):
    state: EvidenceState
    cache_key: str
    from_cache: bool = False
    reason: str = "mock evaluation"


class EvaluationStatus(BaseModel):
    state: Literal["evaluated", "not_evaluated"] = "not_evaluated"
    reason: str
    backend: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)


class EpisodeCurationResult(BaseModel):
    episode_id: str
    source_id: str
    task_id: str
    duration_seconds: float
    motion: float
    fingerprint: str
    content_hash: str
    schema_fingerprint: str
    lineage: dict[str, Any]
    normalized_task: NormalizedTaskText
    evidence_state: EvidenceState
    evidence: list[EvidenceFrame]
    findings: list[Finding]
    disposition: Disposition
    resample_map: list[ResamplePoint]
    vlm: VlmResult | None = None
    kinematics: EvaluationStatus = Field(
        default_factory=lambda: EvaluationStatus(reason="not configured")
    )
    reprojection: EvaluationStatus = Field(
        default_factory=lambda: EvaluationStatus(reason="not configured")
    )
    qwen_core_score: float | None = Field(default=None, ge=0.0, le=1.0)
    extension_score: float | None = Field(default=None, ge=0.0, le=1.0)
    coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    qwen_core_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    extension_coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    quality_contract_version: str | None = None
    selected_checks: list[str] = Field(default_factory=list)
    selected_denominator: int = Field(default=0, ge=0)
    evaluated_count: int = Field(default=0, ge=0)
    final_decision: Literal["pass", "review", "mask", "reject"] = "review"
    metric_evidence: list[MetricEvidence] = Field(default_factory=list)


class CoverageSummary(BaseModel):
    by_source: dict[str, int] = Field(default_factory=dict)
    by_task: dict[str, int] = Field(default_factory=dict)
    by_duration: dict[str, int] = Field(default_factory=dict)
    by_motion: dict[str, int] = Field(default_factory=dict)


class PilotResult(BaseModel):
    processed: int
    resumed: int
    profile_hash: str
    manifest_hash: str
    episodes: list[EpisodeCurationResult]
    coverage: CoverageSummary
    reason_counts: dict[str, int]
    sampler_advice: dict[str, Any]

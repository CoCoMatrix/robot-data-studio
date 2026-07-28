from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


FilterStageId = Literal[
    "visual_quality",
    "sudden_change",
    "state_action_alignment",
    "extreme_value",
    "kinematic_consistency",
    "orientation_alignment",
    "metadata_completeness",
]

ALL_FILTER_STAGE_IDS: list[FilterStageId] = [
    "visual_quality",
    "sudden_change",
    "state_action_alignment",
    "extreme_value",
    "kinematic_consistency",
    "orientation_alignment",
    "metadata_completeness",
]
FilterStatus = Literal["passed", "review", "skipped"]
FilterSeverity = Literal["none", "warning", "critical"]


class FilterFinding(BaseModel):
    code: str
    severity: Literal["info", "warn", "error", "critical"]
    message: str


class FilterStageSummary(BaseModel):
    id: FilterStageId
    label: str
    count: int
    status: FilterStatus
    skipped_reason: str | None = None


class EpisodeFilterStageStatus(BaseModel):
    count: int
    status: FilterStatus
    score: float | None = Field(default=None, ge=0, le=1)
    severity: FilterSeverity = "none"
    skipped_reason: str | None = None


class EpisodeFilterSummary(BaseModel):
    episode_index: int
    stage_status: dict[FilterStageId, EpisodeFilterStageStatus]
    critical_findings: list[FilterFinding] = Field(default_factory=list)


class FilterSummary(BaseModel):
    dataset_path: str
    total_episodes: int
    total_frames: int
    stages: list[FilterStageSummary]
    episodes: list[EpisodeFilterSummary]


class VisualQualityEvidenceFrame(BaseModel):
    frame: int
    timestamp: float


class VisualQualityMetricSample(BaseModel):
    frame: int
    timestamp: float
    sharpness: float | None = None
    brightness: float | None = None
    contrast: float | None = None


class VisualQualityIncident(BaseModel):
    id: str
    camera: str
    issue: str
    start_frame: int
    end_frame: int
    start_timestamp: float
    end_timestamp: float
    sample_count: int
    worst_value: float | str
    threshold: float | str
    representative_frames: list[VisualQualityEvidenceFrame] = Field(default_factory=list)
    severity: Literal["warning", "critical"] = "warning"
    hard_failure: bool = False


class VisualCameraScore(BaseModel):
    camera: str
    technical_score: float | None = Field(default=None, ge=0, le=1)
    percentile: float | None = Field(default=None, ge=0, le=1)
    cohort_size: int = Field(default=0, ge=0)
    priority: Literal["normal", "review", "high"] = "normal"
    model_status: Literal["scored", "unavailable", "failed"] = "unavailable"
    error_code: str | None = None


class VisualQualityDetail(BaseModel):
    sampled_frame_count: int
    camera_count: int
    issue_sample_count: int
    affected_camera_count: int
    episode_frame_count: int
    episode_duration_seconds: float
    incidents: list[VisualQualityIncident] = Field(default_factory=list)
    metrics: dict[str, list[VisualQualityMetricSample]] = Field(default_factory=dict)
    technical_score: float | None = Field(default=None, ge=0, le=1)
    percentile: float | None = Field(default=None, ge=0, le=1)
    camera_scores: list[VisualCameraScore] = Field(default_factory=list)
    hard_failure: bool = False
    hard_failure_reasons: list[str] = Field(default_factory=list)
    requested_backend: Literal["auto", "rules", "dover_mobile"] = "auto"
    effective_backend: Literal["rules", "dover_mobile"] = "rules"
    model_status: Literal["scored", "partial", "unavailable", "failed"] = "unavailable"
    artifact_fingerprint: str | None = None


class FilterDetail(BaseModel):
    stage_id: FilterStageId
    episode_index: int
    title: str
    status: FilterStatus
    series: dict[str, list[float]] = Field(default_factory=dict)
    thresholds: dict[str, dict[str, float]] = Field(default_factory=dict)
    table_rows: list[dict[str, Any]] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    findings: list[FilterFinding] = Field(default_factory=list)
    skipped_reason: str | None = None
    visual_quality: VisualQualityDetail | None = None


class FilterRun(BaseModel):
    run_id: str
    status: Literal["succeeded", "failed"]
    summary: FilterSummary
    visual_quality_details: dict[int, FilterDetail] = Field(
        default_factory=dict,
        exclude=True,
    )


class FilterKinematicsConfig(BaseModel):
    urdf_path: str | None = None
    end_effector_link: str | None = None
    joint_names: list[str] = Field(default_factory=list)
    joint_state_indices: list[int] = Field(default_factory=list)
    eef_position_indices: list[int] = Field(default_factory=list)
    position_tolerance: float = 0.05
    resolve_tcp_offset: bool = True


class MetadataCompletenessConfig(BaseModel):
    require_task_description: bool = True
    min_camera_count: int = Field(default=2, ge=1)
    expected_camera_prefix: str = "observation.images."


class FilterVisualQualityConfig(BaseModel):
    backend: Literal["auto", "rules", "dover_mobile"] = "auto"
    dover_python_path: str | None = None
    checkpoint_path: str | None = None
    model_device: Literal["auto", "cpu", "cuda"] = "auto"
    batch_size: int = Field(default=8, ge=1, le=64)
    review_percentile: float = Field(default=0.05, gt=0, le=1)
    high_priority_percentile: float = Field(default=0.01, gt=0, le=1)
    min_cohort_size: int = Field(default=20, ge=1)
    camera_aggregation: Literal["min", "median"] = "min"
    sample_fps: float = Field(default=2.0, gt=0)
    max_frames_per_video: int = Field(default=48, ge=1, deprecated=True)
    max_parallel_video_decodes: int = Field(default=4, ge=1, le=8)
    sample_width: int = Field(default=160, ge=16)
    sample_height: int = Field(default=120, ge=16)
    blur_laplacian_threshold: float = Field(default=18.0, ge=0)
    dark_mean_threshold: float = Field(default=25.0, ge=0, le=255)
    bright_mean_threshold: float = Field(default=235.0, ge=0, le=255)
    dark_global_mean_threshold: float = Field(default=85.0, ge=0, le=255)
    dark_global_p75_threshold: float = Field(default=120.0, ge=0, le=255)
    bright_global_mean_threshold: float = Field(default=155.0, ge=0, le=255)
    bright_global_p75_threshold: float = Field(default=185.0, ge=0, le=255)
    low_contrast_std_threshold: float = Field(default=12.0, ge=0)
    freeze_mse_threshold: float = Field(default=1.0, ge=0)
    freeze_min_run: int = Field(default=4, ge=2)
    black_pixel_threshold: int = Field(default=10, ge=0, le=255)
    white_pixel_threshold: int = Field(default=245, ge=0, le=255)
    saturated_area_threshold: float = Field(default=0.80, gt=0, le=1)
    hard_failure_duration_seconds: float = Field(default=2.0, gt=0)
    freeze_duration_seconds: float = Field(default=2.0, gt=0)
    robot_motion_threshold: float = Field(default=1e-4, ge=0)


class FilterTimeSyncConfig(BaseModel):
    timestamp_jitter_seconds: float = Field(default=0.01, ge=0)
    timestamp_jitter_ratio: float = Field(default=0.25, ge=0)
    duration_tolerance_seconds: float = Field(default=0.1, ge=0)
    video_boundary_tolerance_seconds: float = Field(default=0.1, ge=0)


class FilterConfig(BaseModel):
    gripper_indices: list[int] = Field(default_factory=list)
    enabled_filter_stages: list[FilterStageId] = Field(
        default_factory=lambda: ALL_FILTER_STAGE_IDS.copy()
    )
    kinematics: FilterKinematicsConfig = Field(default_factory=FilterKinematicsConfig)
    visual_quality: FilterVisualQualityConfig = Field(default_factory=FilterVisualQualityConfig)
    time_sync: FilterTimeSyncConfig = Field(default_factory=FilterTimeSyncConfig)
    metadata_completeness: MetadataCompletenessConfig = Field(
        default_factory=MetadataCompletenessConfig
    )


class FilterConfigPatch(BaseModel):
    gripper_indices: list[int] | None = None
    kinematics: FilterKinematicsConfig | None = None
    visual_quality: FilterVisualQualityConfig | None = None
    time_sync: FilterTimeSyncConfig | None = None
    metadata_completeness: MetadataCompletenessConfig | None = None

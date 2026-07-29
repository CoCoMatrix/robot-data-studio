from __future__ import annotations

import fnmatch
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .human_checks import HumanExtensionConfig
from .sam3 import Sam3Config
from robot_data_studio.quality.contracts import CORE_CONTRACT_IDS, HUMAN_EXTENSION_IDS


class SelectionProfile(BaseModel):
    train_allowlist: list[str] = Field(min_length=1)
    deny_patterns: list[str] = Field(default_factory=list)
    real_episode_allowlist: list[str] = Field(default_factory=list)


class TaskProfile(BaseModel):
    text: str = Field(min_length=1)
    events: list[Literal["start", "contact", "end"]] = Field(min_length=1)


class ModalityProfile(BaseModel):
    name: str
    required: bool = True


class ProgressProfile(BaseModel):
    mode: Literal["fixed_time", "event_or_motion"] = "fixed_time"
    target_points: int = Field(default=64, ge=1, le=4096)


class ChannelProfile(BaseModel):
    name: str
    state_index: int = Field(ge=0)
    action_index: int = Field(ge=0)
    action_min: float | None = None
    action_max: float | None = None
    allow_sign_flip: bool = False


class KinematicsProfile(BaseModel):
    backend: Literal["qwen_urdf", "pinocchio"] = "qwen_urdf"
    urdf_path: str | None = None
    urdf_package: str | None = None
    urdf_relative_path: str | None = None
    urdf_sha256: str | None = None
    joint_names: list[str] = Field(default_factory=list)
    joint_state_indices: list[int] = Field(default_factory=list)
    root_link: str = "base_link"
    end_effector_link: str | None = None
    end_effector_links: dict[str, str] = Field(default_factory=dict)
    end_effector_link_candidates: list[str] = Field(default_factory=list)
    action_target_policy: Literal["anchor_state_plus_named_action"] | None = None
    action_target_joint_names: list[str] = Field(default_factory=list)
    authority_evidence: str | None = None
    authority_evidence_path: str | None = None
    authority_evidence_sha256: str | None = None
    logged_eef_key: str | None = None
    logged_eef_indices: list[int] = Field(default_factory=list)
    position_tolerance: float = Field(default=0.05, gt=0)
    explicit_unrecorded_joint_positions: dict[str, float] = Field(default_factory=dict)
    unrecorded_joint_authority: str | None = None
    visual_scope: Literal["full_urdf", "pose_authorized_upper_body"] = "full_urdf"
    excluded_visual_joint_roots: list[str] = Field(default_factory=list)
    visual_scope_authority_path: str | None = None
    visual_scope_authority_sha256: str | None = None


class ReprojectionProfile(BaseModel):
    intrinsics_path: str | None = None
    extrinsics_path: str | None = None
    calibration_sidecar_path: str | None = None
    calibration_sidecar_sha256: str | None = None
    authority_evidence_path: str | None = None
    authority_evidence_sha256: str | None = None
    camera_link: str | None = None
    optical_axis_convention: str | None = None
    eef_image_binding: str | None = None
    calibration_image_size: tuple[int, int] | None = None
    stored_image_size: tuple[int, int] | None = None
    stored_intrinsic: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ] | None = None
    raw_intrinsic: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ] | None = None
    distortion_coefficients: tuple[float, ...] | None = None
    distortion_model: Literal["opencv_14d"] | None = None
    display_transform: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ] | None = None
    stored_video_undistorted: bool | None = None
    projection_lens_contract: Literal[
        "frozen_sidecar_14d", "train_only_stored_k_override_v2"
    ] = "frozen_sidecar_14d"
    projection_override_evidence_path: str | None = None
    projection_override_evidence_sha256: str | None = None
    binding_verified: bool = False
    mask_provider: str | None = None
    sam3: Sam3Config = Field(default_factory=Sam3Config)


class FullSilhouetteC2Profile(BaseModel):
    contract_version: Literal["qwen-robotmanip-c2-full-silhouette-v2"] = (
        "qwen-robotmanip-c2-full-silhouette-v2"
    )
    enabled: bool = False
    calibration_version: str | None = None
    mask_checkpoint_kind: Literal["official_sam3", "fine_tuned_sam3"] = "official_sam3"
    calibrated: bool = False
    threshold_version: str | None = None
    iou_review_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    temporal_offsets_frames: list[int] = Field(default_factory=lambda: [-2, -1, 0, 1, 2])
    keyframes_per_episode: int = Field(default=5, ge=3, le=8)
    projection_scope: Literal["full_urdf_visual_silhouette"] = (
        "full_urdf_visual_silhouette"
    )
    hard_reject_allowed: Literal[False] = False

    @model_validator(mode="after")
    def calibrated_assets_are_explicit(self) -> FullSilhouetteC2Profile:
        if self.calibrated and self.mask_checkpoint_kind != "fine_tuned_sam3":
            raise ValueError("calibrated C2 requires a fine-tuned SAM3 checkpoint")
        if self.calibrated and (not self.calibration_version or not self.threshold_version):
            raise ValueError("calibrated C2 requires versioned calibration and threshold")
        if 0 not in self.temporal_offsets_frames:
            raise ValueError("C2 temporal diagnostic must include exact frame offset 0")
        if self.calibrated and not self.hard_reject_allowed:
            raise ValueError("calibrated C2 must use a future contract that permits decisions")
        return self


class GeometryProfile(BaseModel):
    kinematics: KinematicsProfile = Field(default_factory=KinematicsProfile)
    reprojection: ReprojectionProfile = Field(default_factory=ReprojectionProfile)
    c2_full_silhouette: FullSilhouetteC2Profile = Field(
        default_factory=FullSilhouetteC2Profile
    )


class SourceProfile(BaseModel):
    source_id: str
    embodiment: Literal["human", "robot"]
    adapter: Literal["jsonl", "xjgt_hdf5", "lerobot_v3"] = "jsonl"
    fps: float = Field(gt=0)
    decoded_pts_tolerance_seconds: float = Field(default=0.05, ge=0)
    latency_frames: int = 0
    latency_tolerance_frames: int = Field(default=1, ge=0)
    validity_field: str = "valid"
    resampling: Literal["nearest"] = "nearest"
    state_fields: list[str] = Field(default_factory=list)
    state_key: str = "observation.state"
    action_key: str | None = "action"
    validity_fields: list[str] = Field(default_factory=list)
    video_key: str | None = None
    channels: list[ChannelProfile] = Field(default_factory=list)
    geometry: GeometryProfile = Field(default_factory=GeometryProfile)
    human_extension: HumanExtensionConfig = Field(default_factory=HumanExtensionConfig)


class GateProfile(BaseModel):
    hard: list[str] = Field(default_factory=list)
    review: list[str] = Field(default_factory=list)


class VlmProfile(BaseModel):
    backend: Literal["disabled", "mock"] = "disabled"
    model: str = "mock-v1"
    prompt: str = "Compare task, keyframes, action, and time evidence."
    max_calls: int = Field(default=0, ge=0)
    max_prompt_tokens: int = Field(default=0, ge=0)
    max_cost_usd: float = Field(default=0.0, ge=0.0)
    estimated_cost_per_call_usd: float = Field(default=0.0, ge=0.0)


class NormalizationProfile(BaseModel):
    source_weights: dict[str, float] = Field(default_factory=dict)
    task_weights: dict[str, float] = Field(default_factory=dict)


class SignalCheckProfile(BaseModel):
    boundary_tolerance_seconds: float = Field(default=0.1, ge=0)
    timestamp_jitter_ratio: float = Field(default=0.25, ge=0)
    idle_motion_threshold: float = Field(default=1e-4, ge=0)
    max_idle_ratio: float = Field(default=0.8, ge=0, le=1)
    action_repeat_frames: int = Field(default=8, ge=2)
    quantization_unique_ratio: float = Field(default=0.1, ge=0, le=1)
    saturation_tolerance: float = Field(default=1e-6, ge=0)


class CheckControl(BaseModel):
    enabled: bool = True
    weight: float | None = Field(default=None, ge=0.0)


class QualityContractProfile(BaseModel):
    contract_version: Literal["qwen-cleaning-v2"] = "qwen-cleaning-v2"
    qwen: dict[str, CheckControl]
    human_extensions: dict[str, CheckControl]

    @model_validator(mode="after")
    def known_and_explicit_checks(self) -> QualityContractProfile:
        qwen_ids = set(self.qwen)
        if qwen_ids != set(CORE_CONTRACT_IDS):
            missing = sorted(set(CORE_CONTRACT_IDS) - qwen_ids)
            unknown = sorted(qwen_ids - set(CORE_CONTRACT_IDS))
            raise ValueError(
                f"qwen selection must explicitly contain current checks; missing={missing}, unknown/retired={unknown}"
            )
        human_ids = set(self.human_extensions)
        if human_ids != set(HUMAN_EXTENSION_IDS):
            missing = sorted(set(HUMAN_EXTENSION_IDS) - human_ids)
            unknown = sorted(human_ids - set(HUMAN_EXTENSION_IDS))
            raise ValueError(
                f"human extension selection must be explicit; missing={missing}, unknown={unknown}"
            )
        return self


class CurationProfile(BaseModel):
    profile_version: Literal["1", "2"] = "1"
    name: str = Field(min_length=1)
    selection: SelectionProfile
    task: TaskProfile
    modalities: list[ModalityProfile] = Field(min_length=1)
    progress: ProgressProfile = Field(default_factory=ProgressProfile)
    sources: list[SourceProfile] = Field(min_length=1)
    gates: GateProfile = Field(default_factory=GateProfile)
    checks: SignalCheckProfile = Field(default_factory=SignalCheckProfile)
    vlm: VlmProfile = Field(default_factory=VlmProfile)
    sidecar_root: str = ".rds-artifacts/curation"
    normalization: NormalizationProfile = Field(default_factory=NormalizationProfile)
    quality_contract: QualityContractProfile | None = None

    @model_validator(mode="after")
    def unique_sources(self) -> CurationProfile:
        ids = [source.source_id for source in self.sources]
        if len(ids) != len(set(ids)):
            raise ValueError("source_id values must be unique")
        if self.profile_version == "2" and self.quality_contract is None:
            raise ValueError("quality_contract is required for profile_version 2")
        if self.profile_version == "1" and self.quality_contract is not None:
            raise ValueError("quality_contract requires profile_version 2")
        return self

    def source(self, source_id: str) -> SourceProfile:
        for source in self.sources:
            if source.source_id == source_id:
                return source
        raise KeyError(f"Unknown source_id: {source_id}")


def load_profile(path: str | Path) -> CurationProfile:
    return CurationProfile.model_validate_json(Path(path).read_text(encoding="utf-8"))


def dump_profile_schema(path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(CurationProfile.model_json_schema(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def dump_quality_contract_schema(path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(QualityContractProfile.model_json_schema(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def is_source_allowed(profile: CurationProfile, source_path: str, *, split: str) -> bool:
    if split != "train":
        return False
    normalized = source_path.replace("\\", "/")
    folded = normalized.casefold()
    if any(
        fnmatch.fnmatch(folded, pattern.replace("\\", "/").casefold())
        for pattern in profile.selection.deny_patterns
    ):
        return False
    return any(
        fnmatch.fnmatch(folded, pattern.replace("\\", "/").casefold())
        for pattern in profile.selection.train_allowlist
    )

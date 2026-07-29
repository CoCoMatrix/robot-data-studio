from __future__ import annotations

import hashlib
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Protocol

import numpy as np

from .models import EpisodeRecord, EvaluationStatus, EvidenceFrame
from .profile import SourceProfile


def select_keyframe_evidence(episode: EpisodeRecord) -> list[EvidenceFrame]:
    if not episode.frames:
        return []
    by_index = {frame.frame_index: frame for frame in episode.frames}
    defaults = {"start": episode.frames[0].frame_index, "end": episode.frames[-1].frame_index}
    evidence = []
    for label in ("start", "contact", "end"):
        selected = episode.event_frames.get(label, defaults.get(label))
        frame = by_index.get(selected) if selected is not None else None
        if frame is not None:
            evidence.append(
                EvidenceFrame(
                    label=label,
                    frame_index=frame.frame_index,
                    timestamp=frame.timestamp,
                    uri=frame.video_uri,
                )
            )
    return evidence


class HumanSparseVideoAdapter:
    kind = "human"

    def keyframes(self, episode: EpisodeRecord) -> list[EvidenceFrame]:
        return select_keyframe_evidence(episode)


class RobotSparseVideoAdapter:
    kind = "robot"

    def keyframes(self, episode: EpisodeRecord) -> list[EvidenceFrame]:
        return select_keyframe_evidence(episode)


class ForwardKinematics(Protocol):
    """TODO interface; the research prototype does not supply a robot-specific solver."""

    def forward(self, joints: Sequence[float]) -> Sequence[float]: ...


class Reprojection(Protocol):
    """TODO interface for camera-specific projection evidence."""

    def project(self, points: Sequence[Sequence[float]]) -> Sequence[Sequence[float]]: ...


class ContactSheetBuilder(Protocol):
    """Interface for composing already-selected keyframes; no media is copied."""

    def build(self, evidence: Sequence[EvidenceFrame]) -> str: ...


def resolve_urdf_path(source: SourceProfile) -> tuple[Path | None, str | None]:
    config = source.geometry.kinematics
    if config.urdf_path:
        path = Path(config.urdf_path)
    elif config.urdf_package and config.urdf_relative_path:
        try:
            package = distribution(config.urdf_package)
        except PackageNotFoundError:
            return None, f"optional package {config.urdf_package!r} is not installed"
        path = Path(package.locate_file(config.urdf_relative_path))
    else:
        return None, "urdf path/package locator missing"
    if not path.is_file():
        return None, f"URDF not found: {path}"
    if config.urdf_sha256:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != config.urdf_sha256:
            return None, f"URDF hash mismatch: expected {config.urdf_sha256}, got {digest}"
    return path, None


def evaluate_geometry(
    episode: EpisodeRecord,
    source: SourceProfile,
) -> tuple[EvaluationStatus, EvaluationStatus]:
    config = source.geometry.kinematics
    urdf_path, urdf_error = resolve_urdf_path(source)
    missing = []
    if urdf_path is None:
        missing.append(f"urdf ({urdf_error})")
    if not config.joint_names or not config.joint_state_indices:
        missing.append("joint_name_mapping")
    if not config.end_effector_link:
        missing.append("eef_link")
    if len(config.logged_eef_indices) < 3:
        missing.append("logged_eef_mapping")
    if missing:
        kinematics = EvaluationStatus(
            reason="missing trusted " + ", ".join(missing),
            backend=config.backend,
        )
    else:
        states = np.asarray([frame.state for frame in episode.frames if frame.valid], dtype=np.float64)
        try:
            if config.backend == "pinocchio":
                from robot_data_studio.quality.filter_models import FilterKinematicsConfig
                from robot_data_studio.quality.filter_service import _pinocchio_fk_detail

                detail = _pinocchio_fk_detail(
                    states,
                    FilterKinematicsConfig(
                        urdf_path=str(urdf_path),
                        end_effector_link=config.end_effector_link,
                        joint_names=config.joint_names,
                        joint_state_indices=config.joint_state_indices,
                        eef_position_indices=config.logged_eef_indices,
                        position_tolerance=config.position_tolerance,
                    ),
                )
                errors = detail["errors"]
                mean_error = float(np.mean(errors)) if len(errors) else 0.0
                max_error = float(np.max(errors)) if len(errors) else 0.0
            else:
                from robot_data_studio.quality.qwen_filters import (
                    KinematicsConfig,
                    evaluate_kinematic_consistency,
                )

                joint_states = states[:, config.joint_state_indices]
                logged_eef = states[:, config.logged_eef_indices[:3]]
                result = evaluate_kinematic_consistency(
                    joint_states,
                    logged_eef,
                    KinematicsConfig(
                        urdf_path=urdf_path,
                        end_effector_link=config.end_effector_link,
                        joint_names=config.joint_names,
                        position_tolerance=config.position_tolerance,
                    ),
                )
                mean_error = result.mean_position_error
                max_error = result.max_position_error
            kinematics = EvaluationStatus(
                state="evaluated",
                reason="trusted geometry configuration supplied",
                backend=config.backend,
                metrics={"mean_position_error": mean_error, "max_position_error": max_error},
            )
        except (ImportError, OSError, ValueError, IndexError) as error:
            kinematics = EvaluationStatus(
                reason=f"geometry adapter unavailable: {error}",
                backend=config.backend,
            )

    reprojection_config = source.geometry.reprojection
    reprojection_missing = []
    if not reprojection_config.calibration_sidecar_path:
        if not reprojection_config.intrinsics_path:
            reprojection_missing.append("intrinsics")
        if not reprojection_config.extrinsics_path:
            reprojection_missing.append("extrinsics")
    if not reprojection_config.optical_axis_convention:
        reprojection_missing.append("optical_axis_convention")
    if not reprojection_config.eef_image_binding:
        reprojection_missing.append("eef_image_binding")
    if not reprojection_config.binding_verified:
        reprojection_missing.append("calibration_to_stored_image_binding")
    reprojection = EvaluationStatus(
        reason=(
            "missing trusted " + ", ".join(reprojection_missing)
            if reprojection_missing
            else "projection evidence available; robot-mask provider remains required for C2 IoU"
        )
    )
    return kinematics, reprojection


class MotionProxy(Protocol):
    """Cheap optional motion proxy; motive attribution remains post-training only."""

    def score(self, episode: EpisodeRecord) -> float: ...


class NearestNeighborActionConflict(Protocol):
    """TODO interface intentionally not implemented in the first prototype."""

    def conflicts(self, episode: EpisodeRecord) -> Sequence[int]: ...


class TrajectoryEntropy(Protocol):
    """TODO interface intentionally not implemented in the first prototype."""

    def entropy(self, episode: EpisodeRecord) -> float: ...

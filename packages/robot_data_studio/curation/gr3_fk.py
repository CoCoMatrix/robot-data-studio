from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Sequence

import numpy as np

from robot_data_studio.quality.qwen_filters import (
    _load_urdf_chain,
    _motion_transform,
    _transform_from_xyz_rpy,
)

from .models import RawPose
from .profile import KinematicsProfile, ReprojectionProfile


@lru_cache(maxsize=16)
def _file_sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=16)
def _chain(path: str, link: str):
    return _load_urdf_chain(Path(path), link)


def _urdf_path(config: KinematicsProfile) -> tuple[Path | None, str | None]:
    if config.urdf_path:
        path = Path(config.urdf_path)
    elif config.urdf_package and config.urdf_relative_path:
        try:
            package = distribution(config.urdf_package)
        except PackageNotFoundError:
            return None, f"authority package {config.urdf_package!r} is not installed"
        path = Path(package.locate_file(config.urdf_relative_path))
    else:
        return None, "URDF authority path/package is missing"
    if not path.is_file():
        return None, f"URDF authority file is missing: {path}"
    digest = _file_sha256(str(path.resolve()))
    if not config.urdf_sha256:
        return None, "URDF authority hash is missing"
    if digest != config.urdf_sha256:
        return None, f"URDF hash mismatch: expected {config.urdf_sha256}, got {digest}"
    return path, None


def _missing(config: KinematicsProfile, reason: str) -> tuple[dict[str, RawPose], dict[str, RawPose], dict]:
    links = config.end_effector_links or (
        {"eef": config.end_effector_link}
        if config.end_effector_link
        else {"left": "left_eef", "right": "right_eef"}
    )
    poses = {
        link: RawPose(
            valid=False,
            reference_frame=config.root_link,
            child_frame=link,
            transform_direction=f"T_{config.root_link}_{link}",
            missing_reason=reason,
        )
        for link in links.values()
    }
    return poses, {name: pose.model_copy(deep=True) for name, pose in poses.items()}, {
        "state": "not_evaluated",
        "reason": reason,
        "deployment_tcp_claimed": False,
    }


def _transform(
    path: Path,
    link: str,
    root_link: str,
    joint_values: dict[str, float],
) -> np.ndarray:
    chain = _chain(str(path.resolve()), link)
    if chain[0].parent != root_link:
        raise ValueError(
            f"EEF chain root {chain[0].parent!r} does not match authority {root_link!r}"
        )
    value = np.eye(4, dtype=np.float64)
    for joint in chain:
        value = value @ _transform_from_xyz_rpy(joint.origin_xyz, joint.origin_rpy)
        if joint.joint_type in {"revolute", "continuous", "prismatic"}:
            if joint.name not in joint_values:
                raise ValueError(f"required FK joint names missing: {joint.name}")
            value = value @ _motion_transform(
                joint.joint_type, joint.axis, joint_values[joint.name]
            )
    return value


def build_gr3_pose_features(
    *,
    state: Sequence[float],
    action: Sequence[float],
    state_names: Sequence[str],
    action_names: Sequence[str],
    config: KinematicsProfile,
) -> tuple[dict[str, RawPose], dict[str, RawPose], dict]:
    """Reuse the existing Qwen URDF chain math under an explicit frozen authority.

    The returned frame is the URDF experiment base frame. It is deliberately
    not described as a deployment TCP or world frame.
    """

    path, error = _urdf_path(config)
    if path is None:
        return _missing(config, error or "URDF authority unavailable")
    if not config.authority_evidence:
        return _missing(config, "EEF authority evidence is missing")
    if config.authority_evidence_path:
        evidence_path = Path(config.authority_evidence_path)
        if not evidence_path.is_file():
            return _missing(config, f"EEF authority evidence file is missing: {evidence_path}")
        evidence_sha = _file_sha256(str(evidence_path.resolve()))
        if evidence_sha != config.authority_evidence_sha256:
            return _missing(config, "EEF authority evidence hash mismatch")
    else:
        evidence_sha = None
    if set(config.end_effector_links) != {"left", "right"}:
        return _missing(config, "left/right EEF link authority is incomplete")
    if len(state) != len(state_names) or len(action) != len(action_names):
        return _missing(config, "state/action joint names do not match vector dimensions")
    expected = list(config.joint_names)
    if not expected or len(expected) != len(config.joint_state_indices):
        return _missing(config, "configured FK joint names/indices are incomplete")
    if any(
        index >= len(state_names) or state_names[index] != name
        for name, index in zip(expected, config.joint_state_indices, strict=True)
    ):
        return _missing(config, "source joint names do not match the frozen FK name order")
    if config.action_target_policy != "anchor_state_plus_named_action":
        return _missing(config, "action-target FK policy authority is missing")
    missing_action = sorted(set(config.action_target_joint_names) - set(action_names))
    if missing_action:
        return _missing(config, f"required action joint names missing: {missing_action}")
    state_map = {
        name: float(state[index])
        for name, index in zip(expected, config.joint_state_indices, strict=True)
    }
    action_by_name = dict(zip(action_names, (float(value) for value in action), strict=True))
    target_map = dict(state_map)
    for name in config.action_target_joint_names:
        target_map[name] = action_by_name[name]
    try:
        observed_matrices = {
            link: _transform(path, link, config.root_link, state_map)
            for link in config.end_effector_links.values()
        }
        target_matrices = {
            link: _transform(path, link, config.root_link, target_map)
            for link in config.end_effector_links.values()
        }
    except (KeyError, ValueError) as exc:
        return _missing(config, str(exc))

    def pose(link: str, matrix: np.ndarray) -> RawPose:
        return RawPose(
            matrix=matrix.reshape(-1).tolist(),
            valid=True,
            reference_frame=config.root_link,
            child_frame=link,
            transform_direction=f"T_{config.root_link}_{link}",
        )

    digest = _file_sha256(str(path.resolve()))
    return (
        {link: pose(link, matrix) for link, matrix in observed_matrices.items()},
        {link: pose(link, matrix) for link, matrix in target_matrices.items()},
        {
            "state": "evaluated",
            "authority_evidence": config.authority_evidence,
            "authority_evidence_path": config.authority_evidence_path,
            "authority_evidence_sha256": evidence_sha,
            "urdf_path": str(path),
            "urdf_sha256": digest,
            "root_link": config.root_link,
            "end_effector_links": config.end_effector_links,
            "action_target_policy": config.action_target_policy,
            "action_target_joint_names": config.action_target_joint_names,
            "deployment_tcp_claimed": False,
        },
    )


def build_gr3_camera_pose(
    *,
    state: Sequence[float],
    state_names: Sequence[str],
    kinematics: KinematicsProfile,
    reprojection: ReprojectionProfile,
) -> tuple[RawPose, dict]:
    """Build frozen Camera0 from current named state and the approved OAK sidecar."""

    def missing(reason: str) -> tuple[RawPose, dict]:
        return (
            RawPose(
                valid=False,
                reference_frame=kinematics.root_link,
                child_frame="camera0",
                transform_direction=f"T_{kinematics.root_link}_camera0",
                missing_reason=reason,
            ),
            {"state": "not_evaluated", "reason": reason},
        )

    if not reprojection.binding_verified:
        return missing("Camera0 frozen authority is not enabled")
    if (
        len(state) != len(state_names)
        or len(set(state_names)) != len(state_names)
        or not all(np.isfinite(state))
    ):
        return missing("Camera0 current state names/values are invalid")
    urdf_path, urdf_error = _urdf_path(kinematics)
    if urdf_path is None:
        return missing(urdf_error or "Camera0 URDF unavailable")
    if not reprojection.camera_link:
        return missing("Camera0 head-pitch link authority is missing")
    paths = {
        "sidecar": (reprojection.calibration_sidecar_path, reprojection.calibration_sidecar_sha256),
        "gate": (reprojection.authority_evidence_path, reprojection.authority_evidence_sha256),
    }
    loaded: dict[str, tuple[Path, str]] = {}
    for label, (raw_path, expected_sha) in paths.items():
        if not raw_path or not expected_sha:
            return missing(f"Camera0 {label} path/hash authority is missing")
        path = Path(raw_path)
        if not path.is_file():
            return missing(f"Camera0 {label} is missing: {path}")
        digest = _file_sha256(str(path.resolve()))
        if digest != expected_sha:
            return missing(f"Camera0 {label} hash mismatch")
        loaded[label] = (path, digest)
    try:
        sidecar = json.loads(loaded["sidecar"][0].read_text(encoding="utf-8"))
        gate = json.loads(loaded["gate"][0].read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        return missing(f"Camera0 authority JSON invalid: {error}")
    decision = gate.get("decision", {})
    bindings = gate.get("bindings", {})
    sidecar_binding = bindings.get("sidecar", {})
    pdf_sha256 = "e0d078fe17bec983e37157993b9d2413feff98498519ab230920a2c9c7e4d1ca"
    if (
        sidecar.get("schema") != "robot_camera0_extrinsic_v1"
        or sidecar.get("source_document", {}).get("sha256") != pdf_sha256
        or bindings.get("pdf", {}).get("sha256") != pdf_sha256
        or bindings.get("camera_fk_schema") != "fastwam_final_robot_camera0_anchor_fk_v1"
        or sidecar_binding.get("sha256") != loaded["sidecar"][1]
        or decision.get("camera0_current_ee_proprio_authorized") is not True
        or decision.get("camera0_future_action_label_authorized") is not False
        or decision.get("validation_test_eval_opened") is not False
    ):
        return missing("Camera0 gate contract differs from frozen authority")
    optical = np.asarray(sidecar.get("mechanical_to_rgb_opencv_optical"), dtype=np.float64)
    expected_optical = np.asarray([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float64)
    mechanical = sidecar.get("mechanical_frame", {})
    translation = np.asarray(
        mechanical.get("head_pitch_from_camera_translation_m", []), dtype=np.float64
    )
    intrinsics = sidecar.get("intrinsics", {})
    raw_k = np.asarray(intrinsics.get("raw_K_1280x800", []), dtype=np.float64)
    stored_k = np.asarray(
        intrinsics.get("stored_K_832x480_candidate", []), dtype=np.float64
    )
    affine = np.asarray([[0.65, 0, 0], [0, 0.65, -20], [0, 0, 1]], dtype=np.float64)
    if (
        optical.shape != (3, 3)
        or not np.array_equal(optical, expected_optical)
        or translation.shape != (3,)
        or raw_k.shape != (3, 3)
        or stored_k.shape != (3, 3)
        or not np.allclose(affine @ raw_k, stored_k, rtol=0.0, atol=1e-12)
        or tuple(reprojection.stored_image_size or ()) != (832, 480)
    ):
        return missing("Camera0 optical axis or stored-K binding differs")
    state_map = dict(zip(state_names, (float(value) for value in state), strict=True))
    try:
        base_from_head = _transform(
            urdf_path,
            reprojection.camera_link,
            kinematics.root_link,
            state_map,
        )
    except (KeyError, ValueError) as error:
        return missing(str(error))
    angle = math.radians(40.0)
    c, s = math.cos(angle), math.sin(angle)
    head_from_mechanical = np.eye(4, dtype=np.float64)
    head_from_mechanical[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    head_from_mechanical[:3, 3] = translation
    mechanical_from_optical = np.eye(4, dtype=np.float64)
    mechanical_from_optical[:3, :3] = optical.T
    base_from_camera0 = base_from_head @ head_from_mechanical @ mechanical_from_optical
    return (
        RawPose(
            matrix=base_from_camera0.reshape(-1).tolist(),
            valid=True,
            reference_frame=kinematics.root_link,
            child_frame="camera0",
            transform_direction=f"T_{kinematics.root_link}_camera0",
        ),
        {
            "state": "evaluated",
            "camera_fk_schema": bindings["camera_fk_schema"],
            "sidecar_sha256": loaded["sidecar"][1],
            "authority_evidence_sha256": loaded["gate"][1],
            "urdf_sha256": _file_sha256(str(urdf_path.resolve())),
            "camera_link": reprojection.camera_link,
            "intrinsic": bindings.get("intrinsic"),
            "stored_image_size": list(reprojection.stored_image_size or ()),
            "current_ee_proprio_authorized": True,
            "future_action_label_authorized": False,
        },
    )

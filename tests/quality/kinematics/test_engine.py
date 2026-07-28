from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robot_data_studio.quality.kinematics import (
    JointFieldMapping,
    KinematicChainProfile,
    KinematicProfile,
    PinocchioKinematics,
    PoseFieldMapping,
)


FIXTURE = Path(__file__).parent / "fixtures" / "nonstandard_robot.urdf"


def _transform_z(angle: float, translation: tuple[float, float, float]) -> np.ndarray:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = [
        [cosine, -sine, 0.0],
        [sine, cosine, 0.0],
        [0.0, 0.0, 1.0],
    ]
    transform[:3, 3] = translation
    return transform


def _quaternion_xyzw_z(angle: float) -> list[float]:
    return [0.0, 0.0, float(np.sin(angle / 2.0)), float(np.cos(angle / 2.0))]


def _first_chain_profile(base: np.ndarray, tcp: np.ndarray) -> KinematicChainProfile:
    return KinematicChainProfile(
        id="primary",
        end_effector_frame="camera_mount",
        joints=[
            JointFieldMapping(
                urdf_joint="alpha_axis",
                state_index=0,
                sign=-1,
                zero_offset=0.2,
            )
        ],
        logged_pose=PoseFieldMapping(
            position_indices=(1, 2, 3),
            orientation_indices=(4, 5, 6, 7),
            orientation_representation="quaternion_xyzw",
        ),
        base_transform=base.tolist(),
        tcp_transform=tcp.tolist(),
    )


def test_pinocchio_engine_applies_base_tcp_joint_sign_and_zero_offset() -> None:
    base = _transform_z(0.1, (0.3, -0.2, 0.1))
    tcp = _transform_z(0.05, (0.2, 0.0, 0.0))
    logged_joint_values = [0.0, 0.4, -0.5]
    rows = []
    for logged_joint in logged_joint_values:
        corrected_joint = -logged_joint + 0.2
        fk = _transform_z(corrected_joint, (0.0, 0.0, 0.0))
        fk[:3, 3] = fk[:3, :3] @ np.asarray([1.0, 0.0, 0.0])
        expected = base @ fk @ tcp
        angle = 0.1 + corrected_joint + 0.05
        rows.append(
            [
                logged_joint,
                *expected[:3, 3].tolist(),
                *_quaternion_xyzw_z(angle),
            ]
        )
    states = np.asarray(rows, dtype=np.float64)
    profile = KinematicProfile(
        urdf_path=str(FIXTURE),
        chains=[_first_chain_profile(base, tcp)],
        fingerprint="fixture-v1",
    )

    result = PinocchioKinematics(profile.urdf_path).evaluate(states, profile)

    assert set(result) == {"primary"}
    assert np.max(result["primary"].position_errors) < 1e-9
    assert np.max(result["primary"].orientation_errors) < 1e-9


def test_pinocchio_engine_evaluates_independent_chains() -> None:
    states = np.zeros((3, 16), dtype=np.float64)
    first_angles = np.asarray([0.0, 0.2, -0.4])
    second_angles = np.asarray([0.1, -0.3, 0.5])
    states[:, 0] = first_angles
    states[:, 8] = second_angles
    for index, angle in enumerate(first_angles):
        states[index, 1:4] = [np.cos(angle), np.sin(angle), 0.0]
        states[index, 4:8] = _quaternion_xyzw_z(angle)
    for index, angle in enumerate(second_angles):
        states[index, 9:12] = [0.5 * np.cos(angle), 2.0 + 0.5 * np.sin(angle), 0.0]
        states[index, 12:16] = _quaternion_xyzw_z(angle)

    profile = KinematicProfile(
        urdf_path=str(FIXTURE),
        chains=[
            KinematicChainProfile(
                id="first",
                end_effector_frame="camera_mount",
                joints=[JointFieldMapping(urdf_joint="alpha_axis", state_index=0)],
                logged_pose=PoseFieldMapping(
                    position_indices=(1, 2, 3),
                    orientation_indices=(4, 5, 6, 7),
                    orientation_representation="quaternion_xyzw",
                ),
            ),
            KinematicChainProfile(
                id="second",
                end_effector_frame="sensor_mount",
                joints=[JointFieldMapping(urdf_joint="gamma_axis", state_index=8)],
                logged_pose=PoseFieldMapping(
                    position_indices=(9, 10, 11),
                    orientation_indices=(12, 13, 14, 15),
                    orientation_representation="quaternion_xyzw",
                ),
            ),
        ],
    )

    result = PinocchioKinematics(profile.urdf_path).evaluate(states, profile)

    assert set(result) == {"first", "second"}
    assert max(item.position.maximum for item in result.values()) < 1e-9
    assert max(item.orientation.maximum for item in result.values()) < 1e-9


def test_pinocchio_engine_reports_unknown_frame_before_evaluation() -> None:
    profile = KinematicProfile(
        urdf_path=str(FIXTURE),
        chains=[
            _first_chain_profile(np.eye(4), np.eye(4)).model_copy(
                update={"end_effector_frame": "missing_frame"}
            )
        ],
    )

    with pytest.raises(ValueError, match="missing_frame"):
        PinocchioKinematics(profile.urdf_path).evaluate(np.zeros((1, 8)), profile)

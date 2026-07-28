from __future__ import annotations

from pathlib import Path

import numpy as np
import pinocchio as pin

from robot_data_studio.quality.kinematics import DiscoveryRequest, discover_profile


FIXTURE = Path(__file__).parent / "fixtures" / "nonstandard_robot.urdf"


def _matrix_to_quaternion_xyzw(matrix: np.ndarray) -> np.ndarray:
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        diagonal = int(np.argmax(np.diag(matrix)))
        if diagonal == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            w = (matrix[2, 1] - matrix[1, 2]) / scale
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
        elif diagonal == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            w = (matrix[0, 2] - matrix[2, 0]) / scale
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            w = (matrix[1, 0] - matrix[0, 1]) / scale
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale
    quaternion = np.asarray([x, y, z, w], dtype=np.float64)
    return quaternion / np.linalg.norm(quaternion)


def _matrix_to_rpy(matrix: np.ndarray) -> np.ndarray:
    pitch = np.arcsin(np.clip(-matrix[2, 0], -1.0, 1.0))
    roll = np.arctan2(matrix[2, 1], matrix[2, 2])
    yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
    return np.asarray([roll, pitch, yaw], dtype=np.float64)


def _encode_rotation(matrix: np.ndarray, representation: str) -> np.ndarray:
    quaternion = _matrix_to_quaternion_xyzw(matrix)
    if representation == "quaternion_xyzw":
        return quaternion
    if representation == "quaternion_wxyz":
        return quaternion[[3, 0, 1, 2]]
    if representation == "rotation_6d":
        return np.concatenate((matrix[:, 0], matrix[:, 1]))
    if representation == "rotation_matrix":
        return matrix.reshape(-1)
    if representation == "rotation_vector":
        return np.asarray(pin.log3(matrix), dtype=np.float64)
    if representation == "rpy":
        return _matrix_to_rpy(matrix)
    raise AssertionError(representation)


def _states_for_representation(representation: str) -> np.ndarray:
    model = pin.buildModelFromUrdf(str(FIXTURE))
    data = model.createData()
    frame_id = model.getFrameId("prism_mount")
    joint_id = model.getJointId("alpha_axis")
    rows = []
    for angle in np.linspace(-0.9, 1.1, 25):
        configuration = np.asarray(pin.neutral(model), dtype=np.float64)
        configuration[model.joints[joint_id].idx_q] = angle
        pin.forwardKinematics(model, data, configuration)
        pin.updateFramePlacements(model, data)
        transform = np.asarray(data.oMf[frame_id].homogeneous, dtype=np.float64)
        rows.append(
            np.concatenate(
                (
                    [angle],
                    transform[:3, 3],
                    _encode_rotation(transform[:3, :3], representation),
                )
            )
        )
    return np.asarray(rows, dtype=np.float64)


def test_discovery_uses_geometry_when_urdf_and_dataset_names_do_not_match() -> None:
    states = _states_for_representation("quaternion_xyzw")
    request = DiscoveryRequest(
        states=states,
        feature_names=[f"channel_{index}" for index in range(states.shape[1])],
        max_candidates=12,
        sample_count=25,
    )

    candidates = discover_profile(FIXTURE, request)

    best = candidates[0]
    chain = best.profile.chains[0]
    assert chain.end_effector_frame == "prism_mount"
    assert [(item.urdf_joint, item.state_index) for item in chain.joints] == [
        ("alpha_axis", 0)
    ]
    assert chain.logged_pose.position_indices == (1, 2, 3)
    assert chain.logged_pose.orientation_indices == (4, 5, 6, 7)
    assert chain.logged_pose.orientation_representation == "quaternion_xyzw"
    assert best.held_out.position.maximum < 1e-8
    assert best.held_out.orientation.maximum < 1e-8
    assert any("geometric" in message.lower() for message in best.evidence)


def test_discovery_identifies_supported_orientation_representations() -> None:
    representations = [
        "quaternion_xyzw",
        "quaternion_wxyz",
        "rotation_6d",
        "rotation_matrix",
        "rotation_vector",
        "rpy",
    ]
    for representation in representations:
        states = _states_for_representation(representation)
        candidates = discover_profile(
            FIXTURE,
            DiscoveryRequest(
                states=states,
                feature_names=[f"signal_{index}" for index in range(states.shape[1])],
                max_candidates=12,
                sample_count=25,
            ),
        )

        best = candidates[0]
        assert best.profile.chains[0].logged_pose.orientation_representation == representation
        assert best.held_out.position.maximum < 1e-8
        assert best.held_out.orientation.maximum < 1e-8

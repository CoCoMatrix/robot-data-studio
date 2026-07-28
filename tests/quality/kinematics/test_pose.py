from __future__ import annotations

import numpy as np
import pytest

from robot_data_studio.quality.kinematics import (
    InvalidPoseRepresentation,
    compute_se3_residuals,
    decode_rotations,
    encode_rotations,
)


def _rotation_z(angle: float) -> np.ndarray:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.asarray(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


@pytest.mark.parametrize(
    ("representation", "values"),
    [
        ("quaternion_xyzw", [0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)]),
        ("quaternion_wxyz", [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]),
        ("rotation_6d", [0.0, 1.0, 0.0, -1.0, 0.0, 0.0]),
        ("rotation_matrix", _rotation_z(np.pi / 2).reshape(-1).tolist()),
        ("rotation_vector", [0.0, 0.0, np.pi / 2]),
        ("rpy", [0.0, 0.0, np.pi / 2]),
    ],
)
def test_decode_rotations_supports_common_robot_pose_representations(
    representation: str,
    values: list[float],
) -> None:
    decoded = decode_rotations(np.asarray([values], dtype=np.float64), representation)

    assert decoded.shape == (1, 3, 3)
    assert np.allclose(decoded[0], _rotation_z(np.pi / 2), atol=1e-7)


def test_decode_rotations_projects_small_matrix_drift() -> None:
    slightly_drifted = _rotation_z(np.pi / 2)
    slightly_drifted[0, 1] -= 1e-6

    decoded = decode_rotations(slightly_drifted.reshape(1, 9), "rotation_matrix")

    assert np.allclose(decoded[0].T @ decoded[0], np.eye(3), atol=1e-10)
    assert np.linalg.det(decoded[0]) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "representation",
    [
        "quaternion_xyzw",
        "quaternion_wxyz",
        "rotation_6d",
        "rotation_matrix",
        "rotation_vector",
        "rpy",
    ],
)
def test_encode_rotations_round_trips_all_supported_representations(
    representation: str,
) -> None:
    rotations = decode_rotations([[0.31, -0.42, 0.57]], "rpy")

    encoded = encode_rotations(rotations, representation)
    decoded = decode_rotations(encoded, representation)

    assert np.allclose(decoded, rotations, atol=1e-8)


@pytest.mark.parametrize(
    ("values", "representation", "message"),
    [
        ([[0.0, 0.0, 0.0, 0.0]], "quaternion_xyzw", "zero-norm quaternion"),
        ([[1.0, 0.0, 0.0]], "quaternion_xyzw", "requires 4 values"),
        ([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0]], "rotation_matrix", "invalid rotation matrix"),
        ([[np.nan, 0.0, 0.0]], "rotation_vector", "non-finite"),
    ],
)
def test_decode_rotations_rejects_invalid_pose_values(
    values: list[list[float]],
    representation: str,
    message: str,
) -> None:
    with pytest.raises(InvalidPoseRepresentation, match=message):
        decode_rotations(np.asarray(values, dtype=np.float64), representation)


def test_compute_se3_residuals_reports_translation_and_orientation_separately() -> None:
    predicted = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], 2, axis=0)
    logged = predicted.copy()
    logged[0, :3, 3] = [0.03, 0.04, 0.0]
    logged[1, :3, :3] = _rotation_z(np.pi / 2)

    result = compute_se3_residuals(predicted, logged)

    assert np.allclose(result.position_errors, [0.05, 0.0], atol=1e-9)
    assert np.allclose(result.orientation_errors, [0.0, np.pi / 2], atol=1e-9)
    assert result.position.median == pytest.approx(0.025)
    assert result.position.p95 == pytest.approx(0.0475)
    assert result.position.maximum == pytest.approx(0.05)
    assert result.orientation.median == pytest.approx(np.pi / 4)
    assert result.orientation.maximum == pytest.approx(np.pi / 2)
    assert np.allclose(result.error_transforms, logged, atol=1e-9)


def test_compute_se3_residuals_rejects_mismatched_batches() -> None:
    with pytest.raises(ValueError, match="same shape"):
        compute_se3_residuals(
            np.repeat(np.eye(4)[None, :, :], 2, axis=0),
            np.repeat(np.eye(4)[None, :, :], 3, axis=0),
        )

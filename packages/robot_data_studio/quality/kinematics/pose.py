from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .models import OrientationRepresentation, ResidualSummary, SE3ResidualResult


class InvalidPoseRepresentation(ValueError):
    pass


_REPRESENTATION_WIDTHS: dict[str, int] = {
    "quaternion_xyzw": 4,
    "quaternion_wxyz": 4,
    "rotation_6d": 6,
    "rotation_matrix": 9,
    "rotation_vector": 3,
    "rpy": 3,
}


def decode_rotations(
    values: ArrayLike,
    representation: OrientationRepresentation | str,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        raise InvalidPoseRepresentation("orientation values must be a 2D array")
    width = _REPRESENTATION_WIDTHS.get(representation)
    if width is None:
        raise InvalidPoseRepresentation(f"unsupported orientation representation: {representation}")
    if array.shape[1] != width:
        raise InvalidPoseRepresentation(f"{representation} requires {width} values")
    if not np.all(np.isfinite(array)):
        raise InvalidPoseRepresentation("orientation values contain non-finite values")

    if representation in {"quaternion_xyzw", "quaternion_wxyz"}:
        return _decode_quaternions(array, representation)
    if representation == "rotation_6d":
        return _decode_rotation_6d(array)
    if representation == "rotation_matrix":
        return _decode_rotation_matrices(array)
    if representation == "rotation_vector":
        return np.asarray([_rotation_from_vector(row) for row in array])
    if representation == "rpy":
        return np.asarray([_rotation_from_rpy(*row) for row in array])
    raise InvalidPoseRepresentation(f"unsupported orientation representation: {representation}")


def compute_se3_residuals(predicted: ArrayLike, logged: ArrayLike) -> SE3ResidualResult:
    predicted_array = _validate_transforms(predicted, "predicted")
    logged_array = _validate_transforms(logged, "logged")
    if predicted_array.shape != logged_array.shape:
        raise ValueError("predicted and logged transforms must have the same shape")

    predicted_inverse = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], len(predicted_array), axis=0)
    predicted_inverse[:, :3, :3] = np.transpose(predicted_array[:, :3, :3], (0, 2, 1))
    predicted_inverse[:, :3, 3] = -np.einsum(
        "tij,tj->ti",
        predicted_inverse[:, :3, :3],
        predicted_array[:, :3, 3],
    )
    errors = predicted_inverse @ logged_array
    position_errors = np.linalg.norm(errors[:, :3, 3], axis=1)
    traces = np.trace(errors[:, :3, :3], axis1=1, axis2=2)
    cosines = np.clip((traces - 1.0) * 0.5, -1.0, 1.0)
    orientation_errors = np.arccos(cosines)
    return SE3ResidualResult(
        position_errors=position_errors,
        orientation_errors=orientation_errors,
        error_transforms=errors,
        position=_summarize(position_errors),
        orientation=_summarize(orientation_errors),
    )


def _decode_quaternions(array: np.ndarray, representation: str) -> np.ndarray:
    if representation == "quaternion_wxyz":
        w = array[:, 0]
        x = array[:, 1]
        y = array[:, 2]
        z = array[:, 3]
    else:
        x = array[:, 0]
        y = array[:, 1]
        z = array[:, 2]
        w = array[:, 3]
    norms = np.sqrt(w * w + x * x + y * y + z * z)
    if np.any(norms <= 1e-12):
        raise InvalidPoseRepresentation("zero-norm quaternion")
    w, x, y, z = (component / norms for component in (w, x, y, z))
    matrices = np.empty((len(array), 3, 3), dtype=np.float64)
    matrices[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrices[:, 0, 1] = 2.0 * (x * y - z * w)
    matrices[:, 0, 2] = 2.0 * (x * z + y * w)
    matrices[:, 1, 0] = 2.0 * (x * y + z * w)
    matrices[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrices[:, 1, 2] = 2.0 * (y * z - x * w)
    matrices[:, 2, 0] = 2.0 * (x * z - y * w)
    matrices[:, 2, 1] = 2.0 * (y * z + x * w)
    matrices[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return matrices


def _decode_rotation_6d(array: np.ndarray) -> np.ndarray:
    first = array[:, :3]
    second = array[:, 3:]
    first_norms = np.linalg.norm(first, axis=1)
    if np.any(first_norms <= 1e-12):
        raise InvalidPoseRepresentation("rotation_6d contains a zero first axis")
    first = first / first_norms[:, None]
    second = second - np.sum(first * second, axis=1)[:, None] * first
    second_norms = np.linalg.norm(second, axis=1)
    if np.any(second_norms <= 1e-12):
        raise InvalidPoseRepresentation("rotation_6d contains collinear axes")
    second = second / second_norms[:, None]
    third = np.cross(first, second)
    return np.stack((first, second, third), axis=2)


def _decode_rotation_matrices(array: np.ndarray) -> np.ndarray:
    matrices = array.reshape(-1, 3, 3)
    projected = []
    for matrix in matrices:
        orthogonality_error = float(np.linalg.norm(matrix.T @ matrix - np.eye(3), ord="fro"))
        determinant = float(np.linalg.det(matrix))
        if orthogonality_error > 1e-3 or abs(determinant - 1.0) > 1e-3:
            raise InvalidPoseRepresentation("invalid rotation matrix")
        left, _, right = np.linalg.svd(matrix)
        rotation = left @ right
        if np.linalg.det(rotation) < 0:
            left[:, -1] *= -1.0
            rotation = left @ right
        projected.append(rotation)
    return np.asarray(projected, dtype=np.float64)


def _rotation_from_vector(vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(vector))
    if angle <= 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = vector / angle
    x, y, z = axis
    skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def _rotation_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    sr, cr = np.sin(roll), np.cos(roll)
    sp, cp = np.sin(pitch), np.cos(pitch)
    sy, cy = np.sin(yaw), np.cos(yaw)
    rotation_x = np.asarray([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    rotation_y = np.asarray([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rotation_z = np.asarray([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rotation_z @ rotation_y @ rotation_x


def _validate_transforms(values: ArrayLike, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 3 or array.shape[1:] != (4, 4):
        raise ValueError(f"{label} transforms must have shape (T, 4, 4)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} transforms contain non-finite values")
    if not np.allclose(array[:, 3, :], np.asarray([0.0, 0.0, 0.0, 1.0]), atol=1e-9):
        raise ValueError(f"{label} transforms have invalid homogeneous rows")
    return array


def _summarize(values: np.ndarray) -> ResidualSummary:
    if not len(values):
        return ResidualSummary(mean=0.0, median=0.0, p95=0.0, maximum=0.0)
    return ResidualSummary(
        mean=float(np.mean(values)),
        median=float(np.median(values)),
        p95=float(np.quantile(values, 0.95)),
        maximum=float(np.max(values)),
    )

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator


OrientationRepresentation = Literal[
    "quaternion_xyzw",
    "quaternion_wxyz",
    "rotation_6d",
    "rotation_matrix",
    "rotation_vector",
    "rpy",
]


class ResidualSummary(BaseModel):
    mean: float
    median: float
    p95: float
    maximum: float


@dataclass(frozen=True)
class SE3ResidualResult:
    position_errors: np.ndarray
    orientation_errors: np.ndarray
    error_transforms: np.ndarray
    position: ResidualSummary
    orientation: ResidualSummary


class JointFieldMapping(BaseModel):
    urdf_joint: str
    state_index: int = Field(ge=0)
    sign: Literal[-1, 1] = 1
    zero_offset: float = 0.0


class PoseFieldMapping(BaseModel):
    position_indices: tuple[int, int, int]
    orientation_indices: tuple[int, ...]
    orientation_representation: OrientationRepresentation

    @field_validator("position_indices", "orientation_indices")
    @classmethod
    def _indices_are_unique_and_non_negative(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(index < 0 for index in value):
            raise ValueError("pose indices must be non-negative")
        if len(set(value)) != len(value):
            raise ValueError("pose indices must be unique")
        return value

    @model_validator(mode="after")
    def _orientation_width_matches_representation(self) -> "PoseFieldMapping":
        expected = {
            "quaternion_xyzw": 4,
            "quaternion_wxyz": 4,
            "rotation_6d": 6,
            "rotation_matrix": 9,
            "rotation_vector": 3,
            "rpy": 3,
        }[self.orientation_representation]
        if len(self.orientation_indices) != expected:
            raise ValueError(
                f"{self.orientation_representation} requires {expected} orientation indices"
            )
        if set(self.position_indices) & set(self.orientation_indices):
            raise ValueError("position and orientation indices must not overlap")
        return self


def _identity_transform() -> list[list[float]]:
    return np.eye(4, dtype=np.float64).tolist()


class KinematicChainProfile(BaseModel):
    id: str
    end_effector_frame: str
    joints: list[JointFieldMapping] = Field(min_length=1)
    logged_pose: PoseFieldMapping
    base_transform: list[list[float]] = Field(default_factory=_identity_transform)
    tcp_transform: list[list[float]] = Field(default_factory=_identity_transform)

    @field_validator("base_transform", "tcp_transform")
    @classmethod
    def _transform_is_rigid(cls, value: list[list[float]]) -> list[list[float]]:
        matrix = np.asarray(value, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise ValueError("kinematic transforms must be finite 4x4 matrices")
        if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
            raise ValueError("kinematic transforms must have a homogeneous last row")
        rotation = matrix[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
            raise ValueError("kinematic transform rotation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
            raise ValueError("kinematic transform rotation must have determinant one")
        return matrix.tolist()

    @model_validator(mode="after")
    def _joint_mappings_are_unique(self) -> "KinematicChainProfile":
        joint_names = [mapping.urdf_joint for mapping in self.joints]
        state_indices = [mapping.state_index for mapping in self.joints]
        if len(set(joint_names)) != len(joint_names):
            raise ValueError("URDF joints must be unique within a chain")
        if len(set(state_indices)) != len(state_indices):
            raise ValueError("joint state indices must be unique within a chain")
        return self


class KinematicProfile(BaseModel):
    urdf_path: str
    chains: list[KinematicChainProfile] = Field(min_length=1)
    fingerprint: str = ""

    @model_validator(mode="after")
    def _chain_ids_are_unique(self) -> "KinematicProfile":
        identifiers = [chain.id for chain in self.chains]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("kinematic chain IDs must be unique")
        return self

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel


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

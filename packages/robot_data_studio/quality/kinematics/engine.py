from __future__ import annotations

from pathlib import Path

import numpy as np

from .models import KinematicChainProfile, KinematicProfile, SE3ResidualResult
from .pose import compute_se3_residuals, decode_rotations


class PinocchioKinematics:
    def __init__(self, urdf_path: str | Path) -> None:
        import pinocchio as pin  # type: ignore[import-not-found]

        self.pin = pin
        self.urdf_path = Path(urdf_path).expanduser().resolve()
        if not self.urdf_path.is_file():
            raise ValueError(f"URDF file not found: {self.urdf_path}")
        self.model = pin.buildModelFromUrdf(str(self.urdf_path))

    def evaluate(
        self,
        states: np.ndarray,
        profile: KinematicProfile,
    ) -> dict[str, SE3ResidualResult]:
        state_array = np.asarray(states, dtype=np.float64)
        if state_array.ndim != 2 or not np.all(np.isfinite(state_array)):
            raise ValueError("states must be a finite 2D array")
        results: dict[str, SE3ResidualResult] = {}
        for chain in profile.chains:
            predicted = self.predict_chain(state_array, chain)
            logged = self.logged_chain_transforms(state_array, chain)
            results[chain.id] = compute_se3_residuals(predicted, logged)
        return results

    def predict_chain(
        self,
        states: np.ndarray,
        chain: KinematicChainProfile,
    ) -> np.ndarray:
        frame_id = self.model.getFrameId(chain.end_effector_frame)
        if frame_id >= len(self.model.frames):
            raise ValueError(
                f"End-effector frame {chain.end_effector_frame!r} was not found in {self.urdf_path}"
            )
        mappings = []
        for mapping in chain.joints:
            if mapping.state_index >= states.shape[1]:
                raise ValueError(
                    f"State index {mapping.state_index} for joint {mapping.urdf_joint!r} "
                    f"is outside state width {states.shape[1]}"
                )
            joint_id = self.model.getJointId(mapping.urdf_joint)
            if joint_id == 0:
                raise ValueError(f"Joint {mapping.urdf_joint!r} was not found in {self.urdf_path}")
            joint = self.model.joints[joint_id]
            if joint.nq != 1:
                raise ValueError(
                    f"Joint {mapping.urdf_joint!r} has nq={joint.nq}; only scalar joints are supported"
                )
            mappings.append((mapping, joint.idx_q))

        base = np.asarray(chain.base_transform, dtype=np.float64)
        tcp = np.asarray(chain.tcp_transform, dtype=np.float64)
        predicted = []
        data = self.model.createData()
        for state in states:
            configuration = np.asarray(self.pin.neutral(self.model), dtype=np.float64)
            for mapping, q_index in mappings:
                configuration[q_index] = (
                    mapping.sign * state[mapping.state_index] + mapping.zero_offset
                )
            self.pin.forwardKinematics(self.model, data, configuration)
            self.pin.updateFramePlacements(self.model, data)
            frame_transform = np.asarray(data.oMf[frame_id].homogeneous, dtype=np.float64)
            predicted.append(base @ frame_transform @ tcp)
        return np.asarray(predicted, dtype=np.float64)

    @staticmethod
    def logged_chain_transforms(
        states: np.ndarray,
        chain: KinematicChainProfile,
    ) -> np.ndarray:
        pose = chain.logged_pose
        indexes = (*pose.position_indices, *pose.orientation_indices)
        if max(indexes) >= states.shape[1]:
            raise ValueError(
                f"Logged pose index {max(indexes)} is outside state width {states.shape[1]}"
            )
        positions = states[:, pose.position_indices]
        rotations = decode_rotations(
            states[:, pose.orientation_indices],
            pose.orientation_representation,
        )
        transforms = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], len(states), axis=0)
        transforms[:, :3, :3] = rotations
        transforms[:, :3, 3] = positions
        return transforms

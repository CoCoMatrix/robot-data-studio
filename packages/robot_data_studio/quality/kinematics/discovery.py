from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from .engine import PinocchioKinematics
from .models import (
    JointFieldMapping,
    KinematicChainProfile,
    KinematicProfile,
    OrientationRepresentation,
    PoseFieldMapping,
    SE3ResidualResult,
)
from .pose import InvalidPoseRepresentation, compute_se3_residuals, decode_rotations


@dataclass(frozen=True)
class DiscoveryRequest:
    states: np.ndarray
    feature_names: list[str]
    max_candidates: int = 20
    sample_count: int = 64
    max_joint_mappings: int = 256

    def __post_init__(self) -> None:
        states = np.asarray(self.states, dtype=np.float64)
        if states.ndim != 2 or not len(states) or not np.all(np.isfinite(states)):
            raise ValueError("discovery states must be a non-empty finite 2D array")
        if self.feature_names and len(self.feature_names) != states.shape[1]:
            raise ValueError("feature_names must match the state width")
        if self.max_candidates < 1 or self.sample_count < 2 or self.max_joint_mappings < 1:
            raise ValueError("discovery limits must be positive")
        object.__setattr__(self, "states", states)


@dataclass(frozen=True)
class ProfileCandidate:
    profile: KinematicProfile
    training: SE3ResidualResult
    held_out: SE3ResidualResult
    score: float
    evidence: list[str]


_ORIENTATION_WIDTHS: dict[OrientationRepresentation, int] = {
    "quaternion_xyzw": 4,
    "quaternion_wxyz": 4,
    "rotation_6d": 6,
    "rotation_matrix": 9,
    "rotation_vector": 3,
    "rpy": 3,
}


def discover_profile(
    urdf_path: str | Path,
    request: DiscoveryRequest,
) -> list[ProfileCandidate]:
    engine = PinocchioKinematics(urdf_path)
    sampled = request.states[_representative_indices(request.states, request.sample_count)]
    training_indexes = np.arange(0, len(sampled), 2, dtype=int)
    held_out_indexes = np.arange(1, len(sampled), 2, dtype=int)
    if not len(held_out_indexes):
        held_out_indexes = training_indexes[-1:]
        training_indexes = training_indexes[:-1]
    if not len(training_indexes):
        training_indexes = held_out_indexes

    feature_names = request.feature_names or [f"state_{index}" for index in range(sampled.shape[1])]
    fingerprint = _fingerprint(engine.urdf_path, sampled.shape, feature_names)
    pose_candidates = _pose_candidates(sampled)
    candidates: list[ProfileCandidate] = []

    for frame_name, joint_names in _frame_paths(engine):
        if not joint_names:
            continue
        for joint_mappings, mapping_prior in _joint_mapping_candidates(
            engine,
            sampled,
            feature_names,
            joint_names,
            request.max_joint_mappings,
        ):
            mapped_indices = {mapping.state_index for mapping in joint_mappings}
            compatible_poses = [
                (pose, pose_prior)
                for pose, pose_prior in pose_candidates
                if not mapped_indices.intersection(
                    (*pose.position_indices, *pose.orientation_indices)
                )
            ]
            if not compatible_poses:
                continue
            prediction_chain = KinematicChainProfile(
                id="arm",
                end_effector_frame=frame_name,
                joints=joint_mappings,
                logged_pose=compatible_poses[0][0],
            )
            try:
                predicted = engine.predict_chain(sampled, prediction_chain)
            except ValueError:
                continue
            for pose, pose_prior in compatible_poses:
                chain = prediction_chain.model_copy(update={"logged_pose": pose})
                try:
                    logged = engine.logged_chain_transforms(sampled, chain)
                except (ValueError, InvalidPoseRepresentation):
                    continue
                training = compute_se3_residuals(
                    predicted[training_indexes],
                    logged[training_indexes],
                )
                held_out = compute_se3_residuals(
                    predicted[held_out_indexes],
                    logged[held_out_indexes],
                )
                geometric_score = (
                    held_out.position.median
                    + held_out.position.p95
                    + 0.25 * (held_out.orientation.median + held_out.orientation.p95)
                )
                tie_breaker = 1e-9 * (mapping_prior + pose_prior)
                profile = KinematicProfile(
                    urdf_path=str(engine.urdf_path),
                    chains=[chain],
                    fingerprint=fingerprint,
                )
                candidates.append(
                    ProfileCandidate(
                        profile=profile,
                        training=training,
                        held_out=held_out,
                        score=float(geometric_score - tie_breaker),
                        evidence=[
                            "Selected by held-out geometric SE(3) residual; names are only a tie-breaking prior.",
                            f"URDF frame {frame_name!r} uses actuated path {joint_names!r}.",
                            f"Logged pose uses position indices {pose.position_indices!r}, "
                            f"orientation indices {pose.orientation_indices!r}, and "
                            f"{pose.orientation_representation!r}.",
                        ],
                    )
                )
            if len(candidates) > request.max_candidates * 100:
                candidates = sorted(candidates, key=_candidate_sort_key)[
                    : request.max_candidates * 20
                ]

    if not candidates:
        raise ValueError("No geometrically valid kinematic profile candidates were found")
    return sorted(candidates, key=_candidate_sort_key)[: request.max_candidates]


def _frame_paths(engine: PinocchioKinematics) -> list[tuple[str, list[str]]]:
    paths = []
    seen = set()
    for frame in engine.model.frames:
        if frame.type != engine.pin.FrameType.BODY:
            continue
        frame_name = str(frame.name)
        if not frame_name or frame_name in seen or int(frame.parentJoint) == 0:
            continue
        seen.add(frame_name)
        joint_id = int(frame.parentJoint)
        joints = []
        while joint_id > 0:
            joint = engine.model.joints[joint_id]
            if joint.nq == 1:
                joints.append(str(engine.model.names[joint_id]))
            joint_id = int(engine.model.parents[joint_id])
        paths.append((frame_name, list(reversed(joints))))
    return paths


def _joint_mapping_candidates(
    engine: PinocchioKinematics,
    states: np.ndarray,
    feature_names: list[str],
    joint_names: list[str],
    limit: int,
) -> list[tuple[list[JointFieldMapping], float]]:
    width = states.shape[1]
    per_joint_rankings = []
    for joint_name in joint_names:
        joint_id = engine.model.getJointId(joint_name)
        joint = engine.model.joints[joint_id]
        lower = float(engine.model.lowerPositionLimit[joint.idx_q])
        upper = float(engine.model.upperPositionLimit[joint.idx_q])
        ranking = []
        for state_index in range(width):
            values = states[:, state_index]
            variation = float(np.std(values))
            if variation <= 1e-10:
                continue
            span = max(upper - lower, 1e-9)
            within = float(
                np.mean((values >= lower - 0.1 * span) & (values <= upper + 0.1 * span))
            )
            name_similarity = _name_similarity(joint_name, feature_names[state_index])
            ranking.append((state_index, 2.0 * name_similarity + within + min(variation, 1.0)))
        per_joint_rankings.append(sorted(ranking, key=lambda item: (-item[1], item[0])))
    if any(not ranking for ranking in per_joint_rankings):
        return []

    pool_size = min(width, max(len(joint_names) + 5, 8))
    pools = [ranking[:pool_size] for ranking in per_joint_rankings]
    combinations = []
    for selection in itertools.product(*pools):
        indices = [item[0] for item in selection]
        if len(set(indices)) != len(indices):
            continue
        prior = float(sum(item[1] for item in selection))
        mappings = [
            JointFieldMapping(urdf_joint=joint_name, state_index=state_index)
            for joint_name, state_index in zip(joint_names, indices, strict=True)
        ]
        combinations.append((mappings, prior))
    return sorted(combinations, key=lambda item: -item[1])[:limit]


def _pose_candidates(states: np.ndarray) -> list[tuple[PoseFieldMapping, float]]:
    width = states.shape[1]
    positions = []
    for start in range(max(width - 2, 0)):
        indices = (start, start + 1, start + 2)
        values = states[:, indices]
        if np.max(np.abs(values)) > 100.0:
            continue
        positions.append((indices, float(np.mean(np.std(values, axis=0)))))

    orientations = []
    for representation, field_width in _ORIENTATION_WIDTHS.items():
        for start in range(max(width - field_width + 1, 0)):
            indices = tuple(range(start, start + field_width))
            try:
                decode_rotations(states[:, indices], representation)
            except InvalidPoseRepresentation:
                continue
            orientations.append((indices, representation))

    candidates = []
    for position_indices, position_variation in positions:
        for orientation_indices, representation in orientations:
            if set(position_indices) & set(orientation_indices):
                continue
            candidates.append(
                (
                    PoseFieldMapping(
                        position_indices=position_indices,
                        orientation_indices=orientation_indices,
                        orientation_representation=representation,
                    ),
                    position_variation,
                )
            )
    return candidates


def _representative_indices(states: np.ndarray, sample_count: int) -> np.ndarray:
    if len(states) <= sample_count:
        return np.arange(len(states), dtype=int)
    indexes = set(np.linspace(0, len(states) - 1, sample_count, dtype=int).tolist())
    for dimension in range(states.shape[1]):
        indexes.add(int(np.argmin(states[:, dimension])))
        indexes.add(int(np.argmax(states[:, dimension])))
    selected = sorted(indexes)
    if len(selected) > sample_count:
        positions = np.linspace(0, len(selected) - 1, sample_count, dtype=int)
        selected = [selected[index] for index in positions]
    return np.asarray(selected, dtype=int)


def _fingerprint(path: Path, state_shape: tuple[int, ...], feature_names: list[str]) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    digest.update(repr(state_shape).encode("utf-8"))
    digest.update("\0".join(feature_names).encode("utf-8"))
    return digest.hexdigest()


def _name_similarity(left: str, right: str) -> float:
    normalized_left = "".join(character.lower() for character in left if character.isalnum())
    normalized_right = "".join(character.lower() for character in right if character.isalnum())
    if not normalized_left or not normalized_right:
        return 0.0
    return SequenceMatcher(None, normalized_left, normalized_right).ratio()


def _candidate_sort_key(candidate: ProfileCandidate) -> tuple[float, float, float]:
    return (
        candidate.score,
        candidate.held_out.position.maximum,
        candidate.held_out.orientation.maximum,
    )

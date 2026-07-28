from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from .discovery import ProfileCandidate
from .engine import PinocchioKinematics
from .models import KinematicChainProfile, KinematicProfile, ResidualSummary, SE3ResidualResult
from .pose import compute_se3_residuals


@dataclass(frozen=True)
class CorrectionOptions:
    allow_base: bool = True
    allow_tcp: bool = True
    allow_joint_sign: bool = True
    allow_joint_offset: bool = True
    min_candidate_margin: float = 1e-4
    min_rotation_excitation: float = 0.05
    min_joint_excitation: float = 1e-3
    position_tolerance: float = 0.05
    orientation_tolerance: float = 0.1
    max_iterations: int = 40
    max_sign_hypotheses: int = 256


class CorrectionMetrics(BaseModel):
    position: ResidualSummary
    orientation: ResidualSummary
    inlier_rate: float = Field(ge=0.0, le=1.0)


class DiscoveryRefusal(BaseModel):
    reason: Literal["insufficient_excitation", "ambiguous", "no_candidates", "fit_failed"]
    evidence: list[str] = Field(default_factory=list)


class CorrectionProposal(BaseModel):
    id: str
    fingerprint: str
    profile: KinematicProfile
    before_training: CorrectionMetrics
    after_training: CorrectionMetrics
    before_held_out: CorrectionMetrics
    after_held_out: CorrectionMetrics
    confidence: float = Field(ge=0.0, le=1.0)
    candidate_margin: float
    evidence: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    status: Literal["proposed", "accepted", "rejected", "stale"] = "proposed"


def fit_correction(
    states: np.ndarray,
    candidates: list[ProfileCandidate],
    options: CorrectionOptions | None = None,
) -> CorrectionProposal | DiscoveryRefusal:
    options = options or CorrectionOptions()
    state_array = np.asarray(states, dtype=np.float64)
    if state_array.ndim != 2 or not len(state_array) or not np.all(np.isfinite(state_array)):
        raise ValueError("correction states must be a non-empty finite 2D array")
    if not candidates:
        return DiscoveryRefusal(reason="no_candidates", evidence=["Discovery returned no candidates."])
    ordered = sorted(candidates, key=lambda candidate: candidate.score)
    margin = float("inf")
    if len(ordered) > 1:
        margin = float(ordered[1].score - ordered[0].score)
        if margin < options.min_candidate_margin:
            return DiscoveryRefusal(
                reason="ambiguous",
                evidence=[
                    f"The top candidate margin {margin:.6g} is below "
                    f"{options.min_candidate_margin:.6g}."
                ],
            )

    initial_profile = ordered[0].profile
    engine = PinocchioKinematics(initial_profile.urdf_path)
    excitation_failure = _excitation_failure(engine, state_array, initial_profile, options)
    if excitation_failure:
        return DiscoveryRefusal(reason="insufficient_excitation", evidence=excitation_failure)

    training_indexes = np.arange(0, len(state_array), 2, dtype=int)
    held_out_indexes = np.arange(1, len(state_array), 2, dtype=int)
    if not len(held_out_indexes):
        held_out_indexes = training_indexes[-1:]
        training_indexes = training_indexes[:-1]
    if not len(training_indexes):
        training_indexes = held_out_indexes

    corrected_chains = []
    fit_evidence = list(ordered[0].evidence)
    for chain in initial_profile.chains:
        fitted = _fit_chain(engine, state_array[training_indexes], chain, options)
        if fitted is None:
            return DiscoveryRefusal(
                reason="fit_failed",
                evidence=[f"Continuous correction fitting failed for chain {chain.id!r}."],
            )
        corrected_chain, chain_cost, signs_tested = fitted
        corrected_chains.append(corrected_chain)
        fit_evidence.append(
            f"Chain {chain.id!r}: selected a hypothesis after testing "
            f"{signs_tested} joint-sign convention(s); training cost={chain_cost:.6g}."
        )

    corrected_profile = initial_profile.model_copy(update={"chains": corrected_chains})
    before_training_result = _profile_residuals(
        engine, state_array[training_indexes], initial_profile
    )
    after_training_result = _profile_residuals(
        engine, state_array[training_indexes], corrected_profile
    )
    before_held_out_result = _profile_residuals(
        engine, state_array[held_out_indexes], initial_profile
    )
    after_held_out_result = _profile_residuals(
        engine, state_array[held_out_indexes], corrected_profile
    )
    before_training = _metrics(before_training_result, options)
    after_training = _metrics(after_training_result, options)
    before_held_out = _metrics(before_held_out_result, options)
    after_held_out = _metrics(after_held_out_result, options)
    confidence = _confidence(before_held_out, after_held_out, margin, options)
    payload = corrected_profile.model_dump_json() + initial_profile.fingerprint
    proposal_id = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    warnings = []
    if options.allow_base and options.allow_tcp and options.allow_joint_offset:
        warnings.append(
            "Base, TCP, and joint-zero corrections can share equivalent parameterizations; "
            "review the evidence before acceptance."
        )
    return CorrectionProposal(
        id=proposal_id,
        fingerprint=initial_profile.fingerprint,
        profile=corrected_profile,
        before_training=before_training,
        after_training=after_training,
        before_held_out=before_held_out,
        after_held_out=after_held_out,
        confidence=confidence,
        candidate_margin=margin,
        evidence=fit_evidence,
        warnings=warnings,
    )


def _fit_chain(
    engine: PinocchioKinematics,
    states: np.ndarray,
    chain: KinematicChainProfile,
    options: CorrectionOptions,
) -> tuple[KinematicChainProfile, float, int] | None:
    sign_hypotheses = [tuple(mapping.sign for mapping in chain.joints)]
    if options.allow_joint_sign:
        total = 2 ** len(chain.joints)
        if total <= options.max_sign_hypotheses:
            sign_hypotheses = list(itertools.product((-1, 1), repeat=len(chain.joints)))
        else:
            baseline = tuple(mapping.sign for mapping in chain.joints)
            sign_hypotheses = [baseline]
            for index in range(len(chain.joints)):
                changed = list(baseline)
                changed[index] *= -1
                sign_hypotheses.append(tuple(changed))

    best: tuple[KinematicChainProfile, float] | None = None
    for signs in sign_hypotheses:
        signed_joints = [
            mapping.model_copy(update={"sign": sign})
            for mapping, sign in zip(chain.joints, signs, strict=True)
        ]
        signed_chain = chain.model_copy(update={"joints": signed_joints})
        fitted_chain, cost = _fit_continuous(engine, states, signed_chain, options)
        if best is None or cost < best[1]:
            best = (fitted_chain, cost)
    if best is None:
        return None
    return best[0], best[1], len(sign_hypotheses)


def _fit_continuous(
    engine: PinocchioKinematics,
    states: np.ndarray,
    chain: KinematicChainProfile,
    options: CorrectionOptions,
) -> tuple[KinematicChainProfile, float]:
    base = np.asarray(chain.base_transform, dtype=np.float64)
    tcp = np.asarray(chain.tcp_transform, dtype=np.float64)
    offsets = np.asarray([mapping.zero_offset for mapping in chain.joints], dtype=np.float64)
    base, tcp = _alternating_transform_initialization(engine, states, chain, base, tcp, options)
    parts = []
    if options.allow_base:
        parts.extend(_transform_to_parameters(base))
    if options.allow_tcp:
        parts.extend(_transform_to_parameters(tcp))
    if options.allow_joint_offset:
        parts.extend(offsets.tolist())
    parameters = np.asarray(parts, dtype=np.float64)
    if not len(parameters):
        residual = _chain_residual_vector(engine, states, chain)
        return chain, float(np.mean(residual * residual))

    def build(values: np.ndarray) -> KinematicChainProfile:
        cursor = 0
        updates: dict[str, object] = {}
        if options.allow_base:
            updates["base_transform"] = _parameters_to_transform(values[cursor : cursor + 6]).tolist()
            cursor += 6
        if options.allow_tcp:
            updates["tcp_transform"] = _parameters_to_transform(values[cursor : cursor + 6]).tolist()
            cursor += 6
        if options.allow_joint_offset:
            new_joints = []
            for mapping in chain.joints:
                new_joints.append(mapping.model_copy(update={"zero_offset": float(values[cursor])}))
                cursor += 1
            updates["joints"] = new_joints
        return chain.model_copy(update=updates)

    damping = 1e-4
    current_chain = build(parameters)
    residual = _weighted_chain_residual(engine, states, current_chain)
    cost = float(np.mean(residual * residual))
    for _ in range(options.max_iterations):
        jacobian = np.empty((len(residual), len(parameters)), dtype=np.float64)
        for index in range(len(parameters)):
            step = 1e-6 * max(1.0, abs(float(parameters[index])))
            changed = parameters.copy()
            changed[index] += step
            changed_residual = _weighted_chain_residual(engine, states, build(changed))
            jacobian[:, index] = (changed_residual - residual) / step
        normal = jacobian.T @ jacobian + damping * np.eye(len(parameters))
        gradient = jacobian.T @ residual
        try:
            delta = np.linalg.solve(normal, -gradient)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(normal, -gradient, rcond=None)[0]
        if np.linalg.norm(delta) < 1e-9:
            break
        proposed_parameters = parameters + delta
        proposed_chain = build(proposed_parameters)
        proposed_residual = _weighted_chain_residual(engine, states, proposed_chain)
        proposed_cost = float(np.mean(proposed_residual * proposed_residual))
        if proposed_cost < cost:
            parameters = proposed_parameters
            current_chain = proposed_chain
            residual = proposed_residual
            cost = proposed_cost
            damping = max(damping * 0.5, 1e-9)
        else:
            damping = min(damping * 10.0, 1e9)
        if damping >= 1e9:
            break
    return current_chain, cost


def _alternating_transform_initialization(
    engine: PinocchioKinematics,
    states: np.ndarray,
    chain: KinematicChainProfile,
    base: np.ndarray,
    tcp: np.ndarray,
    options: CorrectionOptions,
) -> tuple[np.ndarray, np.ndarray]:
    identity_chain = chain.model_copy(
        update={"base_transform": np.eye(4).tolist(), "tcp_transform": np.eye(4).tolist()}
    )
    raw = engine.predict_chain(states, identity_chain)
    logged = engine.logged_chain_transforms(states, chain)
    if options.allow_base and not options.allow_tcp:
        return _mean_transform(logged @ np.linalg.inv(raw)), tcp
    if options.allow_tcp and not options.allow_base:
        return base, _mean_transform(np.linalg.inv(raw) @ logged)
    if options.allow_base and options.allow_tcp:
        for _ in range(8):
            base = _mean_transform(logged @ np.linalg.inv(raw @ tcp))
            tcp = _mean_transform(np.linalg.inv(base @ raw) @ logged)
    return base, tcp


def _chain_residual_vector(
    engine: PinocchioKinematics,
    states: np.ndarray,
    chain: KinematicChainProfile,
) -> np.ndarray:
    predicted = engine.predict_chain(states, chain)
    logged = engine.logged_chain_transforms(states, chain)
    errors = np.linalg.inv(predicted) @ logged
    vectors = [
        np.concatenate((error[:3, 3], _matrix_to_rotation_vector(error[:3, :3])))
        for error in errors
    ]
    return np.asarray(vectors, dtype=np.float64).reshape(-1)


def _weighted_chain_residual(
    engine: PinocchioKinematics,
    states: np.ndarray,
    chain: KinematicChainProfile,
) -> np.ndarray:
    rows = _chain_residual_vector(engine, states, chain).reshape(-1, 6)
    norms = np.linalg.norm(rows, axis=1)
    median = float(np.median(norms))
    mad = float(np.median(np.abs(norms - median)))
    scale = max(1.4826 * mad, 1e-6)
    cutoff = 1.345 * scale
    weights = np.ones_like(norms)
    outside = norms > cutoff
    weights[outside] = cutoff / np.maximum(norms[outside], 1e-12)
    return (rows * np.sqrt(weights)[:, None]).reshape(-1)


def _profile_residuals(
    engine: PinocchioKinematics,
    states: np.ndarray,
    profile: KinematicProfile,
) -> SE3ResidualResult:
    predicted = []
    logged = []
    for chain in profile.chains:
        predicted.append(engine.predict_chain(states, chain))
        logged.append(engine.logged_chain_transforms(states, chain))
    return compute_se3_residuals(np.concatenate(predicted), np.concatenate(logged))


def _metrics(result: SE3ResidualResult, options: CorrectionOptions) -> CorrectionMetrics:
    inliers = (result.position_errors <= options.position_tolerance) & (
        result.orientation_errors <= options.orientation_tolerance
    )
    return CorrectionMetrics(
        position=result.position,
        orientation=result.orientation,
        inlier_rate=float(np.mean(inliers)) if len(inliers) else 0.0,
    )


def _excitation_failure(
    engine: PinocchioKinematics,
    states: np.ndarray,
    profile: KinematicProfile,
    options: CorrectionOptions,
) -> list[str]:
    failures = []
    for chain in profile.chains:
        if options.allow_base and options.allow_tcp:
            predicted = engine.predict_chain(states, chain)
            reference = np.repeat(predicted[:1], len(predicted), axis=0)
            excitation = compute_se3_residuals(reference, predicted).orientation.maximum
            if excitation < options.min_rotation_excitation:
                failures.append(
                    f"Chain {chain.id!r} rotation excitation {excitation:.6g} rad is below "
                    f"{options.min_rotation_excitation:.6g}; base and TCP cannot be separated."
                )
        if options.allow_joint_sign or options.allow_joint_offset:
            for mapping in chain.joints:
                variation = float(np.std(states[:, mapping.state_index]))
                if variation < options.min_joint_excitation:
                    failures.append(
                        f"Joint {mapping.urdf_joint!r} variation {variation:.6g} is below "
                        f"{options.min_joint_excitation:.6g}."
                    )
    return failures


def _confidence(
    before: CorrectionMetrics,
    after: CorrectionMetrics,
    margin: float,
    options: CorrectionOptions,
) -> float:
    before_error = before.position.p95 + 0.25 * before.orientation.p95
    after_error = after.position.p95 + 0.25 * after.orientation.p95
    improvement = max(0.0, 1.0 - after_error / max(before_error, 1e-12))
    margin_score = (
        1.0
        if np.isinf(margin)
        else min(1.0, margin / max(options.min_candidate_margin, 1e-12))
    )
    return float(
        np.clip(
            0.5 * improvement + 0.35 * after.inlier_rate + 0.15 * margin_score,
            0.0,
            1.0,
        )
    )


def _mean_transform(transforms: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = np.median(transforms[:, :3, 3], axis=0)
    rotation_sum = np.sum(transforms[:, :3, :3], axis=0)
    left, _, right = np.linalg.svd(rotation_sum)
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1.0
        rotation = left @ right
    result[:3, :3] = rotation
    return result


def _transform_to_parameters(transform: np.ndarray) -> list[float]:
    return [
        *transform[:3, 3].tolist(),
        *_matrix_to_rotation_vector(transform[:3, :3]).tolist(),
    ]


def _parameters_to_transform(parameters: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = parameters[:3]
    transform[:3, :3] = _rotation_from_vector(parameters[3:6])
    return transform


def _rotation_from_vector(vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(vector))
    if angle <= 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = vector / angle
    x, y, z = axis
    skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def _matrix_to_rotation_vector(matrix: np.ndarray) -> np.ndarray:
    cosine = float(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0))
    vector = 0.5 * np.asarray(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ],
        dtype=np.float64,
    )
    sine = float(np.linalg.norm(vector))
    angle = float(np.arctan2(sine, cosine))
    if sine <= 1e-12:
        return np.zeros(3, dtype=np.float64)
    return vector * (angle / sine)

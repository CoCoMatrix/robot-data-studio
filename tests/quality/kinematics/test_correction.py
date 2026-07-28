from __future__ import annotations

from pathlib import Path

import numpy as np

from robot_data_studio.quality.kinematics import (
    CorrectionOptions,
    CorrectionProposal,
    DiscoveryRefusal,
    JointFieldMapping,
    KinematicChainProfile,
    KinematicProfile,
    PinocchioKinematics,
    PoseFieldMapping,
    ProfileCandidate,
    compute_se3_residuals,
    fit_correction,
)


FIXTURE = Path(__file__).parent / "fixtures" / "nonstandard_robot.urdf"


def _rotation_from_vector(vector: tuple[float, float, float]) -> np.ndarray:
    angle = float(np.linalg.norm(vector))
    if angle == 0.0:
        return np.eye(3)
    axis = np.asarray(vector) / angle
    x, y, z = axis
    skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def _transform(
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotation_vector: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = _rotation_from_vector(rotation_vector)
    value[:3, 3] = translation
    return value


def _matrix_to_quaternion_xyzw(matrix: np.ndarray) -> np.ndarray:
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        # All correction fixtures stay close enough to identity to use this independent branch.
        raise AssertionError("fixture rotation unexpectedly has a non-positive trace")
    return quaternion / np.linalg.norm(quaternion)


def _profile(
    *,
    signs: tuple[int, int] = (1, 1),
    offsets: tuple[float, float] = (0.0, 0.0),
    base: np.ndarray | None = None,
    tcp: np.ndarray | None = None,
) -> KinematicProfile:
    return KinematicProfile(
        urdf_path=str(FIXTURE),
        chains=[
            KinematicChainProfile(
                id="primary",
                end_effector_frame="vector_mount",
                joints=[
                    JointFieldMapping(
                        urdf_joint="alpha_axis",
                        state_index=0,
                        sign=signs[0],
                        zero_offset=offsets[0],
                    ),
                    JointFieldMapping(
                        urdf_joint="epsilon_axis",
                        state_index=1,
                        sign=signs[1],
                        zero_offset=offsets[1],
                    ),
                ],
                logged_pose=PoseFieldMapping(
                    position_indices=(2, 3, 4),
                    orientation_indices=(5, 6, 7, 8),
                    orientation_representation="quaternion_xyzw",
                ),
                base_transform=(base if base is not None else np.eye(4)).tolist(),
                tcp_transform=(tcp if tcp is not None else np.eye(4)).tolist(),
            )
        ],
        fingerprint="correction-fixture-v1",
    )


def _states(
    *,
    true_signs: tuple[int, int] = (1, 1),
    true_offsets: tuple[float, float] = (0.0, 0.0),
    true_base: np.ndarray | None = None,
    true_tcp: np.ndarray | None = None,
    static: bool = False,
) -> np.ndarray:
    count = 32
    first = np.zeros(count) if static else np.linspace(-0.9, 1.0, count)
    second = np.zeros(count) if static else 0.7 * np.sin(np.linspace(0.0, 2.7, count))
    states = np.zeros((count, 9), dtype=np.float64)
    states[:, 0] = first
    states[:, 1] = second
    true_profile = _profile(
        signs=true_signs,
        offsets=true_offsets,
        base=true_base,
        tcp=true_tcp,
    )
    predicted = PinocchioKinematics(FIXTURE).predict_chain(
        states,
        true_profile.chains[0],
    )
    for index, transform in enumerate(predicted):
        states[index, 2:5] = transform[:3, 3]
        states[index, 5:9] = _matrix_to_quaternion_xyzw(transform[:3, :3])
    return states


def _candidate(states: np.ndarray, profile: KinematicProfile, score: float = 1.0) -> ProfileCandidate:
    engine = PinocchioKinematics(FIXTURE)
    predicted = engine.predict_chain(states, profile.chains[0])
    logged = engine.logged_chain_transforms(states, profile.chains[0])
    training = compute_se3_residuals(predicted[::2], logged[::2])
    held_out = compute_se3_residuals(predicted[1::2], logged[1::2])
    return ProfileCandidate(
        profile=profile,
        training=training,
        held_out=held_out,
        score=score,
        evidence=["synthetic candidate"],
    )


def test_fit_correction_separates_world_base_and_rotating_local_tcp() -> None:
    true_base = _transform((0.25, -0.12, 0.08), (0.08, -0.04, 0.12))
    true_tcp = _transform((0.13, 0.025, -0.04), (-0.05, 0.07, 0.03))
    states = _states(true_base=true_base, true_tcp=true_tcp)

    result = fit_correction(
        states,
        [_candidate(states, _profile())],
        CorrectionOptions(
            allow_base=True,
            allow_tcp=True,
            allow_joint_sign=False,
            allow_joint_offset=False,
        ),
    )

    assert isinstance(result, CorrectionProposal)
    corrected = result.profile.chains[0]
    assert np.allclose(corrected.base_transform, true_base, atol=2e-3)
    assert np.allclose(corrected.tcp_transform, true_tcp, atol=2e-3)
    assert result.after_held_out.position.maximum < 2e-4
    assert result.after_held_out.orientation.maximum < 2e-4
    assert result.after_held_out.position.maximum < result.before_held_out.position.maximum


def test_fit_correction_recovers_joint_signs_and_offsets_when_frames_are_fixed() -> None:
    true_signs = (-1, 1)
    true_offsets = (0.18, -0.11)
    states = _states(true_signs=true_signs, true_offsets=true_offsets)

    result = fit_correction(
        states,
        [_candidate(states, _profile())],
        CorrectionOptions(
            allow_base=False,
            allow_tcp=False,
            allow_joint_sign=True,
            allow_joint_offset=True,
        ),
    )

    assert isinstance(result, CorrectionProposal)
    joints = result.profile.chains[0].joints
    assert tuple(joint.sign for joint in joints) == true_signs
    assert np.allclose([joint.zero_offset for joint in joints], true_offsets, atol=2e-3)
    assert result.after_held_out.position.maximum < 2e-4
    assert result.after_held_out.orientation.maximum < 2e-4


def test_fit_correction_refuses_unexcited_or_ambiguous_data() -> None:
    static_states = _states(static=True)
    static_result = fit_correction(
        static_states,
        [_candidate(static_states, _profile())],
        CorrectionOptions(allow_base=True, allow_tcp=True),
    )

    assert isinstance(static_result, DiscoveryRefusal)
    assert static_result.reason == "insufficient_excitation"

    states = _states()
    candidate = _candidate(states, _profile(), score=0.5)
    ambiguous_result = fit_correction(
        states,
        [candidate, _candidate(states, _profile(), score=0.5000001)],
        CorrectionOptions(min_candidate_margin=1e-3),
    )

    assert isinstance(ambiguous_result, DiscoveryRefusal)
    assert ambiguous_result.reason == "ambiguous"


def test_fit_correction_fits_independent_base_transforms_per_chain() -> None:
    first_base = _transform((0.2, 0.0, 0.05), (0.03, 0.0, 0.08))
    second_base = _transform((-0.15, 0.25, 0.02), (-0.02, 0.06, -0.04))
    first_states = _states(true_base=first_base)
    second_states = _states(true_base=second_base)
    states = np.concatenate((first_states, second_states[:, 2:]), axis=1)
    base_profile = _profile()
    second_chain = base_profile.chains[0].model_copy(
        update={
            "id": "secondary",
            "logged_pose": PoseFieldMapping(
                position_indices=(9, 10, 11),
                orientation_indices=(12, 13, 14, 15),
                orientation_representation="quaternion_xyzw",
            ),
        }
    )
    profile = base_profile.model_copy(update={"chains": [base_profile.chains[0], second_chain]})
    engine = PinocchioKinematics(FIXTURE)
    before_first = compute_se3_residuals(
        engine.predict_chain(states, profile.chains[0]),
        engine.logged_chain_transforms(states, profile.chains[0]),
    )
    candidate = ProfileCandidate(
        profile=profile,
        training=before_first,
        held_out=before_first,
        score=1.0,
        evidence=["synthetic bimanual candidate"],
    )

    result = fit_correction(
        states,
        [candidate],
        CorrectionOptions(
            allow_base=True,
            allow_tcp=False,
            allow_joint_sign=False,
            allow_joint_offset=False,
        ),
    )

    assert isinstance(result, CorrectionProposal)
    assert np.allclose(result.profile.chains[0].base_transform, first_base, atol=2e-3)
    assert np.allclose(result.profile.chains[1].base_transform, second_base, atol=2e-3)

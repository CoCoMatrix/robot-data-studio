from __future__ import annotations

import numpy as np
import pytest

from robot_data_studio.curation.human_checks import (
    HumanExtensionConfig,
    analyze_human_arrays,
    build_chunk_hand_validity,
    camera_relative_transforms,
    confidence_track_summary,
    evaluate_hand_object_interaction,
    evaluate_task_conditioned_hand_availability,
    project_camera_points,
)


def _transforms(points: list[tuple[float, float, float]]) -> np.ndarray:
    output = np.repeat(np.eye(4)[None, :, :], len(points), axis=0)
    output[:, :3, 3] = np.asarray(points)
    return output


def test_confidence_spans_keep_short_holes_and_never_bridge_long_invalid_runs() -> None:
    summary = confidence_track_summary(
        np.asarray([1, 1, 0, 1, 0, 0, 0, 1], dtype=float),
        threshold=0.5,
        short_hole_max_frames=1,
    )

    assert summary.valid_spans == [(0, 1), (3, 3), (7, 7)]
    assert summary.short_holes == [(2, 2)]
    assert summary.long_invalid_spans == [(4, 6)]
    assert summary.valid_mask == [True, True, False, True, False, False, False, True]


def test_camera_relative_transform_cancels_equal_camera_and_joint_motion() -> None:
    camera = _transforms([(0, 0, 0), (1, 0, 0)])
    joint = _transforms([(0, 0, 2), (1, 0, 2)])

    relative = camera_relative_transforms(camera, joint)

    assert np.allclose(relative[:, :3, 3], [[0, 0, 2], [0, 0, 2]])


def test_projection_uses_camera_space_depth_and_calibration_canvas() -> None:
    intrinsic = np.asarray([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]], dtype=float)
    points = np.asarray([[0, 0, 2], [2, 0, 1], [0, 0, -1]], dtype=float)

    projection = project_camera_points(points, intrinsic, image_size=(1280, 720))

    assert np.allclose(projection.uv[0], [640, 360])
    assert projection.in_fov.tolist() == [True, False, False]
    assert projection.positive_depth.tolist() == [True, True, False]


def test_human_extension_preserves_per_hand_masks_and_marks_unbound_video_projection_na() -> None:
    transforms = {
        "camera": _transforms([(0, 0, 0)] * 5),
        "leftHand": _transforms([(0, 0, 2), (0.01, 0, 2), (0.02, 0, 2), (0.03, 0, 2), (0.04, 0, 2)]),
        "rightHand": _transforms([(0.2, 0, 2)] * 5),
        "leftForearm": _transforms([(0, 0, 1.7)] * 5),
        "rightForearm": _transforms([(0.2, 0, 1.7)] * 5),
    }
    confidences = {
        "leftHand": np.ones(5),
        "rightHand": np.asarray([1, 1, 0, 0, 0], dtype=float),
        "leftForearm": np.ones(5),
        "rightForearm": np.ones(5),
    }
    intrinsic = np.asarray([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]], dtype=float)

    result = analyze_human_arrays(
        transforms=transforms,
        confidences=confidences,
        intrinsic=intrinsic,
        fps=30.0,
        config=HumanExtensionConfig(short_hole_max_frames=1, confidence_threshold=0.5),
        calibration_image_size=(1280, 720),
        stored_video_size=(720, 1280),
        video_frame_count=5,
        expected_frame_count=5,
        raw_text="move the object",
        events=[{"start_time": 0.0, "end_time": 0.1}],
    )

    assert result.per_hand_valid_masks["left"] == [True] * 5
    assert result.per_hand_valid_masks["right"] == [True, True, False, False, False]
    assert result.bimanual_valid_mask == [True, True, False, False, False]
    assert result.either_hand_valid_mask == [True] * 5
    assert result.metrics_by_id["H1_tracking_confidence"].status == "evaluated"
    assert result.metrics_by_id["H6_camera_projection"].status == "evaluated"
    assert result.metrics_by_id["H6b_stored_video_projection"].status == "not_evaluated"
    assert "binding" in result.metrics_by_id["H6b_stored_video_projection"].reason


def test_chunk_validity_masks_each_hand_from_first_dropout_to_window_end() -> None:
    validity = {
        "left": [True, True, False, True, True, True],
        "right": [True, True, True, True, False, True],
    }

    first = build_chunk_hand_validity(validity, t0=0, horizon=5, stride=1)
    reentry = build_chunk_hand_validity(validity, t0=3, horizon=3, stride=1)

    assert first.source_indices == [0, 1, 2, 3, 4]
    assert first.per_hand["left"] == [True, True, False, False, False]
    assert first.per_hand["right"] == [True, True, True, True, False]
    assert reentry.per_hand["left"] == [True, True, True]
    assert reentry.per_hand["right"] == [True, False, False]


def test_chunk_validity_keeps_one_hand_supervised_and_rejects_boundary_crossing() -> None:
    one_hand = build_chunk_hand_validity(
        {"left": [True, True, True], "right": [False, False, False]},
        t0=0,
        horizon=3,
        stride=1,
    )
    assert one_hand.per_hand == {
        "left": [True, True, True],
        "right": [False, False, False],
    }
    with pytest.raises(ValueError, match="episode boundary"):
        build_chunk_hand_validity(
            {"left": [True, True], "right": [True, True]},
            t0=1,
            horizon=2,
            stride=1,
        )


def test_left_right_identity_is_not_evaluated_without_joint_valid_overlap() -> None:
    transforms = {
        "camera": _transforms([(0, 0, 0)] * 4),
        "leftHand": _transforms([(0, 0, 2)] * 4),
        "rightHand": _transforms([(0.2, 0, 2)] * 4),
        "leftForearm": _transforms([(0, 0, 1.7)] * 4),
        "rightForearm": _transforms([(0.2, 0, 1.7)] * 4),
    }
    confidences = {
        "leftHand": np.asarray([1, 1, 0, 0]),
        "leftForearm": np.asarray([1, 1, 0, 0]),
        "rightHand": np.asarray([0, 0, 1, 1]),
        "rightForearm": np.asarray([0, 0, 1, 1]),
    }
    result = analyze_human_arrays(
        transforms=transforms,
        confidences=confidences,
        intrinsic=np.eye(3),
        fps=30,
        config=HumanExtensionConfig(),
        calibration_image_size=(8, 8),
        stored_video_size=None,
        video_frame_count=None,
        expected_frame_count=4,
        raw_text="wave with one hand",
        events=[],
    )
    identity = result.metrics_by_id["H3_left_right_identity"]
    assert identity.status == "not_evaluated"
    assert "jointly valid" in identity.reason


def _mask(row: int, column: int) -> np.ndarray:
    mask = np.zeros((8, 8), dtype=bool)
    mask[row : row + 2, column : column + 2] = True
    return mask


def test_hand_object_interaction_is_task_conditioned_and_emits_event_states() -> None:
    metric = evaluate_hand_object_interaction(
        frame_indexes=[0, 1, 2, 3],
        hand_masks={"left": [_mask(2, 0), _mask(2, 2), _mask(2, 3), _mask(2, 6)]},
        object_masks=[_mask(2, 3), _mask(2, 3), _mask(2, 4), _mask(2, 4)],
        hand_positions={"left": np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]])},
        object_positions=np.asarray([[3, 0, 0], [3, 0, 0], [4, 0, 0], [4, 0, 0]]),
        structured_task={"verb": "pick", "object": "block", "active_hands": ["left"]},
        c1_manipulation_candidate=True,
        mask_cache_keys=["m0", "m1", "m2", "m3"],
    )

    assert metric.status == "evaluated"
    assert metric.fidelity == "provisional"
    assert metric.evidence["states"] == [
        "approach",
        "contact_or_grasp",
        "manipulate_or_carry",
        "release",
    ]
    assert metric.evidence["hard_reject_allowed"] is False
    assert metric.evidence["reused_mask_cache_keys"] == ["m0", "m1", "m2", "m3"]


def test_hand_object_interaction_skips_non_candidates_and_missing_segmentation() -> None:
    skipped = evaluate_hand_object_interaction(
        frame_indexes=[0],
        hand_masks={"left": [None]},
        object_masks=[None],
        hand_positions={"left": np.zeros((1, 3))},
        object_positions=np.zeros((1, 3)),
        structured_task={"verb": "wave"},
        c1_manipulation_candidate=False,
        mask_cache_keys=[],
    )
    missing = evaluate_hand_object_interaction(
        frame_indexes=[0],
        hand_masks={"left": [None]},
        object_masks=[None],
        hand_positions={"left": np.zeros((1, 3))},
        object_positions=np.zeros((1, 3)),
        structured_task={"verb": "pick"},
        c1_manipulation_candidate=True,
        mask_cache_keys=[],
    )
    assert skipped.status == "not_evaluated"
    assert "C1 manipulation candidate" in skipped.reason
    assert missing.status == "not_evaluated"
    assert "segmentation" in missing.reason


def test_task_conditioned_absence_ignores_inactive_hand_but_reviews_active_contact_gap() -> None:
    okay = evaluate_task_conditioned_hand_availability(
        per_hand_validity={"left": [True, True, True, True], "right": [False] * 4},
        active_hands=["left"],
        event_windows=[("contact_or_grasp", 1, 2)],
    )
    gap = evaluate_task_conditioned_hand_availability(
        per_hand_validity={"left": [True, False, False, True], "right": [True] * 4},
        active_hands=["left"],
        event_windows=[("contact_or_grasp", 1, 2)],
    )

    assert okay.score == 1.0
    assert okay.evidence["inactive_hand_absence_penalized"] is False
    assert gap.score == 0.0
    assert gap.evidence["review_recommended"] is True
    assert gap.evidence["whole_episode_reject_allowed"] is False

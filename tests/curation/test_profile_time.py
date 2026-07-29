from __future__ import annotations

import pytest
from pydantic import ValidationError

from robot_data_studio.curation.models import FrameSample
from robot_data_studio.curation.profile import CurationProfile, is_source_allowed
from robot_data_studio.curation.semantics import normalize_task_text
from robot_data_studio.curation.time_mapping import (
    EpisodeBoundaryError,
    PtsToleranceError,
    iter_episode_chunks,
    progress_resample,
)


def profile_payload() -> dict:
    return {
        "profile_version": "1",
        "name": "generic-manipulation",
        "selection": {
            "train_allowlist": ["train/*.jsonl"],
            "deny_patterns": ["*validation*", "*test*", "*eval*"],
        },
        "task": {
            "text": "move object to target carefully",
            "events": ["start", "contact", "end"],
        },
        "modalities": [
            {"name": "state", "required": True},
            {"name": "action", "required": True},
            {"name": "video", "required": False},
        ],
        "progress": {"mode": "fixed_time", "target_points": 64},
        "sources": [
            {
                "source_id": "robot-a",
                "embodiment": "robot",
                "fps": 10.0,
                "latency_frames": 0,
                "latency_tolerance_frames": 1,
                "validity_field": "valid",
                "resampling": "nearest",
                "channels": [
                    {
                        "name": "x",
                        "state_index": 0,
                        "action_index": 0,
                        "action_min": -1.0,
                        "action_max": 1.0,
                    }
                ],
            }
        ],
        "gates": {
            "hard": ["non_finite", "dimension_mismatch"],
            "review": ["timestamp_jitter", "idle_ratio"],
        },
        "vlm": {
            "backend": "disabled",
            "model": "mock-v1",
            "max_calls": 0,
            "max_prompt_tokens": 0,
            "max_cost_usd": 0.0,
            "estimated_cost_per_call_usd": 0.0,
        },
        "sidecar_root": ".rds-artifacts/curation",
        "normalization": {"source_weights": {}, "task_weights": {}},
    }


def frame(
    episode_id: str,
    frame_index: int,
    timestamp: float,
    *,
    valid: bool = True,
    decoded_pts: float | None = None,
) -> FrameSample:
    return FrameSample(
        episode_id=episode_id,
        frame_index=frame_index,
        timestamp=timestamp,
        decoded_pts=decoded_pts,
        valid=valid,
        state=[float(frame_index)],
        action=[float(frame_index)],
    )


def test_profile_requires_nonempty_train_allowlist() -> None:
    payload = profile_payload()
    payload["selection"]["train_allowlist"] = []

    with pytest.raises(ValidationError, match="train_allowlist"):
        CurationProfile.model_validate(payload)


def test_deny_pattern_wins_over_train_allowlist() -> None:
    profile = CurationProfile.model_validate(profile_payload())

    assert is_source_allowed(profile, "train/robot-a.jsonl", split="train") is True
    assert is_source_allowed(profile, "train/eval-leak.jsonl", split="train") is False
    assert is_source_allowed(profile, "train/robot-a.jsonl", split="validation") is False


def test_v2_profile_requires_explicit_current_check_selection_and_rejects_retired_ids() -> None:
    payload = profile_payload()
    payload["profile_version"] = "2"
    with pytest.raises(ValidationError, match="quality_contract"):
        CurationProfile.model_validate(payload)

    payload["quality_contract"] = {
        "contract_version": "qwen-cleaning-v2",
        "qwen": {
            "S1": {"enabled": True},
            "S2": {"enabled": False},
            "S3": {"enabled": True},
            "C1": {"enabled": True},
            "C2": {"enabled": False},
            "C3": {"enabled": True},
        },
        "human_extensions": {
            "H1_tracking_confidence": {"enabled": True},
            "H2_bone_topology": {"enabled": True},
            "H3_left_right_identity": {"enabled": True},
            "H4_se3_dynamics": {"enabled": True},
            "H5_quality_hand_crosscheck": {"enabled": True},
            "H6_camera_projection": {"enabled": True},
            "H6b_stored_video_projection": {"enabled": True},
            "H7_bimanual_validity": {"enabled": True},
            "H8_text_event_consistency": {"enabled": True},
            "H9_hand_object_interaction": {"enabled": False},
        },
    }
    profile = CurationProfile.model_validate(payload)
    assert profile.quality_contract is not None
    assert profile.quality_contract.qwen["S2"].enabled is False

    payload["quality_contract"]["qwen"]["S4"] = {"enabled": True}
    with pytest.raises(ValidationError, match="S4"):
        CurationProfile.model_validate(payload)


def test_language_normalization_retains_raw_and_slots() -> None:
    normalized = normalize_task_text("Move the red cup onto the shelf carefully")

    assert normalized.raw_text == "Move the red cup onto the shelf carefully"
    assert normalized.verb == "move"
    assert normalized.object == "the red cup"
    assert normalized.target == "the shelf"
    assert normalized.modifiers == ["carefully"]


def test_v3_time_uses_row_timestamp_plus_from_timestamp() -> None:
    rows = [
        frame("ep-1", 10, 0.0, decoded_pts=12.5),
        frame("ep-1", 11, 0.1, decoded_pts=12.6),
    ]

    points = progress_resample(
        rows,
        from_timestamp=12.5,
        target_points=2,
        mode="fixed_time",
        decoded_pts_tolerance=0.005,
    )

    assert [point.episode_time for point in points] == pytest.approx([0.0, 0.1])
    assert [point.file_time for point in points] == pytest.approx([12.5, 12.6])
    assert [point.original_frame_index for point in points] == [10, 11]


def test_v3_decoded_pts_is_only_a_tolerance_check() -> None:
    rows = [frame("ep-1", 0, 0.0, decoded_pts=5.25)]

    with pytest.raises(PtsToleranceError, match="decoded PTS"):
        progress_resample(
            rows,
            from_timestamp=5.0,
            target_points=1,
            mode="fixed_time",
            decoded_pts_tolerance=0.01,
        )


def test_resampling_keeps_sparse_invalid_track_invalid() -> None:
    rows = [
        frame("ep-1", 0, 0.0),
        frame("ep-1", 1, 0.5, valid=False),
        frame("ep-1", 2, 1.0),
    ]

    points = progress_resample(rows, from_timestamp=3.0, target_points=3, mode="fixed_time")

    assert [point.original_frame_index for point in points] == [0, 1, 2]
    assert [point.valid for point in points] == [True, False, True]


def test_chunks_never_cross_episodes_and_maps_have_64_points() -> None:
    rows = [
        *(frame("ep-1", index, index / 10) for index in range(3)),
        *(frame("ep-2", index + 20, index / 10) for index in range(3)),
    ]

    chunks = list(iter_episode_chunks(rows, chunk_size=2))
    assert [{item.episode_id for item in chunk} for chunk in chunks] == [
        {"ep-1"},
        {"ep-1"},
        {"ep-2"},
        {"ep-2"},
    ]
    assert len(progress_resample(rows[:3], from_timestamp=0.0, target_points=64)) == 64

    with pytest.raises(EpisodeBoundaryError, match="one episode"):
        progress_resample(rows, from_timestamp=0.0, target_points=64)

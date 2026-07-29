from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from robot_data_studio.curation.adapters import (
    ContactSheetBuilder,
    ForwardKinematics,
    HumanSparseVideoAdapter,
    NearestNeighborActionConflict,
    Reprojection,
    RobotSparseVideoAdapter,
    TrajectoryEntropy,
    select_keyframe_evidence,
)
from robot_data_studio.curation.checks import (
    check_action_artifacts,
    check_boundaries,
    check_idle_stuck,
    check_state_action_alignment,
)
from robot_data_studio.curation.dispositions import disposition_from_findings
from robot_data_studio.curation.models import (
    EpisodeRecord,
    EvidenceFrame,
    Finding,
    FrameSample,
)
from robot_data_studio.curation.pilot import run_pilot
from robot_data_studio.curation.profile import CurationProfile
from robot_data_studio.curation.vlm import BudgetExceeded, BudgetLedger, MockVlmEvaluator


def make_profile(*, vlm_backend: str = "disabled") -> CurationProfile:
    return CurationProfile.model_validate(
        {
            "profile_version": "1",
            "name": "synthetic",
            "selection": {
                "train_allowlist": ["train/*.jsonl"],
                "deny_patterns": ["*validation*", "*test*", "*eval*"],
            },
            "task": {"text": "move block to target", "events": ["start", "contact", "end"]},
            "modalities": [{"name": "state"}, {"name": "action"}],
            "progress": {"mode": "event_or_motion", "target_points": 64},
            "sources": [
                {
                    "source_id": "robot-a",
                    "embodiment": "robot",
                    "fps": 10.0,
                    "latency_frames": 1,
                    "latency_tolerance_frames": 0,
                    "validity_field": "valid",
                    "resampling": "nearest",
                    "channels": [
                        {
                            "name": "joint-x",
                            "state_index": 0,
                            "action_index": 0,
                            "action_min": -1.0,
                            "action_max": 1.0,
                        }
                    ],
                }
            ],
            "gates": {"hard": ["non_finite"], "review": ["episode_start_truncated"]},
            "checks": {
                "boundary_tolerance_seconds": 0.05,
                "idle_motion_threshold": 0.001,
                "max_idle_ratio": 0.5,
                "action_repeat_frames": 3,
                "quantization_unique_ratio": 0.4,
                "saturation_tolerance": 1e-6,
            },
            "vlm": {
                "backend": vlm_backend,
                "model": "mock-v1",
                "prompt": "judge evidence",
                "max_calls": 1 if vlm_backend == "mock" else 0,
                "max_prompt_tokens": 100 if vlm_backend == "mock" else 0,
                "max_cost_usd": 0.01 if vlm_backend == "mock" else 0.0,
                "estimated_cost_per_call_usd": 0.01 if vlm_backend == "mock" else 0.0,
            },
            "sidecar_root": ".rds-artifacts/curation",
            "normalization": {
                "source_weights": {"robot-a": 1.0},
                "task_weights": {"move-block": 2.0},
            },
        }
    )


def sample(
    index: int,
    *,
    state: float,
    action: float,
    contact_hold: bool = False,
) -> FrameSample:
    return FrameSample(
        episode_id="ep-1",
        frame_index=index,
        timestamp=index / 10,
        state=[state],
        action=[action],
        contact_hold=contact_hold,
    )


def episode(episode_id: str = "ep-1", *, task_id: str = "move-block") -> EpisodeRecord:
    frames = [
        FrameSample(
            episode_id=episode_id,
            frame_index=index,
            timestamp=index / 10,
            state=[float(index)],
            action=[float(index) / 10],
        )
        for index in range(6)
    ]
    return EpisodeRecord(
        episode_id=episode_id,
        source_id="robot-a",
        task_id=task_id,
        split="train",
        from_timestamp=4.0,
        expected_duration=0.5,
        frames=frames,
        event_frames={"start": 0, "contact": 3, "end": 5},
        lineage={"shard": "synthetic-000"},
    )


def test_dispositions_keep_local_mask_separate_from_episode_reject() -> None:
    disposition = disposition_from_findings(
        [
            Finding(code="local_repeat", outcome="window_mask", start_frame=3, end_frame=7),
            Finding(code="ambiguous_text", outcome="review"),
            Finding(code="minor_noise", outcome="soft_weight", weight_multiplier=0.8),
        ]
    )

    assert disposition.hard_reject is False
    assert disposition.review is True
    assert disposition.window_masks[0].reason == "local_repeat"
    assert disposition.soft_weight == pytest.approx(0.8)


def test_sparse_adapters_emit_start_contact_end_refs_and_keep_advanced_interfaces() -> None:
    record = episode()

    evidence = select_keyframe_evidence(record)

    assert [item.label for item in evidence] == ["start", "contact", "end"]
    assert HumanSparseVideoAdapter.kind == "human"
    assert RobotSparseVideoAdapter.kind == "robot"
    assert all(protocol is not None for protocol in (
        ForwardKinematics,
        Reprojection,
        ContactSheetBuilder,
        NearestNeighborActionConflict,
        TrajectoryEntropy,
    ))


def test_configured_decoded_pts_tolerance_is_enforced(tmp_path: Path) -> None:
    profile = make_profile()
    source = profile.sources[0].model_copy(
        update={"decoded_pts_tolerance_seconds": 0.001}
    )
    profile = profile.model_copy(update={"sources": [source]})
    record = episode()
    record.frames[0] = record.frames[0].model_copy(
        update={"decoded_pts": record.from_timestamp + record.frames[0].timestamp + 0.01}
    )

    result = run_pilot(
        profile,
        iter([record]),
        source_path="train/synthetic.jsonl",
        sidecar_root=tmp_path / "sidecars",
        max_episodes=1,
    ).episodes[0]

    assert result.disposition.hard_reject is True
    assert "time_mapping_invalid" in {finding.code for finding in result.findings}


def test_non_finite_episode_reject_still_writes_finite_sidecar(tmp_path: Path) -> None:
    profile = make_profile()
    record = episode()
    record.frames[0] = record.frames[0].model_copy(update={"state": [float("nan")]})
    sidecar_root = tmp_path / "sidecars"

    result = run_pilot(
        profile,
        iter([record]),
        source_path="train/synthetic.jsonl",
        sidecar_root=sidecar_root,
        max_episodes=1,
    ).episodes[0]

    assert result.disposition.hard_reject is True
    assert "non_finite" in {finding.code for finding in result.findings}
    assert math.isfinite(result.motion)
    assert (sidecar_root / "manifest.json").is_file()


def test_profile_gate_escalates_without_erasing_window_mask(tmp_path: Path) -> None:
    profile = make_profile()
    profile = profile.model_copy(
        update={"gates": profile.gates.model_copy(update={"hard": ["action_long_repeat"]})}
    )
    record = episode()
    record.frames = [frame.model_copy(update={"action": [0.25]}) for frame in record.frames]

    result = run_pilot(
        profile,
        iter([record]),
        source_path="train/synthetic.jsonl",
        sidecar_root=tmp_path / "sidecars",
        max_episodes=1,
    ).episodes[0]

    assert result.disposition.hard_reject is True
    assert result.disposition.hard_reasons == ["action_long_repeat"]
    assert result.disposition.window_masks[0].reason == "action_long_repeat"


def test_boundary_and_contact_hold_aware_idle_checks() -> None:
    profile = make_profile()
    source = profile.source("robot-a")
    held = [sample(index, state=0.0, action=0.0, contact_hold=True) for index in range(6)]
    truncated = EpisodeRecord(
        episode_id="ep-1",
        source_id="robot-a",
        task_id="hold",
        split="train",
        from_timestamp=0.0,
        expected_duration=1.0,
        frames=[item.model_copy(update={"timestamp": item.timestamp + 0.2}) for item in held],
    )

    assert check_idle_stuck(held, profile.checks) == []
    assert {finding.code for finding in check_boundaries(truncated, profile.checks)} == {
        "episode_start_truncated",
        "episode_end_truncated",
    }


def test_action_repeat_quantization_and_saturation_are_reported() -> None:
    profile = make_profile()
    frames = [sample(index, state=float(index), action=1.0) for index in range(6)]

    codes = {
        finding.code
        for finding in check_action_artifacts(frames, profile.source("robot-a"), profile.checks)
    }

    assert {"action_long_repeat", "action_quantization", "action_saturation"} <= codes


def test_source_channel_latency_and_sign_mismatch_are_reported() -> None:
    profile = make_profile()
    source = profile.source("robot-a")
    action = [0.0, 1.0, 0.0, -1.0, 0.0, 1.0]
    delayed_state = [0.0, 0.0, 1.0, 0.0, -1.0, 0.0]
    delayed = [sample(i, state=delayed_state[i], action=action[i]) for i in range(len(action))]

    assert check_state_action_alignment(delayed, source) == []

    wrong_sign = [sample(i, state=-action[i], action=action[i]) for i in range(len(action))]
    codes = {finding.code for finding in check_state_action_alignment(wrong_sign, source)}
    assert "state_action_latency_mismatch" in codes
    assert "state_action_sign_mismatch" in codes


def test_mock_vlm_reserves_budget_once_and_uses_hash_cache(tmp_path: Path) -> None:
    profile = make_profile(vlm_backend="mock")
    ledger = BudgetLedger.from_profile(profile.vlm)
    evaluator = MockVlmEvaluator(tmp_path / "cache", ledger)
    evidence = [EvidenceFrame(label="contact", frame_index=3, timestamp=0.3, uri="video.mp4")]

    first = evaluator.evaluate(
        evidence=evidence,
        prompt=profile.vlm.prompt,
        model=profile.vlm.model,
        mock_state="consistent",
    )
    second = evaluator.evaluate(
        evidence=evidence,
        prompt=profile.vlm.prompt,
        model=profile.vlm.model,
        mock_state="conflict",
    )

    assert first.from_cache is False
    assert second.from_cache is True
    assert second.state == "consistent"
    assert ledger.calls == 1
    with pytest.raises(BudgetExceeded):
        evaluator.evaluate(
            evidence=evidence,
            prompt="different prompt",
            model=profile.vlm.model,
            mock_state="review",
        )


def test_streaming_pilot_writes_sidecars_coverage_and_resumes(tmp_path: Path) -> None:
    profile = make_profile()
    episodes = [episode("ep-1"), episode("ep-2", task_id="place-block")]
    sidecar_root = tmp_path / "sidecars"

    first = run_pilot(
        profile,
        iter(episodes),
        source_path="train/synthetic.jsonl",
        sidecar_root=sidecar_root,
        max_episodes=2,
    )
    second = run_pilot(
        profile,
        iter(episodes),
        source_path="train/synthetic.jsonl",
        sidecar_root=sidecar_root,
        max_episodes=2,
    )

    assert first.processed == 2
    assert first.resumed == 0
    assert second.processed == 0
    assert second.resumed == 2
    assert first.coverage.by_source == {"robot-a": 2}
    assert first.coverage.by_task == {"move-block": 1, "place-block": 1}
    assert sum(first.coverage.by_duration.values()) == 2
    assert sum(first.coverage.by_motion.values()) == 2
    assert len(first.episodes[0].resample_map) == 64
    assert (sidecar_root / "manifest.json").is_file()
    manifest = json.loads((sidecar_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["episodes"][0]["sidecar"].startswith("episodes/")
    assert "sidecar_hash" in manifest["episodes"][0]
    assert "lineage_hash" in manifest["episodes"][0]
    assert "resample_map" not in manifest["episodes"][0]
    assert not list(sidecar_root.rglob("*.mp4"))

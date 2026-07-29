from __future__ import annotations

import numpy as np
import pytest

from robot_data_studio.quality.contracts import (
    CURRENT_QWEN_CONTRACT_VERSION,
    CORE_CONTRACT_IDS,
    HISTORICAL_QWEN_CORE_IDS,
    QWEN_CORE_DEFAULT_WEIGHTS,
    RETIRED_QWEN_CORE_IDS,
    FrameInterval,
    MetricEvidence,
    ScorecardDisposition,
    build_c1_instruction_evidence,
    build_quality_scorecard,
    evaluate_c2_video_state_consistency,
)
from robot_data_studio.quality.filter_models import (
    EpisodeFilterStageStatus,
    EpisodeFilterSummary,
)
from robot_data_studio.quality.scorecard import build_qwen_scorecard
from robot_data_studio.quality.execution import (
    build_check_execution_plan,
    intermediate_cache_key,
)


def _metric(metric_id: str, score: float | None, *, weight: float = 1.0) -> MetricEvidence:
    return MetricEvidence(
        metric_id=metric_id,
        label=metric_id,
        group="qwen_core",
        status="evaluated" if score is not None else "not_evaluated",
        score=score,
        weight=weight,
        reason="fixture" if score is not None else "fixture unavailable",
    )


def test_qwen_core_contract_is_explicit_and_preserves_existing_weights() -> None:
    assert CURRENT_QWEN_CONTRACT_VERSION == "qwen-cleaning-v2"
    assert CORE_CONTRACT_IDS == ("S1", "S2", "S3", "C1", "C2", "C3")
    assert RETIRED_QWEN_CORE_IDS == ("S4", "S5")
    assert HISTORICAL_QWEN_CORE_IDS == (
        "S1", "S2", "S3", "S4", "S5", "C1", "C2", "C3"
    )
    assert QWEN_CORE_DEFAULT_WEIGHTS == {
        "S1": 1.5,
        "S2": 1.5,
        "S3": 2.0,
        "S4": 2.0,
        "S5": 1.0,
        "C1": 2.0,
        "C2": 1.0,
        "C3": 1.5,
    }


def test_missing_metrics_are_na_and_lower_coverage_without_becoming_zero() -> None:
    metrics = [
        _metric(metric_id, 0.8 if metric_id in {"S1", "S2", "S3", "C3"} else None)
        for metric_id in CORE_CONTRACT_IDS
    ]

    scorecard = build_quality_scorecard(metrics)

    assert scorecard.qwen_core_score == 0.8
    assert scorecard.coverage == pytest.approx(4 / 6)
    assert scorecard.selected_denominator == 6
    assert scorecard.evaluated_count == 4
    assert scorecard.final_decision == "review"
    assert scorecard.metrics[3].score is None
    assert scorecard.metrics[3].status == "not_evaluated"


def test_disabled_and_retired_checks_are_visible_but_omitted_from_every_denominator() -> None:
    metrics = [
        _metric("S1", 1.0, weight=QWEN_CORE_DEFAULT_WEIGHTS["S1"]),
        _metric("S2", None).model_copy(update={"status": "disabled", "enabled": False}),
        _metric("S3", 0.5, weight=QWEN_CORE_DEFAULT_WEIGHTS["S3"]),
        _metric("C1", None),
        _metric("C2", None).model_copy(update={"status": "disabled", "enabled": False}),
        _metric("C3", 1.0, weight=QWEN_CORE_DEFAULT_WEIGHTS["C3"]),
        _metric("S4", None).model_copy(
            update={"status": "retired", "enabled": False, "contract_state": "retired"}
        ),
        _metric("S5", None).model_copy(
            update={"status": "retired", "enabled": False, "contract_state": "retired"}
        ),
    ]

    scorecard = build_quality_scorecard(metrics)

    assert scorecard.selected_checks == ["S1", "S3", "C1", "C3"]
    assert scorecard.selected_denominator == 4
    assert scorecard.evaluated_count == 3
    assert scorecard.coverage == 0.75
    assert scorecard.qwen_core_score == pytest.approx((1.5 + 1.0 + 1.5) / (1.5 + 2.0 + 1.5))
    assert scorecard.final_decision == "review"


def test_selected_human_extensions_contribute_to_actual_coverage_denominator() -> None:
    metrics = [
        _metric("S1", 1.0),
        MetricEvidence(
            metric_id="H1_tracking_confidence",
            label="tracking",
            group="human_extension",
            status="evaluated",
            score=0.9,
            reason="fixture",
        ),
        MetricEvidence(
            metric_id="H9_hand_object_interaction",
            label="interaction",
            group="human_extension",
            status="not_evaluated",
            reason="mask prerequisite missing",
        ),
        MetricEvidence(
            metric_id="H7_bimanual_validity",
            label="validity",
            group="human_extension",
            status="disabled",
            enabled=False,
            reason="not selected",
        ),
    ]

    scorecard = build_quality_scorecard(metrics)

    assert scorecard.qwen_core_coverage == 1.0
    assert scorecard.extension_coverage == 0.5
    assert scorecard.coverage == pytest.approx(2 / 3)
    assert scorecard.selected_denominator == 3
    assert scorecard.evaluated_count == 2
    assert scorecard.selected_checks == [
        "S1",
        "H1_tracking_confidence",
        "H9_hand_object_interaction",
    ]


def test_disposition_reasons_and_shared_windows_are_aggregated_once() -> None:
    disposition = ScorecardDisposition(
        review=True,
        reasons=["shared-mask", "shared-mask", "independent"],
        window_masks=[
            FrameInterval(start_frame=3, end_frame=5, reason="shared-mask"),
            FrameInterval(start_frame=3, end_frame=5, reason="shared-mask"),
        ],
    )

    assert disposition.reasons == ["shared-mask", "independent"]
    assert len(disposition.window_masks) == 1


def test_historical_evaluated_s4_s5_are_wrapped_as_retired_and_never_change_score() -> None:
    metrics = [_metric(metric_id, 1.0) for metric_id in CORE_CONTRACT_IDS]
    metrics.extend([_metric("S4", 0.0), _metric("S5", 0.0)])

    scorecard = build_quality_scorecard(metrics)
    by_id = {metric.metric_id: metric for metric in scorecard.metrics}

    assert scorecard.qwen_core_score == 1.0
    assert scorecard.coverage == 1.0
    assert by_id["S4"].status == by_id["S5"].status == "retired"
    assert by_id["S4"].evidence["historical_status"] == "evaluated"


def test_check_plan_keeps_selection_independent_and_deduplicates_intermediates() -> None:
    plan = build_check_execution_plan(
        enabled_quality_checks={"C1", "C2", "C3"},
        enabled_human_extension_checks={"H9_hand_object_interaction"},
        embodiment="robot",
        available_intermediates={"task_structure", "sparse_video_decode"},
    )

    by_id = {item.check_id: item for item in plan.checks}
    assert by_id["C1"].state == "ready"
    assert by_id["C2"].state == "missing_prerequisite"
    assert by_id["C2"].missing_prerequisites == [
        "robot_full_mesh_projection", "robot_masks"
    ]
    assert by_id["C3"].state == "ready"
    assert by_id["H9_hand_object_interaction"].state == "missing_prerequisite"
    assert by_id["S1"].state == "disabled"
    assert by_id["S4"].state == "retired"
    assert plan.required_intermediates.count("sparse_video_decode") == 1


def test_shared_intermediate_cache_key_binds_content_model_prompt_config_and_frames() -> None:
    base = intermediate_cache_key(
        content_hash="a" * 64,
        intermediate="sparse_video_decode",
        model="sam3-v1",
        prompt="robot",
        config={"threshold": 0.5},
        frame_indexes=[1, 8],
    )
    same = intermediate_cache_key(
        content_hash="a" * 64,
        intermediate="sparse_video_decode",
        model="sam3-v1",
        prompt="robot",
        config={"threshold": 0.5},
        frame_indexes=[1, 8],
    )
    changed = intermediate_cache_key(
        content_hash="a" * 64,
        intermediate="sparse_video_decode",
        model="sam3-v1",
        prompt="robot",
        config={"threshold": 0.5},
        frame_indexes=[1, 9],
    )
    assert base == same
    assert base != changed


def test_hard_gate_precedes_scores_and_dispositions_remain_independent() -> None:
    metrics = [_metric(metric_id, 1.0) for metric_id in CORE_CONTRACT_IDS]
    scorecard = build_quality_scorecard(
        metrics,
        disposition=ScorecardDisposition(
            hard_reject=True,
            review=True,
            window_masks=[FrameInterval(start_frame=4, end_frame=8, reason="local invalid")],
            soft_weight=0.7,
        ),
    )

    assert scorecard.qwen_core_score == 1.0
    assert scorecard.final_decision == "reject"
    assert scorecard.disposition.review is True
    assert scorecard.disposition.window_masks[0].start_frame == 4
    assert scorecard.disposition.soft_weight == 0.7


def test_provisional_c2_can_request_review_but_cannot_cause_rejection() -> None:
    provisional = _metric("C2", 0.1).model_copy(update={"fidelity": "provisional"})
    calibrated = _metric("C2", 0.1).model_copy(update={"fidelity": "calibrated"})

    provisional_scorecard = build_quality_scorecard([provisional])
    calibrated_scorecard = build_quality_scorecard([calibrated])

    assert provisional_scorecard.qwen_core_score == 0.1
    assert provisional_scorecard.final_decision == "review"
    assert calibrated_scorecard.final_decision == "reject"


def test_c1_disabled_is_structured_na_and_legacy_task_success_is_marked_approximate() -> None:
    disabled = build_c1_instruction_evidence(
        raw_text="Pick up the red block",
        structured_task={"verb": "pick", "object": "block", "modifiers": ["red"]},
        events={"start": 0, "contact": 8, "end": 16},
    )
    approximate = build_c1_instruction_evidence(
        raw_text="Pick up the red block",
        structured_task={"verb": "pick", "object": "block", "modifiers": ["red"]},
        events={"start": 0, "contact": 8, "end": 16},
        approximate_task_success_score=0.75,
    )

    assert disabled.status == "not_evaluated"
    assert disabled.score is None
    assert disabled.evidence["provider"] == "disabled"
    assert approximate.status == "evaluated"
    assert approximate.score == 0.75
    assert approximate.evidence["approximate"] is True


def test_c2_iou_uses_deterministic_masks_and_is_na_without_mask_provider() -> None:
    projected = np.asarray([[True, True], [False, False]])
    observed = np.asarray([[True, False], [False, False]])

    evaluated = evaluate_c2_video_state_consistency(
        projected_robot_mask=projected,
        observed_robot_mask=observed,
        iou_threshold=0.4,
    )
    missing = evaluate_c2_video_state_consistency(
        projected_robot_mask=projected,
        observed_robot_mask=None,
        missing_reason="SAM3/provider unavailable",
    )

    assert evaluated.status == "evaluated"
    assert evaluated.score == 0.5
    assert evaluated.evidence["iou"] == 0.5
    assert missing.status == "not_evaluated"
    assert missing.score is None
    assert "SAM3" in missing.reason


def test_scorecard_bridge_keeps_paper_s2_separate_from_legacy_time_sync() -> None:
    passed = EpisodeFilterStageStatus(count=0, status="passed", score=1.0)
    stages = {
        "visual_quality": passed,
        "sudden_change": passed,
        "state_action_alignment": EpisodeFilterStageStatus(
            count=1, status="review", score=0.4
        ),
        "extreme_value": passed,
        "kinematic_consistency": passed,
        "orientation_alignment": passed,
        "metadata_completeness": passed,
    }
    episode = EpisodeFilterSummary(episode_index=0, stage_status=stages)
    action = np.arange(12, dtype=float).reshape(-1, 1)
    state = np.concatenate([np.zeros((1, 1)), action[:-1]], axis=0)

    scorecard = build_qwen_scorecard(
        episode,
        states=state,
        actions=action,
        raw_task_text="move the block",
        events={"start": 0, "end": 11},
    )

    metrics = {metric.metric_id: metric for metric in scorecard.metrics}
    assert metrics["S2"].status == "evaluated"
    assert metrics["S2"].evidence["implementation"] == "detect_state_action_trend_alignment"
    assert metrics["legacy_time_sync"].score == 0.4
    assert metrics["legacy_time_sync"].group == "robot_extension"
    assert metrics["C1"].status == "not_evaluated"
    assert metrics["C2"].status == "not_evaluated"
    assert scorecard.coverage == pytest.approx(4 / 6)


def test_selected_missing_human_check_is_explicit_na_in_generic_robot_scorer() -> None:
    passed = EpisodeFilterStageStatus(count=0, status="passed", score=1.0)
    episode = EpisodeFilterSummary(
        episode_index=0,
        stage_status={
            "visual_quality": passed,
            "sudden_change": passed,
            "state_action_alignment": passed,
            "extreme_value": passed,
            "kinematic_consistency": passed,
            "orientation_alignment": passed,
            "metadata_completeness": passed,
        },
    )

    scorecard = build_qwen_scorecard(
        episode,
        states=np.zeros((3, 1)),
        actions=np.zeros((3, 1)),
        raw_task_text="move",
        events={"start": 0, "end": 2},
        embodiment="robot",
        enabled_quality_checks={"S1"},
        enabled_extension_checks={"H9_hand_object_interaction"},
    )

    metrics = {metric.metric_id: metric for metric in scorecard.metrics}
    assert metrics["H9_hand_object_interaction"].status == "not_evaluated"
    assert "Human-only" in metrics["H9_hand_object_interaction"].reason
    assert scorecard.selected_denominator == 2
    assert scorecard.coverage == 0.5

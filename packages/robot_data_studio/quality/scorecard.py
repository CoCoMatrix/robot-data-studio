from __future__ import annotations

import numpy as np

from .contracts import (
    CORE_CONTRACT_IDS,
    HUMAN_EXTENSION_IDS,
    QWEN_CORE_DEFAULT_WEIGHTS,
    QWEN_CORE_LABELS,
    MetricEvidence,
    QualityScorecard,
    ScorecardDisposition,
    build_c1_instruction_evidence,
    build_quality_scorecard,
    disabled_metric,
    evaluate_c2_video_state_consistency,
    retired_metric,
)
from .filter_models import EpisodeFilterStageStatus, EpisodeFilterSummary
from .qwen_filters import StateActionAlignmentConfig, detect_state_action_trend_alignment


_CORE_STAGE_MAP = {
    "S1": "sudden_change",
    "S3": "extreme_value",
    "C3": "visual_quality",
}


def _stage_metric(
    metric_id: str,
    stage: EpisodeFilterStageStatus,
) -> MetricEvidence:
    if stage.score is None or stage.skipped_reason:
        return MetricEvidence(
            metric_id=metric_id,
            label=QWEN_CORE_LABELS[metric_id],
            group="qwen_core",
            status="not_evaluated",
            weight=QWEN_CORE_DEFAULT_WEIGHTS[metric_id],
            reason=stage.skipped_reason or "stage did not produce a score",
            evidence={"legacy_stage_status": stage.status, "count": stage.count},
        )
    return MetricEvidence(
        metric_id=metric_id,
        label=QWEN_CORE_LABELS[metric_id],
        group="qwen_core",
        status="evaluated",
        score=stage.score,
        weight=QWEN_CORE_DEFAULT_WEIGHTS[metric_id],
        reason="preserved quality filter stage",
        evidence={
            "legacy_stage_status": stage.status,
            "count": stage.count,
            "severity": stage.severity,
        },
    )


def _s2_metric(
    states: np.ndarray,
    actions: np.ndarray,
    *,
    embodiment: str,
) -> MetricEvidence:
    if embodiment == "human":
        return MetricEvidence(
            metric_id="S2",
            label=QWEN_CORE_LABELS["S2"],
            group="qwen_core",
            status="not_evaluated",
            weight=QWEN_CORE_DEFAULT_WEIGHTS["S2"],
            reason="robot state-action trend alignment is not applicable to Human tracks",
        )
    states = np.asarray(states, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    if (
        states.ndim != 2
        or actions.ndim != 2
        or len(states) < 3
        or states.shape != actions.shape
        or not np.all(np.isfinite(states))
        or not np.all(np.isfinite(actions))
    ):
        return MetricEvidence(
            metric_id="S2",
            label=QWEN_CORE_LABELS["S2"],
            group="qwen_core",
            status="not_evaluated",
            weight=QWEN_CORE_DEFAULT_WEIGHTS["S2"],
            reason="compatible finite state/action channels are unavailable",
        )
    result = detect_state_action_trend_alignment(
        states,
        actions,
        StateActionAlignmentConfig(),
    )
    dimensions = len(result.dimension_results)
    score = 1.0 - len(result.flagged_dimensions) / max(dimensions, 1)
    return MetricEvidence(
        metric_id="S2",
        label=QWEN_CORE_LABELS["S2"],
        group="qwen_core",
        status="evaluated",
        score=score,
        weight=QWEN_CORE_DEFAULT_WEIGHTS["S2"],
        reason="paper S2 trend alignment; intentionally separate from legacy UI Time sync",
        evidence={
            "implementation": "detect_state_action_trend_alignment",
            "flagged_dimensions": result.flagged_dimensions,
            "dimensions": dimensions,
            "lags": [item.lag for item in result.dimension_results],
            "directional_agreement": [
                item.directional_agreement for item in result.dimension_results
            ],
        },
    )


def _legacy_extension(
    metric_id: str,
    label: str,
    stage: EpisodeFilterStageStatus,
    *,
    embodiment: str,
) -> MetricEvidence:
    group = "human_extension" if embodiment == "human" else "robot_extension"
    if stage.score is None or stage.skipped_reason:
        return MetricEvidence(
            metric_id=metric_id,
            label=label,
            group=group,
            status="not_evaluated",
            reason=stage.skipped_reason or "stage did not produce a score",
        )
    return MetricEvidence(
        metric_id=metric_id,
        label=label,
        group=group,
        status="evaluated",
        score=stage.score,
        reason="preserved legacy stage, not mixed into Qwen Core",
        evidence={"count": stage.count, "status": stage.status},
    )


def build_qwen_scorecard(
    filter_episode: EpisodeFilterSummary,
    *,
    states: np.ndarray,
    actions: np.ndarray,
    raw_task_text: str,
    events: dict[str, int],
    task_success_score: float | None = None,
    c2_metric: MetricEvidence | None = None,
    extension_metrics: list[MetricEvidence] | None = None,
    embodiment: str = "robot",
    enabled_quality_checks: set[str] | None = None,
    enabled_extension_checks: set[str] | None = None,
) -> QualityScorecard:
    selected = set(CORE_CONTRACT_IDS if enabled_quality_checks is None else enabled_quality_checks)
    unknown = selected - set(CORE_CONTRACT_IDS)
    if unknown:
        raise ValueError(f"unknown or retired quality checks selected: {sorted(unknown)}")
    core = [
        _stage_metric(metric_id, filter_episode.stage_status[stage_id])
        for metric_id, stage_id in _CORE_STAGE_MAP.items()
    ]
    core.append(_s2_metric(states, actions, embodiment=embodiment))
    core.append(
        build_c1_instruction_evidence(
            raw_text=raw_task_text,
            structured_task={"raw_text": raw_task_text},
            events=events,
            approximate_task_success_score=task_success_score,
        )
    )
    core.append(
        c2_metric
        or evaluate_c2_video_state_consistency(
            projected_robot_mask=None,
            observed_robot_mask=None,
            missing_reason="C2 requires trusted calibration projection and a robot-mask provider",
        )
    )
    core_by_id = {metric.metric_id: metric for metric in core}
    ordered_core = [
        core_by_id[metric_id] if metric_id in selected else disabled_metric(metric_id)
        for metric_id in CORE_CONTRACT_IDS
    ]
    retired = [retired_metric(metric_id) for metric_id in ("S4", "S5")]

    extensions = list(extension_metrics or [])
    extensions.extend(
        [
            _legacy_extension(
                "legacy_time_sync",
                "Legacy timestamp / media time sync",
                filter_episode.stage_status["state_action_alignment"],
                embodiment=embodiment,
            ),
            _legacy_extension(
                "metadata_completeness",
                "Metadata completeness",
                filter_episode.stage_status["metadata_completeness"],
                embodiment=embodiment,
            ),
        ]
    )
    if enabled_extension_checks is not None:
        extension_by_id = {metric.metric_id: metric for metric in extensions}
        selected_extensions: list[MetricEvidence] = []
        for metric_id in HUMAN_EXTENSION_IDS:
            if metric_id not in enabled_extension_checks:
                selected_extensions.append(
                    disabled_metric(metric_id, group="human_extension")
                )
            elif metric_id in extension_by_id:
                selected_extensions.append(extension_by_id[metric_id])
            else:
                selected_extensions.append(
                    MetricEvidence(
                        metric_id=metric_id,
                        label=metric_id,
                        group="human_extension",
                        status="not_evaluated",
                        reason=(
                            "Human-only check selected for a Robot episode"
                            if embodiment != "human"
                            else "required Human curation intermediate was not supplied"
                        ),
                        evidence={"missing_prerequisite": "human_curation_adapter"},
                    )
                )
        extensions = [
            *selected_extensions,
            *[
                metric
                if metric.metric_id in enabled_extension_checks
                else disabled_metric(metric.metric_id, group=metric.group)
                for metric in extensions
                if metric.metric_id not in HUMAN_EXTENSION_IDS
            ],
        ]
    active_stage_ids = {
        _CORE_STAGE_MAP[metric_id]
        for metric_id in selected
        if metric_id in _CORE_STAGE_MAP
    }
    if "S2" in selected:
        active_stage_ids.add("state_action_alignment")
    active_stage_ids.add("metadata_completeness")
    active_findings = [
        finding
        for finding in filter_episode.critical_findings
        if not (
            finding.code.startswith("kinematic_consistency")
            or finding.code.startswith("orientation_alignment")
        )
    ]
    hard_reject = bool(active_findings)
    review = any(
        stage_id in active_stage_ids
        and stage.status == "review"
        and not stage.skipped_reason
        for stage_id, stage in filter_episode.stage_status.items()
    )
    safety = [
        MetricEvidence(
            metric_id=f"gate_{finding.code}",
            label=finding.code.replace("_", " ").title(),
            group="safety_gate",
            status="evaluated",
            score=0.0,
            reason=finding.message,
            evidence={"severity": finding.severity},
        )
        for finding in active_findings
    ]
    return build_quality_scorecard(
        [*ordered_core, *retired, *extensions, *safety],
        disposition=ScorecardDisposition(
            hard_reject=hard_reject,
            review=review,
            reasons=[finding.code for finding in active_findings],
        ),
    )

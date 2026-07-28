from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from robot_data_studio.quality.contracts import (
    CORE_CONTRACT_IDS,
    QWEN_CORE_DEFAULT_WEIGHTS,
    QWEN_CORE_LABELS,
    FrameInterval,
    MetricEvidence,
    ScorecardDisposition,
    build_c1_instruction_evidence,
    build_quality_scorecard,
    disabled_metric,
    evaluate_c2_video_state_consistency,
)
from robot_data_studio.quality.qwen_filters import (
    StateActionAlignmentConfig,
    detect_state_action_trend_alignment,
)

from .adapters import evaluate_geometry, select_keyframe_evidence
from .checks import run_cheap_checks
from .dispositions import disposition_from_findings
from .models import (
    CoverageSummary,
    Disposition,
    EpisodeCurationResult,
    EpisodeRecord,
    Finding,
    PilotResult,
)
from .profile import CurationProfile, SourceProfile, is_source_allowed
from .semantics import normalize_task_text
from .time_mapping import EpisodeBoundaryError, PtsToleranceError, progress_resample
from .vlm import MockVlmEvaluator


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _profile_hash(profile: CurationProfile) -> str:
    return _canonical_hash(profile.model_dump(mode="json"))


def _episode_fingerprint(episode: EpisodeRecord) -> str:
    return _canonical_hash(episode.model_dump(mode="json"))


def _content_hash(episode: EpisodeRecord) -> str:
    payload = episode.model_dump(mode="json", exclude={"episode_id", "lineage"})
    return _canonical_hash(payload)


def _schema_fingerprint(episode: EpisodeRecord) -> str:
    dimensions = [
        sorted({len(frame.state) for frame in episode.frames}),
        sorted({len(frame.action) for frame in episode.frames}),
    ]
    return _canonical_hash({"dimensions": dimensions, "split": episode.split})


def _episode_motion(episode: EpisodeRecord) -> float:
    total = 0.0
    for previous, frame in zip(episode.frames, episode.frames[1:], strict=False):
        if not previous.valid or not frame.valid:
            continue
        if len(previous.state) == len(frame.state):
            delta = sum(abs(a - b) for a, b in zip(previous.state, frame.state, strict=True))
            if math.isfinite(delta):
                total += delta
    return total


def _duration_bucket(duration: float) -> str:
    if duration < 1.0:
        return "lt_1s"
    if duration < 10.0:
        return "1_to_10s"
    return "gte_10s"


def _motion_bucket(motion: float) -> str:
    if motion <= 1e-6:
        return "idle"
    if motion < 1.0:
        return "low"
    return "high"


def _coverage(results: list[EpisodeCurationResult]) -> CoverageSummary:
    return CoverageSummary(
        by_source=dict(sorted(Counter(result.source_id for result in results).items())),
        by_task=dict(sorted(Counter(result.task_id for result in results).items())),
        by_duration=dict(
            sorted(Counter(_duration_bucket(result.duration_seconds) for result in results).items())
        ),
        by_motion=dict(sorted(Counter(_motion_bucket(result.motion) for result in results).items())),
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _disposition(profile: CurationProfile, findings: list[Finding]) -> Disposition:
    return disposition_from_findings(
        findings,
        hard_gates=profile.gates.hard,
        review_gates=profile.gates.review,
    )


def _core_na(metric_id: str, reason: str) -> MetricEvidence:
    return MetricEvidence(
        metric_id=metric_id,
        label=QWEN_CORE_LABELS[metric_id],
        group="qwen_core",
        status="not_evaluated",
        weight=QWEN_CORE_DEFAULT_WEIGHTS[metric_id],
        reason=reason,
    )


def _curation_scorecard(
    profile: CurationProfile,
    episode: EpisodeRecord,
    source: SourceProfile,
    findings: list[Finding],
    disposition: Disposition,
    *,
    kinematics,
    reprojection,
) -> object:
    source_metrics: list[MetricEvidence] = []
    source_kind = source.embodiment
    if source_kind == "human":
        core = [
            _core_na(
                metric_id,
                (
                    "robot state-action metric is not applicable to Human tracks"
                    if metric_id == "S2"
                    else "robot FK metric is not applicable to Human tracks"
                    if metric_id == "S4"
                    else "Qwen Core bridge was not run on Human video/state evidence"
                ),
            )
            for metric_id in CORE_CONTRACT_IDS
        ]
    else:
        valid_frames = [frame for frame in episode.frames if frame.valid]
        states = np.asarray([frame.state for frame in valid_frames], dtype=np.float64)
        actions = np.asarray([frame.action for frame in valid_frames], dtype=np.float64)
        s1 = _core_na(
            "S1",
            "source/task train-only normalization reference was not fitted by this bounded pilot",
        )
        mapped_states = states
        mapped_actions = actions
        if source.channels and len(states) == len(actions):
            if all(
                channel.state_index < states.shape[1]
                and channel.action_index < actions.shape[1]
                for channel in source.channels
            ):
                mapped_states = states[:, [channel.state_index for channel in source.channels]]
                mapped_actions = actions[:, [channel.action_index for channel in source.channels]]
        if (
            mapped_states.ndim == 2
            and mapped_actions.ndim == 2
            and mapped_states.shape == mapped_actions.shape
            and len(mapped_states) >= 3
            and mapped_states.shape[1] > 0
        ):
            alignment = detect_state_action_trend_alignment(
                mapped_states, mapped_actions, StateActionAlignmentConfig()
            )
            s2 = MetricEvidence(
                metric_id="S2",
                label=QWEN_CORE_LABELS["S2"],
                group="qwen_core",
                status="evaluated",
                score=1.0 - len(alignment.flagged_dimensions) / len(alignment.dimension_results),
                weight=QWEN_CORE_DEFAULT_WEIGHTS["S2"],
                reason="existing trend-alignment implementation on explicit name/index mapped channels",
                evidence={
                    "flagged_dimensions": alignment.flagged_dimensions,
                    "channels": [channel.name for channel in source.channels],
                },
            )
        else:
            s2 = _core_na("S2", "no explicit equal-width state/action name mapping")
        if kinematics.state == "evaluated":
            tolerance = source.geometry.kinematics.position_tolerance
            maximum = float(kinematics.metrics.get("max_position_error", tolerance))
            s4 = MetricEvidence(
                metric_id="S4",
                label=QWEN_CORE_LABELS["S4"],
                group="qwen_core",
                status="evaluated",
                score=max(0.0, 1.0 - maximum / tolerance),
                weight=QWEN_CORE_DEFAULT_WEIGHTS["S4"],
                reason=kinematics.reason,
                evidence=kinematics.metrics,
            )
        else:
            s4 = _core_na("S4", kinematics.reason)
        c2 = (
            _core_na("C2", reprojection.reason)
            if reprojection.state != "evaluated"
            else evaluate_c2_video_state_consistency(
                projected_robot_mask=None,
                observed_robot_mask=None,
                missing_reason="projection evidence exists but robot-mask provider is unavailable",
            )
        )
        c1 = build_c1_instruction_evidence(
            raw_text=episode.raw_task_text or "",
            structured_task={"task_id": episode.task_id},
            events=episode.event_frames,
        )
        core = [
            s1,
            s2,
            _core_na("S3", "train-only source/task reference bounds were not fitted by this bounded episode pilot"),
            s4,
            _core_na("S5", "trusted base/EEF orientation mapping is not configured"),
            c1,
            c2,
            _core_na("C3", "video frames were not decoded by the metadata-only pilot"),
        ]
        non_hard = [finding for finding in findings if finding.outcome != "hard_reject"]
        source_metrics.append(
            MetricEvidence(
                metric_id="R1_cheap_signal_checks",
                label="Robot cheap signal checks",
                group="robot_extension",
                status="evaluated",
                score=max(0.0, 1.0 - len(non_hard) / max(len(episode.frames), 1)),
                reason="T0/T1 deterministic findings; separate from Qwen Core",
                evidence={"reason_codes": [finding.code for finding in non_hard]},
            )
        )
    safety_metrics = [
        MetricEvidence(
            metric_id=f"gate_{index}_{finding.code}",
            label=finding.code.replace("_", " ").title(),
            group="safety_gate",
            status="evaluated",
            score=0.0 if finding.outcome == "hard_reject" else 0.5,
            reason=finding.message or finding.code,
            abnormal_intervals=(
                [
                    FrameInterval(
                        start_frame=finding.start_frame,
                        end_frame=finding.end_frame,
                        reason=finding.code,
                    )
                ]
                if finding.start_frame is not None and finding.end_frame is not None
                else []
            ),
            evidence={"outcome": finding.outcome},
        )
        for index, finding in enumerate(findings)
    ]
    scorecard_disposition = ScorecardDisposition(
        hard_reject=disposition.hard_reject,
        review=disposition.review,
        window_masks=[
            FrameInterval(
                start_frame=mask.start_frame,
                end_frame=mask.end_frame,
                reason=mask.reason,
            )
            for mask in disposition.window_masks
        ],
        soft_weight=disposition.soft_weight,
        reasons=[*disposition.hard_reasons, *disposition.review_reasons],
    )
    selected_core = (
        {
            metric_id
            for metric_id, control in profile.quality_contract.qwen.items()
            if control.enabled
        }
        if profile.quality_contract is not None
        else set(CORE_CONTRACT_IDS)
    )
    configured_core: list[MetricEvidence] = []
    for metric in core:
        if metric.metric_id not in CORE_CONTRACT_IDS:
            configured_core.append(metric)
        elif metric.metric_id not in selected_core:
            configured_core.append(disabled_metric(metric.metric_id))
        elif profile.quality_contract is None:
            configured_core.append(metric)
        else:
            configured_weight = profile.quality_contract.qwen[metric.metric_id].weight
            configured_core.append(
                metric.model_copy(
                    update={
                        "weight": metric.weight
                        if configured_weight is None
                        else configured_weight
                    }
                )
            )
    core = configured_core
    extensions = list(episode.extension_metrics)
    if profile.quality_contract is not None:
        selected_extensions = {
            metric_id
            for metric_id, control in profile.quality_contract.human_extensions.items()
            if control.enabled
        }
        extension_by_id = {metric.metric_id: metric for metric in extensions}
        selected_metrics: list[MetricEvidence] = []
        for metric_id, control in profile.quality_contract.human_extensions.items():
            if not control.enabled:
                selected_metrics.append(disabled_metric(metric_id, group="human_extension"))
                continue
            metric = extension_by_id.get(metric_id)
            if metric is None:
                metric = MetricEvidence(
                    metric_id=metric_id,
                    label=metric_id,
                    group="human_extension",
                    status="not_evaluated",
                    reason=(
                        "Human-only check is not applicable to this Robot source"
                        if source_kind != "human"
                        else "selected Human check did not receive its required cached intermediate"
                    ),
                )
            if control.weight is not None:
                metric = metric.model_copy(update={"weight": control.weight})
            selected_metrics.append(metric)
        extensions = [
            *selected_metrics,
            *[
                metric
                for metric in extensions
                if metric.metric_id not in profile.quality_contract.human_extensions
            ],
        ]
    return build_quality_scorecard(
        [*core, *extensions, *source_metrics, *safety_metrics],
        disposition=scorecard_disposition,
    )


def _process_episode(
    profile: CurationProfile,
    episode: EpisodeRecord,
    *,
    evaluator: MockVlmEvaluator | None,
) -> EpisodeCurationResult:
    source = profile.source(episode.source_id)
    findings = run_cheap_checks(episode, source, profile.checks)
    evidence = select_keyframe_evidence(episode)
    disposition = _disposition(profile, findings)
    resample_map = []
    if episode.frames and not disposition.hard_reject:
        event_timestamps = [
            frame.timestamp
            for frame in episode.frames
            if frame.frame_index in set(episode.event_frames.values())
        ]
        try:
            resample_map = progress_resample(
                episode.frames,
                from_timestamp=episode.from_timestamp,
                target_points=profile.progress.target_points,
                mode=profile.progress.mode,
                decoded_pts_tolerance=source.decoded_pts_tolerance_seconds,
                event_timestamps=event_timestamps,
            )
        except (EpisodeBoundaryError, PtsToleranceError) as error:
            findings.append(
                Finding(code="time_mapping_invalid", outcome="hard_reject", tier="T1", message=str(error))
            )
            disposition = _disposition(profile, findings)
    vlm_result = None
    evidence_state = "review" if disposition.review else "consistent"
    if evaluator is not None and profile.vlm.backend == "mock" and not disposition.hard_reject:
        vlm_result = evaluator.evaluate(
            evidence=evidence,
            prompt=profile.vlm.prompt,
            model=profile.vlm.model,
            mock_state="review",
        )
        evidence_state = vlm_result.state
        if vlm_result.state == "conflict":
            findings.append(Finding(code="vlm_conflict", outcome="review", tier="T3"))
            disposition = _disposition(profile, findings)
    duration = episode.frames[-1].timestamp - episode.frames[0].timestamp if episode.frames else 0.0
    if not math.isfinite(duration):
        duration = 0.0
    task_text = episode.raw_task_text or profile.task.text
    kinematics, reprojection = evaluate_geometry(episode, source)
    for metric in episode.extension_metrics:
        if metric.metric_id != "H1_tracking_confidence":
            continue
        for interval in metric.abnormal_intervals:
            findings.append(
                Finding(
                    code="human_confidence_long_invalid",
                    outcome="window_mask",
                    tier="T2",
                    message=interval.reason,
                    start_frame=interval.start_frame,
                    end_frame=interval.end_frame,
                )
            )
    disposition = _disposition(profile, findings)
    scorecard = _curation_scorecard(
        profile,
        episode,
        source,
        findings,
        disposition,
        kinematics=kinematics,
        reprojection=reprojection,
    )
    return EpisodeCurationResult(
        episode_id=episode.episode_id,
        source_id=episode.source_id,
        task_id=episode.task_id,
        duration_seconds=max(0.0, duration),
        motion=_episode_motion(episode),
        fingerprint=_episode_fingerprint(episode),
        content_hash=_content_hash(episode),
        schema_fingerprint=_schema_fingerprint(episode),
        lineage=episode.lineage,
        normalized_task=normalize_task_text(task_text),
        evidence_state=evidence_state,
        evidence=evidence,
        findings=findings,
        disposition=disposition,
        resample_map=resample_map,
        vlm=vlm_result,
        kinematics=kinematics,
        reprojection=reprojection,
        qwen_core_score=scorecard.qwen_core_score,
        extension_score=scorecard.extension_score,
        coverage=scorecard.coverage,
        qwen_core_coverage=scorecard.qwen_core_coverage,
        extension_coverage=scorecard.extension_coverage,
        quality_contract_version=scorecard.contract_version,
        selected_checks=scorecard.selected_checks,
        selected_denominator=scorecard.selected_denominator,
        evaluated_count=scorecard.evaluated_count,
        final_decision=scorecard.final_decision,
        metric_evidence=scorecard.metrics,
    )


def run_pilot(
    profile: CurationProfile,
    episodes: Iterable[EpisodeRecord],
    *,
    source_path: str,
    sidecar_root: str | Path | None = None,
    max_episodes: int,
    evaluator: MockVlmEvaluator | None = None,
) -> PilotResult:
    if max_episodes < 1:
        raise ValueError("max_episodes must be positive; unbounded scans are not supported")
    if not is_source_allowed(profile, source_path, split="train"):
        raise PermissionError(f"source is not in the explicit train allowlist: {source_path}")
    root = Path(sidecar_root or profile.sidecar_root)
    episode_root = root / "episodes"
    profile_hash = _profile_hash(profile)
    results: list[EpisodeCurationResult] = []
    processed = 0
    resumed = 0
    seen_content: dict[str, str] = {}
    for episode_index, episode in enumerate(episodes):
        if episode_index >= max_episodes:
            break
        fingerprint = _episode_fingerprint(episode)
        path = episode_root / f"{fingerprint}.json"
        if path.is_file():
            result = EpisodeCurationResult.model_validate_json(path.read_text(encoding="utf-8"))
            resumed += 1
        else:
            result = _process_episode(profile, episode, evaluator=evaluator)
            processed += 1
        duplicate_of = seen_content.get(result.content_hash)
        if duplicate_of is not None and duplicate_of != result.episode_id:
            duplicate = Finding(code="duplicate_content", outcome="review", tier="T4", message=duplicate_of)
            if all(finding.code != duplicate.code for finding in result.findings):
                result.findings.append(duplicate)
                result.disposition = _disposition(profile, result.findings)
        else:
            seen_content[result.content_hash] = result.episode_id
        if not path.is_file():
            _write_json(path, result.model_dump(mode="json"))
        results.append(result)
    coverage = _coverage(results)
    reason_counts = dict(
        sorted(Counter(finding.code for result in results for finding in result.findings).items())
    )
    sampler_advice = {
        "source_weights": profile.normalization.source_weights,
        "task_weights": profile.normalization.task_weights,
        "observed_by_source": coverage.by_source,
        "observed_by_task": coverage.by_task,
        "note": "Apply configured train-only weights; no Human/Robot ratio is hardcoded.",
        "todo": ["nearest-neighbor action conflict", "trajectory diversity entropy"],
    }
    episode_entries = [
        {
            "episode_id": result.episode_id,
            "source_id": result.source_id,
            "task_id": result.task_id,
            "duration_seconds": result.duration_seconds,
            "motion": result.motion,
            "fingerprint": result.fingerprint,
            "content_hash": result.content_hash,
            "schema_fingerprint": result.schema_fingerprint,
            "sidecar": f"episodes/{result.fingerprint}.json",
            "sidecar_hash": _canonical_hash(result.model_dump(mode="json")),
            "lineage_hash": _canonical_hash(result.lineage),
            "reason_codes": sorted({finding.code for finding in result.findings}),
            "disposition": result.disposition.model_dump(mode="json"),
            "qwen_core_score": result.qwen_core_score,
            "extension_score": result.extension_score,
            "coverage": result.coverage,
            "qwen_core_coverage": result.qwen_core_coverage,
            "extension_coverage": result.extension_coverage,
            "quality_contract_version": result.quality_contract_version,
            "selected_checks": result.selected_checks,
            "selected_denominator": result.selected_denominator,
            "evaluated_count": result.evaluated_count,
            "final_decision": result.final_decision,
        }
        for result in results
    ]
    stable_manifest = {
        "schema": "robot-data-studio.curation-manifest.v1",
        "profile_hash": profile_hash,
        "episodes": episode_entries,
        "coverage": coverage.model_dump(mode="json"),
        "reason_counts": reason_counts,
        "sampler_advice": sampler_advice,
    }
    manifest_hash = _canonical_hash(stable_manifest)
    _write_json(root / "manifest.json", {**stable_manifest, "manifest_hash": manifest_hash})
    return PilotResult(
        processed=processed,
        resumed=resumed,
        profile_hash=profile_hash,
        manifest_hash=manifest_hash,
        episodes=results,
        coverage=coverage,
        reason_counts=reason_counts,
        sampler_advice=sampler_advice,
    )

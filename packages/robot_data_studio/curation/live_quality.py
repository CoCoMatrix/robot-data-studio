from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import threading
import time
from collections.abc import Callable, Sequence
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from robot_data_studio.quality.contracts import (
    QWEN_CORE_DEFAULT_WEIGHTS,
    QWEN_CORE_LABELS,
    FrameInterval,
    MetricEvidence,
    QualityScorecard,
    ScorecardDisposition,
    build_quality_scorecard,
)
from robot_data_studio.quality.filter_models import FilterVisualQualityConfig
from robot_data_studio.quality.visual_quality import (
    aggregate_visual_quality_incidents,
    analyze_sampled_frames,
)

from .c2_projection import evaluate_c2_keyframes, project_eef_support_mask
from .sam3 import Sam3Provider, Sam3Result


_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_C1_SCHEMA_VERSION = "robot-data-studio.c1-qwen-structured.v1"


class C1AxisEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: Literal["consistent", "review", "conflict", "not_visible"]
    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1, max_length=320)
    evidence_positions: list[int] = Field(default_factory=list, max_length=5)

    @model_validator(mode="after")
    def evidence_matches_state(self) -> C1AxisEvidence:
        if self.state == "not_visible" and self.evidence_positions:
            raise ValueError("not_visible axis cannot claim evidence positions")
        if self.state != "not_visible" and not self.evidence_positions:
            raise ValueError("visible axis requires at least one evidence position")
        if len(set(self.evidence_positions)) != len(self.evidence_positions):
            raise ValueError("evidence positions must be unique")
        return self


class C1StructuredEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instruction_video: C1AxisEvidence
    object: C1AxisEvidence
    action_phase: C1AxisEvidence
    success: C1AxisEvidence
    overall_state: Literal["consistent", "review", "conflict"]
    overall_score: float = Field(ge=0.0, le=1.0)
    overall_confidence: float = Field(ge=0.0, le=1.0)
    summary: str = Field(min_length=1, max_length=480)


class LiveBudget:
    def __init__(self, *, max_calls: int, max_evidence_frames: int) -> None:
        if max_calls < 1 or max_evidence_frames < 1:
            raise ValueError("live budget limits must be positive")
        self.max_calls = int(max_calls)
        self.max_evidence_frames = int(max_evidence_frames)
        self.calls = 0
        self.evidence_frames = 0
        self._lock = threading.Lock()

    def reserve(self, frame_count: int) -> None:
        with self._lock:
            if frame_count < 1:
                raise ValueError("C1 requires at least one evidence frame")
            if self.calls + 1 > self.max_calls:
                raise ValueError("live C1 call budget exceeded")
            if self.evidence_frames + frame_count > self.max_evidence_frames:
                raise ValueError("live C1 evidence-frame budget exceeded")
            self.calls += 1
            self.evidence_frames += frame_count


RequestFn = Callable[[dict[str, Any]], dict[str, Any]]


def _dashscope_http_client() -> Any:
    import httpx

    return httpx.Client(trust_env=False, timeout=90.0)


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _uint8_rgb(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image)
    if values.ndim == 3 and values.shape[0] == 3 and values.shape[-1] != 3:
        values = np.moveaxis(values, 0, -1)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("evidence image must be HxWx3 or 3xHxW")
    if np.issubdtype(values.dtype, np.floating):
        maximum = float(np.nanmax(values)) if values.size else 0.0
        values = values * 255.0 if maximum <= 1.0 + 1e-6 else values
    if not np.all(np.isfinite(values)):
        raise ValueError("evidence image contains non-finite values")
    return np.clip(values, 0, 255).astype(np.uint8)


def _jpeg_data_url(image: np.ndarray, *, max_width: int = 640) -> tuple[str, str]:
    rgb = _uint8_rgb(image)
    pil = Image.fromarray(rgb, mode="RGB")
    if pil.width > max_width:
        height = max(1, round(pil.height * max_width / pil.width))
        pil = pil.resize((max_width, height), Image.Resampling.BILINEAR)
    stream = io.BytesIO()
    pil.save(stream, format="JPEG", quality=82, optimize=True)
    payload = stream.getvalue()
    return (
        "data:image/jpeg;base64," + base64.b64encode(payload).decode("ascii"),
        hashlib.sha256(payload).hexdigest(),
    )


class DashScopeC1Provider:
    def __init__(
        self,
        *,
        cache_root: str | Path,
        budget: LiveBudget,
        model: str = "qwen3.7-plus",
        request_fn: RequestFn | None = None,
    ) -> None:
        self.cache_root = Path(cache_root)
        self.budget = budget
        self.model = model
        self.request_fn = request_fn or self._request_dashscope

    def _request_dashscope(self, payload: dict[str, Any]) -> dict[str, Any]:
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is unset")
        from openai import OpenAI

        with _dashscope_http_client() as http_client:
            response = OpenAI(
                api_key=api_key,
                base_url=_DASHSCOPE_BASE_URL,
                http_client=http_client,
                max_retries=0,
            ).chat.completions.create(**payload)
        usage = response.usage
        return {
            "content": response.choices[0].message.content,
            "model": response.model,
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
        }

    @staticmethod
    def _prompt(task_text: str, frame_indexes: Sequence[int], timestamps: Sequence[float]) -> str:
        evidence = [
            {"position": position, "frame_index": int(frame), "timestamp_seconds": float(time)}
            for position, (frame, time) in enumerate(zip(frame_indexes, timestamps, strict=True))
        ]
        return (
            "You are auditing one train episode from sparse ordered keyframes. "
            "Check instruction-video consistency, required objects, action phase order, and visible success. "
            "Do not infer hidden events. Use not_visible when sparse evidence is insufficient. "
            "evidence_positions must reference only the supplied position integers. "
            f"Instruction: {task_text}\nEvidence map: {json.dumps(evidence, separators=(',', ':'))}"
        )

    def evaluate(
        self,
        *,
        source_video_sha256: str,
        episode_id: str,
        task_text: str,
        images: Sequence[np.ndarray],
        frame_indexes: Sequence[int],
        timestamps: Sequence[float],
    ) -> MetricEvidence:
        if not (len(images) == len(frame_indexes) == len(timestamps)):
            raise ValueError("C1 images/frame indexes/timestamps must have equal lengths")
        encoded = [_jpeg_data_url(image) for image in images]
        image_hashes = [item[1] for item in encoded]
        key_inputs = {
            "schema": _C1_SCHEMA_VERSION,
            "source_video_sha256": source_video_sha256,
            "episode_id": str(episode_id),
            "task_text": task_text,
            "frame_indexes": [int(value) for value in frame_indexes],
            "timestamps": [float(value) for value in timestamps],
            "image_sha256": image_hashes,
            "model": self.model,
            "enable_thinking": False,
        }
        key = _canonical_hash(key_inputs)
        cache_path = self.cache_root / key[:2] / f"{key}.json"
        from_cache = cache_path.is_file()
        try:
            if from_cache:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                structured = C1StructuredEvidence.model_validate(cached["structured"])
                usage = cached.get("usage", {})
                response_model = cached.get("model", self.model)
            else:
                self.budget.reserve(len(images))
                content: list[dict[str, Any]] = [
                    {"type": "text", "text": self._prompt(task_text, frame_indexes, timestamps)}
                ]
                content.extend(
                    {"type": "image_url", "image_url": {"url": data_url}}
                    for data_url, _ in encoded
                )
                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": content}],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "c1_embodied_episode_evidence",
                            "strict": True,
                            "schema": C1StructuredEvidence.model_json_schema(),
                        },
                    },
                    "temperature": 0,
                    "max_tokens": 900,
                    "extra_body": {"enable_thinking": False},
                }
                response = self.request_fn(payload)
                structured = C1StructuredEvidence.model_validate_json(response["content"])
                usage = {
                    "prompt_tokens": response.get("prompt_tokens"),
                    "completion_tokens": response.get("completion_tokens"),
                }
                response_model = response.get("model", self.model)
                cache_payload = {
                    "schema": _C1_SCHEMA_VERSION,
                    "cache_key": key,
                    "inputs": key_inputs,
                    "model": response_model,
                    "usage": usage,
                    "structured": structured.model_dump(mode="json"),
                }
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache_path.with_suffix(".tmp")
                temporary.write_text(
                    json.dumps(cache_payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                temporary.replace(cache_path)
            axes: dict[str, Any] = {}
            for name in ("instruction_video", "object", "action_phase", "success"):
                axis = getattr(structured, name)
                if any(position < 0 or position >= len(images) for position in axis.evidence_positions):
                    raise ValueError(f"C1 {name} returned out-of-range evidence position")
                axes[name] = {
                    **axis.model_dump(mode="json"),
                    "evidence_frames": [int(frame_indexes[pos]) for pos in axis.evidence_positions],
                    "evidence_timestamps": [float(timestamps[pos]) for pos in axis.evidence_positions],
                }
            return MetricEvidence(
                metric_id="C1",
                label=QWEN_CORE_LABELS["C1"],
                group="qwen_core",
                status="evaluated",
                score=structured.overall_score,
                weight=QWEN_CORE_DEFAULT_WEIGHTS["C1"],
                reason="real DashScope structured instruction/video evidence",
                evidence={
                    "provider": "dashscope",
                    "provider_mode": "real",
                    "model": response_model,
                    "schema": _C1_SCHEMA_VERSION,
                    "cache_key": key,
                    "from_cache": from_cache,
                    "overall_state": structured.overall_state,
                    "overall_confidence": structured.overall_confidence,
                    "summary": structured.summary,
                    "usage": usage,
                    "axes": axes,
                },
            )
        except Exception as error:
            return MetricEvidence(
                metric_id="C1",
                label=QWEN_CORE_LABELS["C1"],
                group="qwen_core",
                status="not_evaluated",
                weight=QWEN_CORE_DEFAULT_WEIGHTS["C1"],
                reason=f"real DashScope C1 failed closed: {type(error).__name__}",
                evidence={
                    "provider": "dashscope",
                    "provider_mode": "real_failed_closed",
                    "model": self.model,
                    "schema": _C1_SCHEMA_VERSION,
                    "cache_key": key,
                    "from_cache": from_cache,
                },
            )


def evaluate_c1_concurrently(
    provider: DashScopeC1Provider,
    requests: Sequence[dict[str, Any]],
    *,
    max_workers: int,
) -> list[MetricEvidence]:
    if max_workers < 1 or max_workers > 8:
        raise ValueError("C1 concurrency must be between one and eight")
    if not requests:
        return []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(requests))) as executor:
        futures = [executor.submit(provider.evaluate, **request) for request in requests]
        return [future.result() for future in futures]


def evaluate_c3_sampled_frames(
    images: Sequence[np.ndarray],
    *,
    frame_indexes: Sequence[int],
    timestamps: Sequence[float],
    has_motion: bool,
    config: FilterVisualQualityConfig | None = None,
) -> MetricEvidence:
    if not (len(images) == len(frame_indexes) == len(timestamps)) or not images:
        raise ValueError("C3 requires non-empty equal-length frames/indexes/timestamps")
    config = config or FilterVisualQualityConfig(max_frames_per_video=len(images))
    resized = [
        np.asarray(
            Image.fromarray(_uint8_rgb(image), mode="RGB").resize(
                (config.sample_width, config.sample_height), Image.Resampling.BILINEAR
            )
        )
        for image in images
    ]
    result = analyze_sampled_frames(
        resized,
        config,
        has_motion=has_motion,
        camera="camera0",
        timestamps=[float(value) for value in timestamps],
        frame_indexes=[int(value) for value in frame_indexes],
    )
    deltas = np.diff(np.asarray(timestamps, dtype=np.float64))
    interval = float(np.median(deltas)) if len(deltas) else 1.0 / config.sample_fps
    incidents = aggregate_visual_quality_incidents(
        result.rows,
        sample_interval_seconds=max(interval, 1e-6),
    )
    return MetricEvidence(
        metric_id="C3",
        label=QWEN_CORE_LABELS["C3"],
        group="qwen_core",
        status="evaluated",
        score=result.score,
        weight=QWEN_CORE_DEFAULT_WEIGHTS["C3"],
        reason="existing deterministic visual-quality scorer on official-loader decoded frames",
        abnormal_intervals=[
            FrameInterval(
                start_frame=incident.start_frame,
                end_frame=incident.end_frame,
                reason=incident.issue,
            )
            for incident in incidents
        ],
        evidence={
            "provider": "existing_visual_quality",
            "provider_mode": "deterministic_real_frames",
            "sample_count": result.frame_count,
            "sampled_frame_indexes": [int(value) for value in frame_indexes],
            "sampled_timestamps": [float(value) for value in timestamps],
            "issue_count": result.issue_count,
            "issue_counts": result.issue_counts,
            "critical_issue": result.critical_issue,
            "config": config.model_dump(mode="json"),
        },
    )


def merge_live_core_evidence(
    metrics: Sequence[MetricEvidence],
    *,
    replacements: Sequence[MetricEvidence],
    disposition: ScorecardDisposition | None = None,
) -> QualityScorecard:
    def supersede_support_disc(metric: MetricEvidence) -> MetricEvidence:
        scope = str(metric.evidence.get("projection_scope", "")).casefold()
        if metric.metric_id != "C2" or "support disc" not in scope:
            return metric
        return metric.model_copy(
            update={
                "metric_id": "historical_support_disc_projection",
                "label": "Historical EEF support-disc projection",
                "group": "robot_extension",
                "fidelity": "historical",
                "weight": 0.0,
                "reason": "historical provisional support-disc evidence; not current Robot C2",
            }
        )

    metrics = [supersede_support_disc(metric) for metric in metrics]
    replacements = [supersede_support_disc(metric) for metric in replacements]
    replacement_by_id = {metric.metric_id: metric for metric in replacements}
    merged = [replacement_by_id.get(metric.metric_id, metric) for metric in metrics]
    known = {metric.metric_id for metric in merged}
    merged.extend(metric for metric in replacements if metric.metric_id not in known)
    known = {metric.metric_id for metric in merged}
    if "C2" not in known:
        merged.append(
            MetricEvidence(
                metric_id="C2",
                label=QWEN_CORE_LABELS["C2"],
                group="qwen_core",
                status="not_evaluated",
                weight=QWEN_CORE_DEFAULT_WEIGHTS["C2"],
                fidelity="provisional",
                reason="current C2 full-URDF-visual evidence was not run",
                evidence={"projection_scope": "full URDF visual silhouette"},
            )
        )
    return build_quality_scorecard(merged, disposition=disposition)


def uniform_episode_indices(start: int, stop: int, count: int) -> list[int]:
    if start < 0 or stop <= start or count < 1:
        raise ValueError("invalid bounded episode index range")
    length = stop - start
    if length == 1:
        return [start]
    values = np.linspace(start, stop - 1, min(count, length))
    return list(dict.fromkeys(int(round(value)) for value in values))


def _scalar(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _write_json_once(path: Path, payload: object) -> str:
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"refusing to overwrite prior live evidence: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(path)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _c2_fail_closed(sam3_evidence: dict[str, Any]) -> MetricEvidence:
    return MetricEvidence(
        metric_id="C2",
        label=QWEN_CORE_LABELS["C2"],
        group="qwen_core",
        status="not_evaluated",
        weight=QWEN_CORE_DEFAULT_WEIGHTS["C2"],
        reason="SAM3 checkpoint/inference did not complete; projection IoU was not computed",
        evidence={
            "provider": "facebook/sam3",
            "provider_mode": "real_failed_closed",
            "mock_counted_as_real": False,
            **sam3_evidence,
        },
    )


def _s4_s5_fail_closed(embodiment: str) -> list[MetricEvidence]:
    if embodiment == "human":
        s4_reason = "Robot FK consistency is not applicable to Human tracks"
        s5_reason = "Robot base/EEF orientation alignment is not applicable to Human tracks"
    else:
        s4_reason = (
            "GR3 FK authority exists, but no independent logged EEF is available; comparing "
            "against the published FK-derived pose would be tautological"
        )
        s5_reason = (
            "base/Camera0/EEF transforms exist, but no reviewed expected orientation target "
            "and threshold authority exists"
        )
    return [
        MetricEvidence(
            metric_id="S4",
            label=QWEN_CORE_LABELS["S4"],
            group="qwen_core",
            status="not_evaluated",
            weight=QWEN_CORE_DEFAULT_WEIGHTS["S4"],
            reason=s4_reason,
            evidence={"independent_reference_available": False},
        ),
        MetricEvidence(
            metric_id="S5",
            label=QWEN_CORE_LABELS["S5"],
            group="qwen_core",
            status="not_evaluated",
            weight=QWEN_CORE_DEFAULT_WEIGHTS["S5"],
            reason=s5_reason,
            evidence={"reviewed_expected_orientation_available": False},
        ),
    ]


def run_live_quality_dataset(
    dataset: Any,
    *,
    prior_root: str | Path,
    output_root: str | Path,
    source_episode_ids: Sequence[str],
    max_episodes: int,
    c1_provider: DashScopeC1Provider,
    source_video_sha256: str,
    sam3_evidence: dict[str, Any],
    c3_samples: int = 16,
    c1_samples: int = 5,
    embodiment: Literal["human", "robot"] = "robot",
) -> dict[str, Any]:
    if max_episodes < 1 or len(source_episode_ids) > max_episodes:
        raise ValueError("live quality requires a positive bound covering explicit episodes")
    if not source_episode_ids or len(source_episode_ids) > 5:
        raise ValueError("live quality accepts one to five explicit episodes")
    if c1_samples < 1 or c1_samples > 5 or c3_samples < 1 or c3_samples > 48:
        raise ValueError("live quality sparse sample bounds exceeded")
    prior_root = Path(prior_root)
    output_root = Path(output_root)
    manifest_path = output_root / "manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {**existing, "resumed": True}

    prior_manifest = json.loads((prior_root / "manifest.json").read_text(encoding="utf-8"))
    prior_entries = prior_manifest.get("episodes", [])
    if [str(item.get("episode_id")) for item in prior_entries] != [
        str(value) for value in source_episode_ids
    ]:
        raise ValueError("explicit live episode ids do not exactly match prior bounded manifest order")
    episode_rows = dataset.meta.episodes.to_pandas().to_dict(orient="records")
    if len(episode_rows) != len(source_episode_ids):
        raise ValueError("official dataset/prior episode count mismatch")

    results: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    for source_episode_id, metadata, prior_entry in zip(
        source_episode_ids, episode_rows, prior_entries, strict=True
    ):
        start = int(metadata["dataset_from_index"])
        stop = int(metadata["dataset_to_index"])
        c1_global = uniform_episode_indices(start, stop, c1_samples)
        c3_global = uniform_episode_indices(start, stop, c3_samples)
        decoded: dict[int, np.ndarray] = {}
        raw: dict[int, dict[str, Any]] = {}
        for global_index in sorted(set(c1_global) | set(c3_global)):
            raw[global_index] = dataset.get_raw_item(global_index)
            image = dataset[global_index]["observation.images.camera0"]
            if hasattr(image, "detach"):
                image = image.detach().cpu().numpy()
            decoded[global_index] = _uint8_rgb(np.asarray(image))

        def source_frame(global_index: int) -> int:
            return int(round(_scalar(raw[global_index]["source.frame_index"])))

        def source_time(global_index: int) -> float:
            return _scalar(raw[global_index]["source.timestamp"])

        tasks = metadata.get("tasks", [])
        task_text = str(tasks[0]) if len(tasks) else ""
        c1 = c1_provider.evaluate(
            source_video_sha256=source_video_sha256,
            episode_id=str(source_episode_id),
            task_text=task_text,
            images=[decoded[index] for index in c1_global],
            frame_indexes=[source_frame(index) for index in c1_global],
            timestamps=[source_time(index) for index in c1_global],
        )
        prior_path = prior_root / str(prior_entry["sidecar"])
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        c3 = evaluate_c3_sampled_frames(
            [decoded[index] for index in c3_global],
            frame_indexes=[source_frame(index) for index in c3_global],
            timestamps=[source_time(index) for index in c3_global],
            has_motion=float(prior.get("motion", 0.0)) > 1e-6,
        )
        disposition_data = prior.get("disposition", {})
        disposition = ScorecardDisposition(
            hard_reject=bool(disposition_data.get("hard_reject", False)),
            review=bool(disposition_data.get("review", False)),
            window_masks=[
                FrameInterval(
                    start_frame=int(item["start_frame"]),
                    end_frame=int(item["end_frame"]),
                    reason=str(item.get("reason", "")),
                )
                for item in disposition_data.get("window_masks", [])
            ],
            soft_weight=float(disposition_data.get("soft_weight", 1.0)),
            reasons=[
                *[str(value) for value in disposition_data.get("hard_reasons", [])],
                *[str(value) for value in disposition_data.get("review_reasons", [])],
            ],
        )
        previous_metrics = [
            MetricEvidence.model_validate(item) for item in prior.get("metric_evidence", [])
        ]
        scorecard = merge_live_core_evidence(
            previous_metrics,
            replacements=[c1, _c2_fail_closed(sam3_evidence), c3, *_s4_s5_fail_closed(embodiment)],
            disposition=disposition,
        )
        updated = {
            **prior,
            "qwen_core_score": scorecard.qwen_core_score,
            "extension_score": scorecard.extension_score,
            "coverage": scorecard.coverage,
            "qwen_core_coverage": scorecard.qwen_core_coverage,
            "extension_coverage": scorecard.extension_coverage,
            "quality_contract_version": scorecard.contract_version,
            "selected_checks": scorecard.selected_checks,
            "selected_denominator": scorecard.selected_denominator,
            "evaluated_count": scorecard.evaluated_count,
            "final_decision": scorecard.final_decision,
            "metric_evidence": [metric.model_dump(mode="json") for metric in scorecard.metrics],
            "evidence_state": (
                c1.evidence.get("overall_state", prior.get("evidence_state", "review"))
                if c1.status == "evaluated"
                else "review"
            ),
            "live_quality": {
                "schema": "robot-data-studio.live-quality.v1",
                "official_loader_version": str(
                    getattr(dataset.meta, "_version", getattr(dataset.meta, "codebase_version", "unknown"))
                ),
                "source_video_sha256": source_video_sha256,
                "c1_sample_count": len(c1_global),
                "c3_sample_count": len(c3_global),
                "sam3": sam3_evidence,
                "prior_sidecar_sha256": hashlib.sha256(prior_path.read_bytes()).hexdigest(),
            },
        }
        safe_episode = str(source_episode_id).replace("/", "_")
        episode_path = output_root / "episodes" / f"{safe_episode}.json"
        episode_hash = _write_json_once(episode_path, updated)
        for metric in scorecard.metrics:
            if metric.group == "qwen_core":
                status_counts[f"{metric.metric_id}:{metric.status}"] += 1
        results.append(
            {
                "episode_id": str(source_episode_id),
                "sidecar": str(episode_path.relative_to(output_root)),
                "sidecar_sha256": episode_hash,
                "qwen_core_score": scorecard.qwen_core_score,
                "coverage": scorecard.coverage,
                "final_decision": scorecard.final_decision,
            }
        )

    report = {
        "schema": "robot-data-studio.live-quality-manifest.v1",
        "processed_episode_count": len(results),
        "source_episode_ids": [str(value) for value in source_episode_ids],
        "official_loader_reload": "verified",
        "c1_provider": "dashscope/qwen3.7-plus",
        "c1_calls": c1_provider.budget.calls,
        "c1_evidence_frames": c1_provider.budget.evidence_frames,
        "c2_status": "not_evaluated",
        "sam3": sam3_evidence,
        "status_counts": dict(sorted(status_counts.items())),
        "episodes": results,
        "resumed": False,
    }
    _write_json_once(manifest_path, report)
    return report


def _disposition_from_prior(prior: dict[str, Any]) -> ScorecardDisposition:
    data = prior.get("disposition", {})
    return ScorecardDisposition(
        hard_reject=bool(data.get("hard_reject", False)),
        review=bool(data.get("review", False)),
        window_masks=[
            FrameInterval(
                start_frame=int(item["start_frame"]),
                end_frame=int(item["end_frame"]),
                reason=str(item.get("reason", "")),
            )
            for item in data.get("window_masks", [])
        ],
        soft_weight=float(data.get("soft_weight", 1.0)),
        reasons=[
            *[str(value) for value in data.get("hard_reasons", [])],
            *[str(value) for value in data.get("review_reasons", [])],
        ],
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


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def run_fast_live_quality_dataset(
    dataset: Any,
    *,
    prior_root: str | Path,
    output_root: str | Path,
    source_episode_ids: Sequence[str],
    max_episodes: int,
    c1_provider: DashScopeC1Provider,
    source_video_sha256: str,
    sam3_entity_provider: Sam3Provider,
    sam3_object_provider: Sam3Provider,
    sam3_checkpoint_evidence: dict[str, Any],
    intrinsic: Sequence[Sequence[float]],
    embodiment: Literal["human", "robot"],
    c1_concurrency: int = 2,
    c1_samples: int = 5,
    c3_samples: int = 16,
    sam3_samples: int = 3,
    support_radius_pixels: int = 24,
) -> dict[str, Any]:
    """Bounded research fast path over a previously promoted official V3 shard."""

    if not source_episode_ids or len(source_episode_ids) > max_episodes or len(source_episode_ids) > 5:
        raise ValueError("fast live quality requires one to five explicit bounded episodes")
    if not 1 <= c1_samples <= 5 or not 1 <= c3_samples <= 48 or not 1 <= sam3_samples <= 8:
        raise ValueError("fast live quality sparse sample bounds exceeded")
    if not 1 <= c1_concurrency <= 8:
        raise ValueError("C1 concurrency must be between one and eight")
    if sam3_entity_provider.config.backend != "local" or sam3_object_provider.config.backend != "local":
        raise ValueError("fast C2 accepts only real local SAM3 providers")
    intrinsic_array = np.asarray(intrinsic, dtype=np.float64)
    if intrinsic_array.shape != (3, 3):
        raise ValueError("stored-image intrinsic must be 3x3")
    prior_root = Path(prior_root)
    output_root = Path(output_root)
    manifest_path = output_root / "manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {**existing, "resumed": True}

    started = time.perf_counter()
    timings: Counter[str] = Counter()
    prior_manifest = json.loads((prior_root / "manifest.json").read_text(encoding="utf-8"))
    prior_entries = prior_manifest.get("episodes", [])
    if [str(item.get("episode_id")) for item in prior_entries] != [str(x) for x in source_episode_ids]:
        raise ValueError("explicit fast-path episodes differ from prior bounded manifest")
    episode_rows = dataset.meta.episodes.to_pandas().to_dict(orient="records")
    if len(episode_rows) != len(source_episode_ids):
        raise ValueError("official shard episode count differs from explicit selection")

    contexts: list[dict[str, Any]] = []
    video_seconds = 0.0
    for source_episode_id, metadata, prior_entry in zip(
        source_episode_ids, episode_rows, prior_entries, strict=True
    ):
        rule_started = time.perf_counter()
        prior_path = prior_root / str(prior_entry["sidecar"])
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        disposition = _disposition_from_prior(prior)
        eligible = not disposition.hard_reject
        timings["rule_checks_seconds"] += time.perf_counter() - rule_started
        context = {
            "source_episode_id": str(source_episode_id),
            "metadata": metadata,
            "prior": prior,
            "prior_path": prior_path,
            "disposition": disposition,
            "eligible": eligible,
        }
        if not eligible:
            contexts.append(context)
            continue
        start = int(metadata["dataset_from_index"])
        stop = int(metadata["dataset_to_index"])
        c1_global = uniform_episode_indices(start, stop, c1_samples)
        c3_global = uniform_episode_indices(start, stop, c3_samples)
        sam3_global = uniform_episode_indices(start, stop, sam3_samples)
        decode_started = time.perf_counter()
        decoded: dict[int, np.ndarray] = {}
        raw: dict[int, dict[str, Any]] = {}
        for global_index in sorted(set(c1_global) | set(c3_global) | set(sam3_global)):
            raw[global_index] = dataset.get_raw_item(global_index)
            decoded[global_index] = _uint8_rgb(
                _array(dataset[global_index]["observation.images.camera0"])
            )
        timings["video_decode_seconds"] += time.perf_counter() - decode_started

        def source_frame(index: int, rows: dict[int, dict[str, Any]] = raw) -> int:
            return int(round(_scalar(rows[index]["source.frame_index"])))

        def source_time(index: int, rows: dict[int, dict[str, Any]] = raw) -> float:
            return _scalar(rows[index]["source.timestamp"])

        boundary = sorted(set(c1_global) | set(c3_global) | set(sam3_global))
        video_seconds += max(0.0, source_time(boundary[-1]) - source_time(boundary[0]))
        tasks = metadata.get("tasks", [])
        context.update(
            {
                "c1_global": c1_global,
                "c3_global": c3_global,
                "sam3_global": sam3_global,
                "decoded": decoded,
                "raw": raw,
                "source_frame": source_frame,
                "source_time": source_time,
                "task_text": str(tasks[0]) if len(tasks) else "",
            }
        )
        contexts.append(context)

    eligible = [context for context in contexts if context["eligible"]]
    c1_requests = [
        {
            "source_video_sha256": source_video_sha256,
            "episode_id": context["source_episode_id"],
            "task_text": context["task_text"],
            "images": [context["decoded"][index] for index in context["c1_global"]],
            "frame_indexes": [context["source_frame"](index) for index in context["c1_global"]],
            "timestamps": [context["source_time"](index) for index in context["c1_global"]],
        }
        for context in eligible
    ]
    qwen_started = time.perf_counter()
    c1_results = evaluate_c1_concurrently(
        c1_provider,
        c1_requests,
        max_workers=c1_concurrency,
    )
    timings["c1_qwen_seconds"] = time.perf_counter() - qwen_started
    for context, result in zip(eligible, c1_results, strict=True):
        context["c1"] = result

    flattened_images = [
        context["decoded"][index]
        for context in eligible
        for index in context["sam3_global"]
    ]
    flattened_global = [
        index for context in eligible for index in context["sam3_global"]
    ]
    sam_started = time.perf_counter()
    def sam_batch(provider: Sam3Provider, label: str) -> list[Sam3Result]:
        if not flattened_images:
            return []
        try:
            return provider.segment_batch(
                source_video_sha256=source_video_sha256,
                frame_indexes=flattened_global,
                images=flattened_images,
            )
        except Exception as error:
            return [
                Sam3Result(
                    status="not_evaluated",
                    reason=f"local SAM3 {label} batch failed closed: {type(error).__name__}",
                    cache_key=_canonical_hash(
                        {
                            "source_video_sha256": source_video_sha256,
                            "global_frame_index": index,
                            "model": provider.config.model_version,
                            "prompt": provider.config.prompt,
                        }
                    ),
                )
                for index in flattened_global
            ]

    entity_results = sam_batch(sam3_entity_provider, "entity")
    object_results = sam_batch(sam3_object_provider, "object")
    timings["sam3_seconds"] = time.perf_counter() - sam_started
    sam3_checkpoint_evidence = {
        **sam3_checkpoint_evidence,
        "checkpoint_loaded": bool(entity_results)
        and all(result.status == "evaluated" for result in entity_results),
    }
    if sam3_checkpoint_evidence["checkpoint_loaded"]:
        sam3_checkpoint_evidence["status"] = "evaluated"
        sam3_checkpoint_evidence["reason"] = "official local SAM3 checkpoint loaded and produced masks"
    elif entity_results:
        sam3_checkpoint_evidence["status"] = "not_evaluated"
        sam3_checkpoint_evidence["reason"] = entity_results[0].reason
    cursor = 0
    for context in eligible:
        count = len(context["sam3_global"])
        context["sam3_entity"] = entity_results[cursor : cursor + count]
        context["sam3_object"] = object_results[cursor : cursor + count]
        cursor += count

    results: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    sam_cache_hits = 0
    sam_evaluated = 0
    c1_cache_hits = 0
    for context in contexts:
        prior = context["prior"]
        if context["eligible"]:
            c1 = context["c1"]
            c1_cache_hits += int(bool(c1.evidence.get("from_cache")))
            c3_started = time.perf_counter()
            c3 = evaluate_c3_sampled_frames(
                [context["decoded"][index] for index in context["c3_global"]],
                frame_indexes=[context["source_frame"](index) for index in context["c3_global"]],
                timestamps=[context["source_time"](index) for index in context["c3_global"]],
                has_motion=float(prior.get("motion", 0.0)) > 1e-6,
            )
            timings["c3_seconds"] += time.perf_counter() - c3_started
            projected_masks = []
            projection_rows = []
            for index in context["sam3_global"]:
                row = context["raw"][index]
                image = context["decoded"][index]
                projected, projection = project_eef_support_mask(
                    eef_pose_absolute=_array(row["observation.eef_pose_absolute"]),
                    eef_validity=_array(row["observation.eef_pose_absolute.validity"]),
                    camera_pose_absolute=_array(row["observation.camera0_pose_absolute"]),
                    camera_valid=bool(_scalar(row["observation.camera0_pose_absolute.validity"])),
                    intrinsic=intrinsic_array,
                    image_shape=image.shape[:2],
                    support_radius_pixels=support_radius_pixels,
                )
                projected_masks.append(projected)
                projection_rows.append(projection)
            c2 = evaluate_c2_keyframes(
                projected_masks=projected_masks,
                projection_evidence=projection_rows,
                sam_results=context["sam3_entity"],
                frame_indexes=[context["source_frame"](index) for index in context["sam3_global"]],
                projection_kind=(
                    "robot_urdf_fk_eef_support" if embodiment == "robot" else "human_measured_hand_support"
                ),
            )
            object_evidence = []
            for index, item in zip(context["sam3_global"], context["sam3_object"], strict=True):
                object_evidence.append(
                    {
                        "frame_index": context["source_frame"](index),
                        "status": item.status,
                        "confidence": item.confidence,
                        "cache_hit": item.cache_hit,
                        "cache_key": item.cache_key,
                        "mask_pixels": int(np.count_nonzero(item.mask)) if item.mask is not None else None,
                    }
                )
            c2 = c2.model_copy(
                update={
                    "evidence": {
                        **c2.evidence,
                        "checkpoint_loaded": sam3_checkpoint_evidence.get("checkpoint_loaded", False),
                        "mock_counted_as_real": False,
                        "object_prompt": sam3_object_provider.config.prompt,
                        "object_masks": object_evidence,
                    }
                }
            )
            for item in [*context["sam3_entity"], *context["sam3_object"]]:
                sam_cache_hits += int(item.cache_hit)
                sam_evaluated += int(item.status == "evaluated")
        else:
            reason = "cheap hard gate rejected the episode before video/model checks"
            c1, c2, c3 = (_core_na(metric_id, reason) for metric_id in ("C1", "C2", "C3"))
        previous = [MetricEvidence.model_validate(item) for item in prior.get("metric_evidence", [])]
        scorecard = merge_live_core_evidence(
            previous,
            replacements=[c1, c2, c3, *_s4_s5_fail_closed(embodiment)],
            disposition=context["disposition"],
        )
        updated = {
            **prior,
            "qwen_core_score": scorecard.qwen_core_score,
            "extension_score": scorecard.extension_score,
            "coverage": scorecard.coverage,
            "qwen_core_coverage": scorecard.qwen_core_coverage,
            "extension_coverage": scorecard.extension_coverage,
            "quality_contract_version": scorecard.contract_version,
            "selected_checks": scorecard.selected_checks,
            "selected_denominator": scorecard.selected_denominator,
            "evaluated_count": scorecard.evaluated_count,
            "final_decision": scorecard.final_decision,
            "metric_evidence": [metric.model_dump(mode="json") for metric in scorecard.metrics],
            "evidence_state": (
                c1.evidence.get("overall_state", "review") if c1.status == "evaluated" else "review"
            ),
            "live_quality_fast": {
                "schema": "robot-data-studio.live-quality-fast.v1",
                "cheap_gate_eligible": context["eligible"],
                "source_video_sha256": source_video_sha256,
                "sam3_checkpoint": sam3_checkpoint_evidence,
                "c1_concurrency": c1_concurrency,
                "c1_sample_count": len(context.get("c1_global", [])),
                "c3_sample_count": len(context.get("c3_global", [])),
                "sam3_sample_count": len(context.get("sam3_global", [])),
                "prior_sidecar_sha256": hashlib.sha256(context["prior_path"].read_bytes()).hexdigest(),
            },
        }
        write_started = time.perf_counter()
        episode_path = output_root / "episodes" / f"{context['source_episode_id'].replace('/', '_')}.json"
        episode_hash = _write_json_once(episode_path, updated)
        timings["sidecar_write_seconds"] += time.perf_counter() - write_started
        for metric in scorecard.metrics:
            if metric.group == "qwen_core":
                status_counts[f"{metric.metric_id}:{metric.status}"] += 1
        results.append(
            {
                "episode_id": context["source_episode_id"],
                "sidecar": str(episode_path.relative_to(output_root)),
                "sidecar_sha256": episode_hash,
                "qwen_core_score": scorecard.qwen_core_score,
                "coverage": scorecard.coverage,
                "final_decision": scorecard.final_decision,
            }
        )

    total_seconds = time.perf_counter() - started
    report = {
        "schema": "robot-data-studio.live-quality-fast-manifest.v1",
        "processed_episode_count": len(results),
        "source_episode_ids": [str(value) for value in source_episode_ids],
        "official_loader_reload": "verified",
        "cheap_gate_candidate_count": len(eligible),
        "c1_provider": f"dashscope/{c1_provider.model}",
        "c1_api_calls": c1_provider.budget.calls,
        "c1_cache_hits": c1_cache_hits,
        "sam3": sam3_checkpoint_evidence,
        "sam3_evaluated_masks": sam_evaluated,
        "sam3_cache_hits": sam_cache_hits,
        "timing": {
            **{key: round(value, 6) for key, value in sorted(timings.items())},
            "total_seconds": round(total_seconds, 6),
            "video_seconds": round(video_seconds, 6),
            "episodes_per_hour": len(results) * 3600.0 / max(total_seconds, 1e-9),
            "video_hours_per_hour": video_seconds / max(total_seconds, 1e-9),
        },
        "status_counts": dict(sorted(status_counts.items())),
        "episodes": results,
        "resumed": False,
    }
    _write_json_once(manifest_path, report)
    return report

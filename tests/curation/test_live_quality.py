from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from robot_data_studio.curation.live_quality import (
    C1AxisEvidence,
    C1StructuredEvidence,
    DashScopeC1Provider,
    LiveBudget,
    evaluate_c3_sampled_frames,
    evaluate_c1_concurrently,
    merge_live_core_evidence,
    run_live_quality_dataset,
    run_fast_live_quality_dataset,
    uniform_episode_indices,
    _dashscope_http_client,
)
from robot_data_studio.quality.contracts import (
    QWEN_CORE_DEFAULT_WEIGHTS,
    QWEN_CORE_LABELS,
    MetricEvidence,
)
from robot_data_studio.curation.c2_projection import matrix_to_xyz_rot6d
from robot_data_studio.curation.sam3 import Sam3Config, Sam3Provider


def _axis(state: str = "consistent", score: float = 0.9) -> dict:
    return {
        "state": state,
        "score": score,
        "confidence": 0.8,
        "reason": "visible evidence",
        "evidence_positions": [0, 2],
    }


def _response() -> dict:
    return {
        "instruction_video": _axis(),
        "object": _axis(),
        "action_phase": _axis("review", 0.6),
        "success": _axis("review", 0.5),
        "overall_state": "review",
        "overall_score": 0.7,
        "overall_confidence": 0.75,
        "summary": "instruction and objects are visible; success needs review",
    }


def test_c1_schema_is_strict_and_requires_all_semantic_axes() -> None:
    parsed = C1StructuredEvidence.model_validate(_response())
    assert parsed.success.state == "review"

    missing = _response()
    missing.pop("action_phase")
    with pytest.raises(ValidationError):
        C1StructuredEvidence.model_validate(missing)

    extra = _response()
    extra["unreviewed_field"] = True
    with pytest.raises(ValidationError):
        C1StructuredEvidence.model_validate(extra)

    with pytest.raises(ValidationError):
        C1AxisEvidence.model_validate(_axis("not_visible", 0.5) | {"evidence_positions": [1]})


def test_dashscope_transport_does_not_inherit_proxy_environment() -> None:
    with _dashscope_http_client() as client:
        assert client._trust_env is False


def test_real_c1_provider_uses_strict_schema_budget_and_secret_free_cache(
    tmp_path: Path,
) -> None:
    calls: list[dict] = []

    def request(payload: dict) -> dict:
        calls.append(payload)
        assert payload["response_format"]["type"] == "json_schema"
        assert payload["response_format"]["json_schema"]["strict"] is True
        assert payload["extra_body"] == {"enable_thinking": False}
        return {
            "content": json.dumps(_response()),
            "model": "qwen3.7-plus",
            "prompt_tokens": 111,
            "completion_tokens": 77,
        }

    provider = DashScopeC1Provider(
        cache_root=tmp_path,
        budget=LiveBudget(max_calls=1, max_evidence_frames=5),
        request_fn=request,
    )
    images = [np.zeros((24, 32, 3), dtype=np.uint8) for _ in range(3)]
    kwargs = {
        "source_video_sha256": "a" * 64,
        "episode_id": "episode-1",
        "task_text": "place the red block on the table",
        "images": images,
        "frame_indexes": [0, 10, 20],
        "timestamps": [0.0, 1.0, 2.0],
    }
    first = provider.evaluate(**kwargs)
    second = provider.evaluate(**kwargs)

    assert len(calls) == 1
    assert first.status == second.status == "evaluated"
    assert first.evidence["provider_mode"] == "real"
    assert first.evidence["from_cache"] is False
    assert second.evidence["from_cache"] is True
    assert first.evidence["axes"]["success"]["evidence_frames"] == [0, 20]
    assert first.evidence["axes"]["success"]["evidence_timestamps"] == [0.0, 2.0]
    cache_text = next(tmp_path.rglob("*.json")).read_text(encoding="utf-8")
    assert "secret" not in cache_text.casefold()
    assert "api_key" not in cache_text.casefold()


def test_c1_provider_fails_closed_on_invalid_evidence_position(tmp_path: Path) -> None:
    payload = _response()
    payload["object"]["evidence_positions"] = [9]
    provider = DashScopeC1Provider(
        cache_root=tmp_path,
        budget=LiveBudget(max_calls=1, max_evidence_frames=5),
        request_fn=lambda _: {"content": json.dumps(payload), "model": "qwen3.7-plus"},
    )
    metric = provider.evaluate(
        source_video_sha256="b" * 64,
        episode_id="episode-1",
        task_text="move an object",
        images=[np.zeros((8, 8, 3), dtype=np.uint8)],
        frame_indexes=[0],
        timestamps=[0.0],
    )
    assert metric.status == "not_evaluated"
    assert metric.score is None
    assert metric.evidence["provider_mode"] == "real_failed_closed"


def test_c1_concurrency_is_bounded_and_warm_cache_makes_no_requests(tmp_path: Path) -> None:
    lock = threading.Lock()
    active = 0
    max_active = 0
    calls = 0

    def request(_: dict) -> dict:
        nonlocal active, max_active, calls
        with lock:
            calls += 1
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.04)
        with lock:
            active -= 1
        return {"content": json.dumps(_response()), "model": "qwen3.7-plus"}

    provider = DashScopeC1Provider(
        cache_root=tmp_path,
        budget=LiveBudget(max_calls=2, max_evidence_frames=6),
        request_fn=request,
    )
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    requests = [
        {
            "source_video_sha256": "d" * 64,
            "episode_id": f"episode-{index}",
            "task_text": "move block",
            "images": [image, image, image],
            "frame_indexes": [0, 1, 2],
            "timestamps": [0.0, 0.1, 0.2],
        }
        for index in range(2)
    ]
    cold = evaluate_c1_concurrently(provider, requests, max_workers=2)
    warm = evaluate_c1_concurrently(provider, requests, max_workers=2)

    assert calls == 2
    assert max_active == 2
    assert all(metric.status == "evaluated" for metric in cold + warm)
    assert all(metric.evidence["from_cache"] is False for metric in cold)
    assert all(metric.evidence["from_cache"] is True for metric in warm)


def test_sparse_c3_scores_real_decoded_arrays_and_reports_intervals() -> None:
    checker = (np.indices((120, 160)).sum(axis=0) % 2 * 255).astype(np.uint8)
    crisp = np.repeat(checker[:, :, None], 3, axis=2)
    dark = np.zeros((120, 160, 3), dtype=np.uint8)
    metric = evaluate_c3_sampled_frames(
        [crisp, crisp.copy(), dark, dark.copy()],
        frame_indexes=[0, 10, 20, 30],
        timestamps=[0.0, 1.0, 2.0, 3.0],
        has_motion=True,
    )
    assert metric.status == "evaluated"
    assert metric.evidence["provider_mode"] == "deterministic_real_frames"
    assert metric.evidence["sample_count"] == 4
    assert metric.evidence["issue_counts"]["dark"] >= 1
    assert any(interval.start_frame >= 20 for interval in metric.abnormal_intervals)


def test_live_core_merge_preserves_na_and_recomputes_coverage() -> None:
    old = [
        MetricEvidence(
            metric_id=metric_id,
            label=QWEN_CORE_LABELS[metric_id],
            group="qwen_core",
            status="not_evaluated",
            weight=QWEN_CORE_DEFAULT_WEIGHTS[metric_id],
            reason="not run",
        )
        for metric_id in ("S1", "S2", "S3", "S4", "S5", "C1", "C2", "C3")
    ]
    c1 = old[5].model_copy(
        update={"status": "evaluated", "score": 0.8, "reason": "real qwen"}
    )
    c3 = old[7].model_copy(
        update={"status": "evaluated", "score": 0.9, "reason": "real decoded frames"}
    )
    scorecard = merge_live_core_evidence(old, replacements=[c1, c3])
    by_id = {metric.metric_id: metric for metric in scorecard.metrics}
    assert scorecard.coverage == pytest.approx(2 / 6)
    assert by_id["S4"].status == by_id["S5"].status == "retired"
    assert by_id["S1"].status == by_id["S3"].status == "not_evaluated"
    assert by_id["C1"].evidence.get("provider_mode") != "mock"
    assert scorecard.final_decision == "review"


def test_uniform_episode_indices_include_boundaries_without_crossing() -> None:
    assert uniform_episode_indices(10, 21, 5) == [10, 12, 15, 18, 20]
    assert uniform_episode_indices(4, 5, 8) == [4]
    with pytest.raises(ValueError):
        uniform_episode_indices(8, 8, 3)


def test_live_quality_runner_writes_additive_json_and_resumes(tmp_path: Path) -> None:
    pandas = pytest.importorskip("pandas")

    class Meta:
        episodes = type(
            "Episodes",
            (),
            {
                "to_pandas": staticmethod(
                    lambda: pandas.DataFrame(
                        [
                            {
                                "episode_index": 0,
                                "tasks": ["move the red block"],
                                "length": 5,
                                "dataset_from_index": 0,
                                "dataset_to_index": 5,
                            }
                        ]
                    )
                )
            },
        )()
        _version = "3.0"

    class Dataset:
        meta = Meta()

        def get_raw_item(self, index: int) -> dict:
            return {
                "source.frame_index": index,
                "source.timestamp": index / 30,
            }

        def __getitem__(self, index: int) -> dict:
            image = np.zeros((3, 24, 32), dtype=np.float32)
            image[:, 4:20, 4 + index : 16 + index] = 1.0
            return {"observation.images.camera0": image}

    prior = tmp_path / "prior"
    (prior / "episodes").mkdir(parents=True)
    metrics = [
        MetricEvidence(
            metric_id=metric_id,
            label=QWEN_CORE_LABELS[metric_id],
            group="qwen_core",
            status="not_evaluated",
            weight=QWEN_CORE_DEFAULT_WEIGHTS[metric_id],
            reason="not run",
        ).model_dump(mode="json")
        for metric_id in ("S1", "S2", "S3", "S4", "S5", "C1", "C2", "C3")
    ]
    prior_episode = {
        "episode_id": "source-episode",
        "source_id": "synthetic",
        "task_id": "move",
        "motion": 1.0,
        "disposition": {
            "hard_reject": False,
            "review": False,
            "window_masks": [],
            "soft_weight": 1.0,
            "hard_reasons": [],
            "review_reasons": [],
        },
        "metric_evidence": metrics,
    }
    (prior / "episodes" / "source-episode.json").write_text(json.dumps(prior_episode))
    (prior / "manifest.json").write_text(
        json.dumps(
            {
                "episodes": [
                    {"episode_id": "source-episode", "sidecar": "episodes/source-episode.json"}
                ]
            }
        )
    )
    provider = DashScopeC1Provider(
        cache_root=tmp_path / "cache",
        budget=LiveBudget(max_calls=1, max_evidence_frames=5),
        request_fn=lambda _: {"content": json.dumps(_response()), "model": "qwen3.7-plus"},
    )
    output = tmp_path / "live"
    first = run_live_quality_dataset(
        Dataset(),
        prior_root=prior,
        output_root=output,
        source_episode_ids=["source-episode"],
        max_episodes=1,
        c1_provider=provider,
        source_video_sha256="c" * 64,
        sam3_evidence={"status": "not_evaluated", "provider_mode": "real_failed_closed"},
        c3_samples=4,
    )
    second = run_live_quality_dataset(
        Dataset(),
        prior_root=prior,
        output_root=output,
        source_episode_ids=["source-episode"],
        max_episodes=1,
        c1_provider=provider,
        source_video_sha256="c" * 64,
        sam3_evidence={"status": "not_evaluated", "provider_mode": "real_failed_closed"},
        c3_samples=4,
    )
    merged = json.loads((output / "episodes/source-episode.json").read_text())
    by_id = {metric["metric_id"]: metric for metric in merged["metric_evidence"]}
    assert first["processed_episode_count"] == 1
    assert second["resumed"] is True
    assert by_id["C1"]["status"] == by_id["C3"]["status"] == "evaluated"
    assert by_id["C2"]["status"] == "not_evaluated"
    assert by_id["S1"]["status"] == by_id["S3"]["status"] == "not_evaluated"
    assert not list(output.rglob("*.mp4"))


def test_fast_runner_freezes_source_rows_per_episode(tmp_path: Path) -> None:
    pandas = pytest.importorskip("pandas")

    class Meta:
        episodes = type(
            "Episodes",
            (),
            {
                "to_pandas": staticmethod(
                    lambda: pandas.DataFrame(
                        [
                            {"episode_index": 0, "tasks": ["move"], "dataset_from_index": 0, "dataset_to_index": 3},
                            {"episode_index": 1, "tasks": ["move"], "dataset_from_index": 3, "dataset_to_index": 6},
                        ]
                    )
                )
            },
        )()

    identity = matrix_to_xyz_rot6d(np.eye(4))
    hand = np.eye(4)
    hand[:3, 3] = [0, 0, 1]
    eef = np.concatenate([matrix_to_xyz_rot6d(hand), matrix_to_xyz_rot6d(hand)])

    class Dataset:
        meta = Meta()

        def get_raw_item(self, index: int) -> dict:
            return {
                "source.frame_index": index % 3,
                "source.timestamp": (index % 3) / 30,
                "observation.eef_pose_absolute": eef,
                "observation.eef_pose_absolute.validity": np.asarray([True, True]),
                "observation.camera0_pose_absolute": identity,
                "observation.camera0_pose_absolute.validity": True,
            }

        def __getitem__(self, index: int) -> dict:
            return {"observation.images.camera0": np.zeros((3, 20, 30), dtype=np.float32)}

    prior = tmp_path / "prior"
    (prior / "episodes").mkdir(parents=True)
    entries = []
    for episode in ("a", "b"):
        sidecar = prior / "episodes" / f"{episode}.json"
        sidecar.write_text(json.dumps({"episode_id": episode, "motion": 1, "disposition": {}, "metric_evidence": []}))
        entries.append({"episode_id": episode, "sidecar": f"episodes/{episode}.json"})
    (prior / "manifest.json").write_text(json.dumps({"episodes": entries}))
    c1 = DashScopeC1Provider(
        cache_root=tmp_path / "c1",
        budget=LiveBudget(max_calls=2, max_evidence_frames=6),
        request_fn=lambda _: {"content": json.dumps(_response()), "model": "qwen3.7-plus"},
    )
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"fixture")
    runner = lambda images, prompt, threshold: [
        (np.ones(np.asarray(image).shape[:2], dtype=bool), 0.9) for image in images
    ]
    config = Sam3Config(backend="local", local_weights_path=str(weights), model_version="fixture")
    report = run_fast_live_quality_dataset(
        Dataset(),
        prior_root=prior,
        output_root=tmp_path / "out",
        source_episode_ids=["a", "b"],
        max_episodes=2,
        c1_provider=c1,
        source_video_sha256="e" * 64,
        sam3_entity_provider=Sam3Provider(config, cache_root=tmp_path / "sam-entity", local_batch_fn=runner),
        sam3_object_provider=Sam3Provider(config.model_copy(update={"prompt": "object"}), cache_root=tmp_path / "sam-object", local_batch_fn=runner),
        sam3_checkpoint_evidence={"status": "available"},
        intrinsic=[[10, 0, 15], [0, 10, 10], [0, 0, 1]],
        embodiment="human",
        c1_samples=3,
        c3_samples=3,
        sam3_samples=2,
    )
    assert report["processed_episode_count"] == 2
    assert report["c1_api_calls"] == 2

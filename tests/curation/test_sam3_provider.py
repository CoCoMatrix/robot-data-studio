from __future__ import annotations

import json

import numpy as np

from robot_data_studio.curation.sam3 import Sam3Config, Sam3Provider


def test_sam3_disabled_is_honest_not_evaluated(tmp_path) -> None:
    result = Sam3Provider(
        Sam3Config(backend="disabled"), cache_root=tmp_path
    ).segment(
        source_video_sha256="a" * 64,
        frame_index=4,
        image=np.zeros((12, 16, 3), dtype=np.uint8),
    )
    assert result.status == "not_evaluated"
    assert result.mask is None
    assert result.cache_hit is False


def test_sam3_mock_is_deterministic_cached_and_secret_free(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SAM3_API_KEY", "never-write-this-secret")
    config = Sam3Config(
        backend="mock",
        model_version="sam3-mock-v1",
        prompt="robot arms and hands",
        api_key_env="SAM3_API_KEY",
        mock_box=(0.25, 0.25, 0.75, 0.75),
    )
    provider = Sam3Provider(config, cache_root=tmp_path)
    kwargs = {
        "source_video_sha256": "b" * 64,
        "frame_index": 9,
        "image": np.zeros((12, 16, 3), dtype=np.uint8),
    }
    first = provider.segment(**kwargs)
    second = provider.segment(**kwargs)

    assert first.status == "evaluated"
    assert first.cache_hit is False
    assert second.cache_hit is True
    np.testing.assert_array_equal(first.mask, second.mask)
    assert first.cache_key == second.cache_key
    assert first.confidence == 1.0
    cache_text = "\n".join(
        path.read_text(errors="ignore")
        for path in tmp_path.rglob("*.json")
    )
    assert "never-write-this-secret" not in cache_text
    metadata = json.loads(next(tmp_path.rglob("*.json")).read_text())
    assert metadata["inputs"]["frame_index"] == 9
    assert metadata["inputs"]["model_version"] == "sam3-mock-v1"


def test_sam3_local_batch_runner_is_called_once_and_warm_cache_is_free(tmp_path) -> None:
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"synthetic-checkpoint")
    calls = []

    def local_batch(images, prompt, confidence_threshold):
        calls.append((len(images), prompt, confidence_threshold))
        return [
            (np.ones(np.asarray(image).shape[:2], dtype=bool), 0.91)
            for image in images
        ]

    provider = Sam3Provider(
        Sam3Config(
            backend="local",
            model_version="facebook-sam3-test",
            prompt="hands",
            local_weights_path=str(weights),
        ),
        cache_root=tmp_path / "cache",
        local_batch_fn=local_batch,
    )
    kwargs = {
        "source_video_sha256": "c" * 64,
        "frame_indexes": [1, 9],
        "images": [
            np.zeros((10, 12, 3), dtype=np.uint8),
            np.zeros((10, 12, 3), dtype=np.uint8),
        ],
    }
    cold = provider.segment_batch(**kwargs)
    warm = provider.segment_batch(**kwargs)

    assert calls == [(2, "hands", 0.5)]
    assert all(result.status == "evaluated" for result in cold + warm)
    assert all(result.cache_hit is False for result in cold)
    assert all(result.cache_hit is True for result in warm)
    assert all(result.reason.startswith("local SAM3") for result in cold)

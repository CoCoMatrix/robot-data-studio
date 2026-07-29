from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field


class Sam3Config(BaseModel):
    backend: Literal["disabled", "mock", "local", "api"] = "disabled"
    model_version: str = "sam3-unconfigured"
    prompt: str = "robot arms and hands"
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    api_key_env: str = "SAM3_API_KEY"
    local_weights_path: str | None = None
    device: str = "cuda:0"
    batch_size: int = Field(default=4, ge=1, le=16)
    mock_box: tuple[float, float, float, float] = (0.25, 0.25, 0.75, 0.75)


class Sam3Result(BaseModel):
    status: Literal["evaluated", "not_evaluated"]
    reason: str
    confidence: float | None = None
    cache_key: str
    cache_hit: bool = False
    mask: np.ndarray | None = None

    model_config = {"arbitrary_types_allowed": True}


def _cache_key(config: Sam3Config, source_video_sha256: str, frame_index: int) -> tuple[str, dict]:
    inputs = {
        "source_video_sha256": source_video_sha256,
        "frame_index": int(frame_index),
        "model_version": config.model_version,
        "prompt": config.prompt,
        "config": {
            "backend": config.backend,
            "confidence_threshold": config.confidence_threshold,
            "mock_box": list(config.mock_box),
            "local_weights_path": config.local_weights_path,
            "device": config.device,
            "batch_size": config.batch_size,
            "api_key_env": config.api_key_env,
        },
    }
    encoded = json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), inputs


LocalBatchFn = Callable[
    [Sequence[np.ndarray], str, float],
    list[tuple[np.ndarray, float]],
]


@lru_cache(maxsize=2)
def _local_runtime(model_dir: str, device: str):
    import torch
    from transformers import Sam3Model, Sam3Processor

    path = Path(model_dir)
    processor = Sam3Processor.from_pretrained(path, local_files_only=True)
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = Sam3Model.from_pretrained(
        path,
        local_files_only=True,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    return processor, model


class Sam3Provider:
    """Fail-closed SAM3 boundary with deterministic cache and bounded local batches."""

    def __init__(
        self,
        config: Sam3Config,
        *,
        cache_root: str | Path,
        local_batch_fn: LocalBatchFn | None = None,
    ) -> None:
        self.config = config
        self.cache_root = Path(cache_root)
        self.local_batch_fn = local_batch_fn or self._run_local_batch

    def _run_local_batch(
        self,
        images: Sequence[np.ndarray],
        prompt: str,
        confidence_threshold: float,
    ) -> list[tuple[np.ndarray, float]]:
        import torch
        from PIL import Image

        weights = Path(self.config.local_weights_path or "")
        model_dir = weights if weights.is_dir() else weights.parent
        processor, model = _local_runtime(str(model_dir.resolve()), self.config.device)
        results: list[tuple[np.ndarray, float]] = []
        for offset in range(0, len(images), self.config.batch_size):
            batch = [np.asarray(image, dtype=np.uint8) for image in images[offset : offset + self.config.batch_size]]
            pil_images = [Image.fromarray(image, mode="RGB") for image in batch]
            encoded = processor(
                images=pil_images,
                text=[prompt] * len(pil_images),
                return_tensors="pt",
            )
            encoded = {
                key: value.to(self.config.device) if hasattr(value, "to") else value
                for key, value in encoded.items()
            }
            device_type = "cuda" if self.config.device.startswith("cuda") else "cpu"
            with torch.inference_mode(), torch.autocast(
                device_type=device_type,
                dtype=torch.float16,
                enabled=device_type == "cuda",
            ):
                outputs = model(**encoded)
            processed = processor.post_process_instance_segmentation(
                outputs,
                threshold=confidence_threshold,
                target_sizes=[image.shape[:2] for image in batch],
            )
            for image, item in zip(batch, processed, strict=True):
                masks = item["masks"].detach().cpu().numpy().astype(bool)
                scores = item["scores"].detach().float().cpu().numpy()
                union = (
                    np.any(masks, axis=0)
                    if masks.size
                    else np.zeros(image.shape[:2], dtype=bool)
                )
                confidence = float(np.max(scores)) if scores.size else 0.0
                results.append((union, confidence))
            del encoded, outputs, processed
        return results

    def _cache_paths(self, key: str) -> tuple[Path, Path]:
        metadata_path = self.cache_root / key[:2] / f"{key}.json"
        return metadata_path, metadata_path.with_suffix(".npz")

    def _load_cached(self, key: str) -> Sam3Result | None:
        metadata_path, mask_path = self._cache_paths(key)
        if not metadata_path.is_file() or not mask_path.is_file():
            return None
        metadata = json.loads(metadata_path.read_text())
        mask = np.load(mask_path, allow_pickle=False)["mask"].astype(np.bool_)
        return Sam3Result(
            status="evaluated",
            reason=metadata["reason"],
            confidence=float(metadata["confidence"]),
            cache_key=key,
            cache_hit=True,
            mask=mask,
        )

    def _write_cache(
        self,
        *,
        key: str,
        inputs: dict,
        mask: np.ndarray,
        confidence: float,
        reason: str,
    ) -> Sam3Result:
        metadata_path, mask_path = self._cache_paths(key)
        metadata = {
            "schema": "robot-data-studio.sam3-cache.v1",
            "inputs": inputs,
            "status": "evaluated",
            "reason": reason,
            "confidence": confidence,
        }
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = metadata_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        temporary.replace(metadata_path)
        np.savez_compressed(mask_path, mask=np.asarray(mask, dtype=np.bool_))
        return Sam3Result(
            status="evaluated",
            reason=reason,
            confidence=confidence,
            cache_key=key,
            mask=np.asarray(mask, dtype=np.bool_),
        )

    def segment_batch(
        self,
        *,
        source_video_sha256: str,
        frame_indexes: Sequence[int],
        images: Sequence[np.ndarray],
    ) -> list[Sam3Result]:
        if len(frame_indexes) != len(images) or not images:
            raise ValueError("SAM3 batch requires equal non-empty frames and images")
        if self.config.backend != "local":
            return [
                self.segment(
                    source_video_sha256=source_video_sha256,
                    frame_index=frame_index,
                    image=image,
                )
                for frame_index, image in zip(frame_indexes, images, strict=True)
            ]
        weights = Path(self.config.local_weights_path or "")
        if not self.config.local_weights_path or not weights.is_file():
            return [
                Sam3Result(
                    status="not_evaluated",
                    reason="local SAM3 weights unavailable; C2 remains N/A",
                    cache_key=_cache_key(self.config, source_video_sha256, frame_index)[0],
                )
                for frame_index in frame_indexes
            ]

        results: list[Sam3Result | None] = []
        missing_positions: list[int] = []
        missing_images: list[np.ndarray] = []
        keys_and_inputs = [
            _cache_key(self.config, source_video_sha256, frame_index)
            for frame_index in frame_indexes
        ]
        for position, ((key, _), image) in enumerate(zip(keys_and_inputs, images, strict=True)):
            cached = self._load_cached(key)
            results.append(cached)
            if cached is None:
                missing_positions.append(position)
                missing_images.append(np.asarray(image))
        if missing_images:
            inferred = self.local_batch_fn(
                missing_images,
                self.config.prompt,
                self.config.confidence_threshold,
            )
            if len(inferred) != len(missing_images):
                raise RuntimeError("local SAM3 batch returned an unexpected result count")
            for position, (mask, confidence) in zip(missing_positions, inferred, strict=True):
                image = np.asarray(images[position])
                if np.asarray(mask).shape != image.shape[:2]:
                    raise ValueError("local SAM3 mask shape differs from input image")
                key, inputs = keys_and_inputs[position]
                results[position] = self._write_cache(
                    key=key,
                    inputs=inputs,
                    mask=np.asarray(mask),
                    confidence=float(confidence),
                    reason="local SAM3 instance masks from official facebook/sam3 checkpoint",
                )
        return [result for result in results if result is not None]

    def segment(
        self,
        *,
        source_video_sha256: str,
        frame_index: int,
        image: np.ndarray,
    ) -> Sam3Result:
        key, inputs = _cache_key(self.config, source_video_sha256, frame_index)
        if cached := self._load_cached(key):
            return cached
        if self.config.backend == "disabled":
            return Sam3Result(
                status="not_evaluated",
                reason="SAM3 provider disabled; C2 remains N/A",
                cache_key=key,
            )
        if self.config.backend == "local":
            return self.segment_batch(
                source_video_sha256=source_video_sha256,
                frame_indexes=[frame_index],
                images=[image],
            )[0]
        if self.config.backend == "api":
            if not os.environ.get(self.config.api_key_env):
                return Sam3Result(
                    status="not_evaluated",
                    reason=f"SAM3 API credential environment variable {self.config.api_key_env} is unset; C2 remains N/A",
                    cache_key=key,
                )
            return Sam3Result(
                status="not_evaluated",
                reason="SAM3 API execution is prohibited in this bounded pilot",
                cache_key=key,
            )

        height, width = np.asarray(image).shape[:2]
        x0, y0, x1, y1 = self.config.mock_box
        x_start, x_end = int(round(x0 * width)), int(round(x1 * width))
        y_start, y_end = int(round(y0 * height)), int(round(y1 * height))
        mask = np.zeros((height, width), dtype=np.bool_)
        mask[max(0, y_start) : min(height, y_end), max(0, x_start) : min(width, x_end)] = True
        return self._write_cache(
            key=key,
            inputs=inputs,
            mask=mask,
            confidence=1.0,
            reason="deterministic synthetic SAM3 mock; never evidence for real C2 pass",
        )

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch DOVER-Mobile technical quality runner")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    import decord
    import numpy as np
    import torch
    from dover.datasets.dover_datasets import UnifiedFrameSampler, get_spatial_fragments
    from dover.models import DOVER

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    batch_size = args.batch_size if device == "cuda" else 1
    model = DOVER(
        backbone={"technical": {"type": "conv_femto"}},
        backbone_preserve_keys="technical",
        divide_head=True,
        vqa_head={"in_channels": 384, "hidden_channels": 32},
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    state = normalize_checkpoint_state(model, state)
    model.load_state_dict(state, strict=False)
    model.eval()
    samples = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    results: list[dict] = []
    mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1, 1)
    std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1, 1)

    for offset in range(0, len(samples), batch_size):
        prepared: list[tuple[dict, object]] = []
        for sample in samples[offset : offset + batch_size]:
            try:
                reader = decord.VideoReader(sample["video_path"])
                fps = float(reader.get_avg_fps())
                start = max(0, round(float(sample["start_seconds"]) * fps))
                stop = min(
                    len(reader),
                    round(
                        (float(sample["start_seconds"]) + float(sample["duration_seconds"]))
                        * fps
                    ),
                )
                if stop <= start:
                    raise ValueError("empty_video_segment")
                seed = int(sample["episode_index"]) + sum(
                    ord(char) for char in sample["camera"]
                )
                np.random.seed(seed)
                random.seed(seed)
                torch.manual_seed(seed)
                sampler = UnifiedFrameSampler(32, 3, 2)
                indices = sampler.get_frame_indices(stop - start) + start
                indices = np.clip(indices, start, stop - 1)
                frames = reader.get_batch(indices).permute(3, 0, 1, 2)
                technical = get_spatial_fragments(
                    frames,
                    fragments_h=7,
                    fragments_w=7,
                    fsize_h=32,
                    fsize_w=32,
                    aligned=32,
                )
                prepared.append(
                    (sample, reshape_technical_clips(technical, mean, std))
                )
            except Exception as error:
                results.append(failure(sample, error))
        if not prepared:
            continue
        try:
            raw = infer_technical_scores(model, torch, prepared, device)
            for (sample, _), value in zip(prepared, raw, strict=True):
                results.append(success(sample, value))
        except Exception:
            for item in prepared:
                sample = item[0]
                try:
                    value = infer_technical_scores(model, torch, [item], device)[0]
                    results.append(success(sample, value))
                except Exception as error:
                    results.append(failure(sample, error))

    Path(args.output).write_text(json.dumps(results), encoding="utf-8")


def normalize_checkpoint_state(model, state):
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model" in state:
        state = state["model"]
    if not isinstance(state, dict):
        raise RuntimeError("invalid_checkpoint")
    normalized = {
        key.removeprefix("module."): value
        for key, value in state.items()
        if isinstance(key, str)
    }
    required = set(model.state_dict())
    if required - normalized.keys():
        raise RuntimeError("checkpoint_missing_technical_weights")
    return normalized


def reshape_technical_clips(
    technical,
    mean,
    std,
    *,
    num_clips: int = 3,
    clip_len: int = 32,
):
    normalized = (technical - mean) / std
    return normalized.reshape(3, num_clips, clip_len, *technical.shape[2:]).swapaxes(0, 1)


def infer_technical_scores(model, torch, chunk, device: str) -> list[float]:
    inputs = torch.cat([item[1] for item in chunk], dim=0).to(device)
    with torch.no_grad():
        raw = model({"technical": inputs}, reduce_scores=False)[0]
    return raw.reshape(len(chunk), 3, -1).mean(dim=(1, 2)).cpu().tolist()


def success(sample: dict, value: float) -> dict:
    score = 1.0 / (1.0 + math.exp(-((float(value) - 0.1107) / 0.07355)))
    return {
        "episode_index": sample["episode_index"],
        "camera": sample["camera"],
        "technical_score": score,
        "error_code": None,
    }


def failure(sample: dict, error: Exception) -> dict:
    return {
        "episode_index": sample["episode_index"],
        "camera": sample["camera"],
        "technical_score": None,
        "error_code": type(error).__name__,
    }


if __name__ == "__main__":
    main()

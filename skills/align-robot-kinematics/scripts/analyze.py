#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Discover a reviewable SE(3) kinematic correction without modifying source data."
    )
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--format", default="auto")
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--episodes", default="", help="Optional comma-separated episode indexes.")
    parser.add_argument("--sample-budget", type=int, default=64)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--max-joint-mappings", type=int, default=256)
    parser.add_argument("--min-candidate-margin", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.sample_budget < 4:
        raise SystemExit("--sample-budget must be at least 4")
    repository = args.repository.expanduser().resolve()
    packages = repository / "packages"
    if not packages.is_dir():
        raise SystemExit(f"Repository packages directory not found: {packages}")
    sys.path.insert(0, str(packages))
    sys.path.insert(0, str(repository))

    from robot_data_studio.formats import FormatRegistry
    from robot_data_studio.quality.kinematics import (
        CorrectionOptions,
        DiscoveryRefusal,
        DiscoveryRequest,
        discover_profile,
        fit_correction,
    )

    reader = FormatRegistry.default().open_dataset(
        args.dataset.expanduser().resolve(),
        format_hint=None if args.format == "auto" else args.format,
    )
    available = [episode.episode_index for episode in reader.list_episodes()]
    requested = _parse_episode_indexes(args.episodes) or available[: min(4, len(available))]
    missing = sorted(set(requested) - set(available))
    if missing:
        raise SystemExit(f"Unknown episode indexes: {missing}")
    states = _sample_states(reader, requested, args.sample_budget)
    feature_names = _feature_names(reader.metadata().features, states.shape[1])
    try:
        candidates = discover_profile(
            args.urdf.expanduser().resolve(),
            DiscoveryRequest(
                states=states,
                feature_names=feature_names,
                max_candidates=args.max_candidates,
                sample_count=len(states),
                max_joint_mappings=args.max_joint_mappings,
            ),
        )
        result = fit_correction(
            states,
            candidates,
            CorrectionOptions(min_candidate_margin=args.min_candidate_margin),
        )
    except ValueError as error:
        result = DiscoveryRefusal(reason="no_candidates", evidence=[str(error)])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "result": _result_label(result)}))
    return 0


def _parse_episode_indexes(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _sample_states(reader: object, episode_indexes: list[int], budget: int) -> np.ndarray:
    rows = [
        frame.observation_state
        for episode_index in episode_indexes
        for frame in reader.read_episode_frames(episode_index)
    ]
    states = np.asarray(rows, dtype=np.float64)
    if states.ndim != 2 or not len(states) or not np.all(np.isfinite(states)):
        raise ValueError("Selected episodes contain no finite observation.state matrix")
    if len(states) <= budget:
        return states
    indexes = np.linspace(0, len(states) - 1, budget, dtype=int)
    return states[indexes]


def _feature_names(features: dict[str, dict], width: int) -> list[str]:
    feature = features.get("observation.state", {})
    names = feature.get("names", {}) if isinstance(feature, dict) else {}
    if isinstance(names, dict):
        flattened = [
            str(name)
            for group in names.values()
            if isinstance(group, list)
            for name in group
        ]
    elif isinstance(names, list):
        flattened = [str(name) for name in names]
    else:
        flattened = []
    return flattened if len(flattened) == width else [f"state_{index}" for index in range(width)]


def _result_label(result: object) -> str:
    status = getattr(result, "status", None)
    return str(status or getattr(result, "reason", "unknown"))


if __name__ == "__main__":
    raise SystemExit(main())

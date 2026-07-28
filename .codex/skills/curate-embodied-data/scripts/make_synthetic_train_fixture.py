#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    episode = {
        "episode_id": "synthetic-000",
        "source_id": "robot-example",
        "task_id": "move-object",
        "split": "train",
        "from_timestamp": 5.0,
        "expected_duration": 0.3,
        "event_frames": {"start": 0, "contact": 2, "end": 3},
        "lineage": {"generator": "make_synthetic_train_fixture.py", "seed": 0},
        "frames": [
            {
                "episode_id": "synthetic-000",
                "frame_index": index,
                "timestamp": index / 10,
                "decoded_pts": 5.0 + index / 10,
                "valid": True,
                "state": [float(index)],
                "action": [float(index) / 10],
            }
            for index in range(4)
        ],
    }
    output.write_text(json.dumps(episode, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

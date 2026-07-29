#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "packages"))

from robot_data_studio.curation.profile import load_profile  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile")
    args = parser.parse_args()
    profile = load_profile(args.profile)
    print(
        json.dumps(
            {
                "valid": True,
                "name": profile.name,
                "sources": [source.source_id for source in profile.sources],
                "target_points": profile.progress.target_points,
                "vlm_backend": profile.vlm.backend,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

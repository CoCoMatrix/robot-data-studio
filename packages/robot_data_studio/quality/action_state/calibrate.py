"""Propose (never silently apply) channel-specific limits from reviewed, known-good episodes."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

from .contract import VERSION
from .run_operator import atomic_json


def propose(reports, minimum=10):
    if minimum < 3:
        raise ValueError("at least three distinct reference episodes required")
    seen = set()
    groups = defaultdict(list)
    identity = None
    for r in reports:
        key = (r["dataset"], r["profile_sha256"], r["version"])
        if (
            r["operator"] != "action_state"
            or r["version"] != VERSION
            or (identity and key != identity)
        ):
            raise ValueError(
                "use one operator version, dataset, and frozen profile per fit"
            )
        if r["episode_id"] in seen:
            raise ValueError("duplicate calibration episode")
        seen.add(r["episode_id"])
        identity = key
        for name, c in r["channels"].items():
            q = c.get("alignment", {})
            if c["route"] == "command" and q.get("active_transitions", 0) >= 30:
                groups[name].append((q, c["thresholds"], r["native_dt_s"]))
    if not seen:
        raise ValueError("no reference reports")
    output = {}
    for name, rows in groups.items():
        if len(rows) < minimum:
            output[name] = dict(support=len(rows), status="insufficient_support")
            continue
        base = rows[0][1]
        # Each episode has one vote irrespective of length. Wide proposals remain visible for review.
        vals = lambda field: [r[0][field] for r in rows if r[0].get(field) is not None]
        error = float(np.quantile(vals("active_error_p95"), 0.90))
        step = float(np.quantile(vals("active_action_step_p999"), 0.90))
        lag = [
            m["best_lag_s"] + dt
            for m, _, dt in rows
            if m["best_lag_s"] is not None
            and m["best_lag_s"] >= 0
            and m["delta_correlation"] >= 0.6
            and not m["lag_at_search_limit"]
        ]
        suggested = dict(
            error=max(base["error"], 1.5 * error), jump=max(base["jump"], 1.5 * step)
        )
        if len(lag) >= minimum:
            suggested["response_s"] = max(
                base["response_s"], float(np.quantile(lag, 0.95))
            )
        output[name] = dict(
            status="proposal",
            support=len(rows),
            lag_support=len(lag),
            thresholds=suggested,
            wide=[k for k, v in suggested.items() if v > 2 * base[k]],
        )
    return dict(
        dataset=identity[0],
        profile_sha256=identity[1],
        operator_version=identity[2],
        reference_episode_ids=sorted(seen),
        channels=output,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reports", type=Path, nargs="+")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--minimum", type=int, default=10)
    args = p.parse_args()
    atomic_json(
        args.out,
        propose((json.loads(f.read_text()) for f in args.reports), args.minimum),
    )

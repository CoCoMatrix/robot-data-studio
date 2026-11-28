"""Bounded replay from the 1127 train allowlist and the recorded v1.0 channel decisions."""

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import argparse
import fnmatch
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path
import numpy as np
from robot_data_studio.quality.action_state import analyze, usable_mask
from .profiles import joint
from .run_operator import atomic_json

TRAIN_ALLOWLIST = ("agibotworld-beta/train/*", "robocoin/train/*")
PACKAGES = {
    "agibotworld-beta": "agibotworld-beta/v3-subtask-20261116",
    "robocoin": "robocoin/v3-subtask-20261113",
}


def plan(cache_run, old_results, limit):
    manifest = cache_run / "sample/selected.jsonl"
    samples = {sample["uid"]: sample for line in manifest.open() if (sample := json.loads(line))}
    jobs = defaultdict(list)
    expected = {
        uid
        for uid, sample in samples.items()
        if sample["release"] in PACKAGES.values()
        and any(fnmatch.fnmatchcase(sample["episode"], pat) for pat in TRAIN_ALLOWLIST)
    }
    seen = set()
    for path in sorted(old_results.glob("labels-*.jsonl.gz")):
        with gzip.open(path, "rt") as f:
            for line in f:
                baseline = json.loads(line)
                sample = samples[baseline["uid"]]
                if not any(fnmatch.fnmatchcase(sample["episode"], pat) for pat in TRAIN_ALLOWLIST):
                    continue
                if PACKAGES.get(sample["dataset"]) != sample["release"]:
                    continue
                if sample["uid"] in seen:
                    raise ValueError(f"duplicate baseline clip {sample['uid']}")
                seen.add(sample["uid"])
                dataset = sample["dataset"]
                if dataset == "agibotworld-beta":
                    pick = [
                        previous
                        for previous in baseline["channels"]
                        if previous.get("valid") and previous.get("role") == "target"
                    ]
                else:
                    pick = [
                        previous
                        for previous in baseline["channels"]
                        if previous.get("route") == "command"
                        and previous.get("semantic") in ("same_state", "next_state")
                    ]
                if pick:
                    jobs[sample["batch"]].append((sample, pick))
    if seen != expected:
        raise ValueError("missing baseline decisions for frozen training clips")
    size = sum(map(len, jobs.values()))
    if not 0 < size <= limit:
        raise ValueError(f"{size} selected train clips outside pilot limit {limit}")
    return [(cache_run, batch, batch_samples) for batch, batch_samples in jobs.items()]


def replay(job):
    root, batch, batch_samples = job
    stats = Counter()
    examples = []
    with tarfile.open(root / "inputs" / (batch + ".tar")) as archive:
        for sample, channels_to_check in batch_samples:
            row_count = sample["rows"]
            with np.load(io.BytesIO(archive.extractfile(sample["uid"] + ".npz").read())) as arrays:
                timestamps = arrays["t"][:row_count]
                action = arrays["a"][:row_count]
                state = arrays["s"][:row_count]
                action_valid = arrays["am"][:row_count]
                state_valid = arrays["sm"][:row_count]
                cols = arrays["cols"].tolist()
            signals = {}
            channels = []
            for previous in channels_to_check:
                slot = previous["k"]
                column = cols.index(slot)
                kind = (
                    "arm"
                    if slot < 14 or 74 <= slot < 80
                    else "hand"
                    if 30 <= slot < 74
                    else "aperture"
                    if slot in (28, 29)
                    else "linear"
                )
                unit = "m" if kind == "linear" else "fraction" if kind == "aperture" else "rad"
                ch = joint(
                    str(slot),
                    "action",
                    "state",
                    slot,
                    kind=kind,
                    evidence="published robot100 slot",
                    unit=unit,
                )
                if kind == "aperture":
                    ch["anchors"] = dict(
                        action=[0, 1], state=[0, 1], evidence="released aperture mapping"
                    )
                if kind == "hand":
                    ch["contact_possible"] = True
                channels.append(ch)
                signals[str(slot)] = dict(
                    action=action[:, column],
                    state=state[:, column],
                    action_valid=action_valid[:, column],
                    state_valid=state_valid[:, column],
                )
            profile = dict(
                version=1,
                dataset=sample["dataset"],
                channels=channels,
                cache_source=dict(
                    release=sample["release"], robot=sample["robot"], layout="robot100"
                ),
            )
            report = analyze(
                sample["episode"],
                timestamps,
                signals,
                profile,
                start_s=min(0.0, float(timestamps[0])),
                end_s=float(timestamps[-1]) + 1000,
            )
            stats[sample["dataset"] + "_clips"] += 1
            for previous in channels_to_check:
                slot = previous["k"]
                column = cols.index(slot)
                name = str(slot)
                new = report["channels"][name]
                valid = action_valid[:, column] & state_valid[:, column]
                mask = usable_mask(report, [name])[:, 0]
                proxy = usable_mask(report, [name], supervision="state_proxy")[:, 0]
                old = valid.copy()
                for lo, hi in previous["excluded"]:
                    old[lo:hi] = False
                if sample["dataset"] == "agibotworld-beta" and slot < 14:
                    assert np.array_equal(action[valid, column], state[valid, column])
                    assert not mask.any() and np.array_equal(proxy, valid)
                    stats["agibot_arm_channel_clips"] += 1
                    stats["agibot_arm_proxy_values"] += int(proxy.sum())
                    if old.any():
                        assert previous["semantic"] == "insufficient_motion"
                        stats["agibot_prior_command_channel_clips"] += 1
                        stats["agibot_prior_command_rows"] += int(old.sum())
                elif sample["dataset"] == "robocoin":
                    assert (
                        new["semantic"]["verdict"] == previous["semantic"]
                        and new["route"] == "review"
                    )
                    assert not mask.any() and not proxy.any()
                    stats["robocoin_prior_command_rows"] += int(old.sum())
                    stats["robocoin_" + previous["semantic"]] += 1
                    examples.append(
                        dict(
                            uid=sample["uid"],
                            episode=sample["episode"],
                            slot=slot,
                            signature=previous["semantic"],
                            before=int(old.sum()),
                            after=int(mask.sum()),
                        )
                    )
                else:
                    assert new["route"] == previous["route"] and np.array_equal(mask, old), (
                        sample["uid"],
                        slot,
                        new["route"],
                        previous["route"],
                    )
                    stats["agibot_nonarm_unchanged_channel_clips"] += 1
                    stats["agibot_nonarm_unchanged_command_rows"] += int(mask.sum())
    return stats, examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-run", required=True, type=Path, help="1127_actionstate frozen labels and selection"
    )
    parser.add_argument(
        "--old-results", required=True, type=Path, help="v1.0 results/cache-v11 decision shards"
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--limit", default=900, type=int)
    parser.add_argument("--workers", default=8, type=int)
    args = parser.parse_args()
    jobs = plan(args.cache_run, args.old_results, args.limit)
    counts = Counter()
    examples_out = []
    with ProcessPoolExecutor(args.workers) as pool:
        for stats, examples in pool.map(replay, jobs):
            counts.update(stats)
            examples_out.extend(examples)
    if not counts["robocoin_prior_command_rows"] or not counts["agibot_prior_command_rows"]:
        raise ValueError("neither affected cohort may be empty in this historical regression")
    print(json.dumps(counts, indent=2))
    summary = dict(
        counts=counts,
        examples=examples_out,
        reference_commit="50128205e0fe117591642b5806aec5a4efb50036",
        selection_sha256=hashlib.sha256(
            (args.cache_run / "sample/selected.jsonl").read_bytes()
        ).hexdigest(),
    )

    atomic_json(args.out, summary)


if __name__ == "__main__":
    main()

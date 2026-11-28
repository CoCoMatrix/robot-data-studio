"""Replay raw native episodes. Compares decisions to independently stored cache/reference records."""

import argparse
import csv
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import pyarrow.parquet as pq
from .profiles import preset
from .run_operator import atomic_json, run_parquet

ROOT = Path("/mnt/oss/data_pipeline/robot-clean")
RAW = Path("/mnt/ccm-raw")
SHARPA = ROOT / "cache/sharpadex/v3-subtask-20261117"
SAMPLES = Path("/mnt/workspace/robot-action-state-samples")
ARCH = {
    "Astribot S1": "astribot-s1",
    "SharpaDex": "sharpadex",
    "LingYu": "lingyu",
    "FineART": "fineart",
}
DEV_TASKS = {
    "transfer_salt_between_cups_with_spoon",
    "nest_cups",
    "clean_garment_with_lint_roller",
}


@lru_cache(maxsize=80)
def sharp_files(season_dir):
    loc = {}
    for f in sorted((season_dir / "meta/episodes").glob("*/*.parquet")):
        for r in pq.read_table(
            f, columns=["episode_index", "data/chunk_index", "data/file_index"]
        ).to_pylist():
            loc[r["episode_index"]] = season_dir / (
                "data/chunk-%03d/file-%03d.parquet"
                % (r["data/chunk_index"], r["data/file_index"])
            )
    return loc


def sharp_path(key):
    task, sep, ep = key.replace("/episode_", "__episode_").rpartition("__episode_")
    if not sep:
        raise ValueError(key)
    return str(
        sharp_files(RAW / "public/SharpaDex-v1.0" / task / "lerobot_v3.0")[int(ep)]
    )


def curated_cases(samples):
    out = []
    for file, normal in [
        ("normal-source-frames.csv", True),
        ("source-frames.csv", False),
    ]:
        with (samples / file).open(encoding="utf-8-sig") as f:
            entries = {}
            for row in csv.DictReader(f):
                d = ARCH.get(row["dataset"], row["dataset"])
                key = row.get("episode") or row.get("episode_key")
                c = entries.setdefault(
                    (d, key),
                    dict(
                        dataset=d,
                        key=key,
                        source=row["source_parquet"],
                        channel=row["action_name"],
                        frame_times=[],
                        expect="normal" if normal else "example",
                    ),
                )
                c["frame_times"].append(float(row["time_s"]))
        for c in entries.values():
            times = c.pop("frame_times")
            c["window"] = [min(times), max(times) + 1e-6]
            if c["dataset"] == "lingyu":
                c["channel"] = "left_grip"
                if not normal:
                    c["expect"] = "normal_calibrated_gripper"
            if c["dataset"] == "fineart":
                c.update(channel="joint_0", expect="next_state")
            if c["dataset"] == "sharpadex" and not normal:
                c.update(expect="handover_step", window=[2.3, 2.9])
            if c["dataset"] == "astribot-s1" and not normal:
                c.update(expect="unresponsive_feedback")
            out.append(c)
    for i in (0, 1):
        key = f"nest_cups/season_POC22054_2026_05_23_15_03_06_train__episode_{i:06d}"
        out.append(
            dict(
                dataset="sharpadex",
                key=key,
                source=sharp_path(key),
                expect="persistent_hand_mismatch",
            )
        )
    for batch, uid, ep in [
        ("01", "b81cf533-d219-4c64-a0bd-4a8c2ec66b0b", 46),
        ("02", "528961a6-c5ea-43f9-a9c9-7d8557d227cd", 8),
    ]:
        key = f"A2D-{batch}/{uid}/episode_{ep:06d}"
        f = (
            RAW
            / f"vendor/VENDOR-000002/智元G1/A2D-{batch}/{uid}/data/chunk-000/episode_{ep:06d}.parquet"
        )
        out.append(
            dict(
                dataset="zhiyuan-g1",
                key=key,
                source=str(f),
                channel="arm_0",
                expect="unknown_command",
            )
        )
    return out


def stratified(rows, k):
    # One case per season before reusing a season; deterministic and independent of detector scores.
    key = lambda r: hashlib.sha256(("action-state-v1/" + r["key"]).encode()).digest()
    ordered = sorted(rows, key=key)
    seen = set()
    first = []
    rest = []
    for r in ordered:
        season = r["key"].split("__episode_")[0]
        (rest if season in seen else first).append(r)
        seen.add(season)
    return (first + rest)[:k]


def holdout_cases(count):
    excluded = json.loads((SHARPA / "audit/EXCLUDED.json").read_text())
    buckets = {"hand_mismatch": {}, "arm_handover": {}}
    for row in excluded:
        k = row["episode"]
        reason = row["reason"]
        window = row["window"]
        if k.split("/")[0] in DEV_TASKS or reason not in buckets:
            continue
        if reason == "arm_handover" and (
            window[0] > 4.5 or window[1] - window[0] < 0.2
        ):
            continue
        buckets[reason].setdefault(
            k,
            dict(
                dataset="sharpadex",
                key=k,
                window=window,
                expect="persistent_hand_mismatch"
                if reason == "hand_mismatch"
                else "handover_step",
            ),
        )
    samples = []
    for rows in buckets.values():
        samples += stratified(rows.values(), count)
    index = SHARPA / json.loads((SHARPA / "PACKAGE_READY.json").read_text())["index"]
    retained = {}
    used_keys = {r["key"] for r in samples}
    with index.open() as f:
        for line in f:
            r = json.loads(line)
            k = r["metadata"]["episode_key"]
            lo, hi = r["segment_time_s"]
            if k in used_keys or k.split("/")[0] in DEV_TASKS or hi - lo < 2:
                continue
            retained.setdefault(
                k,
                dict(
                    dataset="sharpadex", key=k, window=[lo, hi], expect="retained_all"
                ),
            )
            if len(retained) >= 1024:
                break
    samples += stratified(retained.values(), count)
    for case in samples:
        case["source"] = sharp_path(case["key"])
    return samples


def other_retained(count, skip):
    out = []
    for ds in ("astribot-s1", "lingyu", "roco_taskboard"):
        release = next(
            f.parent for f in (ROOT / "cache" / ds).glob("v*/PACKAGE_READY.json")
        )
        index = (
            release / json.loads((release / "PACKAGE_READY.json").read_text())["index"]
        )
        candidates = {}
        with index.open() as f:
            for line in f:
                r = json.loads(line)
                key = r["metadata"]["episode_key"]
                lo, hi = r["segment_time_s"]
                if (ds, key) in skip or hi - lo < 2:
                    continue
                candidates.setdefault(
                    key,
                    dict(
                        dataset=ds, key=key, window=[lo, hi], expect="retained_targets"
                    ),
                )
                if len(candidates) >= 512:
                    break
        for case in stratified(candidates.values(), count):
            key = case["key"]
            if ds == "astribot-s1":
                task, ep = key.rsplit("/", 1)
                d = RAW / "vendor/VENDOR-000002/交付-星尘S1-110小时/星尘S1数据" / task
                path = (
                    d
                    / "data"
                    / f"chunk-{int(ep.split('_')[-1]) // 1000:03d}"
                    / (ep + ".parquet")
                )
            elif ds == "lingyu":
                task, ep = key.rsplit("/", 1)
                d = RAW / "public/lingyu-realworld-robotics-dataset" / task
                path = (
                    d
                    / "data"
                    / f"chunk-{int(ep.split('_')[-1]) // 1000:03d}"
                    / (ep + ".parquet")
                )
            else:
                _, rest = key.split("/", 1)
                season, ep = rest.split("__episode_")
                d = RAW / "public/RoCo_TaskBoardAssembly" / season / "lerobotv2.1"
                path = (
                    d
                    / "data"
                    / f"chunk-{int(ep) // 1000:03d}"
                    / ("episode_" + ep + ".parquet")
                )
            case["source"] = str(path)
            out.append(case)
    return out


def overlap(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def evaluate(c):
    ds = c["dataset"]
    k = c["key"]
    p = preset(ds)
    ep = int(k.split("episode_")[-1])
    kwargs = {"single_episode": True} if ds == "astribot-s1" else {"episode_index": ep}
    report = run_parquet(p, c["source"], k, **kwargs)
    return score(c, report)


def score(c, r):
    expect = c["expect"]
    channel = c.get("channel")
    window = c.get("window")
    chs = r["channels"]
    vals = {channel: chs[channel]} if channel else chs
    details = {}
    if expect in (
        "normal",
        "normal_calibrated_gripper",
        "retained_all",
        "retained_targets",
    ):
        fractions = {}
        reasons = {}
        for name, x in vals.items():
            if expect == "retained_targets" and x["role"] != "target":
                continue
            area = sum(
                overlap(window, [y["start_s"], y["end_s"]])
                for y in x["command_mask_windows"]
            )
            if area > 1e-6:
                fractions[name] = round(area / (window[1] - window[0]), 6)
                reasons[name] = [
                    e
                    for e in x["events"]
                    if e["effect"] == "mask"
                    and (
                        e.get("scope") == "whole_group"
                        or overlap(window, [e["start_s"], e["end_s"]]) > 0
                    )
                ] or [x["reason"]]
        details.update(
            masked_fraction_by_channel=fractions, disagreement_reasons=reasons
        )
        passed = not fractions
    elif expect in ("next_state", "unknown_command"):
        x = next(iter(vals.values()))
        passed = x["command_rows"] == 0 and (
            x["route"] == "state_proxy" and x["alignment"]["signature"] == "next_state"
            if expect == "next_state"
            else x["route"] == "review"
        )
        details = dict(
            route=x["route"], reason=x["reason"], alignment=x.get("alignment")
        )
    elif expect == "persistent_hand_mismatch":
        hits = [
            (n, e) for n, v in vals.items() for e in v["events"] if e["code"] == expect
        ]
        details["masked_hands"] = sorted({e["group"] for n, e in hits})
        passed = bool(hits)
    else:
        hits = [
            (n, e)
            for n, v in vals.items()
            for e in v["events"]
            if e["code"] == expect
            and e["effect"] == "mask"
            and overlap(window, [e["start_s"], e["end_s"]]) > 0
        ]
        # Deduplicate overlaps across joints before evaluating window recall.
        pieces = sorted(
            (max(window[0], e["start_s"]), min(window[1], e["end_s"])) for _, e in hits
        )
        merged = []
        for a, b in pieces:
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(b, merged[-1][1])
            else:
                merged.append([a, b])
        fraction = sum(b - a for a, b in merged) / (window[1] - window[0])
        details = dict(
            reference_time_recall=round(fraction, 6), detected_intersections=merged
        )
        passed = fraction >= 0.5
    return dict(**c, passed=passed, details=details, profile_sha256=r["profile_sha256"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--samples",
        type=Path,
        default=SAMPLES,
        help="original chart CSVs, for revising the case list only",
    )
    ap.add_argument(
        "--case-list",
        type=Path,
        default=Path(__file__).parent / "validation/cases.json",
    )
    ap.add_argument(
        "--resample",
        action="store_true",
        help="regenerate frozen references from original sample/audit files",
    )
    ap.add_argument("--heldout", type=int, default=24)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "validation/real-replay.json",
    )
    args = ap.parse_args()
    if args.resample:
        cases = curated_cases(args.samples)
        n = len(cases)
        cases += holdout_cases(args.heldout)
        cases += other_retained(args.heldout, {(c["dataset"], c["key"]) for c in cases})
        atomic_json(args.case_list, dict(curated_count=n, cases=cases))
    else:
        frozen = json.loads(args.case_list.read_text())
        cases = frozen["cases"]
        n = frozen["curated_count"]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(evaluate, cases))
    counts = {}
    for i, r in enumerate(results):
        cohort = "curated" if i < n else "heldout_" + r["dataset"]
        r["cohort"] = cohort
        label = cohort + "/" + r["expect"]
        a = counts.setdefault(label, dict(passed=0, total=0))
        a["passed"] += int(r["passed"])
        a["total"] += 1
    atomic_json(
        args.out,
        dict(
            summary=counts,
            case_list_sha256=hashlib.sha256(args.case_list.read_bytes()).hexdigest(),
            code_sha256={
                p: hashlib.sha256((Path(__file__).parent / p).read_bytes()).hexdigest()
                for p in (
                    "action_state.py",
                    "contract.py",
                    "cache_roles.py",
                    "probe.py",
                    "profiles.py",
                    "run_operator.py",
                    "validate_real.py",
                )
            },
            cases=results,
            references=[
                str(SHARPA / "audit/EXCLUDED.json"),
                str(SHARPA / "PACKAGE_READY.json"),
            ],
            note="Curated replay and held-out agreement with cache audit, not independent physical ground truth.",
        ),
    )
    print(json.dumps(counts, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""CLI for a source-native, single-episode Parquet adapter. Also import run_parquet()."""

import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pyarrow.parquet as pq

from .action_state import analyze
from .contract import validate_profile


def atomic_json(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", dir=path.parent, encoding="utf-8", delete=False
        ) as f:
            tmp = f.name
            json.dump(result, f, ensure_ascii=False, allow_nan=False, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)


def run_parquet(
    profile,
    path,
    episode_id,
    *,
    episode_index=None,
    single_episode=False,
    keep_windows=None,
    start_s=None,
    end_s=None,
):
    if int(episode_index is not None) + int(single_episode) != 1:
        raise ValueError(
            "provide an episode_index or explicitly declare a single-episode file"
        )
    validate_profile(profile)
    tcol = profile.get("time_column", "timestamp")
    ecol = profile.get("episode_column", "episode_index")
    cols = {tcol}
    for c in profile["channels"]:
        for role in ("action", "state"):
            m = c[role]
            cols.add(m["column"])
            if m.get("valid_column"):
                cols.add(m["valid_column"])
    schema = pq.ParquetFile(path).schema_arrow.names
    if tcol not in schema:
        raise ValueError("time column missing")
    if single_episode and ecol in schema:
        cols.add(ecol)
    table = pq.read_table(
        path,
        columns=sorted(cols.intersection(schema)),
        filters=None if single_episode else [(ecol, "=", episode_index)],
    )
    if single_episode and ecol in schema and len(table[ecol].unique()) > 1:
        raise ValueError("single-episode mode received multiple episode IDs")
    t = np.asarray(table[tcol].to_numpy(), dtype=float)
    arrays = {}

    def column(name, component):
        if name not in table.column_names:
            return None
        if name not in arrays:
            arr = table[name].combine_chunks()
            if len(arr) == 0:
                raise ValueError("episode selection is empty")
            # Arrow fixed/variable lists, including nullable rows; preserve NaNs for missing components.
            import pyarrow as pa

            if (
                pa.types.is_list(arr.type)
                or pa.types.is_large_list(arr.type)
                or pa.types.is_fixed_size_list(arr.type)
            ):
                raw = arr.to_pylist()
                widths = {len(x) for x in raw if x is not None}
                if len(widths) > 1:
                    raise ValueError(f"ragged source column: {name}")
                width = next(iter(widths), 0)
                arrays[name] = np.asarray(
                    [([None] * width if x is None else x) for x in raw], dtype=float
                )
            else:
                arrays[name] = arr.to_numpy(zero_copy_only=False)
        ar = arrays[name]
        if ar.ndim == 1:
            if component is not None:
                raise ValueError(f"{name} is scalar")
            return ar
        if (
            ar.ndim != 2
            or type(component) is not int
            or component < 0
            or component >= ar.shape[1]
        ):
            raise ValueError(f"invalid component in {name}")
        return ar[:, component]

    signals = {}
    for c in profile["channels"]:
        pair = signals[c["name"]] = {}
        for role in ("action", "state"):
            m = c[role]
            pair[role] = column(m["column"], m.get("component"))
            if m.get("valid_column"):
                flag = column(m["valid_column"], m.get("valid_component"))
                if flag is None:
                    raise ValueError("declared validity column missing")
                if not np.all(
                    np.isin(flag, [False, True, 0, 1, None])
                    | np.asarray([v != v for v in flag])
                ):
                    raise ValueError("validity flag must be boolean or 0/1")
                pair[role + "_valid"] = flag == 1
    keep = None
    if keep_windows is not None:
        keep = np.zeros(len(t), bool)
        for lo, hi in keep_windows:
            if not np.isfinite([lo, hi]).all() or hi <= lo:
                raise ValueError("invalid upstream keep interval")
            keep |= (t >= lo) & (t < hi)
    result = analyze(
        episode_id, t, signals, profile, keep=keep, start_s=start_s, end_s=end_s
    )
    result["source"] = {
        "path": str(path),
        "episode_index": episode_index,
        "single_episode": single_episode,
    }
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    from .profiles import NAMES, preset

    configs = ap.add_mutually_exclusive_group(required=True)
    configs.add_argument("--profile", type=Path)
    configs.add_argument("--preset", choices=NAMES)
    ap.add_argument("--parquet", type=Path, required=True)
    ap.add_argument("--episode-id", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--episode-index", type=int)
    g.add_argument("--single-episode", action="store_true")
    ap.add_argument(
        "--keep-json", type=Path, help="list of [start_s,end_s), on source clock"
    )
    ap.add_argument(
        "--start-s", type=float, help="true episode start when input is a slice"
    )
    ap.add_argument(
        "--end-s", type=float, help="true exclusive episode end when input is a slice"
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    profile = (
        json.loads(args.profile.read_text()) if args.profile else preset(args.preset)
    )
    result = run_parquet(
        profile,
        args.parquet,
        args.episode_id,
        episode_index=args.episode_index,
        single_episode=args.single_episode,
        start_s=args.start_s,
        end_s=args.end_s,
        keep_windows=json.loads(args.keep_json.read_text()) if args.keep_json else None,
    )
    atomic_json(args.out, result)


if __name__ == "__main__":
    main()

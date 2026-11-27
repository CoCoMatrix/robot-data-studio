"""Single-episode action/state quality operator. No writes and no GPU dependencies."""

from dataclasses import asdict
from hashlib import sha256
import json
import numpy as np
from .contract import (
    VERSION,
    UNIT,
    MappingError,
    check_anchors,
    convert,
    limits,
    validate_profile,
    vector,
)
from .probe import probe

# Probe verdicts that contradict a declared command target. Exact copies are caught by the signature.
SEMANTIC_GATE = {
    "same_state_noisy",
    "future_state",
    "past_state",
    "action_lags_state",
    "sign_flip",
    "unit_mismatch",
    "velocity_like",
    "constant_action",
    "unrelated",
}
SEMANTIC_KEYS = (
    "verdict",
    "detail",
    "lead_frames",
    "lead_s",
    "lead_r2",
    "level_corr",
    "level_slope",
    "offset",
    "error_mad",
    "active_rows",
)
# Fast grippers track within a frame and binary commands defeat derivative tests; contact decouples fingers.
APERTURE_SOFT = {"same_state_noisy", "velocity_like", "unrelated"}
CONTACT_SOFT = {"unrelated", "same_state_noisy", "velocity_like"}
PROXY_MIN_FRACTION = 0.95
SETTLE_S = 2.0


def runs(mask, breaks):
    """Half-open runs of true rows, split on real clock gaps."""
    starts = np.flatnonzero(mask & (np.r_[True, ~mask[:-1]] | breaks))
    ends = np.flatnonzero(mask & (np.r_[~mask[1:], True] | np.r_[breaks[1:], True])) + 1
    return list(zip(starts.tolist(), ends.tolist()))


def alignment(a, s, spans, lim, dt, t):
    # Differences suppress spurious correlations between two slowly drifting levels.
    # Pool only within contiguous runs; lags are restricted to near-uniform cadence.
    lagmax = min(48, int(lim.probe_s / dt))
    cov = np.zeros((2 * lagmax + 1, 6))
    count = np.zeros(2, int)
    copied = np.zeros(2, int)
    sum_error = np.zeros(2)
    moving = 0
    active_error = []
    action_steps = []
    for i, j in spans:
        aa, ss = a[i:j], s[i:j]
        ds = np.diff(ss)
        da = np.diff(aa)
        active = (np.abs(ds) > lim.still * 0.15) | (np.abs(da) > lim.still * 0.15)
        moving += int(active.sum())
        if active.any():
            active_error.append(abs(aa[:-1] - ss[:-1])[active])
            action_steps.append(abs(da[active]))
        for k, ref in enumerate((ss[:-1], ss[1:])):
            err = np.abs(aa[:-1] - ref)[active]
            count[k] += len(err)
            copied[k] += np.count_nonzero(err <= max(1e-7, lim.still * 0.0005))
            sum_error[k] += err.sum()
        if j - i < 8 or np.max(np.abs(np.diff(t[i:j]) - dt)) > 0.15 * dt:
            continue
        for idx, lag in enumerate(range(-lagmax, lagmax + 1)):
            if abs(lag) + 4 >= len(da):
                continue
            if lag < 0:
                x, y = da[-lag:], ds[:lag]
            elif lag > 0:
                x, y = da[:-lag], ds[lag:]
            else:
                x, y = da, ds
            use = (abs(x) > lim.still * 0.15) | (abs(y) > lim.still * 0.15)
            x, y = x[use], y[use]
            if len(x):
                cov[idx] += (len(x), x.sum(), y.sum(), x @ x, y @ y, x @ y)
    frac = copied / np.maximum(count, 1)
    mean = sum_error / np.maximum(count, 1)
    signature = "undetermined"
    if moving >= 20:
        for k, label in enumerate(("same_state", "next_state")):
            if frac[k] >= 0.995 and frac[1 - k] < 0.80:
                signature = label
    scores = []
    for n, x, y, xx, yy, xy in cov:
        denom = np.sqrt(max(0, n * xx - x * x) * max(0, n * yy - y * y))
        scores.append(
            float(np.clip((n * xy - x * y) / denom, -1, 1))
            if n >= 20 and denom > 1e-14
            else None
        )
    eligible = [i for i, c in enumerate(scores) if c is not None]
    best = (
        max(eligible, key=lambda i: (scores[i], -abs(i - lagmax))) if eligible else None
    )
    strongest = (
        max(eligible, key=lambda i: (abs(scores[i]), -abs(i - lagmax)))
        if eligible
        else None
    )
    return dict(
        active_transitions=moving,
        strongest_signed_correlation=scores[strongest]
        if strongest is not None
        else None,
        strongest_signed_lag_s=(strongest - lagmax) * dt
        if strongest is not None
        else None,
        signature=signature,
        active_error_p95=float(np.quantile(np.concatenate(active_error), 0.95))
        if moving
        else None,
        active_action_step_p999=float(np.quantile(np.concatenate(action_steps), 0.999))
        if moving
        else None,
        same_exact_fraction=float(frac[0]) if moving else None,
        next_exact_fraction=float(frac[1]) if moving else None,
        same_mae=float(mean[0]) if moving else None,
        next_mae=float(mean[1]) if moving else None,
        best_lag_s=(best - lagmax) * dt if best is not None else None,
        delta_correlation=scores[best] if best is not None else None,
        lag_at_search_limit=bool(
            best is not None and lagmax and best in (0, 2 * lagmax)
        ),
    )


def aperture_evidence(a, s, t, spans, lim):
    """Use settled commands with observable motion. Contact with no motion casts no vote."""
    votes = []
    excursion_counts = 0
    command_low = np.inf
    command_high = -np.inf
    for i, j in spans:
        if j - i < 2:
            continue
        aa = a[i:j]
        ss = s[i:j]
        command_low = min(command_low, float(aa.min()))
        command_high = max(command_high, float(aa.max()))
        last = None
        # Treat captures already at one endpoint as the starting state, not an event.
        for k in range(i, j):
            side = 1 if a[k] >= 0.85 else 0 if a[k] <= 0.15 else None
            if side is None or side == last:
                continue
            if last is not None:
                stop = min(j, int(np.searchsorted(t, t[k] + max(0.65, lim.response_s))))
                if stop - k >= 3 and t[stop - 1] - t[k] >= lim.response_s * 0.8:
                    dwell = a[k:stop]
                    settled = np.all(dwell >= 0.7) if side else np.all(dwell <= 0.3)
                    pre = max(i, int(np.searchsorted(t, t[k] - 0.18)))
                    baseline = s[pre] if pre == k else np.median(s[pre:k])
                    change = float(np.median(s[max(k, stop - 3) : stop]) - baseline)
                    if settled and abs(change) >= 0.20:
                        votes.append((1 if side else -1) * change)
            last = side
        level = None
        for v in ss:
            current = 1 if v >= 0.65 else 0 if v <= 0.35 else None
            if current is not None and current != level:
                excursion_counts += int(level is not None)
                level = current
    return dict(
        aligned=sum(v > 0 for v in votes),
        opposite=sum(v < 0 for v in votes),
        feedback_traversals=excursion_counts,
        command_range=float(command_high - command_low)
        if command_high >= command_low
        else None,
    )


def edge_windows(a, s, t, i, j, lim, start, end, policy):
    """Only a new discrepancy at an episode edge can trigger a handover cut."""
    err = abs(a[i:j] - s[i:j])
    candidate = (abs(np.diff(a[i:j])) > lim.jump) & (
        abs(np.diff(s[i:j])) < lim.state_jump
    )
    if policy.get("require_error_growth", True):
        candidate &= np.diff(err) > lim.error_growth
    hits = np.flatnonzero(candidate) + i + 1
    for h in hits:
        front = t[h] - start <= lim.edge_s
        tail = end - t[h] <= lim.edge_s
        if not (front or (tail and policy.get("include_tail", True))):
            continue
        lo = max(i, int(np.searchsorted(t, t[h] - lim.pad_s - 1e-9)))
        if "fixed_after_s" in policy:
            hi = min(
                j,
                max(
                    h + 1,
                    int(np.searchsorted(t, t[h] + policy["fixed_after_s"] - 1e-9)),
                ),
            )
            yield (
                lo,
                hi,
                dict(
                    event_s=float(t[h]),
                    action_step=float(a[h] - a[h - 1]),
                    state_step=float(s[h] - s[h - 1]),
                    window_rule="fixed",
                ),
            )
            continue
        caught = j
        for lo, hi in runs(
            abs(a[h:j] - s[h:j]) <= lim.catch_error, np.zeros(j - h, bool)
        ):
            if t[h + hi - 1] - t[h + lo] + 1e-9 >= lim.catch_s:
                caught = h + lo
                break
        lo = max(i, int(np.searchsorted(t, t[h] - lim.pad_s - 1e-9)))
        yield (
            lo,
            max(h + 1, caught),
            dict(
                event_s=float(t[h]),
                action_step=float(a[h] - a[h - 1]),
                state_step=float(s[h] - s[h - 1]),
                caught=bool(caught < j),
            ),
        )


def stalls(a, s, t, i, j, lim):
    """Search for quiet feedback over sustained, changing, substantially different targets."""
    # Sample window starts on the native grid at ~1/4 window strides, never across gaps.
    width = max(lim.response_s * 1.5, lim.stall_s)
    lo = i
    while lo < j:
        hi = int(np.searchsorted(t, t[lo] + width, side="left")) + 1
        if hi > j:
            break
        aa, ss = a[lo:hi], s[lo:hi]
        excursion = float(np.ptp(aa))
        drift = float(np.ptp(ss))
        median_error = float(np.median(abs(aa - ss)))
        # A last-moment target change has not yet had time to produce feedback.
        early = int(np.searchsorted(t, t[hi - 1] - lim.response_s, side="right"))
        early_move = float(np.ptp(a[lo:early])) if early > lo + 1 else 0.0
        if (
            excursion >= lim.move
            and early_move >= 0.25 * lim.move
            and drift <= lim.still
            and median_error >= lim.error
        ):
            yield (
                lo,
                hi,
                dict(
                    action_range=excursion, state_range=drift, median_error=median_error
                ),
            )
        lo = max(lo + 1, int(np.searchsorted(t, t[lo] + width / 4)))


def unanswered_steps(a, s, t, i, j, lim):
    """A target left a previously tracked position and stayed away beyond the response allowance."""
    error = abs(a[i:j] - s[i:j])
    departures = (
        np.flatnonzero((error[:-1] <= lim.error) & (error[1:] > lim.error)) + i + 1
    )
    for h in departures:
        before = max(i, int(np.searchsorted(t, t[h] - lim.stall_s)))
        # Require a witnessed transition out of a previously tracked state.
        good = np.flatnonzero(error[before - i : h - i] < lim.error / 2)
        if not len(good):
            continue
        ref = before + good[-1]
        stop = int(np.searchsorted(t, t[h] + lim.response_s + lim.stall_s)) + 1
        if stop > j or not np.all(error[h - i : stop - i] > lim.error):
            continue
        feedback = float(np.ptp(s[ref:stop]))
        demand = float(np.max(abs(a[h:stop] - a[ref])))
        if demand >= lim.move and feedback <= lim.still:
            yield (
                h,
                stop,
                dict(
                    action_range=demand,
                    state_range=feedback,
                    median_error=float(np.median(error[h - i : stop - i])),
                    pattern="step_and_hold",
                ),
            )


def analyze(
    episode_id, time_s, signals, profile, *, keep=None, end_s=None, start_s=None
):
    """signals[name] contains raw action/state vectors and optional boolean validity vectors."""
    validate_profile(profile)
    t = np.asarray(time_s, dtype=float)
    if t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError(
            "expected >=2 strictly increasing finite native episode timestamps"
        )
    n = len(t)
    dt = float(np.median(np.diff(t)))
    begin = float(t[0] if start_s is None else start_s)
    end = float(t[-1] + dt if end_s is None else end_s)
    if not (np.isfinite([begin, end]).all() and begin <= t[0] and end > t[-1]):
        raise ValueError("episode source bounds do not enclose timestamps")
    gap = profile.get("max_gap_s", 2.5 * dt)
    if not np.isfinite(gap) or gap <= 0:
        raise ValueError("invalid max_gap_s")
    breaks = np.r_[True, np.diff(t) > gap]
    row_end = np.r_[
        np.where(breaks[1:], t[:-1] + np.minimum(np.diff(t), dt), t[1:]),
        min(end, t[-1] + dt),
    ]
    keep = np.ones(n, bool) if keep is None else vector(keep, n, "keep", bool)

    def interval(i, j):
        return {
            "start_s": float(t[i]),
            "end_s": float(row_end[j - 1]),
            "rows": [int(i), int(j)],
        }

    def intervals(mask):
        return [interval(i, j) for i, j in runs(mask, breaks)]

    result = {}
    masks = {}
    group_data = {}
    edge_mask = np.zeros(n, bool)
    for c in profile["channels"]:
        name = c["name"]
        kind = c["kind"]
        role = c["role"]
        lim = limits(
            kind,
            {**profile.get("thresholds", {}).get(kind, {}), **c.get("thresholds", {})},
        )
        flags = []
        row = result[name] = dict(
            role=role,
            kind=kind,
            unit=UNIT[kind],
            events=flags,
            thresholds=asdict(lim),
            mapping_evidence=c.get("evidence"),
        )
        raw = signals.get(name, {})
        policy = np.zeros(n, bool)
        masks[name] = policy
        row["route"] = "review"
        row["usable_proxy_windows"] = []
        if not all(key in raw and raw[key] is not None for key in ("action", "state")):
            row["reason"] = "missing_source"
            continue
        act = vector(raw["action"], n, name + ".action")
        state = vector(raw["state"], n, name + ".state")
        valid = keep & np.isfinite(act) & np.isfinite(state)
        for k in ("action_valid", "state_valid"):
            if k in raw:
                valid &= vector(raw[k], n, k, bool)
        if role == "unsupported":
            row["reason"] = "noncomparable_quantity"
            continue
        try:
            check_anchors(c)
            a = convert(act, c["action"], kind)
            s = convert(state, c["state"], kind)
        except MappingError as err:
            row.update(reason="mapping_unverified", detail=str(err))
            continue
        if kind == "aperture" and np.count_nonzero(valid) >= 20:
            diagnostics = {}
            for key, raw_values in (("action", act), ("state", state)):
                points = c[key].get("points")
                lo, hi = (
                    (points[0][0], points[-1][0]) if points is not None else (0.0, 1.0)
                )
                margin = 0.5 * (hi - lo)
                vals = raw_values[valid]
                diagnostics[key] = float(
                    np.mean((vals < lo - margin) | (vals > hi + margin))
                )
            row["gross_out_of_calibration_fraction"] = diagnostics
            if max(diagnostics.values()) > 0.10:
                row["reason"] = "gripper_scale_conflict"
                continue
        valid &= np.isfinite(a) & np.isfinite(s)
        if kind == "aperture":
            valid &= (a >= 0) & (a <= 1) & (s >= 0) & (s <= 1)
        spans = runs(valid, breaks)
        row["invalid_source_windows"] = intervals(keep & ~valid)
        row["valid_rows"] = int(valid.sum())
        metrics = row["alignment"] = alignment(a, s, spans, lim, dt, t)
        inv = metrics["strongest_signed_correlation"]
        if (
            inv is not None
            and inv < -0.85
            and metrics["delta_correlation"] < 0.6
            and metrics["active_transitions"] >= 40
        ):
            flags.append(
                dict(
                    code="inverse_trend_candidate",
                    effect="review",
                    scope="channel",
                    evidence={
                        "correlation": inv,
                        "lag_s": metrics["strongest_signed_lag_s"],
                    },
                )
            )
        if not valid.any():
            row["reason"] = "no_valid_rows"
            continue
        sig = metrics["signature"]
        if role in ("same_state", "next_state"):
            if not c.get("evidence") or (sig != "undetermined" and sig != role):
                row["reason"] = "unverified_proxy"
                continue
            # Verify the declared alignment directly. Do not cross gaps or upstream masks.
            ok = np.zeros(n, bool)
            tolerance = c.get("proxy_tolerance", max(1e-7, lim.still * 0.0005))
            if not np.isfinite(tolerance) or tolerance < 0:
                raise ValueError("invalid proxy_tolerance")
            lag = int(role == "next_state")
            checkable = np.zeros(n, bool)
            for i, j in spans:
                if j - i > lag:
                    ok[i : j - lag] = abs(a[i : j - lag] - s[i + lag : j]) <= tolerance
                    checkable[i : j - lag] = True
            fraction = float(ok.sum() / max(1, checkable.sum()))
            agrees = fraction >= c.get("proxy_min_fraction", PROXY_MIN_FRACTION)
            row.update(
                route="state_proxy" if agrees else "review",
                reason=role if agrees else "declared_proxy_disagrees",
                proxy_match_fraction=fraction,
                usable_proxy_windows=intervals(ok) if agrees else [],
                proxy_mask_windows=intervals(keep & ~ok) if agrees else intervals(keep),
            )
            continue
        if role == "target" and c.get("evidence") and sig == "undetermined":
            # Start-up motion toward the initial pose belongs to the handover rules, not to channel semantics.
            settled = valid & (t >= begin + SETTLE_S)
            sem = probe(t, a, s, kind, settled if settled.sum() >= 0.5 * valid.sum() else valid)
            row["semantic"] = {k: sem.get(k) for k in SEMANTIC_KEYS if k in sem}
            accepted = c.get("accept_semantic", {})
            verdict = sem["verdict"]
            gate = SEMANTIC_GATE - (APERTURE_SOFT if kind == "aperture" else set())
            if c.get("contact_possible"):
                gate = gate - CONTACT_SOFT
            if verdict in SEMANTIC_GATE - gate:
                flags.append(dict(code="semantic_" + verdict, effect="review", scope="channel", evidence=row["semantic"]))
            elif verdict in gate and not (verdict in accepted.get("verdicts", []) and accepted.get("evidence")):
                row["reason"] = "semantic_" + verdict
                if c.get("group"):
                    group_data[name] = (c["group"], valid, a, s)
                continue
        if role != "target" or not c.get("evidence") or sig != "undetermined":
            row["reason"] = (
                "copy_candidate" if sig != "undetermined" else "unverified_command"
            )
            continue
        if kind == "aperture":
            ge = row["gripper_evidence"] = aperture_evidence(a, s, t, spans, lim)
            if ge["opposite"] >= 2 and ge["aligned"] == 0:
                row["reason"] = "gripper_direction_conflict"
                continue
            if (
                ge["command_range"] is not None
                and ge["command_range"] < 0.01
                and ge["feedback_traversals"] >= 2
            ):
                row["reason"] = "constant_gripper_command_with_feedback_cycles"
                continue
        policy[:] = valid
        row.update(route="command", reason="declared_target")
        # A disagreeing gripper can be holding an object. Joint handover cuts are opt-in.
        edge = c.get("edge", "review") if kind in ("arm", "linear") else "off"
        stalled = np.zeros(n, bool)
        peak = {}
        for i, j in spans:
            if edge != "off":
                for l, h, evidence in edge_windows(
                    a, s, t, i, j, lim, begin, end, c.get("edge_policy", {})
                ):
                    flags.append(
                        dict(
                            code="handover_step",
                            effect=edge,
                            **interval(l, h),
                            evidence=evidence,
                        )
                    )
                    if edge == "mask":
                        policy[l:h] = False
                        edge_mask[l:h] = True
            for detector in (stalls, unanswered_steps):
                for l, h, evidence in detector(a, s, t, i, j, lim):
                    stalled[l:h] = True
                    peak[l] = evidence
        # Continue an established failure through a held target until feedback moves,
        # the discrepancy clears, or a validity/clock boundary is encountered.
        for l, h in runs(stalled, breaks):
            k = h
            while (
                k < n
                and valid[k]
                and not breaks[k]
                and not stalled[k]
                and abs(s[k] - s[h - 1]) <= 2 * lim.still
                and abs(a[k] - s[k]) > lim.error
            ):
                stalled[k] = True
                k += 1
        for i, j in runs(stalled, breaks):
            effect = (
                "review"
                if kind == "aperture" or c.get("contact_possible", False)
                else "mask"
            )
            flags.append(
                dict(
                    code="unresponsive_feedback",
                    effect=effect,
                    **interval(i, j),
                    evidence=peak[i],
                )
            )
            if effect == "mask":
                policy[i:j] = False
        if c.get("group"):
            group_data[name] = (c["group"], valid, a, s)
    # Group rule uses pre-stall validity; removing a related arm handover must not erase other evidence.
    for g, rule in profile.get("hand_groups", {}).items():
        group_channels = [c["name"] for c in profile["channels"] if c.get("group") == g]
        errs = {}
        for name, (grp, valid, a, s) in group_data.items():
            if grp != g:
                continue
            use = valid & ~edge_mask
            if np.sum((row_end - t)[use]) >= rule["min_s"]:
                errs[name] = float(np.median(abs(a[use] - s[use])))
        hits = [name for name, v in errs.items() if v > rule["error"]]
        if len(hits) >= rule["min_channels"]:
            for name in group_channels:
                result[name]["events"].append(
                    dict(
                        code="persistent_hand_mismatch",
                        effect="mask",
                        scope="whole_group",
                        group=g,
                        offending_channels=hits,
                        median_error=errs,
                        threshold=rule["error"],
                        evidence=rule["evidence"],
                    )
                )
                masks[name][:] = False
    for name, r in result.items():
        r["command_keep_windows"] = intervals(masks[name])
        r["command_mask_windows"] = intervals(keep & ~masks[name])
        r["command_rows"] = int(masks[name].sum())
    count = sum(x["command_rows"] for x in result.values())
    original = int(keep.sum()) * len(result)
    return dict(
        operator="action_state",
        version=VERSION,
        dataset=profile["dataset"],
        episode_id=episode_id,
        profile_sha256=sha256(
            json.dumps(profile, sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
        source_rows=n,
        native_dt_s=dt,
        source_keep_windows=intervals(keep),
        channels=result,
        command_disposition="no_command"
        if count == 0
        else "keep"
        if count == original
        else "keep_masked",
        command_rows=count,
    )


def usable_mask(report, names, *, supervision="command"):
    """Native row mask in caller-selected channel order; proxies use a separate target."""
    if supervision not in ("command", "state_proxy"):
        raise ValueError("unknown supervision")
    if len(set(names)) != len(names):
        raise ValueError("duplicate channels")
    result = np.zeros((report["source_rows"], len(names)), dtype=bool)
    field = (
        "command_keep_windows" if supervision == "command" else "usable_proxy_windows"
    )
    for column, name in enumerate(names):
        for window in report["channels"][name][field]:
            lo, hi = window["rows"]
            result[lo:hi, column] = True
    return result

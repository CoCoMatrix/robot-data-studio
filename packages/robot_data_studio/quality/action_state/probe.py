"""Declaration-free action/state semantic probe. Reads arrays only; CPU; no writes.

For one (action, state) pair on a native clock it estimates what the action column
physically is relative to the state column, without trusting the declared role:
exact or noisy state copy at some shift, causal command with an implied lead,
sign or unit mismatch, constant offset, pose/velocity confusion, or no command.
`pairing` searches all action/state columns for the best partner of each column.
"""

import numpy as np

# Speeds (physical units per second) below which a joint is treated as at rest.
REST_SPEED = {
    "arm": 0.03,
    "hand": 0.06,
    "aperture": 0.20,
    "linear": 0.0075,
    "unknown": None,
}
# Median |action - state| below which a lead-free action carries no information beyond the state.
COPY_NOISE = {"arm": 5e-4, "hand": 1e-3, "aperture": 5e-3, "linear": 1.5e-4}
# Constant offsets above these (physical units) are reported as calibration zero mismatch.
OFFSET = {"arm": 0.05, "hand": 0.10, "aperture": 0.15, "linear": 0.01}
UNIT_FACTORS = {
    "deg_vs_rad": 180 / np.pi,
    "rad_vs_deg": np.pi / 180,
    "mm_vs_m": 1000.0,
    "m_vs_mm": 0.001,
    "cm_vs_m": 100.0,
    "m_vs_cm": 0.01,
    "percent_vs_fraction": 100.0,
    "fraction_vs_percent": 0.01,
}


def _runs(valid, t, gap):
    breaks = np.r_[True, np.diff(t) > gap]
    edges = np.flatnonzero(np.diff(np.r_[0, valid.astype(int), 0]))
    out = []
    for i, j in zip(edges[::2], edges[1::2]):
        cut = [i] + [k for k in range(i + 1, j) if breaks[k]] + [j]
        out += [(a, b) for a, b in zip(cut[:-1], cut[1:]) if b - a >= 3]
    return out


def _shift_pairs(spans, k):
    """Index arrays (x, y) with y = x + k inside the same contiguous span."""
    xs = [np.arange(i, j - k) if k >= 0 else np.arange(i - k, j) for i, j in spans]
    xs = [x for x in xs if len(x)]
    if not xs:
        return np.zeros(0, int), np.zeros(0, int)
    x = np.concatenate(xs)
    return x, x + k


def probe(t, a, s, kind="unknown", valid=None, *, max_shift_s=1.0, min_active=30):
    t = np.asarray(t, float)
    a = np.asarray(a, float)
    s = np.asarray(s, float)
    n = len(t)
    dt = float(np.median(np.diff(t)))
    ok = np.isfinite(a) & np.isfinite(s)
    if valid is not None:
        ok &= np.asarray(valid, bool)
    spans = _runs(ok, t, 2.5 * dt)
    out = dict(kind=kind, dt=dt, valid_rows=int(ok.sum()))
    if not spans:
        return dict(out, verdict="no_valid_rows")
    v = np.full(n, np.nan)
    for i, j in spans:
        v[i:j] = np.gradient(s[i:j]) / dt
    a_ok, s_ok = a[ok], s[ok]
    a_ptp, s_ptp = float(np.ptp(a_ok)), float(np.ptp(s_ok))
    rest = REST_SPEED.get(kind) or max(1e-9, 0.02 * s_ptp)
    moving = ok & (np.abs(v) > rest)
    out.update(
        action_ptp=a_ptp,
        state_ptp=s_ptp,
        active_rows=int(moving.sum()),
    )
    if a_ptp <= 1e-9 and moving.sum() >= min_active:
        return dict(out, verdict="constant_action", action_value=float(np.median(a_ok)))
    if moving.sum() < min_active:
        return dict(out, verdict="insufficient_motion")

    # Exact copy of the state at an integer shift, judged only on moving rows.
    K = max(1, int(round(max_shift_s / dt)))
    tol = max(1e-6, 1e-5 * s_ptp)
    exact = {}
    for k in range(-K, K + 1):
        x, y = _shift_pairs(spans, k)
        m = moving[x] & moving[y]
        if m.sum() >= min_active:
            exact[k] = float(np.mean(np.abs(a[x[m]] - s[y[m]]) <= tol))
    best_k = max(exact, key=exact.get) if exact else None
    out["exact_shift"] = best_k
    out["exact_fraction"] = exact.get(best_k) if exact else None

    # Implied lead: on moving rows the tracking error is lead * velocity + offset.
    e = a - s
    X = np.c_[v[moving] * dt, np.ones(moving.sum())]
    coef, *_ = np.linalg.lstsq(X, e[moving], rcond=None)
    resid = e[moving] - X @ coef
    var = float(np.var(e[moving]))
    r2 = 1 - float(np.var(resid)) / var if var > 0 else 1.0
    step = float(np.median(np.abs(v[moving]) * dt))
    xc = X[:, 0] - X[:, 0].mean()
    se = float(np.sqrt(np.var(resid) / max(1e-30, xc @ xc)))
    out.update(
        lead_frames=float(coef[0]),
        lead_s=float(coef[0] * dt),
        lead_r2=r2,
        lead_t=float(coef[0] / se) if se > 0 else float("inf"),
        offset=float(coef[1]),
        residual_mad=float(np.median(np.abs(resid))),
        residual_in_steps=float(np.median(np.abs(resid)) / step) if step else None,
        error_mad=float(np.median(np.abs(e[ok]))),
    )

    # Level and velocity relations, each at the shift that maximises |correlation|.
    def best_corr(target, limit=K):
        """Strongest positive and strongest negative correlation over shifts, with their slopes."""
        pos, neg = (None, None, None), (None, None, None)
        for k in range(-limit, limit + 1):
            x, y = _shift_pairs(spans, k)
            ax, ty = a[x], target[y]
            fin = np.isfinite(ty)
            ax, ty = ax[fin], ty[fin]
            if len(ax) < min_active or np.std(ax) == 0 or np.std(ty) == 0:
                continue
            c = float(np.corrcoef(ax, ty)[0, 1])
            item = (c, k, float(np.sign(c) * np.std(ax) / np.std(ty)))
            if pos[0] is None or c > pos[0]:
                pos = item
            if neg[0] is None or c < neg[0]:
                neg = item
        return pos, neg

    (corr, shift, slope), _ = best_corr(s)
    # Sign is judged over short shifts only: for periodic motion a half-period shift mimics a sign flip.
    near = max(1, int(round(min(max_shift_s, 0.5) / dt)))
    (ncorr_pos, _, _), (ncorr, nshift, nslope) = best_corr(s, near)
    out.update(level_corr=corr, level_shift=shift, level_slope=slope, near_level_corr=ncorr_pos,
               neg_level_corr=ncorr, neg_level_shift=nshift, neg_level_slope=nslope)
    (vcorr, vshift, _), (nvcorr, _, _) = best_corr(v)
    if nvcorr is not None and (vcorr is None or -nvcorr > vcorr):
        vcorr = nvcorr
    out.update(velocity_corr=vcorr, velocity_shift=vshift)
    out["verdict"], out["detail"] = _decide(out, kind)
    return out


def _decide(r, kind):
    ef, ek = r.get("exact_fraction"), r.get("exact_shift")
    if ef is not None and ef >= 0.95:
        name = {0: "same_state", 1: "next_state"}.get(ek)
        if name:
            return name, f"exact copy on {ef:.1%} of moving rows"
        return ("future_state" if ek > 1 else "past_state"), f"exact copy at shift {ek}"
    corr, slope = r.get("level_corr"), r.get("level_slope")
    ncorr = r.get("neg_level_corr")
    # Oscillation can produce a negative correlation at some shift; a sign error has no positive match.
    near = r.get("near_level_corr")
    if ncorr is not None and ncorr <= -0.8 and (near is None or near < 0.3):
        return "sign_flip", f"level correlation {ncorr:.2f} at shift {r['neg_level_shift']}, best positive {near}"
    vc = r.get("velocity_corr")
    if vc is not None and abs(vc) >= 0.8 and (corr is None or (corr < 0.7 and abs(vc) > corr + 0.1)):
        return "velocity_like", f"action follows d(state)/dt (corr {vc:.2f}) not level ({corr})"
    if corr is None or corr < 0.3:
        return "unrelated", f"no positive level relation (best {corr})"
    if corr is not None and corr >= 0.8 and slope:
        for name, f in UNIT_FACTORS.items():
            if abs(np.log(abs(slope) / f)) < np.log(1.25):
                return "unit_mismatch", f"{name}: slope {slope:.4g}"
        if abs(np.log(abs(slope))) > np.log(1.5):
            return "scale_mismatch", f"slope {slope:.3g}"
    lead, r2, tstat = r["lead_frames"], r["lead_r2"], r["lead_t"]
    noise = COPY_NOISE.get(kind)
    # A state copy plus small noise: no lead and an error unrelated to motion, below sensor noise.
    if abs(lead) < 0.15 and r2 < 0.1 and noise is not None and r["error_mad"] < noise:
        return "same_state_noisy", f"lead {lead:.2f} frames, R2 {r2:.2f}, |a-s| {r['error_mad']:.2g}"
    if r.get("level_shift") is not None and r["level_shift"] <= -2 and corr >= 0.9 and lead < 0:
        return "action_lags_state", f"best level match at shift {r['level_shift']}: action follows the measured state"
    if lead >= 0.2 and tstat >= 8:
        off = r["offset"]
        if noise is not None and abs(off) > max(OFFSET.get(kind, np.inf), 3 * r["residual_mad"]):
            return "command_with_offset", f"lead {lead:.2f} frames, constant offset {off:.3g}"
        return "command", f"lead {lead:.2f} frames ({r['lead_s']:.3f}s), R2 {r2:.2f}"
    return "undetermined", f"lead {lead:.2f} (t {tstat:.1f}), R2 {r2:.2f}, corr {corr}"


def pairing(t, A, S, kind="unknown", valid=None, max_lag=8):
    """For each action column, the state column whose increments best explain it at a small lag."""
    t = np.asarray(t, float)
    A = np.asarray(A, float)
    S = np.asarray(S, float)
    ok = np.isfinite(A).all(1) & np.isfinite(S).all(1)
    if valid is not None:
        ok &= valid
    dA = np.diff(A, axis=0)[ok[1:] & ok[:-1]]
    dS = np.diff(S, axis=0)[ok[1:] & ok[:-1]]
    best = np.full((A.shape[1], S.shape[1]), np.nan)
    for lag in range(0, max_lag + 1):
        x = dA[: len(dA) - lag] if lag else dA
        y = dS[lag:]
        xs, ys = x.std(0), y.std(0)
        c = ((x - x.mean(0)).T @ (y - y.mean(0))) / len(x)
        with np.errstate(invalid="ignore", divide="ignore"):
            c = c / np.outer(xs, ys)
        c[~np.isfinite(c)] = np.nan
        best = np.where(np.isnan(best) | (np.abs(c) > np.abs(best)), c, best)
    return best

"""Source-to-physical-channel contract. All times passed to the operator are seconds."""

from dataclasses import asdict, dataclass, replace
import math
import numpy as np

VERSION = "1.0.1"
UNIT = {
    "arm": "rad",
    "hand": "rad",
    "aperture": "fraction",
    "linear": "m",
    "angular_velocity": "rad/s",
    "linear_velocity": "m/s",
}
SCALE = {
    ("rad", "rad"): 1,
    ("deg", "rad"): math.pi / 180,
    ("m", "m"): 1,
    ("cm", "m"): 0.01,
    ("mm", "m"): 0.001,
    ("fraction", "fraction"): 1,
    ("rad/s", "rad/s"): 1,
    ("deg/s", "rad/s"): math.pi / 180,
    ("m/s", "m/s"): 1,
    ("mm/s", "m/s"): 0.001,
}
ROLES = {"target", "same_state", "next_state", "unknown", "unsupported"}


@dataclass(frozen=True)
class Limits:
    move: float = 0.06  # appreciable target excursion in physical units
    still: float = 0.002  # feedback peak-to-peak noise
    error: float = 0.08  # meaningful persistent tracking error
    response_s: float = 0.40
    stall_s: float = 0.65
    jump: float = 0.30  # one-sample jump, native clock
    state_jump: float = 0.10
    error_growth: float = 0.20
    catch_error: float = 0.15
    catch_s: float = 0.13
    edge_s: float = 5.0
    pad_s: float = 0.20
    probe_s: float = 0.45


DEFAULTS = {
    "arm": Limits(),
    "hand": Limits(move=0.12, still=0.004, error=0.25, response_s=0.5, stall_s=0.8),
    "aperture": Limits(move=0.35, still=0.015, error=0.35, response_s=0.5, stall_s=0.8),
    # Velocity profiles must explicitly supply magnitude limits in their own units.
    "angular_velocity": Limits(),
    "linear_velocity": Limits(),
    "linear": Limits(
        move=0.015,
        still=0.0005,
        error=0.015,
        jump=0.05,
        state_jump=0.012,
        error_growth=0.035,
        catch_error=0.018,
    ),
}


def limits(kind, overrides):
    lim = replace(DEFAULTS[kind], **overrides)
    if not all(
        type(x) in (float, int) and math.isfinite(x) and x > 0
        for x in asdict(lim).values()
    ):
        raise ValueError("thresholds must be positive finite numbers")
    if lim.still >= lim.move or lim.state_jump >= lim.jump:
        raise ValueError("still < move and state_jump < jump required")
    return lim


def vector(x, length, name, dtype=float):
    v = np.asarray(x)
    if v.shape != (length,) or (dtype is bool and v.dtype != bool):
        raise ValueError(
            f"{name}: expected {length} values"
            + (" of boolean type" if dtype is bool else "")
        )
    return v.astype(dtype, copy=False)


class MappingError(ValueError):
    """Unverified or contradictory physical mapping: withhold the affected channel."""


def convert(raw, mapping, kind):
    source, dest = mapping.get("unit"), UNIT[kind]
    if not source:
        raise MappingError("missing source unit")
    if "points" in mapping:
        # A cross-quantity calibration must state its physical meaning and provenance.
        if not mapping.get("evidence") or mapping.get("to") != dest:
            raise MappingError("calibration has no evidence or wrong destination unit")
        pts = np.asarray(mapping["points"], dtype=float)
        if (
            pts.ndim != 2
            or pts.shape[1] != 2
            or len(pts) < 2
            or not np.isfinite(pts).all()
        ):
            raise MappingError("invalid calibration points")
        x, y = pts.T
        if np.any(np.diff(x) <= 0) or not (
            np.all(np.diff(y) > 0) or np.all(np.diff(y) < 0)
        ):
            raise MappingError("calibration must be strictly monotone")
        if kind == "aperture" and (np.min(y) < 0 or np.max(y) > 1):
            raise MappingError("aperture outside [0,1]")
        # np.interp saturates at endpoints. Only use it when saturation is declared.
        val = np.interp(raw, x, y)
        if not mapping.get("clip", False):
            val = np.where((raw < x[0]) | (raw > x[-1]), np.nan, val)
        return val
    if (source, dest) not in SCALE:
        raise MappingError(f"{source} cannot be compared as {dest} without calibration")
    return raw * SCALE[source, dest]


def check_anchors(spec):
    """Independent observed OPEN/CLOSED references for each role, in source units."""
    if spec["kind"] != "aperture":
        return
    anchors = spec.get("anchors", {})
    if not anchors.get("evidence"):
        raise MappingError("missing independent open/closed evidence")
    for role in ("action", "state"):
        pair = anchors.get(role)
        if (
            pair is None
            or np.asarray(pair).shape != (2,)
            or not np.all(np.isfinite(pair))
        ):
            raise MappingError(f"missing {role} [closed,open] references")
        closed, opened = convert(np.asarray(pair), spec[role], spec["kind"])
        if not (0 <= closed <= 0.25 and 0.75 <= opened <= 1):
            raise MappingError(
                f"{role} known closed/open values map to {closed:.3g}/{opened:.3g}"
            )


def validate_profile(p):
    if p.get("version") != 1 or not p.get("dataset") or not p.get("channels"):
        raise ValueError("profile requires version=1, dataset and channels")
    groups = p.get("hand_groups", {})
    members = {g: [] for g in groups}
    names = set()
    for c in p["channels"]:
        name = c["name"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("channel names must be unique nonempty strings")
        names.add(name)
        if c["role"] not in ROLES or c["kind"] not in DEFAULTS:
            raise ValueError(f"{name}: invalid role or kind")
        if c.get("edge", "review") not in ("review", "mask", "off"):
            raise ValueError(f"{name}: invalid edge action")
        policy = c.get("edge_policy", {})
        if set(policy) - {"fixed_after_s", "include_tail", "require_error_growth"}:
            raise ValueError("unknown edge policy key")
        if "fixed_after_s" in policy and not (
            math.isfinite(policy["fixed_after_s"]) and policy["fixed_after_s"] > 0
        ):
            raise ValueError("invalid fixed handover duration")
        for key in ("action", "state"):
            if not isinstance(c.get(key), dict):
                raise ValueError(f"{name}: missing {key} mapping")
        if c.get("group") is not None:
            g = c["group"]
            if g not in groups or c["kind"] != "hand" or c["role"] != "target":
                raise ValueError(
                    "persistent-mismatch groups require hand target channels"
                )
            members[g].append(name)
        overrides = {
            **p.get("thresholds", {}).get(c["kind"], {}),
            **c.get("thresholds", {}),
        }
        if c["kind"].endswith("_velocity") and c["role"] != "unsupported":
            required = {
                "move",
                "still",
                "error",
                "jump",
                "state_jump",
                "error_growth",
                "catch_error",
            }
            if not required.issubset(overrides):
                raise ValueError(
                    f"{name}: declare physical thresholds for measured velocity"
                )
        limits(c["kind"], overrides)
    if set(p.get("thresholds", {})) - set(DEFAULTS):
        raise ValueError("unknown threshold family")
    for g, c in groups.items():
        if (
            not c.get("evidence")
            or type(c.get("min_channels")) is not int
            or not 1 <= c["min_channels"] <= len(members[g])
            or any(not math.isfinite(c[k]) or c[k] <= 0 for k in ("error", "min_s"))
        ):
            raise ValueError(f"{g}: invalid persistent mismatch policy")
    return p

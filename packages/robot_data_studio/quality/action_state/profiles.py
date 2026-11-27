"""Auditable adapters for the evaluated signal families. Print one as a portable JSON profile."""

import argparse
import json


def joint(
    name,
    a,
    s,
    i,
    *,
    kind="arm",
    role="target",
    evidence,
    edge="off",
    group=None,
    unit="rad",
):
    c = {
        "name": name,
        "kind": kind,
        "role": role,
        "evidence": evidence,
        "edge": edge,
        "action": {"column": a, "component": i, "unit": unit},
        "state": {"column": s, "component": i, "unit": unit},
    }
    if group:
        c["group"] = group
    if kind == "hand":
        c["contact_possible"] = True
    return c


def aperture(name, act, st, ac_points, st_points, anchors, evidence):
    a, ai, au = act
    s, si, su = st

    def lut(col, i, unit, points):
        return dict(
            column=col,
            component=i,
            unit=unit,
            to="fraction",
            points=points,
            clip=True,
            evidence=evidence,
        )

    return dict(
        name=name,
        kind="aperture",
        role="target",
        evidence=evidence,
        edge="off",
        action=lut(a, ai, au, ac_points),
        state=lut(s, si, su, st_points),
        anchors=anchors,
    )


NAMES = (
    "astribot-s1",
    "sharpadex",
    "roco_taskboard",
    "lingyu",
    "fineart",
    "zhiyuan-g1",
)


def preset(dataset):
    c = []
    p = dict(version=1, dataset=dataset, channels=c)
    if dataset in ("sharpadex", "roco_taskboard"):
        sharpa = dataset == "sharpadex"
        ev = (
            "SharpaDex v3 native absolute joint targets; audit/EXCLUSION-POLICY-V2.json"
            if sharpa
            else "RoCo TaskBoard cache v3 README, robot100 source mapping"
        )
        for side, offset in (("left", 0), ("right", 29)):
            for j in range(7):
                c.append(
                    joint(
                        f"{side}_arm_j{j}",
                        "action",
                        "observation.state",
                        offset + j,
                        evidence=ev,
                        edge="mask" if sharpa else "review",
                    )
                )
            for j in range(22):
                c.append(
                    joint(
                        f"{side}_hand_j{j}",
                        "action",
                        "observation.state",
                        offset + 7 + j,
                        kind="hand",
                        evidence=ev,
                        group=side if sharpa else None,
                    )
                )
        if sharpa:
            p["hand_groups"] = {
                side: dict(error=0.30, min_channels=3, min_s=0.50, evidence=ev)
                for side in ("left", "right")
            }
    elif dataset == "astribot-s1":
        for i in range(14):
            c.append(
                joint(
                    f"{'left' if i < 7 else 'right'}_arm_{i % 7}",
                    "action.arm.position",
                    "observation.state.arm.position",
                    i,
                    evidence="S1 clean-v1 technical/contract.json",
                    edge="review",
                )
            )
        ev = "S1 cache v3 mappings/GRIPPER-ENDPOINTS.json; separate visual open/close review in cleaning audit"
        for i, (side, op, cl) in enumerate(
            [
                ("left", -0.5824275455213308, 99.91081424095046),
                ("right", -0.251804982820367, 97.78277880464874),
            ]
        ):
            c.append(
                aperture(
                    side + "_grip",
                    ("action.effector.position", i, "native_percent"),
                    ("observation.state.effector.position", i, "mm"),
                    [[0, 1], [100, 0]],
                    [[op, 1], [cl, 0]],
                    dict(
                        action=[100, 0],
                        state=[cl, op],
                        evidence="S1 gripper-visual-review-v2.md and endpoints",
                    ),
                    ev,
                )
            )
        for i, k in enumerate(("x", "y", "yaw")):
            c.append(
                joint(
                    "chassis_velocity_" + k,
                    "action.chassis.position",
                    "observation.state.chassis.position",
                    i,
                    kind="angular_velocity" if k == "yaw" else "linear_velocity",
                    role="unsupported",
                    unit="rad" if k == "yaw" else "m",
                    evidence="CHASSIS-AUDIT.json: odometry pose cannot supply a body velocity command",
                )
            )
    elif dataset == "lingyu":
        ev = "LingYu TA2 README 5.4 and cache v3 mappings/robot100.json"
        for side, offset in (("left", 0), ("right", 8)):
            for j in range(7):
                ch = joint(
                    f"{side}_joint{j + 1}",
                    "action",
                    "observation.state",
                    offset + j,
                    evidence=ev,
                    edge="mask",
                )
                ch["edge_policy"] = {
                    "include_tail": False,
                    "require_error_growth": False,
                    "fixed_after_s": 1.0,
                }
                ch["thresholds"] = {"jump": 0.12, "state_jump": 0.03}
                c.append(ch)
        for i, (side, op, cl) in enumerate(
            [
                ("left", 1.168290138244629, -0.14826583862304688),
                ("right", 1.1318464279174805, -0.2978835105895996),
            ]
        ):
            c.append(
                aperture(
                    side + "_grip",
                    ("action", 39 + i * 8, "Nm"),
                    ("observation.state", 7 + i * 8, "rad"),
                    [[-1.6, 0], [0, 0.90], [2, 1]],
                    [[cl, 0], [op, 1]],
                    dict(
                        action=[-1.6, 2],
                        state=[cl, op],
                        evidence="source command endpoints + independent readback plateaus + visual open/closed audit",
                    ),
                    ev,
                )
            )
    elif dataset == "fineart":
        for i in (*range(6), *range(7, 13)):
            c.append(
                joint(
                    f"joint_{i}",
                    "action",
                    "observation.state",
                    i,
                    role="next_state",
                    evidence="clean-v1-native audit/technical/REPORT.md: row-exact next observed state on native clock",
                )
            )
    elif dataset == "zhiyuan-g1":
        for i in range(14):
            c.append(
                joint(
                    f"arm_{i}",
                    "action.arm.position",
                    "observation.state.arm.position",
                    i,
                    role="unknown",
                    evidence="A2D source: close to state; independent controller stream unverified",
                )
            )
        c.append(
            joint(
                "yaw",
                "action.head.position",
                "observation.state.head.position",
                0,
                role="same_state",
                evidence="A2D cache v3 mapping and native head-position readback",
            )
        )
    else:
        raise ValueError("unknown preset")
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", choices=NAMES)
    print(json.dumps(preset(ap.parse_args().dataset), ensure_ascii=False, indent=2))

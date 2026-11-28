"""Verified semantics for released cache layouts, applied before per-episode checks."""

AGIBOT_DATASET = "agibotworld-beta"
AGIBOT_SOURCE = {
    "release": "agibotworld-beta/v3-subtask-20261116",
    "robot": "agibot_g1",
    "layout": "robot100",
}
AGIBOT_ARM_EVIDENCE = (
    "agibotworld-beta/v3-subtask-20261116, agibot_g1 robot100 slots 0-13: "
    "published arm action values matched same-row state; verified again on input rows"
)


def resolve_cache_roles(profile):
    source = profile.get("cache_source", {})
    if not isinstance(source, dict):
        raise ValueError("cache_source must be an object")
    if profile["dataset"] != AGIBOT_DATASET or any(
        source.get(key) != value for key, value in AGIBOT_SOURCE.items()
    ):
        return profile

    channels = []
    for channel in profile["channels"]:
        action, state = channel["action"], channel["state"]
        slot = action.get("component")
        if (
            channel["role"] == "target"
            and channel["kind"] == "arm"
            and type(slot) is int
            and 0 <= slot < 14
            and type(state.get("component")) is int
            and state["component"] == slot
            and action.get("column") == "action"
            and state.get("column") == "state"
            and action.get("unit") == state.get("unit") == "rad"
            and "points" not in action
            and "points" not in state
        ):
            channel = dict(
                channel, role="same_state", evidence=AGIBOT_ARM_EVIDENCE, proxy_tolerance=0.0
            )
        channels.append(channel)
    return dict(profile, channels=channels)

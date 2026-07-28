from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, Field

from .contracts import (
    CORE_CONTRACT_IDS,
    HUMAN_CHECK_DEPENDENCIES,
    HUMAN_EXTENSION_IDS,
    QWEN_CHECK_DEPENDENCIES,
    RETIRED_QWEN_CORE_IDS,
)


class CheckExecutionDecision(BaseModel):
    check_id: str
    state: Literal["ready", "missing_prerequisite", "disabled", "retired"]
    required_intermediates: list[str] = Field(default_factory=list)
    missing_prerequisites: list[str] = Field(default_factory=list)


class CheckExecutionPlan(BaseModel):
    checks: list[CheckExecutionDecision]
    required_intermediates: list[str]


def build_check_execution_plan(
    *,
    enabled_quality_checks: set[str],
    enabled_human_extension_checks: set[str],
    embodiment: str,
    available_intermediates: set[str],
) -> CheckExecutionPlan:
    unknown = enabled_quality_checks - set(CORE_CONTRACT_IDS)
    if unknown:
        raise ValueError(f"unknown or retired quality checks: {sorted(unknown)}")
    unknown_human = enabled_human_extension_checks - set(HUMAN_EXTENSION_IDS)
    if unknown_human:
        raise ValueError(f"unknown Human extension checks: {sorted(unknown_human)}")
    checks: list[CheckExecutionDecision] = []
    required: set[str] = set()
    for check_id in CORE_CONTRACT_IDS:
        if check_id not in enabled_quality_checks:
            checks.append(CheckExecutionDecision(check_id=check_id, state="disabled"))
            continue
        dependencies = list(QWEN_CHECK_DEPENDENCIES[check_id])
        if check_id == "C2" and embodiment != "robot":
            dependencies = [*dependencies, "robot_embodiment"]
        missing = sorted(set(dependencies) - available_intermediates)
        required.update(dependencies)
        checks.append(
            CheckExecutionDecision(
                check_id=check_id,
                state="missing_prerequisite" if missing else "ready",
                required_intermediates=dependencies,
                missing_prerequisites=missing,
            )
        )
    checks.extend(
        CheckExecutionDecision(check_id=check_id, state="retired")
        for check_id in RETIRED_QWEN_CORE_IDS
    )
    for check_id in HUMAN_EXTENSION_IDS:
        if check_id not in enabled_human_extension_checks:
            checks.append(CheckExecutionDecision(check_id=check_id, state="disabled"))
            continue
        dependencies = list(HUMAN_CHECK_DEPENDENCIES[check_id])
        if embodiment != "human":
            dependencies = [*dependencies, "human_embodiment"]
        missing = sorted(set(dependencies) - available_intermediates)
        required.update(dependencies)
        checks.append(
            CheckExecutionDecision(
                check_id=check_id,
                state="missing_prerequisite" if missing else "ready",
                required_intermediates=dependencies,
                missing_prerequisites=missing,
            )
        )
    return CheckExecutionPlan(checks=checks, required_intermediates=sorted(required))


def intermediate_cache_key(
    *,
    content_hash: str,
    intermediate: str,
    model: str,
    prompt: str,
    config: dict[str, Any],
    frame_indexes: list[int],
) -> str:
    payload = {
        "schema": "quality-intermediate-cache-v1",
        "content_hash": content_hash,
        "intermediate": intermediate,
        "model": model,
        "prompt": prompt,
        "config": config,
        "frame_indexes": frame_indexes,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()

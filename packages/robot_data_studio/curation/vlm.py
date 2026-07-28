from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .models import EvidenceFrame, EvidenceState, VlmResult
from .profile import VlmProfile


class BudgetExceeded(RuntimeError):
    pass


class BudgetLedger:
    def __init__(
        self,
        *,
        max_calls: int,
        max_prompt_tokens: int,
        max_cost_usd: float,
        estimated_cost_per_call_usd: float,
    ) -> None:
        self.max_calls = max_calls
        self.max_prompt_tokens = max_prompt_tokens
        self.max_cost_usd = max_cost_usd
        self.estimated_cost_per_call_usd = estimated_cost_per_call_usd
        self.calls = 0
        self.prompt_tokens = 0
        self.cost_usd = 0.0

    @classmethod
    def from_profile(cls, profile: VlmProfile) -> BudgetLedger:
        return cls(
            max_calls=profile.max_calls,
            max_prompt_tokens=profile.max_prompt_tokens,
            max_cost_usd=profile.max_cost_usd,
            estimated_cost_per_call_usd=profile.estimated_cost_per_call_usd,
        )

    def reserve(self, prompt_tokens: int) -> None:
        next_cost = self.cost_usd + self.estimated_cost_per_call_usd
        if self.calls + 1 > self.max_calls:
            raise BudgetExceeded("VLM call budget exceeded")
        if self.prompt_tokens + prompt_tokens > self.max_prompt_tokens:
            raise BudgetExceeded("VLM prompt-token budget exceeded")
        if next_cost > self.max_cost_usd + 1e-12:
            raise BudgetExceeded("VLM cost budget exceeded")
        self.calls += 1
        self.prompt_tokens += prompt_tokens
        self.cost_usd = next_cost


class MockVlmEvaluator:
    """Deterministic, local-only evaluator. It contains no network backend."""

    def __init__(self, cache_dir: str | Path, ledger: BudgetLedger) -> None:
        self.cache_dir = Path(cache_dir)
        self.ledger = ledger

    @staticmethod
    def cache_key(*, evidence: list[EvidenceFrame], prompt: str, model: str) -> str:
        payload = {
            "evidence": [item.model_dump(mode="json") for item in evidence],
            "model": model,
            "prompt": prompt,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def evaluate(
        self,
        *,
        evidence: list[EvidenceFrame],
        prompt: str,
        model: str,
        mock_state: EvidenceState = "review",
    ) -> VlmResult:
        key = self.cache_key(evidence=evidence, prompt=prompt, model=model)
        path = self.cache_dir / f"{key}.json"
        if path.is_file():
            cached = VlmResult.model_validate_json(path.read_text(encoding="utf-8"))
            return cached.model_copy(update={"from_cache": True})
        token_estimate = max(1, len(prompt.split()) + len(evidence) * 12)
        self.ledger.reserve(token_estimate)
        result = VlmResult(state=mock_state, cache_key=key, from_cache=False)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
        return result


def estimate_vlm_cost(profile: VlmProfile, eligible_episodes: int) -> dict[str, float | int]:
    calls = min(eligible_episodes, profile.max_calls) if profile.backend == "mock" else 0
    return {
        "calls": calls,
        "max_prompt_tokens": profile.max_prompt_tokens,
        "estimated_cost_usd": round(calls * profile.estimated_cost_per_call_usd, 8),
    }

from __future__ import annotations

import re

from .models import NormalizedTaskText


_TARGET_PREPOSITIONS = {"to", "into", "onto", "on", "in", "at"}


def normalize_task_text(raw_text: str) -> NormalizedTaskText:
    tokens = re.findall(r"[\w'-]+", raw_text.lower())
    if not tokens:
        return NormalizedTaskText(raw_text=raw_text, verb="", object="")
    verb = tokens[0]
    prep_index = next(
        (index for index, token in enumerate(tokens[1:], start=1) if token in _TARGET_PREPOSITIONS),
        None,
    )
    object_tokens = tokens[1:prep_index] if prep_index is not None else tokens[1:]
    target_tokens = tokens[prep_index + 1 :] if prep_index is not None else []
    modifiers: list[str] = []
    while target_tokens and target_tokens[-1].endswith("ly"):
        modifiers.insert(0, target_tokens.pop())
    return NormalizedTaskText(
        raw_text=raw_text,
        verb=verb,
        object=" ".join(object_tokens),
        target=" ".join(target_tokens) or None,
        modifiers=modifiers,
    )

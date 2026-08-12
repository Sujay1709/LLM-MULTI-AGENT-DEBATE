"""Robust, task-oriented answer parsing and aggregation."""

from __future__ import annotations

import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from typing import TypeVar

T = TypeVar("T")

_NUMBER = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"


def normalize_number(value: str | int | float | Decimal) -> str | None:
    raw = str(value).strip().replace(",", "")
    try:
        number = Decimal(raw)
    except InvalidOperation:
        return None
    if not number.is_finite():
        return None
    normalized = number.normalize()
    if normalized == normalized.to_integral():
        return str(normalized.quantize(Decimal(1)))
    return format(normalized, "f").rstrip("0").rstrip(".")


def parse_numeric_answer(text: str) -> str | None:
    """Prefer boxed answers, then GSM markers, then the final visible number."""
    boxed = re.findall(r"\\boxed\s*\{\s*(" + _NUMBER + r")\s*\}", text)
    if boxed:
        return normalize_number(boxed[-1])
    gsm = re.findall(r"####\s*(" + _NUMBER + r")", text)
    if gsm:
        return normalize_number(gsm[-1])
    numbers = re.findall(_NUMBER, text)
    return normalize_number(numbers[-1]) if numbers else None


def parse_choice_answer(text: str) -> str | None:
    patterns = [
        r"\(([A-Da-d])\)",
        r"(?:final\s+answer|answer)\s*(?:is|:)\s*([A-Da-d])\b",
        r"\b([A-Da-d])\s*$",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].upper()
    return None


def majority_vote(values: list[T | None], fallback: T | None = None) -> tuple[T | None, bool]:
    """Return the unique mode; ties use the supplied deterministic fallback."""
    valid = [value for value in values if value is not None]
    if not valid:
        return fallback, True
    counts = Counter(valid)
    highest = max(counts.values())
    winners = [value for value, count in counts.items() if count == highest]
    if len(winners) == 1:
        return winners[0], False
    return fallback, True


def extract_json_object(text: str) -> str:
    """Extract the outermost JSON object from plain or fenced model output."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("No JSON object found in judge response")
    return text[start : end + 1]


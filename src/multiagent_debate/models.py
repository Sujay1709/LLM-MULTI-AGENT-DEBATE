"""Typed experiment records.

The records deliberately use dataclasses instead of a validation framework so the
research logic stays transparent to students and easy to serialize.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass(slots=True)
class Message:
    role: Literal["system", "user", "assistant"]
    content: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(slots=True)
class Generation:
    text: str
    model: str
    latency_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DebateConfig:
    task: str
    model: str
    num_agents: int
    num_rounds: int
    temperature: float = 0.7
    max_tokens: int = 700
    seed: int = 0
    limit: int = 10
    aggregation: Literal["majority", "paired"] = "majority"
    baseline_strategies: tuple[str, ...] = ()
    max_retries: int = 3
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        # Omitting the new empty field preserves experiment IDs for pre-1.1 resumable runs.
        if not self.baseline_strategies:
            value.pop("baseline_strategies")
        return value


@dataclass(slots=True)
class Example:
    example_id: str
    prompt: str
    reference: str | int | float | list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

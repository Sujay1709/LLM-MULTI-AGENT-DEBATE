"""Provider-independent model clients."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

from dotenv import load_dotenv

from .models import Generation, Message


class LLMClient(Protocol):
    """The small interface used by the debate engine."""

    def generate(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        seed: int | None = None,
    ) -> Generation: ...


def _read_attr(value: object, name: str, default: object = None) -> object:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


class LiteLLMClient:
    """Thin LiteLLM adapter with bounded exponential-backoff retries."""

    def __init__(self, *, max_retries: int = 3, env_file: str | Path | None = None) -> None:
        load_dotenv(dotenv_path=env_file)
        self.max_retries = max_retries

    def generate(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        seed: int | None = None,
    ) -> Generation:
        from litellm import completion, completion_cost

        started = time.perf_counter()
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = completion(
                    model=model,
                    messages=[message.to_dict() for message in messages],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=seed,
                    drop_params=True,
                    num_retries=0,
                )
                choices = _read_attr(response, "choices", [])
                first_choice = choices[0]
                response_message = _read_attr(first_choice, "message", {})
                content = _read_attr(response_message, "content", "")
                usage = _read_attr(response, "usage", {})
                prompt_tokens = int(_read_attr(usage, "prompt_tokens", 0) or 0)
                completion_tokens = int(_read_attr(usage, "completion_tokens", 0) or 0)
                total_tokens = int(
                    _read_attr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0
                )
                try:
                    cost = float(completion_cost(completion_response=response))
                except Exception:
                    cost = None
                return Generation(
                    text=str(content or ""),
                    model=model,
                    latency_seconds=time.perf_counter() - started,
                    input_tokens=prompt_tokens,
                    output_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    cost_usd=cost,
                )
            except Exception as exc:  # provider SDKs expose many exception types
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(min(2**attempt, 8))

        return Generation(
            text="",
            model=model,
            latency_seconds=time.perf_counter() - started,
            error=f"{type(last_error).__name__}: {last_error}",
        )


class FakeClient:
    """Deterministic client for tests, demos, and offline exploration."""

    def __init__(
        self,
        responses: Sequence[str] | None = None,
        responder: Callable[[Sequence[Message], int], str] | None = None,
    ) -> None:
        self.responses = list(responses or [])
        self.responder = responder
        self.calls: list[list[Message]] = []

    @staticmethod
    def _default_response(messages: Sequence[Message]) -> str:
        text = messages[-1].content
        if '"claim_labels"' in text and "Reference facts" in text:
            facts_match = re.search(
                r"Reference facts \(numbered from zero\):\s*(\[.*?\])\s*\n\nGenerated biography:",
                text,
                flags=re.DOTALL,
            )
            facts = json.loads(facts_match.group(1)) if facts_match else []
            return json.dumps(
                {
                    "claim_labels": [
                        {"claim": "Offline placeholder claim", "label": "absent"}
                    ],
                    "reference_coverage": [
                        {"fact_index": index, "covered": False}
                        for index in range(len(facts))
                    ],
                }
            )
        arithmetic = re.search(
            r"(-?\d+)\s*\+\s*(-?\d+)\s*\*\s*(-?\d+)\s*\+\s*(-?\d+)\s*-\s*(-?\d+)\s*\*\s*(-?\d+)",
            text,
        )
        if arithmetic:
            a, b, c, d, e, f = (int(item) for item in arithmetic.groups())
            answer = a + b * c + d - e * f
            return f"Applying multiplication first gives the final answer \\boxed{{{answer}}}."
        if "A)" in text and "B)" in text:
            return "For this offline demonstration I select (A)."
        if "biography" in text.lower():
            return "- This is a deterministic offline biography placeholder."
        return "The deterministic offline model returns \\boxed{0}."

    def generate(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        seed: int | None = None,
    ) -> Generation:
        copied = [Message(message.role, message.content) for message in messages]
        call_index = len(self.calls)
        self.calls.append(copied)
        if self.responder is not None:
            text = self.responder(messages, call_index)
        elif call_index < len(self.responses):
            text = self.responses[call_index]
        else:
            text = self._default_response(messages)
        return Generation(text=text, model=model, total_tokens=0)


def make_client(model: str, *, max_retries: int = 3) -> LLMClient:
    """Create the offline fake client or a live LiteLLM client."""
    if model.startswith("fake/"):
        return FakeClient()
    return LiteLLMClient(max_retries=max_retries)

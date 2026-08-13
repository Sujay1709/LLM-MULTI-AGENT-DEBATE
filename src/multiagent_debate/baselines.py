"""Task-independent strong baselines for debate experiments."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .clients import LLMClient
from .models import DebateConfig, Example, Message
from .parsing import majority_vote
from .prompts import self_refinement_prompt

SELF_REFINEMENT = "self_refinement"
BUDGET_MATCHED_SELF_CONSISTENCY = "budget_matched_self_consistency"
SUPPORTED_BASELINES = {SELF_REFINEMENT, BUDGET_MATCHED_SELF_CONSISTENCY}
AnswerParser = Callable[[str], str | None]


def validate_baseline_strategies(config: DebateConfig) -> None:
    unknown = set(config.baseline_strategies) - SUPPORTED_BASELINES
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown baseline strategies: {names}")
    if (
        BUDGET_MATCHED_SELF_CONSISTENCY in config.baseline_strategies
        and config.aggregation != "majority"
    ):
        raise ValueError("Budget-matched self-consistency requires majority aggregation")


def _turn(
    *,
    messages: list[Message],
    generation: Any,
    parsed_answer: str | None,
    step: int,
) -> dict[str, Any]:
    return {
        "step": step,
        "request_messages": [message.to_dict() for message in messages],
        "response": generation.to_dict(),
        "parsed_answer": parsed_answer,
        "parse_ok": parsed_answer is not None,
    }


def run_self_refinement(
    example: Example,
    config: DebateConfig,
    client: LLMClient,
    parse_answer: AnswerParser,
    *,
    answer_instruction: str,
    initial_response: str,
    initial_answer: str | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Reuse agent 0's answer, then let that agent revise without peer responses."""
    history = [Message("user", example.prompt), Message("assistant", initial_response)]
    revision_turns: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    prediction = initial_answer

    # Matching the number of debate rounds gives both methods the same sequential depth.
    for step in range(1, config.num_rounds):
        history.append(
            Message(
                "user",
                self_refinement_prompt(
                    example.prompt,
                    answer_instruction=answer_instruction,
                ),
            )
        )
        generation = client.generate(
            history,
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            seed=config.seed + 100_000 + step,
        )
        parsed = parse_answer(generation.text) if not generation.error else None
        revision_turns.append(
            _turn(messages=history, generation=generation, parsed_answer=parsed, step=step)
        )
        prediction = parsed
        history.append(Message("assistant", generation.text))
        if generation.error:
            errors.append(
                {
                    "baseline": SELF_REFINEMENT,
                    "step": step,
                    "error": generation.error,
                }
            )

    return (
        {
            "strategy": SELF_REFINEMENT,
            "reused_initial_turn": {"agent_id": 0, "round": 0},
            "revision_turns": revision_turns,
            "prediction": prediction,
        },
        errors,
    )


def run_budget_matched_self_consistency(
    example: Example,
    config: DebateConfig,
    client: LLMClient,
    parse_answer: AnswerParser,
    *,
    round_zero_answers: list[str | None],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Sample as many independent answers as debate uses total agent calls."""
    target_samples = config.num_agents * config.num_rounds
    answers = list(round_zero_answers)
    additional_samples: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for sample_index in range(len(answers), target_samples):
        messages = [Message("user", example.prompt)]
        generation = client.generate(
            messages,
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            seed=config.seed + sample_index,
        )
        parsed = parse_answer(generation.text) if not generation.error else None
        answers.append(parsed)
        additional_samples.append(
            _turn(
                messages=messages,
                generation=generation,
                parsed_answer=parsed,
                step=sample_index,
            )
        )
        if generation.error:
            errors.append(
                {
                    "baseline": BUDGET_MATCHED_SELF_CONSISTENCY,
                    "sample": sample_index,
                    "error": generation.error,
                }
            )

    fallback = answers[0] if answers else None
    selected, tie = majority_vote(answers, fallback=fallback)
    return (
        {
            "strategy": BUDGET_MATCHED_SELF_CONSISTENCY,
            "target_samples": target_samples,
            "reused_round_zero_samples": len(round_zero_answers),
            "additional_samples": additional_samples,
            "parsed_answers": answers,
            "prediction": selected,
            "tie": tie,
        },
        errors,
    )

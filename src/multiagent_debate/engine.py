"""Synchronous multi-agent debate orchestration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .clients import LLMClient
from .models import DebateConfig, Example, Message
from .parsing import majority_vote
from .prompts import peer_review_prompt

AnswerParser = Callable[[str], str | None]


def _round_prediction(
    answers: list[str | None], aggregation: str
) -> tuple[str | None, bool]:
    fallback = answers[0] if answers else None
    if aggregation == "paired":
        return fallback, False
    return majority_vote(answers, fallback=fallback)


def run_debate(
    example: Example,
    config: DebateConfig,
    client: LLMClient,
    parse_answer: AnswerParser,
    *,
    answer_instruction: str,
    experiment_id: str = "",
) -> dict[str, Any]:
    """Run one debate while preventing within-round information leakage."""
    histories = [
        [Message("user", example.prompt)]
        for _ in range(config.num_agents)
    ]
    agents: list[dict[str, Any]] = [
        {"agent_id": agent_id, "turns": []} for agent_id in range(config.num_agents)
    ]
    errors: list[dict[str, Any]] = []
    round_predictions: list[dict[str, Any]] = []

    for round_index in range(config.num_rounds):
        # Snapshot the completed previous round before any agent starts this one.
        previous_responses = [
            agent["turns"][-1]["response"]["text"] if agent["turns"] else ""
            for agent in agents
        ]
        current_answers: list[str | None] = []

        for agent_id in range(config.num_agents):
            if round_index > 0:
                peers = [
                    response
                    for peer_id, response in enumerate(previous_responses)
                    if peer_id != agent_id
                ]
                review = peer_review_prompt(
                    example.prompt,
                    peers,
                    answer_instruction=answer_instruction,
                )
                histories[agent_id].append(Message("user", review))

            request_messages = [message.to_dict() for message in histories[agent_id]]
            generation = client.generate(
                histories[agent_id],
                model=config.model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                seed=config.seed + agent_id + round_index * config.num_agents,
            )
            answer = parse_answer(generation.text) if not generation.error else None
            turn = {
                "round": round_index,
                "request_messages": request_messages,
                "response": generation.to_dict(),
                "parsed_answer": answer,
                "parse_ok": answer is not None,
            }
            agents[agent_id]["turns"].append(turn)
            current_answers.append(answer)
            histories[agent_id].append(Message("assistant", generation.text))
            if generation.error:
                errors.append(
                    {"agent_id": agent_id, "round": round_index, "error": generation.error}
                )

        selected, tie = _round_prediction(current_answers, config.aggregation)
        round_predictions.append(
            {
                "round": round_index,
                "agent_answers": current_answers,
                "selected_answer": selected,
                "tie": tie,
            }
        )

    direct = round_predictions[0]["agent_answers"][0] if round_predictions else None
    final_round = round_predictions[-1] if round_predictions else {}
    return {
        "schema_version": "1.0",
        "experiment_id": experiment_id,
        "task": config.task,
        "example_id": example.example_id,
        "question": example.prompt,
        "reference": example.reference,
        "metadata": example.metadata,
        "config": config.to_dict(),
        "agents": agents,
        "predictions": {
            "direct": direct,
            "rounds": round_predictions,
            "debate": final_round.get("selected_answer"),
            "final_agent_answers": final_round.get("agent_answers", []),
            "tie": bool(final_round.get("tie", False)),
        },
        "status": "partial" if errors else "ok",
        "errors": errors,
    }


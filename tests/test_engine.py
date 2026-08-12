from multiagent_debate.clients import FakeClient
from multiagent_debate.engine import run_debate
from multiagent_debate.models import DebateConfig, Example, Generation, Message
from multiagent_debate.parsing import parse_numeric_answer


def test_rounds_use_only_previous_round_snapshot() -> None:
    client = FakeClient(
        responses=[
            "first agent \\boxed{1}",
            "second agent \\boxed{2}",
            "first revised \\boxed{2}",
            "second revised \\boxed{3}",
        ]
    )
    config = DebateConfig(task="test", model="fake/test", num_agents=2, num_rounds=2)
    record = run_debate(
        Example("one", "Compute something", 2),
        config,
        client,
        parse_numeric_answer,
        answer_instruction="Return a boxed number.",
    )

    agent_zero_review = client.calls[2][-1].content
    agent_one_review = client.calls[3][-1].content
    assert "second agent" in agent_zero_review
    assert "first agent" in agent_one_review
    assert "first revised" not in agent_one_review
    assert record["predictions"]["direct"] == "1"
    assert record["predictions"]["debate"] == "2"
    assert record["predictions"]["tie"] is True


class ErrorClient:
    def generate(
        self,
        messages: list[Message],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        seed: int | None = None,
    ) -> Generation:
        return Generation(text="", model=model, error="simulated provider failure")


def test_provider_errors_are_recorded_instead_of_crashing() -> None:
    config = DebateConfig(task="test", model="fake/error", num_agents=1, num_rounds=1)
    record = run_debate(
        Example("error", "Question", 1),
        config,
        ErrorClient(),
        parse_numeric_answer,
        answer_instruction="Return a boxed number.",
    )
    assert record["status"] == "partial"
    assert record["errors"][0]["error"] == "simulated provider failure"
    assert record["predictions"]["direct"] is None


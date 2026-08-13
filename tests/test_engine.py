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


def test_strong_baselines_reuse_only_appropriate_debate_calls() -> None:
    client = FakeClient(
        responses=[
            "first initial \\boxed{1}",
            "second initial \\boxed{2}",
            "first debate \\boxed{2}",
            "second debate \\boxed{2}",
            "private refinement \\boxed{3}",
            "independent sample \\boxed{4}",
            "another independent sample \\boxed{4}",
        ]
    )
    config = DebateConfig(
        task="test",
        model="fake/test",
        num_agents=2,
        num_rounds=2,
        baseline_strategies=("self_refinement", "budget_matched_self_consistency"),
    )
    record = run_debate(
        Example("baselines", "Compute something", 4),
        config,
        client,
        parse_numeric_answer,
        answer_instruction="Return a boxed number.",
    )

    assert record["predictions"]["direct"] == "1"
    assert record["predictions"]["self_consistency"] == "1"
    assert record["predictions"]["debate"] == "2"
    assert record["predictions"]["self_refinement"] == "3"
    assert record["predictions"]["budget_matched_self_consistency"] == "4"
    assert "first initial" in client.calls[4][1].content
    assert "second initial" not in "\n".join(message.content for message in client.calls[4])
    assert len(client.calls[5]) == 1
    assert len(client.calls[6]) == 1
    budget = record["baselines"]["budget_matched_self_consistency"]
    assert budget["target_samples"] == 4
    assert budget["reused_round_zero_samples"] == 2

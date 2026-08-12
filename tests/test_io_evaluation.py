import csv

import pytest

from multiagent_debate.clients import FakeClient
from multiagent_debate.evaluation import evaluate_objective
from multiagent_debate.io import read_jsonl
from multiagent_debate.models import DebateConfig, Example
from multiagent_debate.parsing import parse_numeric_answer
from multiagent_debate.runner import run_experiment


def test_incremental_output_resume_and_objective_summary(tmp_path) -> None:
    examples = [Example(str(index), f"Problem {index}", index) for index in range(3)]
    config = DebateConfig(
        task="arithmetic",
        model="fake/test",
        num_agents=1,
        num_rounds=1,
        limit=2,
    )
    client = FakeClient(responses=["\\boxed{0}", "\\boxed{1}"])
    results = run_experiment(
        examples,
        config,
        client,
        parse_numeric_answer,
        answer_instruction="Return a box.",
        output_base=tmp_path,
        progress=lambda _: None,
    )
    records = read_jsonl(results)
    assert len(records) == 2

    resumed = FakeClient(responses=["This should never be called"])
    run_experiment(
        examples,
        config,
        resumed,
        parse_numeric_answer,
        answer_instruction="Return a box.",
        resume=results.parent,
        progress=lambda _: None,
    )
    assert resumed.calls == []

    changed_config = DebateConfig(
        task="arithmetic",
        model="fake/test",
        num_agents=2,
        num_rounds=1,
        limit=2,
    )
    with pytest.raises(ValueError, match="does not match"):
        run_experiment(
            examples,
            changed_config,
            resumed,
            parse_numeric_answer,
            answer_instruction="Return a box.",
            resume=results.parent,
            progress=lambda _: None,
        )

    rows = evaluate_objective(results)
    assert {row["strategy"] for row in rows} == {"direct", "debate"}
    with results.with_name("summary.csv").open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 2

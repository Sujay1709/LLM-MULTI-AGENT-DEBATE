import json

from multiagent_debate.clients import FakeClient
from multiagent_debate.evaluation import (
    judge_biography,
    judgment_metrics,
    parse_biography_judgment,
)


def valid_judgment() -> dict:
    return {
        "claim_labels": [
            {"claim": "Fact one", "label": "supported"},
            {"claim": "Wrong", "label": "contradicted"},
            {"claim": "Extra", "label": "absent"},
        ],
        "reference_coverage": [
            {"fact_index": 0, "covered": True},
            {"fact_index": 1, "covered": False},
        ],
    }


def test_biography_judgment_validation_and_metrics() -> None:
    judgment = parse_biography_judgment(f"```json\n{json.dumps(valid_judgment())}\n```", 2)
    metrics = judgment_metrics(judgment)
    assert metrics["supported_precision"] == 1 / 3
    assert metrics["contradiction_rate"] == 1 / 3
    assert metrics["unknown_rate"] == 1 / 3
    assert metrics["reference_coverage"] == 0.5


def test_malformed_judge_output_gets_one_repair_attempt() -> None:
    client = FakeClient(responses=["not json", json.dumps(valid_judgment())])
    judgment, calls = judge_biography(
        client=client,
        model="fake/judge",
        name="Person",
        facts=["Fact one", "Fact two"],
        biography="- Fact one",
    )
    assert judgment["claim_labels"][0]["label"] == "supported"
    assert len(calls) == 2


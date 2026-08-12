"""Objective and reference-grounded biography evaluation."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from statistics import mean
from typing import Any

from .clients import LLMClient
from .io import append_jsonl, read_jsonl, write_csv
from .models import Message
from .parsing import extract_json_object, normalize_number
from .prompts import biography_judge_prompt, json_repair_prompt


def _generation_usage(generation: dict[str, Any]) -> tuple[int, float, float]:
    return (
        int(generation.get("total_tokens") or 0),
        float(generation.get("latency_seconds") or 0.0),
        float(generation.get("cost_usd") or 0.0),
    )


def _strategy_usage(record: dict[str, Any], strategy: str) -> tuple[int, float, float]:
    if strategy == "direct":
        generation = record["agents"][0]["turns"][0]["response"]
        return _generation_usage(generation)
    tokens = 0
    latency = 0.0
    cost = 0.0
    for agent in record["agents"]:
        for turn in agent["turns"]:
            values = _generation_usage(turn["response"])
            tokens += values[0]
            latency += values[1]
            cost += values[2]
    return tokens, latency, cost


def _canonical_reference(reference: Any, task: str) -> str | None:
    if task in {"arithmetic", "gsm8k"}:
        return normalize_number(reference)
    if task == "mmlu":
        return str(reference).strip().upper()
    return str(reference)


def _objective_rows(
    records: Sequence[dict[str, Any]], label: str = "overall"
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    strategies = ["direct", "debate"]
    for strategy in strategies:
        scored = correct = parse_failures = ties = total_tokens = 0
        total_latency = total_cost = 0.0
        for record in records:
            prediction = record["predictions"].get(strategy)
            reference = _canonical_reference(record["reference"], record["task"])
            if prediction is None:
                parse_failures += 1
            else:
                scored += 1
                correct += int(str(prediction) == str(reference))
            if strategy == "debate":
                ties += int(bool(record["predictions"].get("tie")))
            usage = _strategy_usage(record, strategy)
            total_tokens += usage[0]
            total_latency += usage[1]
            total_cost += usage[2]
        rows.append(
            {
                "group": label,
                "strategy": strategy,
                "n": len(records),
                "scored": scored,
                "correct": correct,
                "accuracy": correct / scored if scored else 0.0,
                "parse_failures": parse_failures,
                "ties": ties,
                "total_tokens": total_tokens,
                "mean_latency_seconds": total_latency / len(records) if records else 0.0,
                "total_cost_usd": total_cost,
            }
        )
    return rows


def evaluate_objective(
    results_path: str | Path,
    *,
    summary_path: str | Path | None = None,
    group_by: str | None = None,
) -> list[dict[str, Any]]:
    records = read_jsonl(results_path)
    rows = _objective_rows(records)
    if group_by:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            grouped[str(record.get("metadata", {}).get(group_by, "unknown"))].append(record)
        for label in sorted(grouped):
            rows.extend(_objective_rows(grouped[label], label=label))
    destination = (
        Path(summary_path) if summary_path else Path(results_path).with_name("summary.csv")
    )
    write_csv(destination, rows)
    return rows


def _validate_judgment(value: Any, fact_count: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Judge output must be a JSON object")
    claims = value.get("claim_labels")
    coverage = value.get("reference_coverage")
    if not isinstance(claims, list) or not isinstance(coverage, list):
        raise ValueError("claim_labels and reference_coverage must be lists")
    for claim in claims:
        if not isinstance(claim, dict) or claim.get("label") not in {
            "supported",
            "contradicted",
            "absent",
        }:
            raise ValueError("Each claim needs a supported, contradicted, or absent label")
    coverage_by_index: dict[int, bool] = {}
    for item in coverage:
        if not isinstance(item, dict):
            raise ValueError("Coverage items must be objects")
        index = item.get("fact_index")
        covered = item.get("covered")
        if not isinstance(index, int) or not isinstance(covered, bool):
            raise ValueError("Coverage needs integer fact_index and boolean covered")
        if 0 <= index < fact_count:
            coverage_by_index[index] = covered
    if len(coverage) != fact_count or set(coverage_by_index) != set(range(fact_count)):
        raise ValueError("Coverage must contain every reference fact exactly once")
    return {"claim_labels": claims, "reference_coverage": coverage}


def parse_biography_judgment(text: str, fact_count: int) -> dict[str, Any]:
    return _validate_judgment(json.loads(extract_json_object(text)), fact_count)


def judgment_metrics(judgment: dict[str, Any]) -> dict[str, float | int]:
    claims = judgment["claim_labels"]
    total = len(claims)
    supported = sum(item["label"] == "supported" for item in claims)
    contradicted = sum(item["label"] == "contradicted" for item in claims)
    absent = sum(item["label"] == "absent" for item in claims)
    coverage_items = judgment["reference_coverage"]
    covered = sum(bool(item["covered"]) for item in coverage_items)
    precision = supported / total if total else 0.0
    coverage = covered / len(coverage_items) if coverage_items else 0.0
    f1 = 2 * precision * coverage / (precision + coverage) if precision + coverage else 0.0
    return {
        "claims": total,
        "supported_claims": supported,
        "contradicted_claims": contradicted,
        "absent_claims": absent,
        "supported_precision": precision,
        "contradiction_rate": contradicted / total if total else 0.0,
        "unknown_rate": absent / total if total else 0.0,
        "reference_coverage": coverage,
        "coverage_precision_f1": f1,
    }


def judge_biography(
    *,
    client: LLMClient,
    model: str,
    name: str,
    facts: Sequence[str],
    biography: str,
    temperature: float = 0.0,
    max_tokens: int = 1200,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prompt = biography_judge_prompt(name, facts, biography)
    generations: list[dict[str, Any]] = []
    first = client.generate(
        [Message("user", prompt)],
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=0,
    )
    generations.append(first.to_dict())
    try:
        return parse_biography_judgment(first.text, len(facts)), generations
    except (ValueError, json.JSONDecodeError) as first_error:
        repair = client.generate(
            [Message("user", json_repair_prompt(first.text, str(first_error)))],
            model=model,
            temperature=0.0,
            max_tokens=max_tokens,
            seed=1,
        )
        generations.append(repair.to_dict())
        return parse_biography_judgment(repair.text, len(facts)), generations


def _candidate_text(record: dict[str, Any], agent_id: int, round_index: int) -> str:
    return str(record["agents"][agent_id]["turns"][round_index]["response"]["text"])


def evaluate_biographies(
    results_path: str | Path,
    *,
    client: LLMClient,
    judge_model: str,
    summary_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    records = read_jsonl(results_path)
    details_path = Path(results_path).with_name("biography_judgments.jsonl")
    if details_path.exists():
        details_path.unlink()
    by_strategy: dict[str, list[dict[str, float | int]]] = defaultdict(list)

    for record in records:
        facts = [str(item) for item in record["reference"]]
        name = str(record.get("metadata", {}).get("name", record["example_id"]))
        candidates = {"direct": _candidate_text(record, 0, 0)}
        final_round_index = len(record["agents"][0]["turns"]) - 1
        for agent in record["agents"]:
            agent_id = int(agent["agent_id"])
            candidates[f"debate_agent_{agent_id}"] = _candidate_text(
                record, agent_id, final_round_index
            )
        for strategy, biography in candidates.items():
            judgment, judge_calls = judge_biography(
                client=client,
                model=judge_model,
                name=name,
                facts=facts,
                biography=biography,
            )
            metrics = judgment_metrics(judgment)
            by_strategy[strategy].append(metrics)
            append_jsonl(
                details_path,
                {
                    "example_id": record["example_id"],
                    "strategy": strategy,
                    "judge_model": judge_model,
                    "biography": biography,
                    "judgment": judgment,
                    "metrics": metrics,
                    "judge_calls": judge_calls,
                },
            )

    rows: list[dict[str, Any]] = []
    metric_names = [
        "supported_precision",
        "contradiction_rate",
        "unknown_rate",
        "reference_coverage",
        "coverage_precision_f1",
    ]
    for strategy in sorted(by_strategy):
        values = by_strategy[strategy]
        row: dict[str, Any] = {"strategy": strategy, "n": len(values)}
        row.update(
            {metric: mean(float(item[metric]) for item in values) for metric in metric_names}
        )
        rows.append(row)

    final_agent_rows = [row for row in rows if row["strategy"].startswith("debate_agent_")]
    if final_agent_rows:
        rows.append(
            {
                "strategy": "debate_agents_mean",
                "n": len(records),
                **{
                    metric: mean(float(row[metric]) for row in final_agent_rows)
                    for metric in metric_names
                },
            }
        )
    destination = (
        Path(summary_path) if summary_path else Path(results_path).with_name("summary.csv")
    )
    write_csv(destination, rows)
    return rows

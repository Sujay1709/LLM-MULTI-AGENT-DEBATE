"""Objective and reference-grounded biography evaluation."""

from __future__ import annotations

import json
import math
import random
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

    generations: list[dict[str, Any]] = []
    if strategy == "self_consistency":
        generations = [agent["turns"][0]["response"] for agent in record["agents"]]
    elif strategy == "self_refinement":
        generations = [record["agents"][0]["turns"][0]["response"]]
        baseline = record.get("baselines", {}).get(strategy, {})
        generations.extend(turn["response"] for turn in baseline.get("revision_turns", []))
    elif strategy == "budget_matched_self_consistency":
        generations = [agent["turns"][0]["response"] for agent in record["agents"]]
        baseline = record.get("baselines", {}).get(strategy, {})
        generations.extend(turn["response"] for turn in baseline.get("additional_samples", []))
    else:
        maximum_round: int | None = None
        if strategy.startswith("debate_round_"):
            maximum_round = int(strategy.rsplit("_", 1)[1]) - 1
        for agent in record["agents"]:
            for turn in agent["turns"]:
                if maximum_round is None or int(turn["round"]) <= maximum_round:
                    generations.append(turn["response"])

    usage = [_generation_usage(generation) for generation in generations]
    return (
        sum(item[0] for item in usage),
        sum(item[1] for item in usage),
        sum(item[2] for item in usage),
    )


def _canonical_reference(reference: Any, task: str) -> str | None:
    if task in {"arithmetic", "gsm8k"}:
        return normalize_number(reference)
    if task == "mmlu":
        return str(reference).strip().upper()
    return str(reference)


def _strategy_names(records: Sequence[dict[str, Any]]) -> list[str]:
    names = ["direct", "self_consistency"]
    optional = ["self_refinement", "budget_matched_self_consistency"]
    for name in optional:
        if any(name in record.get("predictions", {}) for record in records):
            names.append(name)

    max_rounds = max(
        (len(record.get("predictions", {}).get("rounds", [])) for record in records),
        default=0,
    )
    names.extend(f"debate_round_{round_number}" for round_number in range(2, max_rounds))
    names.append("debate")
    return names


def _strategy_prediction(record: dict[str, Any], strategy: str) -> str | None:
    if strategy == "self_consistency":
        saved = record.get("predictions", {}).get(strategy)
        if saved is not None:
            return saved
        rounds = record.get("predictions", {}).get("rounds", [])
        return rounds[0].get("selected_answer") if rounds else None
    if strategy.startswith("debate_round_"):
        round_index = int(strategy.rsplit("_", 1)[1]) - 1
        rounds = record.get("predictions", {}).get("rounds", [])
        if round_index >= len(rounds):
            return None
        return rounds[round_index].get("selected_answer")
    return record.get("predictions", {}).get(strategy)


def _strategy_tie(record: dict[str, Any], strategy: str) -> bool:
    if strategy == "self_consistency":
        rounds = record.get("predictions", {}).get("rounds", [])
        return bool(rounds and rounds[0].get("tie"))
    if strategy == "budget_matched_self_consistency":
        return bool(record.get("baselines", {}).get(strategy, {}).get("tie"))
    if strategy.startswith("debate_round_"):
        round_index = int(strategy.rsplit("_", 1)[1]) - 1
        rounds = record.get("predictions", {}).get("rounds", [])
        return bool(round_index < len(rounds) and rounds[round_index].get("tie"))
    if strategy == "debate":
        return bool(record.get("predictions", {}).get("tie"))
    return False


def _is_correct(record: dict[str, Any], strategy: str) -> bool:
    prediction = _strategy_prediction(record, strategy)
    reference = _canonical_reference(record["reference"], record["task"])
    return prediction is not None and str(prediction) == str(reference)


def _objective_rows(
    records: Sequence[dict[str, Any]],
    strategies: Sequence[str],
    label: str = "overall",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for strategy in strategies:
        scored = correct = parse_failures = ties = total_tokens = 0
        total_latency = total_cost = 0.0
        for record in records:
            prediction = _strategy_prediction(record, strategy)
            reference = _canonical_reference(record["reference"], record["task"])
            if prediction is None:
                parse_failures += 1
            else:
                scored += 1
                correct += int(str(prediction) == str(reference))
            ties += int(_strategy_tie(record, strategy))
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
                "accuracy_all": correct / len(records) if records else 0.0,
                "parse_failures": parse_failures,
                "ties": ties,
                "total_tokens": total_tokens,
                "mean_latency_seconds": total_latency / len(records) if records else 0.0,
                "total_cost_usd": total_cost,
            }
        )
    return rows


def paired_bootstrap_difference(
    baseline: Sequence[bool],
    candidate: Sequence[bool],
    *,
    samples: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Return a percentile interval for paired accuracy(candidate) - accuracy(baseline)."""
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("Paired bootstrap inputs must have the same non-zero length")
    if samples < 1:
        raise ValueError("Bootstrap samples must be at least one")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("Confidence level must be between zero and one")

    differences = [int(right) - int(left) for left, right in zip(baseline, candidate, strict=True)]
    rng = random.Random(seed)
    size = len(differences)
    estimates = sorted(
        sum(differences[rng.randrange(size)] for _ in range(size)) / size
        for _ in range(samples)
    )
    tail = (1.0 - confidence_level) / 2.0
    low_index = max(0, math.floor(tail * (samples - 1)))
    high_index = min(samples - 1, math.ceil((1.0 - tail) * (samples - 1)))
    return estimates[low_index], estimates[high_index]


def mcnemar_exact(baseline: Sequence[bool], candidate: Sequence[bool]) -> tuple[int, int, float]:
    """Return discordant counts and the exact two-sided McNemar p-value."""
    if len(baseline) != len(candidate):
        raise ValueError("McNemar inputs must have the same length")
    baseline_only = sum(left and not right for left, right in zip(baseline, candidate, strict=True))
    candidate_only = sum(
        right and not left for left, right in zip(baseline, candidate, strict=True)
    )
    discordant = baseline_only + candidate_only
    if discordant == 0:
        return baseline_only, candidate_only, 1.0

    cutoff = min(baseline_only, candidate_only)
    log_probabilities = [
        math.lgamma(discordant + 1)
        - math.lgamma(index + 1)
        - math.lgamma(discordant - index + 1)
        - discordant * math.log(2.0)
        for index in range(cutoff + 1)
    ]
    maximum = max(log_probabilities)
    cumulative = math.exp(maximum) * sum(
        math.exp(value - maximum) for value in log_probabilities
    )
    return baseline_only, candidate_only, min(1.0, 2.0 * cumulative)


def _comparison_rows(
    records: Sequence[dict[str, Any]],
    strategies: Sequence[str],
    *,
    label: str,
    bootstrap_samples: int,
    confidence_level: float,
) -> list[dict[str, Any]]:
    baseline = [_is_correct(record, "direct") for record in records]
    rows: list[dict[str, Any]] = []
    for strategy in strategies:
        if strategy == "direct":
            continue
        candidate = [_is_correct(record, strategy) for record in records]
        low, high = paired_bootstrap_difference(
            baseline,
            candidate,
            samples=bootstrap_samples,
            confidence_level=confidence_level,
        )
        baseline_only, candidate_only, p_value = mcnemar_exact(baseline, candidate)
        rows.append(
            {
                "group": label,
                "baseline": "direct",
                "strategy": strategy,
                "n": len(records),
                "baseline_correct": sum(baseline),
                "strategy_correct": sum(candidate),
                "accuracy_difference": mean(candidate) - mean(baseline),
                "confidence_level": confidence_level,
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "mcnemar_baseline_only": baseline_only,
                "mcnemar_strategy_only": candidate_only,
                "mcnemar_p_value": p_value,
            }
        )
    return rows


def evaluate_objective(
    results_path: str | Path,
    *,
    summary_path: str | Path | None = None,
    comparisons_path: str | Path | None = None,
    group_by: str | None = None,
    bootstrap_samples: int = 2000,
    confidence_level: float = 0.95,
) -> list[dict[str, Any]]:
    records = read_jsonl(results_path)
    if not records:
        raise ValueError("Cannot evaluate an empty results file")
    strategies = _strategy_names(records)
    rows = _objective_rows(records, strategies)
    comparison_rows = _comparison_rows(
        records,
        strategies,
        label="overall",
        bootstrap_samples=bootstrap_samples,
        confidence_level=confidence_level,
    )
    if group_by:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            grouped[str(record.get("metadata", {}).get(group_by, "unknown"))].append(record)
        for label in sorted(grouped):
            group_records = grouped[label]
            rows.extend(_objective_rows(group_records, strategies, label=label))
            comparison_rows.extend(
                _comparison_rows(
                    group_records,
                    strategies,
                    label=label,
                    bootstrap_samples=bootstrap_samples,
                    confidence_level=confidence_level,
                )
            )
    destination = (
        Path(summary_path) if summary_path else Path(results_path).with_name("summary.csv")
    )
    write_csv(destination, rows)
    comparison_destination = (
        Path(comparisons_path)
        if comparisons_path
        else Path(results_path).with_name("comparisons.csv")
    )
    write_csv(comparison_destination, comparison_rows)
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

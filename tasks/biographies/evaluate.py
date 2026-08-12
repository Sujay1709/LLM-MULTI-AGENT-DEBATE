"""Evaluate biography claims against the closed reference facts with an LLM judge."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from multiagent_debate.cli import resolve_results
from multiagent_debate.clients import make_client
from multiagent_debate.evaluation import evaluate_biographies
from multiagent_debate.io import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL"))
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()
    results = resolve_results(args.input, args.outputs)
    records = read_jsonl(results)
    if not records:
        raise SystemExit("The selected results file is empty")
    judge_model = args.judge_model or str(records[0]["config"]["model"])
    rows = evaluate_biographies(
        results,
        client=make_client(judge_model, max_retries=args.max_retries),
        judge_model=judge_model,
    )
    for row in rows:
        print(
            f"{row['strategy']}: precision={row['supported_precision']:.3f}, "
            f"coverage={row['reference_coverage']:.3f}, f1={row['coverage_precision_f1']:.3f}"
        )
    print(results.with_name("summary.csv").resolve())


if __name__ == "__main__":
    main()


"""Evaluate MMLU overall and by subject."""

from __future__ import annotations

import argparse
from pathlib import Path

from multiagent_debate.cli import resolve_results
from multiagent_debate.evaluation import evaluate_objective


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    args = parser.parse_args()
    results = resolve_results(args.input, args.outputs)
    rows = evaluate_objective(
        results,
        group_by="subject",
        bootstrap_samples=args.bootstrap_samples,
        confidence_level=args.confidence_level,
    )
    for row in rows:
        if row["group"] == "overall":
            print(f"{row['strategy']}: accuracy={row['accuracy']:.3f} (n={row['scored']})")
    print(results.with_name("summary.csv").resolve())
    print(results.with_name("comparisons.csv").resolve())


if __name__ == "__main__":
    main()

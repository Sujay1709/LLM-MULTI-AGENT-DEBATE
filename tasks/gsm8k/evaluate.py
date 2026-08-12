"""Evaluate GSM8K direct and debate predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

from multiagent_debate.cli import resolve_results
from multiagent_debate.evaluation import evaluate_objective


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    results = resolve_results(args.input, args.outputs)
    rows = evaluate_objective(results)
    for row in rows:
        print(f"{row['strategy']}: accuracy={row['accuracy']:.3f} (n={row['scored']})")
    print(results.with_name("summary.csv").resolve())


if __name__ == "__main__":
    main()


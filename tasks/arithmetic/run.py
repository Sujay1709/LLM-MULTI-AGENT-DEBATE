"""Generate and evaluate synthetic arithmetic debates."""

from __future__ import annotations

import argparse
import random

from multiagent_debate.cli import add_baseline_arguments, add_run_arguments, baseline_strategies
from multiagent_debate.clients import make_client
from multiagent_debate.evaluation import evaluate_objective
from multiagent_debate.models import DebateConfig, Example
from multiagent_debate.parsing import parse_numeric_answer
from multiagent_debate.runner import run_experiment


def build_examples(seed: int, count: int = 100) -> list[Example]:
    rng = random.Random(seed)
    examples: list[Example] = []
    for index in range(count):
        a, b, c, d, e, f = [rng.randint(0, 29) for _ in range(6)]
        expression = f"{a}+{b}*{c}+{d}-{e}*{f}"
        answer = a + b * c + d - e * f
        examples.append(
            Example(
                example_id=f"arithmetic-{index:04d}",
                prompt=(
                    f"Calculate {expression}. Explain the order of operations briefly. "
                    "End with one numeric answer in the form \\boxed{answer}."
                ),
                reference=answer,
                metadata={"expression": expression},
            )
        )
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_arguments(parser, default_agents=2, default_rounds=3)
    add_baseline_arguments(parser)
    parser.add_argument("--no-evaluate", action="store_true")
    args = parser.parse_args()
    config = DebateConfig(
        task="arithmetic",
        model=args.model,
        num_agents=args.agents,
        num_rounds=args.rounds,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
        limit=args.limit,
        baseline_strategies=baseline_strategies(args.baselines),
        max_retries=args.max_retries,
    )
    results = run_experiment(
        build_examples(args.seed),
        config,
        make_client(args.model, max_retries=args.max_retries),
        parse_numeric_answer,
        answer_instruction="End with one numeric answer in the form \\boxed{answer}.",
        output_base=args.output_dir,
        resume=args.resume,
    )
    if not args.no_evaluate:
        evaluate_objective(results)
        print(f"Summary: {results.with_name('summary.csv').resolve()}")
        print(f"Comparisons: {results.with_name('comparisons.csv').resolve()}")


if __name__ == "__main__":
    main()

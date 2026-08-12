"""Generate direct and multi-agent debate answers for GSM8K."""

from __future__ import annotations

import argparse
from pathlib import Path

from multiagent_debate.cli import add_run_arguments
from multiagent_debate.clients import make_client
from multiagent_debate.evaluation import evaluate_objective
from multiagent_debate.models import DebateConfig, Example
from multiagent_debate.parsing import parse_numeric_answer
from multiagent_debate.runner import run_experiment


def load_examples(data_dir: Path) -> tuple[list[Example], str]:
    if not data_dir.exists():
        raise SystemExit(f"Dataset not found at {data_dir}. Run: python download_data.py")
    from datasets import load_from_disk

    dataset = load_from_disk(str(data_dir))
    examples = [
        Example(
            example_id=f"gsm8k-test-{index:04d}",
            prompt=(
                f"Solve this grade-school math problem:\n\n{row['question']}\n\n"
                "Explain your reasoning and end with one number in the form \\boxed{answer}."
            ),
            reference=parse_numeric_answer(str(row["answer"])) or "",
            metadata={"source": "openai/gsm8k", "split": "test", "row": index},
        )
        for index, row in enumerate(dataset)
    ]
    return examples, str(dataset._fingerprint)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_arguments(parser, default_agents=3, default_rounds=2)
    parser.add_argument("--data-dir", type=Path, default=Path("data/gsm8k"))
    parser.add_argument("--no-evaluate", action="store_true")
    args = parser.parse_args()
    examples, fingerprint = load_examples(args.data_dir)
    config = DebateConfig(
        task="gsm8k",
        model=args.model,
        num_agents=args.agents,
        num_rounds=args.rounds,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
        limit=args.limit,
        max_retries=args.max_retries,
        metadata={"dataset": "openai/gsm8k", "configuration": "main", "fingerprint": fingerprint},
    )
    results = run_experiment(
        examples,
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


if __name__ == "__main__":
    main()


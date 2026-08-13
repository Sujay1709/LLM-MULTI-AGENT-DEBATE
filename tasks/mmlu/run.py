"""Generate direct and multi-agent debate answers for MMLU."""

from __future__ import annotations

import argparse
from pathlib import Path

from multiagent_debate.cli import add_baseline_arguments, add_run_arguments, baseline_strategies
from multiagent_debate.clients import make_client
from multiagent_debate.evaluation import evaluate_objective
from multiagent_debate.models import DebateConfig, Example
from multiagent_debate.parsing import parse_choice_answer
from multiagent_debate.runner import run_experiment

LETTERS = "ABCD"


def load_examples(data_dir: Path, subject: str | None) -> tuple[list[Example], str]:
    if not data_dir.exists():
        raise SystemExit(f"Dataset not found at {data_dir}. Run: python download_data.py")
    from datasets import load_from_disk

    dataset = load_from_disk(str(data_dir))
    examples: list[Example] = []
    for index, row in enumerate(dataset):
        row_subject = str(row["subject"])
        if subject and row_subject != subject:
            continue
        choices = list(row["choices"])
        formatted = "\n".join(
            f"{letter}) {choice}" for letter, choice in zip(LETTERS, choices, strict=True)
        )
        raw_answer = row["answer"]
        answer = (
            LETTERS[int(raw_answer)]
            if isinstance(raw_answer, int)
            else str(raw_answer).upper()
        )
        examples.append(
            Example(
                example_id=f"mmlu-{row_subject}-{index:05d}",
                prompt=(
                    "Answer this multiple-choice question from "
                    f"{row_subject.replace('_', ' ')}.\n\n"
                    f"{row['question']}\n{formatted}\n\n"
                    "Explain briefly and end with exactly one option in the form (X)."
                ),
                reference=answer,
                metadata={"source": "cais/mmlu", "subject": row_subject, "row": index},
            )
        )
    if not examples:
        raise SystemExit(f"No MMLU examples matched subject={subject!r}")
    return examples, str(dataset._fingerprint)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_arguments(parser, default_agents=3, default_rounds=2)
    add_baseline_arguments(parser)
    parser.add_argument("--data-dir", type=Path, default=Path("data/mmlu"))
    parser.add_argument("--subject", help="For example: machine_learning")
    parser.add_argument("--no-evaluate", action="store_true")
    args = parser.parse_args()
    examples, fingerprint = load_examples(args.data_dir, args.subject)
    config = DebateConfig(
        task="mmlu",
        model=args.model,
        num_agents=args.agents,
        num_rounds=args.rounds,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
        limit=args.limit,
        baseline_strategies=baseline_strategies(args.baselines),
        max_retries=args.max_retries,
        metadata={
            "dataset": "cais/mmlu",
            "configuration": "all",
            "fingerprint": fingerprint,
            "subject_filter": args.subject,
        },
    )
    results = run_experiment(
        examples,
        config,
        make_client(args.model, max_retries=args.max_retries),
        parse_choice_answer,
        answer_instruction="End with exactly one option in the form (X), where X is A, B, C, or D.",
        output_base=args.output_dir,
        resume=args.resume,
    )
    if not args.no_evaluate:
        evaluate_objective(results, group_by="subject")
        print(f"Summary: {results.with_name('summary.csv').resolve()}")
        print(f"Comparisons: {results.with_name('comparisons.csv').resolve()}")


if __name__ == "__main__":
    main()

"""Generate direct and debated biographies from a cited reference set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multiagent_debate.cli import add_run_arguments
from multiagent_debate.clients import make_client
from multiagent_debate.models import DebateConfig, Example
from multiagent_debate.runner import run_experiment


def parse_biography(text: str) -> str | None:
    value = text.strip()
    return value or None


def load_examples(path: Path) -> list[Example]:
    examples: list[Example] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            name = str(row["name"])
            facts = [str(fact) for fact in row["facts"]]
            if not facts:
                raise ValueError(f"Biography row {line_number} has no reference facts")
            examples.append(
                Example(
                    example_id=str(row.get("id", name.lower().replace(" ", "-"))),
                    prompt=(
                        f"Write a concise bullet-point biography of {name}, emphasizing their "
                        "contributions to computing. Put one independently checkable fact per "
                        "bullet."
                    ),
                    reference=facts,
                    metadata={"name": name, "sources": list(row.get("sources", []))},
                )
            )
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_arguments(parser, default_agents=3, default_rounds=2)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).parent / "data" / "computer_scientists.jsonl",
    )
    args = parser.parse_args()
    config = DebateConfig(
        task="biographies",
        model=args.model,
        num_agents=args.agents,
        num_rounds=args.rounds,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
        limit=args.limit,
        aggregation="paired",
        max_retries=args.max_retries,
        metadata={"dataset": str(args.data.resolve()), "reference_mode": "closed_set"},
    )
    run_experiment(
        load_examples(args.data),
        config,
        make_client(args.model, max_retries=args.max_retries),
        parse_biography,
        answer_instruction="Return a revised bullet list with one factual claim per bullet.",
        output_base=args.output_dir,
        resume=args.resume,
    )
    print(
        "Generation complete. Run python evaluate.py with an appropriate judge model to score it."
    )


if __name__ == "__main__":
    main()

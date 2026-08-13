"""Small command-line helpers shared by task scripts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from .baselines import BUDGET_MATCHED_SELF_CONSISTENCY, SELF_REFINEMENT


def add_run_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_agents: int,
    default_rounds: int,
) -> None:
    load_dotenv()
    parser.add_argument("--model", default=os.getenv("MODEL", "fake/deterministic"))
    parser.add_argument("--agents", type=int, default=default_agents)
    parser.add_argument("--rounds", type=int, default=default_rounds)
    parser.add_argument("--limit", type=int, default=10, help="0 means all examples")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--resume", type=Path, help="Existing run directory to continue")


def add_baseline_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--baselines",
        nargs="+",
        choices=["self-refinement", "budget-matched-self-consistency", "all"],
        default=[],
        help=(
            "Optional paid-call baselines. First-round self-consistency is always reported at no "
            "additional cost."
        ),
    )


def baseline_strategies(values: list[str]) -> tuple[str, ...]:
    if "all" in values:
        return (SELF_REFINEMENT, BUDGET_MATCHED_SELF_CONSISTENCY)
    mapping = {
        "self-refinement": SELF_REFINEMENT,
        "budget-matched-self-consistency": BUDGET_MATCHED_SELF_CONSISTENCY,
    }
    return tuple(mapping[value] for value in values)


def resolve_results(path: Path | None, outputs: Path = Path("outputs")) -> Path:
    if path is not None:
        return path
    from .io import latest_results

    return latest_results(outputs)

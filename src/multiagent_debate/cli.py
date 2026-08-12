"""Small command-line helpers shared by task scripts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv


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


def resolve_results(path: Path | None, outputs: Path = Path("outputs")) -> Path:
    if path is not None:
        return path
    from .io import latest_results

    return latest_results(outputs)


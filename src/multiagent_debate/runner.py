"""Dataset-level experiment runner shared by task entry points."""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .clients import LLMClient
from .engine import AnswerParser, run_debate
from .io import append_jsonl, completed_ids, create_run_directory, experiment_id
from .models import DebateConfig, Example


def run_experiment(
    examples: Sequence[Example],
    config: DebateConfig,
    client: LLMClient,
    parse_answer: AnswerParser,
    *,
    answer_instruction: str,
    output_base: str | Path = "outputs",
    resume: str | Path | None = None,
    progress: Callable[[str], Any] = print,
) -> Path:
    if config.num_agents < 1 or config.num_rounds < 1:
        raise ValueError("num_agents and num_rounds must both be at least one")

    if resume:
        run_dir = Path(resume)
        results_path = run_dir / "results.jsonl"
        if not results_path.exists():
            raise FileNotFoundError(f"Resume file not found: {results_path}")
        config_path = run_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Resume configuration not found: {config_path}")
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        expected_id = experiment_id(config)
        if saved.get("experiment_id") != expected_id:
            raise ValueError(
                "Resume configuration does not match this command. Use the original settings or "
                "start a new run."
            )
        done = completed_ids(results_path)
        identifier = expected_id
    else:
        run_dir, identifier = create_run_directory(output_base, config)
        results_path = run_dir / "results.jsonl"
        done = set()

    selected = list(examples)
    random.Random(config.seed).shuffle(selected)
    if config.limit > 0:
        selected = selected[: config.limit]

    pending = [example for example in selected if example.example_id not in done]
    for index, example in enumerate(pending, start=1):
        progress(f"[{index}/{len(pending)}] {example.example_id}")
        record = run_debate(
            example,
            config,
            client,
            parse_answer,
            answer_instruction=answer_instruction,
            experiment_id=identifier,
        )
        append_jsonl(results_path, record)

    progress(f"Results: {results_path.resolve()}")
    return results_path

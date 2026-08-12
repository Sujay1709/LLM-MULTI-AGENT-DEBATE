"""Experiment directories, JSONL persistence, and CSV reports."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models import DebateConfig


def _slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return value[:60] or "model"


def experiment_id(config: DebateConfig) -> str:
    canonical = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def create_run_directory(base: str | Path, config: DebateConfig) -> tuple[Path, str]:
    identifier = experiment_id(config)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(base) / f"{timestamp}_{_slug(config.task)}_{_slug(config.model)}_{identifier}"
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(
        run_dir / "config.json",
        {
            "schema_version": "1.0",
            "experiment_id": identifier,
            "created_at": timestamp,
            "config": config.to_dict(),
        },
    )
    return run_dir, identifier


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(destination)


def append_jsonl(path: str | Path, value: Mapping[str, Any]) -> None:
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), ensure_ascii=False) + "\n")
        handle.flush()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc
    return records


def completed_ids(path: str | Path) -> set[str]:
    destination = Path(path)
    if not destination.exists():
        return set()
    return {str(record["example_id"]) for record in read_jsonl(destination)}


def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise ValueError("Cannot write an empty CSV report")
    fieldnames: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)


def latest_results(directory: str | Path = "outputs") -> Path:
    candidates = sorted(Path(directory).glob("*/results.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"No results.jsonl found under {Path(directory).resolve()}")
    return candidates[-1]


"""Download the Hugging Face MMLU test split for repeatable offline runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/mmlu"))
    args = parser.parse_args()
    from datasets import load_dataset

    dataset = load_dataset("cais/mmlu", "all", split="test")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(args.output))
    metadata = {
        "dataset": "cais/mmlu",
        "configuration": "all",
        "split": "test",
        "fingerprint": dataset._fingerprint,
        "rows": len(dataset),
    }
    args.output.with_name("mmlu_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {len(dataset)} examples to {args.output.resolve()}")


if __name__ == "__main__":
    main()


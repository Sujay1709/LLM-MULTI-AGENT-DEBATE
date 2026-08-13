import os
import subprocess
import sys
from pathlib import Path

from multiagent_debate.io import read_jsonl

ROOT = Path(__file__).parents[1]


def test_arithmetic_cli_smoke(tmp_path) -> None:
    env = {**os.environ, "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}"}
    subprocess.run(
        [
            sys.executable,
            "run.py",
            "--model",
            "fake/deterministic",
            "--limit",
            "2",
            "--output-dir",
            str(tmp_path),
            "--baselines",
            "all",
        ],
        cwd=ROOT / "tasks" / "arithmetic",
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    results = next(tmp_path.glob("*/results.jsonl"))
    assert len(read_jsonl(results)) == 2
    assert results.with_name("summary.csv").exists()
    assert results.with_name("comparisons.csv").exists()

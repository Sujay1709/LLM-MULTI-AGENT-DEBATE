import json
import sys
from types import SimpleNamespace

from tasks.biographies.run import load_examples as load_biographies
from tasks.gsm8k.run import load_examples as load_gsm8k
from tasks.mmlu.run import load_examples as load_mmlu


class FakeDataset(list):
    _fingerprint = "fixture-fingerprint"


def test_gsm8k_adapter_with_fixture(monkeypatch, tmp_path) -> None:
    data_dir = tmp_path / "gsm8k"
    data_dir.mkdir()
    dataset = FakeDataset([{"question": "What is 1+1?", "answer": "work\n#### 2"}])
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_from_disk=lambda _: dataset))
    examples, fingerprint = load_gsm8k(data_dir)
    assert examples[0].reference == "2"
    assert fingerprint == "fixture-fingerprint"


def test_mmlu_adapter_with_fixture(monkeypatch, tmp_path) -> None:
    data_dir = tmp_path / "mmlu"
    data_dir.mkdir()
    dataset = FakeDataset(
        [
            {
                "subject": "machine_learning",
                "question": "Pick",
                "choices": ["x", "y", "z", "w"],
                "answer": 2,
            }
        ]
    )
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_from_disk=lambda _: dataset))
    examples, _ = load_mmlu(data_dir, "machine_learning")
    assert examples[0].reference == "C"
    assert "C) z" in examples[0].prompt


def test_biography_adapter_accepts_custom_jsonl(tmp_path) -> None:
    data_path = tmp_path / "bios.jsonl"
    row = {
        "id": "person",
        "name": "Person",
        "facts": ["A fact"],
        "sources": ["https://example.org"],
    }
    data_path.write_text(
        json.dumps(row) + "\n",
        encoding="utf-8",
    )
    examples = load_biographies(data_path)
    assert examples[0].reference == ["A fact"]
    assert examples[0].metadata["sources"] == ["https://example.org"]

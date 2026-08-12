# MMLU

MMLU measures multiple-choice knowledge and reasoning across 57 subjects. Download the official
`cais/mmlu` aggregate test split once:

```bash
python download_data.py
python run.py --model openai/gpt-5-mini --subject machine_learning
python evaluate.py
```

Omit `--subject` to sample across all subjects. The summary contains overall and per-subject rows.
This harness is zero-shot by design; it does not reproduce MMLU's separate five-shot protocol.


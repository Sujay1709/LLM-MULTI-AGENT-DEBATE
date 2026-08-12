# GSM8K

GSM8K contains multi-step grade-school word problems. Download the official Hugging Face test split
once, then run a seeded 10-example experiment:

```bash
python download_data.py
python run.py --model openai/gpt-5-mini
python evaluate.py
```

The download script records the Hugging Face dataset fingerprint. `--limit 0` runs all 1,319 test
examples and can create substantial API cost with three agents and two rounds.


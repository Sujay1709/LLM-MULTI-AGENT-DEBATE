# Arithmetic

This task generates deterministic expressions of the form `a+b*c+d-e*f`. Python computes the
reference with normal operator precedence; the language model must explain and finish with a boxed
number.

```bash
python run.py --model fake/deterministic
python evaluate.py
```

Compare debate with private revision and compute-matched independent sampling:

```bash
python run.py --model fake/deterministic --baselines all
python evaluate.py --bootstrap-samples 5000
```

Use a live model such as `openai/gpt-5-mini` by configuring its provider key in the root `.env`.

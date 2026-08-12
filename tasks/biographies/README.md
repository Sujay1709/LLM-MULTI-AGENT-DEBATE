# Biographies

This task measures whether debate changes factuality in short computer-scientist biographies. The
included JSONL contains a deliberately small set of facts with institutional source URLs. You can
replace it with any file using `id`, `name`, `facts`, and `sources` fields.

```bash
python run.py --model openai/gpt-5-mini
python evaluate.py --judge-model openai/gpt-5
```

Use a judge stronger than and preferably different from the generation model. The evaluator labels
unsupported-by-reference claims as `absent`; that label means *unverifiable from this closed set*,
not necessarily false. Evaluation requires one judge call for the direct answer and one for each
agent's final biography.


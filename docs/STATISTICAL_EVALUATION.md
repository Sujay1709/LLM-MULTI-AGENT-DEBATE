# Strong Baselines and Statistical Evaluation

Multi-agent debate uses more inference than one direct answer. A fair experiment therefore needs
baselines that separate the effect of *discussion* from the effect of simply sampling or revising
more often.

## Strategies

Every objective-task evaluation reports:

- `direct`: agent 0's independent first answer. The task prompts request an explanation, so this is
  the project's single-agent reasoning baseline.
- `self_consistency`: majority vote over every agent's independent first-round answer. This reuses
  existing calls and measures the benefit of sampling without debate.
- `debate_round_N`: cumulative debate performance after an intermediate round, when applicable.
- `debate`: the final-round majority prediction.

Two optional baselines add model calls:

- `self_refinement`: agent 0 privately audits and revises its own first answer for the same number of
  sequential rounds. It never receives peer responses.
- `budget_matched_self_consistency`: independent answers are sampled until their count equals
  `agents × rounds`, then majority-voted. It uses the same number of conceptual generation calls as
  debate and isolates whether discussion beats additional sampling.

This is call-count matching, not guaranteed token matching: generated response lengths can differ.
Always compare the provider-reported token and cost columns before describing two strategies as
equally expensive.

Run both optional baselines with:

```bash
python run.py --model fake/deterministic --baselines all
```

Or select one:

```bash
python run.py --model openai/gpt-5-mini --baselines self-refinement
python run.py --model openai/gpt-5-mini --baselines budget-matched-self-consistency
```

For `E` examples, `A` agents, and `R` rounds, debate uses `E × A × R` calls. Self-refinement adds
`E × (R - 1)` calls. Budget-matched self-consistency reuses the `A` first-round calls and adds
`E × A × (R - 1)` calls. Use a small `--limit` before starting a live-provider run.

## Reports

`summary.csv` contains per-strategy quality and conceptual efficiency. Calls reused across strategies
are counted in each strategy they support, so the rows should not be summed to estimate total run
cost:

- `accuracy`: correct divided by parseable predictions.
- `accuracy_all`: correct divided by every example, treating parse failures as incorrect.
- Parse failures, ties, conceptual tokens, latency, and provider-reported cost.

`comparisons.csv` compares every strategy with `direct` on the same examples:

- `accuracy_difference`: candidate `accuracy_all` minus direct `accuracy_all`.
- `bootstrap_ci_low` and `bootstrap_ci_high`: percentile interval for the paired accuracy
  difference. Pairing matters because both methods answer the same questions.
- `mcnemar_baseline_only` and `mcnemar_strategy_only`: discordant pairs where only one method was
  correct.
- `mcnemar_p_value`: exact two-sided McNemar test under the null that both directions of answer
  change are equally likely.

The evaluator defaults to 2,000 seeded bootstrap resamples and a 95% interval. Override them with:

```bash
python evaluate.py --bootstrap-samples 5000 --confidence-level 0.95
```

## Interpretation

A positive accuracy difference is descriptive. If its confidence interval crosses zero, the sample
does not clearly distinguish improvement from sampling variability. A small McNemar p-value suggests
asymmetric wins and losses, but it is not the probability that debate is better and does not measure
effect size.

The default 10 examples are a software smoke test, not sufficient evidence for a benchmark claim.
Pre-register the primary comparison, use enough examples, report effect sizes and intervals, and
account for multiple comparisons when testing many strategies, models, tasks, or subjects.

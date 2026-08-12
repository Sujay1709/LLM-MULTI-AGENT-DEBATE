# Instructions for Coding Agents

## Purpose

Maintain this repository as a readable research harness and teaching project. Preserve the distinction
between direct answers, peer debate, aggregation, and evaluation. Never present a smoke-test result as
a benchmark claim.

## Architecture rules

- Keep provider-specific code in `src/multiagent_debate/clients.py`.
- Keep orchestration task-agnostic; benchmark loaders and prompts belong under `tasks/`.
- Do not let an agent see responses from the current incomplete round.
- Treat saved JSONL as a public, versioned research artifact. Prefer additive schema changes.
- Record parse failures and provider errors; never silently discard examples.
- Preserve seeded sampling, dataset fingerprints, configuration metadata, and resumability.

## Security and cost

- Never commit `.env`, API keys, downloaded datasets, or experiment outputs.
- Do not print authorization headers or provider exception payloads that might contain secrets.
- Default new examples to small limits and document call-count/cost implications.
- Do not run live API or full-benchmark experiments without explicit user authorization.

## Changes and tests

Before modifying behavior, explain the experimental consequence. Add focused tests for parsers,
round isolation, ties, serialization, and metrics. Run:

```bash
ruff check .
pytest
python -m compileall -q src tasks tests
```

Mark tests requiring network access or paid APIs with `@pytest.mark.integration`.

## Adding a provider

Prefer a LiteLLM model name. If a custom SDK is necessary, implement `LLMClient.generate`, return a
normalized `Generation`, bound retries, and add an offline fake-client test.

## Adding a task

Create a task folder with `run.py`, `evaluate.py`, and `README.md`. Convert source rows to `Example`,
define an explicit final-answer format, use or add a tested parser, and keep benchmark-specific metrics
out of the debate engine.


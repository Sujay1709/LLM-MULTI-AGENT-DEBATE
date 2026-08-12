"""Prompt construction for peer review and factuality judging."""

from __future__ import annotations

import json
from collections.abc import Sequence


def peer_review_prompt(
    original_prompt: str,
    peer_responses: Sequence[str],
    *,
    answer_instruction: str,
) -> str:
    if not peer_responses:
        return (
            "Re-check your reasoning independently. Correct any mistake you find. "
            + answer_instruction
        )
    peers = "\n\n".join(
        f"Peer agent {index + 1}:\n```\n{response}\n```"
        for index, response in enumerate(peer_responses)
    )
    return (
        "Review the other agents' previous-round solutions below. They are evidence, not "
        "authority: identify disagreements, verify calculations or facts, and revise your answer "
        "only when justified.\n\n"
        f"Original task:\n{original_prompt}\n\n{peers}\n\n{answer_instruction}"
    )


def biography_judge_prompt(name: str, facts: Sequence[str], biography: str) -> str:
    return f"""You are evaluating a generated biography against a closed reference set.

Person: {name}
Reference facts (numbered from zero):
{json.dumps(list(facts), ensure_ascii=False, indent=2)}

Generated biography:
---
{biography}
---

Split the generated biography into factual claims. Label each claim exactly one of:
- supported: directly supported by the reference facts
- contradicted: conflicts with a reference fact
- absent: neither supported nor contradicted by this limited reference

Also mark whether each numbered reference fact is covered by the generated biography.
Return JSON only using this schema:
{{
  "claim_labels": [{{"claim": "...", "label": "supported|contradicted|absent"}}],
  "reference_coverage": [{{"fact_index": 0, "covered": true}}]
}}
Do not use outside knowledge. "absent" means unverifiable here, not necessarily false."""


def json_repair_prompt(malformed: str, error: str) -> str:
    return f"""Repair the following evaluator output into valid JSON matching the requested schema.
Return JSON only. Preserve its intended classifications and do not add new factual judgments.

Validation error: {error}
Malformed output:
{malformed}"""


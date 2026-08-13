# Related Work and Research Roadmap

This project is a clean-room research harness, not a reproduction claim. The papers below were
selected because they cover both evidence *for* multi-agent debate and evidence that simpler or more
diverse baselines can be competitive. Links point to the authors' or publishers' pages; PDFs are not
vendored into this repository.

Machine-readable citations are available in [`references.bib`](../references.bib).

## Core papers

| Paper | Status | Main contribution | Connection to this project |
|---|---|---|---|
| [Improving Factuality and Reasoning in Language Models through Multiagent Debate](https://arxiv.org/abs/2305.14325) (Du et al., 2023) | arXiv preprint | Agents independently answer, inspect peers' answers, and revise over multiple rounds. | Motivates the round-based protocol, arithmetic/GSM8K/MMLU/biography tasks, and direct-versus-debate comparison. |
| [Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate](https://aclanthology.org/2024.emnlp-main.992/) (Liang et al., EMNLP 2024) | Peer reviewed | Studies convergence toward similar reasoning and introduces judge-managed, role-based debate. | Motivates measuring answer convergence and testing distinct agent roles instead of assuming more agents create useful diversity. |
| [ReConcile: Round-Table Conference Improves Reasoning via Consensus among Diverse LLMs](https://aclanthology.org/2024.acl-long.381/) (Chen et al., ACL 2024) | Peer reviewed | Combines heterogeneous models, discussion, confidence scores, and confidence-weighted consensus. | Motivates a future heterogeneous-agent configuration and calibrated confidence-weighted voting baseline. |
| [ChatEval: Towards Better LLM-Based Evaluators Through Multi-Agent Debate](https://openreview.net/forum?id=FQepisCUWu) (Chan et al., ICLR 2024) | Peer reviewed | Uses a multi-agent referee team and studies the importance of diverse role prompts for evaluation. | Informs biography judging and suggests comparing one judge with role-specialized judging, while retaining human audits. |

## Critical evaluation papers

| Paper | Status | Main finding | Experimental consequence |
|---|---|---|---|
| [Rethinking the Bounds of LLM Reasoning: Are Multi-Agent Discussions the Key?](https://aclanthology.org/2024.acl-long.331/) (Wang et al., ACL 2024) | Peer reviewed | Strong single-agent prompts can approach the best multi-agent discussion results in the studied settings. | Debate should be compared with chain-of-thought, self-refinement, and self-consistency—not only a weak direct prompt. |
| [Stop Overvaluing Multi-Agent Debate—We Must Rethink Evaluation and Embrace Model Heterogeneity](https://arxiv.org/abs/2502.08788) (Zhang et al., 2025) | Position paper / arXiv preprint | Reports that several debate methods often fail to beat strong single-agent baselines despite higher inference cost, while heterogeneous models help. | Accuracy, latency, tokens, and cost must be reported together, with homogeneous-versus-heterogeneous ablations. |

## Systems reference

[MALLM: Multi-Agent Large Language Models Framework](https://aclanthology.org/2025.emnlp-demos.29/)
(Becker et al., EMNLP 2025 System Demonstrations) exposes many combinations of personas,
discussion protocols, response generators, and decision rules. It is a useful comparison for
configuration design. This repository intentionally remains smaller and benchmark-focused so its
experimental logic is easy to audit.

## What the literature changes about our claims

The defensible hypothesis is not “debate always makes models better.” It is:

> Under which tasks, model combinations, prompts, and compute budgets does debate improve answer
> quality relative to appropriately strong baselines?

Accordingly, published experiment reports should include:

1. Direct, chain-of-thought, round-zero majority/self-consistency, self-refinement, and debate
   baselines using comparable token budgets.
2. Homogeneous and heterogeneous model teams when provider access permits it.
3. Accuracy or factuality alongside latency, tokens, estimated cost, parse failures, and answer
   changes between rounds.
4. Confidence intervals and paired significance tests on enough examples; the default 10-example
   runs are only smoke tests.
5. A manually audited subset for biography judge results and a clear statement that `absent` means
   unsupported by the closed reference, not necessarily false.

## Three high-value engineering extensions

The first two are the research roadmap; the baseline/statistics portion of item 2 is implemented.

### 1. Heterogeneous agents and calibrated confidence

Allow a run to assign a different provider/model, temperature, system role, and token budget to each
agent. Require structured answers containing both a prediction and confidence, calibrate confidence
on held-out examples, and compare majority vote with confidence-weighted consensus.

**Portfolio signal:** provider abstraction, schema validation, calibration, ablation design, and
cost-aware orchestration.

### 2. Strong baselines and statistical experiment analysis (partially implemented)

The rationale-requesting direct prompt, round-zero self-consistency, self-refinement, and
generation-call-budget-matched self-consistency are now reported with paired bootstrap confidence
intervals and exact McNemar tests. Exact token matching, accuracy-versus-cost Pareto plots, and more
complete environment snapshots remain future work.

**Portfolio signal:** the project evaluates whether a technique works instead of showcasing only
successful examples—a major distinction between an API demo and research engineering.

### 3. Read-only experiment explorer

Build a small Streamlit application that loads sanitized saved runs, replays each debate round,
filters answer flips and regressions, and charts quality, latency, tokens, and cost. The public demo
should use committed sample traces or the deterministic fake client; paid provider execution should
remain disabled by default.

**Portfolio signal:** end-to-end communication of experiments, observability, failure analysis, and
safe deployment without exposing API keys or creating an unlimited public inference endpoint.

## Suggested implementation order

Complete the remaining experiment visualization, implement heterogeneous agents next, and build the
read-only explorer third. That order creates trustworthy results before expanding the presentation
layer.

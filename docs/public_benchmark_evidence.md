# Public benchmark evidence for model selection

This catalog is a starting point for deciding which models deserve a local
test. It is not a universal ranking and does not copy a permanently stale
table of model scores into Evalanche.

Last source review: 2026-08-05

Always open the source and confirm that the exact model version, evaluation
date, prompting mode, tools, token budget, and provider setup are relevant.

## Initial evidence catalog

| Source | Use it for | What it measures | Important limit |
| --- | --- | --- | --- |
| [HELM](https://crfm.stanford.edu/helm/) | Broad capability orientation | Multiple standardized scenarios and metrics under a transparent evaluation framework | Broad averages can hide the one task or constraint that matters locally. |
| [LiveBench](https://github.com/LiveBench/LiveBench) | Current broad reasoning, math, coding, language, data analysis, and instruction-following evidence | Frequently refreshed questions with objective ground-truth scoring | Category results are not a substitute for specialized workflow, modality, or provider testing. |
| [SWE-bench Verified](https://www.swebench.com/verified.html) | Repository-level software issue resolution | Whether an agent can produce patches that pass tests on 500 human-validated tasks | Results depend heavily on the agent harness, tools, budget, and repository distribution, not only the base model. |
| [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) | Function calling, tool selection, multi-turn use, and agentic tool workflows | Executable or structured tool-use behavior across several categories | Real tool schemas, permissions, error recovery, and orchestration may differ materially. |
| [LongBench v2](https://longbench2.github.io/) | Long-context understanding and reasoning | Challenging single-document, multi-document, dialogue, code, and structured-data tasks over long contexts | Multiple-choice accuracy does not prove reliable extraction, citation, or retrieval on a particular document collection. |
| [DocVQA](https://www.docvqa.org/) | Reading document images and using layout | Question answering over document images and OCR-relevant visual content | It does not directly measure full-PDF transport, long multi-page retrieval, or a custom structured schema. |
| [MMMU](https://mmmu-benchmark.github.io/) and [MMMU-Pro](https://aclanthology.org/2025.acl-long.736/) | Multimodal knowledge and reasoning | College-level questions across disciplines using text and images, with MMMU-Pro designed to reduce text-only shortcuts | Academic question answering may transfer poorly to operational images, diagrams, forms, or regulated documents. |

## How to use the catalog

1. Start from the real task, not the source with the most familiar overall
   score.
2. Select the smallest group of evidence sources that measures the required
   capabilities.
3. Record the exact model and setup represented by each result.
4. Note missing candidates, stale results, self-reported submissions, and
   harness differences.
5. Build a shortlist, then use local representative cases for unanswered
   questions.

## Evidence-strength labels

Use these labels when summarizing a model choice:

| Strength | Meaning |
| --- | --- |
| Strong | Same model version, relevant input and output contract, comparable tools and inference settings, transparent reproducible harness, and current results. |
| Moderate | Closely related task or model family, but one or more material setup differences remain. |
| Weak | Marketing claim, broad aggregate, old model version, undisclosed harness, self-report without reproducible evidence, or a task with poor transfer. |
| Missing | The required capability, model, or deployment route was not evaluated. |

Missing evidence is not a zero. It is a reason to verify capabilities or run a
local test.

## What should not be combined into one score

Do not average unrelated public benchmark percentages. Metrics may use
different denominators, prompting policies, tool access, model versions, and
test populations. Keep evidence grouped by the question it can answer:

- Can the model perform the required task?
- Can it accept the required input and output contract?
- Is it reliable on representative cases?
- Is the provider route available and compliant?
- What does it cost at the observed token usage?
- Is latency and throughput acceptable?

Evalanche local benchmarks can answer the last four questions for a specific
deployment. Public benchmarks primarily help avoid testing every model in the
market.

## Maintenance policy

- Retain the source URL and the date it was reviewed.
- Prefer the benchmark owner's site, paper, repository, and evaluation logs.
- Do not preserve a current leaderboard snapshot as evergreen advice.
- Recheck evidence before a consequential decision or when a shortlisted model
  version changes.
- Treat provider-published benchmark numbers as declared evidence until the
  harness and result can be independently checked.
- Add a new source only when it answers a model-choice question not already
  covered well.

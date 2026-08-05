# Choosing a model with Evalanche

Evalanche exists to answer one practical question:

> Which model should I use for my problem, and what evidence supports that
> choice?

There is no universal best model. A defensible choice depends on the task,
required capabilities, acceptable errors, deployment limits, cost, latency,
and the evidence available for the exact model version and setup.

## The shortest useful workflow

1. Describe the real task using representative inputs and required outputs.
2. Remove models that fail a hard capability or deployment requirement.
3. Use relevant public evidence to make a small shortlist.
4. Decide whether the public evidence is close enough to the real task.
5. If it is not, compare the shortlist on a small local evaluation.
6. Choose using quality, cost, latency, reliability, and risk, then record why.

The goal is usually a defensible shortlist, not a ranking of every model that
exists.

## Decision worksheet

Answer these before looking at leaderboard positions:

| Question | Why it matters |
| --- | --- |
| What exact work will the model perform? | Evidence from a different task may not transfer. |
| What input types are required? | Text, image, PDF, audio, tools, and long context are hard capability filters. |
| What output contract is required? | Strict JSON, citations, code patches, and open prose need different tests. |
| Which languages must work? | Average multilingual claims can hide weak performance in a required language. |
| Which errors are unacceptable? | Strict pass rate should reflect real failure severity. |
| Where may data be processed and stored? | Privacy, hosting, region, and approval requirements can remove candidates immediately. |
| What are the cost and throughput limits? | A small quality difference may not justify a large operating-cost difference. |
| What latency is acceptable? | Interactive and batch workloads have different constraints. |
| How reversible is a wrong choice? | High-consequence or hard-to-reverse uses need stronger local evidence. |
| How often will the decision be revisited? | Model versions, pricing, and provider behavior change. |

Write hard requirements as pass or fail rules. Keep preferences, such as
lower cost or lower latency, separate so they do not masquerade as facts.

## Match evidence to the task

Start with the [public benchmark evidence guide](public_benchmark_evidence.md).
Use more than one source when the task crosses categories.

| Task family | Useful public evidence | Local evidence usually needed? |
| --- | --- | --- |
| Broad reasoning and instruction following | HELM and LiveBench | Yes when formatting, domain knowledge, or error costs are specific. |
| Repository-level coding | SWE-bench Verified | Yes when the languages, repositories, tools, or agent harness differ. |
| Function and tool calling | BFCL | Yes when the real tool schemas and multi-step workflow matter. |
| Long-document reasoning | LongBench v2 | Yes when document type, retrieval behavior, or context length differs. |
| Document images and layout | DocVQA | Usually, because actual document formats and extraction fields matter. |
| Multimodal subject reasoning | MMMU or MMMU-Pro | Yes for specialized images, diagrams, or regulated domains. |
| Creative or preference-based work | Human-preference evidence | Yes, with task owners reviewing blinded outputs. |

A public score is strong evidence only when the evaluated model version,
prompting setup, tools, input contract, and task are relevant to the intended
use.

## Decide whether to run a local evaluation

Run one when any of these are true:

- the task is important and public evidence is only loosely related;
- the required modality or provider route is not represented publicly;
- strict structured output matters;
- the language, domain, or document format is specialized;
- two credible candidates are close enough that cost or reliability could
  change the decision;
- an error could create material operational, legal, safety, or reputational
  harm.

Skip or keep the local test very small when public evidence clearly separates
the candidates and the choice is cheap to reverse.

## Use staged local testing

Do not pay for the largest run first.

| Stage | Typical purpose | Promotion rule |
| --- | --- | --- |
| Smoke | Confirm transport, permissions, parsing, token reporting, and cost | The route works and artifacts are complete. |
| Screen | Detect obvious quality or compatibility problems on a small representative set | Only credible candidates continue. |
| Standard | Compare finalists on one fixed benchmark release | Every compared model receives identical cases and scoring. |
| Extended | Increase precision for a consequential or publishable decision | Use only when the extra evidence can change a decision. |

The exact case count depends on task diversity and decision stakes. A model
that fails the screen should not consume a full-run budget merely to make a
table symmetrical. A published leaderboard is different, every listed model
must complete the same frozen release.

## Add and compare models without changing analysis

Each model is one independent file under `configs/models/`. A model manifest
declares its ID, provider route, safe request settings, capabilities, version
metadata, and pricing reference. It does not require a benchmark-specific
analysis edit.

Validate the registry and plan every compatible model:

```bash
python -m evalanche.cli registry-validate
python -m evalanche.cli run-benchmark \
  --benchmark <benchmark_id>@<version> \
  --all-compatible \
  --plan-only
```

For a draft benchmark, run local experiments without registration or
leaderboard publication:

```bash
python -m evalanche.cli run-benchmark \
  --benchmark <benchmark_id>@<version> \
  --model <model_id> \
  --experiment
```

Summarize every completed compatible evaluation, including any model added
later:

```bash
python -m evalanche.cli summarize-benchmark \
  --benchmark <benchmark_id>@<version>
```

The summary discovers results from compatibility-checked run plans. It
rebuilds model totals, cost per request, token use, latency, slices,
field-level accuracy, case outcomes, and every pairwise comparison. No model
name appears in the analysis code.

## Make the final decision explicit

Record:

- the intended task and date;
- hard requirements and who approved them;
- shortlisted model versions and provider routes;
- public evidence used and its access date;
- local dataset, prompt, scorer, and run fingerprints;
- quality, cost, latency, reliability, and missing evidence;
- why the selected model won for this use;
- what event should trigger reconsideration.

Evalanche's [constraint-aware selection policy](model_selection.md) can apply
approved thresholds and weights after evaluation. It does not decide whether
the thresholds or weights are sensible, that remains a stakeholder decision.

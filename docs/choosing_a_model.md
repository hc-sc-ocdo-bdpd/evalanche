# The Evalanche model selection handbook

This guide is for anyone who needs to decide which language or multimodal
model to use for a real task. Reading it does not require installing or running
the Evalanche software.

The central principle is simple:

> Start with the task and the model deployments you can access. Use public
> benchmarks as scoped evidence. Run a local comparison only when it can
> resolve an important unknown.

The result may be one model deployment, a conditional shortlist, or a
well-documented tradeoff that the decision owner resolves. Forcing a single
winner is not a requirement.

**Terminology:**

- model: the underlying model family or checkpoint;
- model deployment: the exact provider route, version, and config available to the user;
- candidate: a model deployment that passes access and hard requirements;
- benchmark: a standardized evidence source or test suite;
- evaluation: any benchmark, rubric, or local comparison used to measure performance.

Using the same term for all of them makes the decision logic harder to follow.

## 1. Overview

### The complete process

| Stage | Question | Output |
| --- | --- | --- |
| 1. Frame | What decision are we making? | Task and decision statement |
| 2. Confirm access | Which exact model deployments can we use? | Access-confirmed candidate set |
| 3. Filter | Which candidates meet every hard requirement? | Feasible candidates and exclusions |
| 4. Gather | What public evidence is relevant? | Source notes, not one universal score |
| 5. Assess | What does the evidence establish and leave unknown? | Evidence matrix and gap list |
| 6. Test if needed | Could representative local cases resolve a gap? | Small comparison plan |
| 7. Interpret | What quality and operational tradeoffs were observed? | Decision brief |
| 8. Decide and revisit | Who accepts the tradeoff, and when will it be reviewed? | Decision record and review trigger |

The order matters. Starting from a leaderboard and working backward often
creates a false sense of objectivity.

## 2. Frame the actual decision

Write one sentence that names the work and the intended environment:

> Choose a model deployment for extracting specified fields from bilingual
> regulatory PDFs into validated JSON for analyst review.

This is stronger than "choose the smartest model" because it exposes what the
evidence must cover.

### Describe the input

Record the actual input form, variation, and constraints: plain text, PDFs,
images, audio, repositories, or tool state; typical and maximum length; language
and regional variation; layout, OCR, tables, or handwriting issues; and any
privacy, residency, or policy constraints.

**Important:** a result on text pasted from a PDF does not establish that the
provider route can accept or reason over the full PDF workflow.

### Describe the required output

State what a usable response looks like: a class label, valid JSON, a
factually grounded answer with citations, a code patch that passes tests, a
verifiable tool sequence, or another output with explicit acceptance criteria.

Include the failures that invalidate an otherwise fluent answer: invented facts,
invalid JSON, unsupported citations, omitted safety information, or unauthorized
tool actions.

### Define success before seeing outputs

Use criteria tied to the real decision: strict pass rate, per-field accuracy,
recall for a high-cost error class, groundedness or citation accuracy,
executable task success, failure rate, cost per completed case, and p95 latency
or throughput.

**Do not choose a metric only because a public leaderboard already reports it.**
The [HELM taxonomy](https://arxiv.org/abs/2211.09110) is useful here because it
separates scenarios, metrics, and evaluation conditions rather than reducing
performance to one number.

### Record stakes and reversibility

Ask:

- Who is affected by a wrong answer?
- How quickly will a person detect the error?
- Can the output be corrected before it causes harm?
- Is the choice cheap to change later?
- Does it create legal, safety, financial, privacy, or reputational exposure?

Higher consequence and lower reversibility justify stronger evidence, wider
failure review, and more human oversight. The [NIST AI Risk Management
Framework](https://www.nist.gov/itl/ai-risk-management-framework) treats
measurement as part of a broader context and governance process, not a single
benchmark event.

### Consider workflow fit

A useful model does not exist in isolation. A weaker model in a well-designed
workflow can outperform a stronger model in a poor workflow.

Ask whether the bottleneck is the model or the surrounding workflow: the task
setup, retrieval, prompts, tool use, human review, and operational controls. A
model that looks strong on paper can still be a poor fit for a process that is
mis-specified, brittle, or missing review gates.

## 3. Confirm model access first

A model is a candidate only if the user or organization can use the exact
deployment route in the intended environment.

Confirm:

- the model or deployment identifier;
- provider, API, local runtime, or approved platform;
- account and quota access;
- region, hosting, and data-processing approval;
- required modalities and features on that route;
- provider model version or alias behavior;
- pricing terms and rate limits;
- any procurement, licensing, or policy restriction.

Public sources may discuss inaccessible models because they help explain the
state of the field. Those models remain context, not candidates.

### Access is not capability

These statements are different:

- the deployment appears in a registry;
- the deployment declares PDF or tool support;
- the route is compatible with a benchmark manifest;
- the user can actually call it with the required permissions.

**Only the last statement confirms access.** Capability and compatibility are
still checked afterward.

### Access changes over time

Record the confirmation date. Recheck when:

- an alias moves to a new model version;
- a provider retires a deployment;
- a region or organizational policy changes;
- price, quota, or rate limits materially change;
- a self-hosted environment changes weights, quantization, serving stack, or
  hardware.

The software workflow uses a dated [access-set file](local_comparison.md#2-confirm-access-to-each-candidate)
for reusable comparisons. Repeated `--model` arguments are a one-run access
confirmation.

**Access is the right starting place, not an absolute rule.** If every currently
accessible option fails the hard requirements or is materially worse than the
non-accessible alternatives, consider expanding access, changing the workflow,
or revisiting whether a local model comparison is the right tool.

## 4. Apply hard requirements before preferences

Hard requirements are pass or fail. Preferences are tradeoffs. Mixing them can
make an unavailable or noncompliant model look attractive because of a high
quality score.

### Common hard requirements

A hard requirement is a pass/fail gate: approved route, legal and data-location
constraints, required modality and context length, structured-output or tool
support, mandated language behavior, budget ceiling, or required reliability or
latency thresholds.

### Common preferences

Preferences are tradeoffs: lower cost, lower latency, simpler operations,
better observability, more stable versioning, or stronger evidence for a
specialized language or domain.

**Keep the exclusion reason for every removed candidate.** "Unknown" is not the
same as "fails." An unknown capability may require provider verification or a
small test.

## 5. Find public evidence that matches the task

Public evidence is useful when it helps answer a specific question. It is weak
when the evaluated construct, inputs, outputs, model version, or harness differ
materially from the intended use.

Start with the [research and evidence directory](evidence/README.md), which is
organized by task and evidence type.

### A transfer checklist

For every source, compare:

| Dimension | What to verify |
| --- | --- |
| Construct | Does it measure the ability or property the decision needs? |
| Task | Are the cases similar in domain, difficulty, language, and format? |
| Input | Is the modality and context delivery comparable? |
| Output | Is the required answer contract comparable? |
| System | Is the result for a raw model, an agent, retrieval system, or provider product? |
| Harness | What prompt, tools, retries, sampling, context, and budget were allowed? |
| Scoring | Is correctness objective, executable, human-rated, or LLM-judged? |
| Version | Is the exact model or deployment version represented? |
| Submission | Was the result independently run, self-reported, or provider-reported? |
| Freshness | Could model, harness, data exposure, or provider behavior have changed? |

The [Benchmark Lottery](https://arxiv.org/abs/2107.07002) demonstrates that
relative conclusions can change with benchmark choice. This does not make
benchmarks useless. It means the choice of benchmark is part of the claim and
must be justified.

### Do not average unrelated public percentages

Different benchmarks use different tasks, denominators, model versions,
sampling policies, tool budgets, and scoring methods. A 70 percent result on
one benchmark and 80 percent on another are not automatically values on a
shared scale.

Keep findings grouped by question:

- Can the system complete a related task?
- Can the deployment accept the required input?
- Does it follow the required output contract?
- How well does it handle the required language or domain?
- Does it meet operational requirements?
- What uncertainty and limitations remain?

### Prefer source-first evidence

For an important decision, inspect the benchmark paper or official
methodology, not only a leaderboard row. Record:

- source URL and review date;
- task actually measured;
- dataset and benchmark version;
- evaluated model version;
- prompt, tools, and inference settings;
- metric and uncertainty, if reported;
- important finding;
- limitation for this decision;
- whether the result is independent, self-submitted, or provider-declared.

Provider documentation is the preferred source for current availability,
input support, pricing, region, and service limits. Provider benchmark claims
should still be labelled as provider-declared until the evaluation can be
independently checked.

## 6. Judge the strength of the evidence

Use four labels consistently:

| Strength | Meaning |
| --- | --- |
| Strong | Same model version, highly relevant task and contract, comparable tools and settings, transparent scoring, and current evidence |
| Moderate | Closely related task with one or more material differences that are understood |
| Weak | Broad aggregate, old or unclear version, poor task transfer, undisclosed harness, or marketing-only claim |
| Missing | Required model, capability, deployment route, or operating condition was not evaluated |

Evidence can also conflict. Do not hide the conflict. Investigate whether the
sources used different versions, prompts, tool budgets, task distributions, or
scoring methods.

### Construct validity matters

A benchmark may be reliable at scoring its own questions yet weak at measuring
the capability named in a broad headline. Recent work on benchmark construct
validity has documented recurring gaps between claimed and operationalized
constructs. See the 2025 preprint [Construct Validity in AI Benchmarks](https://arxiv.org/abs/2511.04703)
for a systematic treatment. Because it is a preprint, use it as methodological
guidance rather than a settled standard.

## 7. Decide whether public evidence is enough

### Decision confidence

Use a five-point decision-confidence label to summarize how strongly the
available evidence supports the choice.

| Confidence | Meaning |
| --- | --- |
| 5 - Very high | Strong, current, task-relevant evidence with a clear margin and limited uncertainty |
| 4 - High | Good evidence with minor gaps or a small margin between candidates |
| 3 - Moderate | Relevant evidence exists, but some important questions remain or the gap is narrow |
| 2 - Low | Evidence is weak, mixed, or only loosely matched to the actual task |
| 1 - Very low | The evidence is thin, conflicting, or not informative enough to justify a confident choice |

This label should reflect not just benchmark scores, but also evidence quality,
model fit, task relevance, operational risk, and the cost of being wrong.

A close result with weak evidence should not be treated as confident. A model
with moderate evidence and a large operational advantage can still be a sensible
choice if the decision owner documents the uncertainty.

Public evidence may be sufficient when:

- one access-confirmed candidate clearly satisfies the hard requirements and
  the others do not;
- strong, current evidence closely matches the task and intended system;
- the decision is low consequence and easy to reverse;
- remaining uncertainty is unlikely to change the choice;
- a short operational check confirms the actual route behaves as expected.

Run a local comparison when:

- the task, language, modality, domain, or output contract is specialized;
- model versions or provider routes differ from public results;
- strict JSON, citations, or unacceptable error types matter;
- tool schemas, repository environments, or retrieved evidence are specific;
- two feasible candidates remain close enough that local failures, cost, or
  latency could change the choice;
- the choice has material consequences;
- public evidence is missing or conflicting.

**Do not run a local comparison solely to produce another leaderboard.** State
which decision-relevant unknown it is intended to resolve.

## 8. Design representative local cases

The quality of a local evaluation depends more on the cases and scoring than
on the number of model deployments being compared.

### Required case contract

Evalanche's universal case format begins with a case ID, input, expected output,
and evaluation type. Additional columns can describe language, domain, risk,
difficulty, source, or document family.

```text
case_id
input
expected_output
evaluation_type
```

### Expected inputs and outputs by task

The expected output should reflect the real contract, not an idealized answer
that is unrelated to how the output will be used.

| Task | What to supply | Preferred scoring |
| --- | --- | --- |
| Classification | Representative inputs and labels | Deterministic scoring with class-level analysis |
| Structured extraction | Source, schema, and expected JSON | Parsed JSON and field-level scoring |
| Short factual answer | Question, evidence, and accepted answer | Normalized exact or reference-based scoring |
| RAG answer | Query, context, and supported claims | Separate groundedness and answer quality |
| Coding | Repo state, task, environment, and tests | Executable tests |
| Tool workflow | Tool definitions and valid end state | Executable or state-based evaluation |
| Summary or rewrite | Source, rubric, and unacceptable errors | Human review, optionally supported by a calibrated LLM judge |

### Sample from real variation

Include:

- common cases;
- important subgroups;
- difficult but realistic cases;
- cases representing high-cost errors;
- multiple languages or formats when required;
- grouped items that should not be split across development and test sets.

Do not build the primary test set from cases chosen because one current model
failed them. Those cases can become a separate challenge set. A fixed standard
comparison should be selected using task metadata and decision relevance.

### Protect the test set

If prompts, labels, canonicalization, or scoring rules change after inspecting
test failures, those cases have become development evidence. A publishable or
high-consequence claim should use a fresh unseen test cohort.

**A benchmark or local comparison is only as strong as its frozen
case set and scoring protocol.**

Dynamic and human-in-the-loop benchmark work, such as
[Dynabench](https://aclanthology.org/2021.naacl-main.324/), illustrates why
static datasets can saturate and why new failure-driven data should be managed
carefully rather than silently mixed into the original test claim.

## 9. Choose the least subjective valid scorer

Use this order of preference when it matches the construct:

1. Executable end-state checks.
2. Deterministic exact, schema, field, or canonicalized scoring.
3. Structured human review against explicit criteria.
4. A calibrated LLM judge as supporting measurement.

Deterministic scoring is not automatically valid. An exact match is a poor
measure if several semantically correct answers are acceptable. Conversely,
an LLM judge adds avoidable subjectivity when a schema or test suite can decide
the outcome directly.

For open-ended work, define criteria separately. "Overall quality" is rarely
specific enough. Correctness, completeness, groundedness, style, policy
compliance, and task success may need distinct treatment.

The current Evalanche LLM-judge path is exploratory unless it has been
calibrated for the task. Read the [LLM judge method](llm_judges.md) before
using it for an important decision.

## 10. Test in stages

Start with the least expensive run that can reveal a blocker.

| Stage | Purpose | Continue when |
| --- | --- | --- |
| Smoke | Confirm credentials, transport, parsing, token use, artifact capture, and cost | Every route works and outputs are complete |
| Screen | Detect obvious quality, compatibility, and reliability weaknesses | Candidate remains credible for the decision |
| Standard | Compare finalists on one fixed representative cohort | Evidence is precise enough for the intended decision |
| Extended | Increase precision for a consequential or publishable claim | Additional cases can materially change or strengthen the claim |

The exact case count depends on task diversity and decision stakes. Evalanche's
Health Canada PDF example uses 2, 20, and 80 cumulative cases, but those counts
are not universal defaults.

A candidate that fails a local screen does not need a full run for an internal
decision. A published leaderboard is different: every listed model must
complete the same frozen benchmark release.

## 11. Interpret results without hiding tradeoffs

Review more than average score:

- strict passes and failures;
- confidence intervals;
- paired disagreements on identical cases;
- results by important field and subgroup;
- shared failures;
- generation and scoring failures;
- total cost and cost per completed request;
- token use and coverage of cost data;
- p50 and p95 latency where meaningful;
- retry and failure rate;
- qualitative failure severity;
- missing evidence.

Paired comparisons are usually more informative than comparing two unrelated
confidence intervals because every candidate answered the same cases. The
[statistical methods note](statistical_methods.md) explains Evalanche's Wilson
intervals, exact McNemar tests, and Holm correction.

### Do not confuse statistical and practical importance

A detectable difference may be too small to matter operationally. A large
observed difference may remain uncertain in a small sample. A rare but severe
failure can matter more than average accuracy.

Ask what would change the decision:

- Is the quality difference larger than the accepted margin?
- Does the cheaper model still clear the minimum quality bar?
- Are failures concentrated in an unacceptable subgroup?
- Is a slower model still fast enough for the workflow?
- Does one route create operational or compliance risk?

### Default output: a decision brief

A useful brief contains:

- decision and task scope;
- access-confirmed candidates;
- hard requirements and exclusions;
- public evidence used;
- local method and exact versions;
- quality and failure patterns;
- cost, latency, and reliability;
- important unknowns and limitations;
- tradeoffs;
- selected option, if the owner chooses one;
- review trigger.

Evalanche reports comparison evidence by default. A policy-selected model is
allowed only when a user explicitly enables a policy with access profiles,
hard constraints, weights, and a minimum margin. The policy is a declared
preference rule, not an empirical truth.

Use the [decision record template](decision_record.md) to preserve the choice.

## Common mistakes

### Starting from all models on a leaderboard

This wastes attention on options that may be unavailable or unsuitable. Start
from access and hard requirements.

### Treating compatibility as access

An API declaration or local manifest does not prove account, region, approval,
quota, or licensing access.

### Treating a model result as a system result, or the reverse

SWE-bench, WebArena, and tool-use evaluations often measure a complete agent
with prompts, tools, budgets, and orchestration. Record the whole system.

### Copying current scores into permanent guidance

Leaderboards change. Preserve methodology notes and links, then recheck live
results for the decision date.

### Using one judge model once

LLM judges can show position, verbosity, style, self-preference, reference,
prompt, and run-to-run effects. Calibrate and test stability.

### Selecting the top row automatically

Rank order alone does not encode access, risk, uncertainty, minimum thresholds,
or stakeholder priorities.

### Expanding a benchmark whenever a new model appears

Adding a model to a frozen benchmark requires only that model's run. Changing
the cases creates a new benchmark release and may require comparable reruns.

## Three short examples

### High-volume classification

The decision owner confirms three available deployments. All support the input
and output contract. Public broad reasoning scores are only weak evidence.
A stratified local set with exact labels shows that two clear the required
recall. The cheaper of those two is chosen because the quality threshold was
defined before the run.

### Repository coding assistant

SWE-bench Verified and LiveCodeBench help identify relevant systems and coding
capabilities. The organization can access only two of them through approved
routes. Because its repositories, tools, and test harness differ, it runs a
small executable comparison on representative internal tasks. The final brief
attributes results to each complete agent setup, not only the model name.

### Open-ended client summaries

Public evidence does not represent the client's tone, omissions, and factual
risk. A blinded human review defines correctness, coverage, and usefulness
separately. An LLM judge is tested against the human-reviewed calibration set
and used only to support triage. The brief retains a shortlist because the
remaining difference is a preference tradeoff.

## Continue by depth

- [Research and evidence directory](evidence/README.md)
- [How to read benchmarks and leaderboards](evidence/benchmarks.md)
- [Foundational papers and guidance](evidence/foundations.md)
- [Evaluation frameworks](evidence/frameworks.md)
- [LLM judge method](llm_judges.md)
- [Local comparison guide](local_comparison.md)
- [Operational metrics](operational_metrics.md)
- [Statistical methods](statistical_methods.md)
- [Optional decision policy](model_selection.md)
- [Health Canada case studies](case_studies/health_canada.md)

Last substantive review: 2026-08-06.

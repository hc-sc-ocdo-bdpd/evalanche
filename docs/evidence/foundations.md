# Foundational papers and guidance

This is a compact reading list for designing and interpreting model
evaluations. It favors sources that change how a decision is made, not papers
that merely introduce another score.

For live task-specific sources, see [benchmarks.md](benchmarks.md).

## If you read only five

1. [Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110)
   for scenarios, adaptations, metrics, and transparency.
2. [The Benchmark Lottery](https://arxiv.org/abs/2107.07002) for why benchmark
   choice can change relative conclusions.
3. [NIST AI Risk Management Framework 1.0](https://nvlpubs.nist.gov/nistpubs/ai/nist.ai.100-1.pdf)
   for connecting measurement to context, risk, governance, and lifecycle.
4. [Dynabench](https://aclanthology.org/2021.naacl-main.324/) for the limits of
   static saturated datasets and the role of human-and-model-in-the-loop data.
5. [A Practical Guide to Evaluating Large Language Models](https://arxiv.org/abs/2506.13023)
   for a broad current synthesis. This fifth source is a 2025 preprint and
   should be treated as guidance under review.

## Evaluation is a claim, not a table

An evaluation supports a claim of this form:

> Under these tasks, cases, prompts, tools, versions, scoring rules, and
> operating conditions, this system showed these measured behaviors, with
> these uncertainties and limitations.

Every omitted condition broadens the claim. The goal of rigorous reporting is
not to make the claim sound universal, but to make its valid scope clear.

## Context, risk, and lifecycle

### NIST AI Risk Management Framework 1.0

- **Status:** Official NIST framework, 2023.
- **Source:** [NIST AI RMF 1.0](https://www.nist.gov/itl/ai-risk-management-framework).
- **Helps answer:** How should evaluation connect to the intended context,
  affected people, risk, governance, and monitoring?
- **Core contribution:** Organizes work around Govern, Map, Measure, and
  Manage. It treats test, evaluation, verification, and validation as ongoing
  lifecycle activities.
- **Important limit:** It does not select a benchmark or model for you. It
  provides a risk-management structure within which those choices are made.

### NIST Generative AI Profile

- **Status:** Official NIST profile, NIST AI 600-1, 2024.
- **Source:** [Generative AI Profile](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf).
- **Helps answer:** Which generative-AI risks and measurement practices should
  be considered before and after deployment?
- **Core contribution:** Extends the AI RMF to generative AI, including
  pre-deployment testing, incident information, content risks, human factors,
  and ongoing measurement.
- **Important limit:** It is risk guidance, not evidence that any particular
  model is safe or suitable.

### NIST evaluation reporting and statistical guidance

- **Status:** Official NIST initial public draft and report, 2026.
- **Sources:** [NIST AI 800-2 initial public draft](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.800-2.ipd.pdf)
  and [NIST AI 800-3](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.800-3.pdf).
- **Helps answer:** How should an evaluation objective, benchmark relevance,
  uncertainty, and claim qualification be documented?
- **Core contribution:** AI 800-2 proposes a structured relationship among
  evaluation objective, relevant benchmark, statistical analysis, reported
  details, and qualified claims. AI 800-3 discusses statistical models and
  uncertainty assumptions for benchmark analysis.
- **Important limit:** AI 800-2 is an initial public draft, not a final NIST
  standard. Verify its publication status before citing it as current policy.

## Scope, validity, and benchmark choice

### Holistic Evaluation of Language Models, HELM

- **Status:** Research paper and maintained Stanford framework.
- **Sources:** [Paper](https://arxiv.org/abs/2211.09110) and
  [official site](https://crfm.stanford.edu/helm/).
- **Helps answer:** How can evaluations make scenarios, adaptation methods,
  metrics, and missing coverage visible?
- **Core contribution:** Separates the task scenario from prompting or
  adaptation and from multiple metrics such as accuracy, calibration,
  robustness, fairness, bias, toxicity, and efficiency.
- **Important limit:** A broad evaluation suite still cannot represent every
  local domain, deployment route, or unacceptable error.

### The Benchmark Lottery

- **Status:** Research paper, 2021.
- **Source:** [The Benchmark Lottery](https://arxiv.org/abs/2107.07002).
- **Helps answer:** Why can a model look better or worse depending on which
  reasonable benchmark was selected?
- **Core contribution:** Shows that benchmark choice can influence relative
  model conclusions and argues that benchmark selection must be treated as a
  consequential methodological decision.
- **Important limit:** It does not imply that all benchmarks are equally weak.
  Relevance and validity can still be argued and tested.

### Construct Validity in AI Benchmarks

- **Status:** 2025 preprint.
- **Source:** [Systematic review preprint](https://arxiv.org/abs/2511.04703).
- **Helps answer:** Does a benchmark operationalize the capability named in
  its claim?
- **Core contribution:** Applies construct-validity analysis across a large
  benchmark sample and documents recurring gaps between broad claims and the
  actual tasks and measures.
- **Important limit:** It is recent and not yet a settled standard. Use its
  questions as a validity checklist, and verify later publication changes.

### Dynabench

- **Status:** Peer-reviewed NAACL 2021 paper and platform proposal.
- **Source:** [Dynabench paper](https://aclanthology.org/2021.naacl-main.324/).
- **Helps answer:** What should happen when static benchmarks saturate or
  models expose new weaknesses?
- **Core contribution:** Proposes dynamic, human-and-model-in-the-loop data
  creation and evaluation so failures can drive new rounds.
- **Important limit:** Failure-driven challenge data should not be silently
  mixed into an unchanged held-out claim. It measures a changing frontier and
  requires careful versioning.

## Contamination, freshness, and changing systems

### LiveBench

- **Status:** Research paper and maintained benchmark.
- **Sources:** [Paper](https://arxiv.org/abs/2406.19314) and
  [official repository](https://github.com/LiveBench/LiveBench).
- **Helps answer:** How can a broad benchmark reduce reliance on static,
  potentially contaminated questions?
- **Core contribution:** Uses frequently refreshed questions and objective
  ground-truth scoring across several broad categories.
- **Important limit:** Freshness reduces one contamination risk. It does not
  make broad categories a substitute for a specialized task or deployment
  test.

### Reproducibility requires system identity

There is no single paper that removes model-version ambiguity. A defensible
record should preserve:

- provider and route;
- exact version when exposed;
- alias and evaluation date when exact version is not exposed;
- prompt and system instructions;
- tools, retrieval, memory, and context management;
- sampling and reasoning settings;
- token and action budget;
- retries and stopping rules;
- scorer version;
- dataset version and hashes.

This is especially important for agent benchmarks, where the measured unit is
often the complete system rather than the base model.

## Human evaluation

### Human Evaluation of Automatically Generated Text

- **Status:** Peer-reviewed INLG 2024 tutorial paper.
- **Source:** [Tutorial paper](https://aclanthology.org/2024.inlg-tutorials.1/).
- **Helps answer:** How should human evaluation constructs, questions,
  instructions, participants, scales, and analysis be designed and reported?
- **Core contribution:** Provides a structured introduction to the many design
  choices that determine whether human ratings support the intended claim.
- **Important limit:** A tutorial cannot replace domain-expert protocol design
  for a high-consequence use.

### Chatbot Arena

- **Status:** ICML 2024 research paper and maintained public platform.
- **Sources:** [Paper](https://arxiv.org/abs/2403.04132) and
  [current methodology](https://lmarena.ai/how-it-works).
- **Helps answer:** Which anonymous response users prefer in open-ended,
  organic prompts under the arena protocol?
- **Core contribution:** Uses blinded pairwise human preferences at scale and
  statistical ranking methods.
- **Important limit:** Preference is not the same construct as factual
  correctness, policy compliance, structured extraction, or executable task
  success. The population and prompts also determine transfer.

## Statistical interpretation

### Report uncertainty that matches the design

At minimum, ask:

- Are model results paired on identical cases?
- Are cases independent, or clustered by document, product, user, or task?
- Does the interval cover case sampling only, or also generation variability?
- Were multiple model pairs or metrics tested?
- Was the sample fixed before results were inspected?
- Is the reported difference practically meaningful?

Evalanche uses Wilson intervals for pass rates, exact McNemar tests for paired
pass and fail disagreements, and Holm correction across model pairs. The
[statistical methods note](../statistical_methods.md) defines that scope. These
methods do not cover prompt sensitivity, model sampling, judge variation, or
provider drift unless the evaluation design explicitly adds those sources.

### Avoid rank-only reasoning

A leaderboard rank hides:

- the magnitude of the difference;
- uncertainty and ties;
- subgroup failures;
- operational cost;
- hard constraints;
- missing evidence;
- whether the evaluated system is available.

Use rank to navigate evidence, then inspect the actual measure and conditions.

## LLM judging is a separate measurement problem

LLM judges require their own validity, bias, agreement, and stability checks.
The relevant foundational and current papers are annotated in the
[LLM judge method](../llm_judges.md). Evalanche's general rule is to prefer
executable or deterministic scoring whenever that scoring validly captures the
task.

## Reading sequence by role in a decision

| If you need to... | Read first |
| --- | --- |
| Define the claim and context | NIST AI RMF and HELM |
| Choose among candidate benchmarks | Benchmark Lottery and construct-validity review |
| Manage a saturated or challenge set | Dynabench |
| Design human review | Human-evaluation tutorial |
| Interpret paired local results | Evalanche statistical methods and NIST statistical guidance |
| Use an automated judge | Evalanche LLM judge method |

Last source review: 2026-08-06.

# LLM judges: evidence, limits, and Evalanche method

An LLM judge is a measurement instrument. It is not an answer key and it is
not automatically a substitute for human review.

Evalanche supports rubric-based judging because some important properties,
such as clarity, coverage, usefulness, and writing quality, cannot always be
reduced to exact matching. The feature should be used only when that judgment
is part of the real construct.

## Bottom line

1. Use executable or deterministic scoring when it validly captures success.
2. Use task-specific criteria, not an undefined overall-quality score.
3. Provide a reference answer or source evidence when the task permits it.
4. Blind candidate identity and control response order.
5. Calibrate on human-reviewed strong, weak, and adversarial examples.
6. Measure chance-corrected agreement and error patterns, not only raw match.
7. Test run-to-run and prompt stability.
8. Preserve the exact judge model, prompt, version, settings, and outputs.
9. Use human or expert review when consequences justify it.
10. Treat current Evalanche judge results as exploratory unless this protocol
    has been completed for the task.

## First ask whether a judge is needed

| Property | Preferred method |
| --- | --- |
| Valid JSON | Parser and schema validation |
| Exact class | Deterministic label comparison |
| Required fields | Field-level checks after canonicalization |
| Code correctness | Executable tests |
| Tool outcome | Environment state or executable assertions |
| Source-grounded factual claims | Reference or source-based checks, then human review for unresolved cases |
| Completeness of an open-ended summary | Human rubric, optionally supported by a calibrated judge |
| Tone or style for a target audience | Blinded target-user review, optionally supported by a calibrated judge |

An LLM judge adds noise and bias when a deterministic answer exists. An exact
match adds invalid rigidity when several materially correct answers exist.

## What the research establishes

### MT-Bench and Chatbot Arena judge study

[Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)
reported that strong judges could reach high agreement with human preferences
in the studied settings. It also documented position bias, verbosity bias,
self-enhancement bias, and limits on reasoning and grading.

Useful implications:

- agreement is task- and protocol-specific;
- pairwise judging can be easier than assigning an absolute score;
- response order should be swapped and inconsistent pairs handled explicitly;
- judge identity and candidate identity matter.

The frequently quoted raw agreement from this paper is not a universal judge
reliability guarantee.

### G-Eval

[G-Eval](https://aclanthology.org/2023.emnlp-main.153/) used criteria,
evaluation steps, and probability-weighted score outputs to evaluate generated
text. It reported stronger correlation with human judgments than several prior
automatic metrics on the studied summarization and dialogue tasks.

Useful implications:

- criteria and evaluation form matter;
- decomposition is stronger than an undefined holistic score;
- correlation on one task does not validate a judge elsewhere;
- the paper also raises concern about bias toward model-generated text.

### LLM-Rubric

[LLM-Rubric](https://aclanthology.org/2024.acl-long.745/) evaluates text with
multiple rubric questions and calibrates the resulting judge signals against
human annotations. In its studied dialogue setting, the calibrated,
multidimensional approach predicted human ratings more accurately than the
uncalibrated baseline. The paper also reports imperfect agreement both between
LLMs and people and among human judges.

Useful implications:

- preserve dimensions separately before deriving any overall judgment;
- calibration requires real human annotations, not merely a detailed prompt;
- human disagreement is part of the measurement problem and should be
  reported;
- one calibrated task does not establish validity for another rubric or use.

### LLMBar

[LLMBar](https://arxiv.org/abs/2310.07641) and its
[official repository](https://github.com/princeton-nlp/LLMBar) evaluate whether
judges can distinguish instruction-following outputs in adversarial pairs.
The benchmark shows that superficial style can mislead judges away from the
instruction and correctness.

Useful implications:

- calibration needs adversarial cases, not only obvious good and bad outputs;
- a fluent, detailed answer should not receive credit for missing the task;
- the rubric should make instruction adherence explicit when it matters.

### JudgeBench

[JudgeBench](https://arxiv.org/abs/2410.12784) focuses on challenging
knowledge, reasoning, mathematics, and coding response pairs with objective
correctness signals. It reports that many strong models remain weak judges on
hard pairs.

Useful implications:

- a capable generator is not automatically a valid judge;
- correctness-heavy tasks should include known-answer calibration;
- judge selection must be empirical for the target construct.

### Position bias study

[A Systematic Study of Position Bias in LLM-as-a-Judge](https://arxiv.org/abs/2406.07791)
tests multiple judges and tasks at large scale. It finds that position effects
vary by judge and task and recommends stronger control than assuming one
ordering is neutral.

Useful implications:

- evaluate both A/B and B/A orders for pairwise judgments;
- use double-blind candidate labels;
- report inconsistency rather than silently choosing one order;
- do not assume a low-temperature call removes order effects.

### Prompt-paraphrase robustness

[All Prompts Are Created Equal?](https://aclanthology.org/2026.findings-acl.1929/)
tests semantically equivalent judge-prompt variants across several generation
tasks. It reports an accuracy-robustness gap, especially for subjective
attributes, and no judge that is uniformly strongest across the studied
conditions.

Useful implications:

- human alignment under one prompt does not establish prompt robustness;
- use non-adversarial paraphrases that preserve the rubric's meaning;
- record label flips and chance-corrected stability across variants;
- keep the exact production prompt in the validated contract.

### Prometheus 2

[Prometheus 2](https://aclanthology.org/2024.emnlp-main.248/) and its
[repository](https://github.com/prometheus-eval/prometheus-eval) develop open
evaluator models that support direct assessment and pairwise ranking with
rubrics and reference answers.

Useful implications:

- reference answers and explicit scoring rubrics can materially structure the
  task;
- pointwise and pairwise evaluation are distinct protocols;
- a specialized judge still requires target-task validation.

### Reliability without validity

[Reliability Without Validity: The Failure of LLM Judges in Real-World
Evaluations](https://arxiv.org/abs/2606.19544) is a June 2026 preprint. Across a
large multi-judge study, it reports that raw agreement can substantially
overstate chance-corrected agreement, judge rankings can shift across
validation benchmarks, and high test-retest consistency can coexist with
systematic bias.

The paper proposes a multi-view validation protocol including chance-corrected
agreement, A/B and B/A order checks, repeated trials, cross-benchmark
validation, and bias analysis.

Useful implications:

- report Cohen's kappa or Krippendorff's alpha where appropriate;
- separate repeatability from validity;
- validate on more than one slice or benchmark when making broad claims;
- label this source correctly as a recent preprint.

### Reference-free generosity

[LLM Judges Can Be Too Generous When There Is No Reference
Answer](https://arxiv.org/abs/2607.12885) is a July 2026 preprint. It reports
that reference-free judges can over-credit incorrect answers and that adding a
reference can change judgments materially across studied languages and tasks.

Useful implications:

- include reference answers or source evidence when available;
- calibrate reference-free criteria against known-answer examples;
- do not generalize the reported effect sizes beyond the studied settings;
- label the source as a recent preprint.

## Choose pointwise or pairwise judging deliberately

### Pointwise

The judge evaluates one response against criteria and may assign scores.

Advantages:

- cost scales linearly with candidate responses;
- a response can be evaluated without a competitor;
- criteria-level diagnostics are natural.

Risks:

- score scales can drift or compress;
- absolute thresholds may not be calibrated;
- verbosity or style can inflate scores;
- scores from different judge prompts may not be comparable.

Use pointwise judging when the rubric has anchored examples and the absolute
score has demonstrated meaning.

### Pairwise

The judge compares two responses and selects A, B, or tie.

Advantages:

- relative preference can be easier than an absolute score;
- direct paired evidence matches the candidate decision;
- differences can be explained criterion by criterion.

Risks:

- cost grows with the number of model pairs;
- position effects require both orders;
- transitivity is not guaranteed;
- one weak opponent can make another response appear stronger than it is.

Use pairwise judging when direct tradeoff is the construct and both response
orders can be evaluated.

## Build a calibration set

A calibration set should be separate from the final test set and contain:

- clear passes;
- clear failures;
- borderline examples near the intended threshold;
- concise correct and verbose incorrect responses;
- fluent responses that miss the instruction;
- responses with unsupported claims;
- subgroup examples for every required language or format;
- known adversarial judge cases;
- disagreements among qualified human reviewers.

For each example, preserve:

- input and source evidence;
- candidate response;
- reference answer where possible;
- criterion-level human labels;
- reviewer instructions;
- adjudicated outcome;
- rationale and ambiguity notes.

Do not tune the judge on the final held-out test examples.

## Human reference protocol

The human review must be designed, not improvised after seeing a model score.

1. Define the construct and criteria.
2. Select reviewers with appropriate language and domain knowledge.
3. Blind candidate and judge identity.
4. Randomize presentation order.
5. Train reviewers on anchored examples.
6. Collect independent labels before adjudication.
7. Measure agreement and inspect criterion-level confusion.
8. Adjudicate disagreements under a written rule.
9. Preserve original and adjudicated labels.
10. Document reviewer population, sample, compensation or role, and conflicts.

For a casual internal screen, one domain owner may be enough to identify gross
errors. For a decision-grade or publishable claim, independent review and an
agreement protocol become much more important.

## Metrics for judge validation

Report more than raw agreement.

### Classification or pass and fail

- confusion matrix;
- accuracy;
- precision and recall for the failure class;
- false-approval rate, which is the false-negative rate for the failure class,
  especially when the judge may approve bad outputs;
- Cohen's kappa for two raters;
- Krippendorff's alpha when there are multiple raters, missing labels, or
  suitable scale types;
- results by task, language, difficulty, and error type.

### Ordinal or continuous scores

- score distribution and calibration plot;
- rank and linear association where justified;
- weighted kappa for ordinal categories;
- mean absolute error against anchored human scores;
- threshold-specific confusion if scores become pass and fail decisions;
- inter-rater reliability among the human reference labels.

Correlation alone can be high while threshold errors remain unacceptable.

### Stability

- exact repeat rate;
- chance-corrected repeat agreement;
- A/B versus B/A consistency;
- repeated-call variance;
- prompt-paraphrase sensitivity;
- judge-version and judge-family comparison;
- missing, malformed, or refused judgment rate.

## Minimum protocol by evidence level

### Exploratory

Appropriate for early development and failure discovery.

- explicit task-specific rubric;
- blind candidate name;
- temperature zero when supported;
- preserve raw judge output and metadata;
- manual inspection of a sample;
- clear exploratory label.

This is the default when no matching validation report is attached.

### Calibrated

Appropriate when a judge has been tested as a measurement aid for one exact
task and rubric.

- everything in exploratory;
- independent human annotations and adjudicated reference labels;
- a separate validation split;
- chance-corrected human-human and judge-human agreement;
- failure precision, failure recall, and false-approval analysis;
- declared acceptable thresholds and complete baseline coverage;
- exact task, prompt, rubric, model version, settings, and data fingerprints.

`calibrated` is task-specific. It does not mean the same judge can be trusted
for a different language, rubric, model version, or evaluation construct.

### Decision-grade

Appropriate when the result may influence a material internal choice.

- everything in calibrated;
- human-reviewed calibration set;
- reference or source evidence where possible;
- chance-corrected agreement;
- false-approval analysis;
- repeated judgments on a stability sample;
- order swapping for pairwise judgments;
- subgroup validation;
- human review of material disagreements and high-risk failures.

### Publishable

Appropriate for a public benchmark or research claim.

- preregistered or frozen protocol before final test inspection;
- independent human labels and adjudication;
- documented reviewer qualifications and agreement;
- held-out validation separated from judge development;
- multiple judge views or a justified judge selection study;
- full prompt, model, version, settings, and code release when permitted;
- uncertainty and multiple-comparison handling;
- bias and sensitivity analysis;
- reproducible raw judgment artifacts subject to privacy constraints;
- explicit publication limitations.

## Judge prompt requirements

A defensible prompt should include:

- the task instruction;
- the exact criterion being measured;
- criterion-specific anchors;
- source evidence and reference answer where available;
- instruction to ignore style features that are not criteria;
- instruction not to infer facts outside the provided evidence when
  groundedness is being measured;
- a structured output schema;
- an explicit tie or insufficient-evidence result when appropriate;
- no candidate model identity.

Blinding is not achieved merely by omitting a dedicated model-name field.
Inspect inputs, response text, filenames, and metadata for identity leakage,
and document unavoidable leakage as a limitation.

Do not ask the judge to combine correctness, style, safety, and completeness
into one unexplained score.

## Current Evalanche implementation status

The current judge path provides:

- one pointwise judge call per judge-routed response;
- user-defined weighted criteria;
- temperature zero when supported;
- retry and error recording;
- raw and parsed judge outputs;
- judge token, cost, latency, and failure accounting;
- deterministic routes kept separate from judge routes;
- task, prompt, rubric, model, and version identifiers in the judge contract;
- reference-required, reference-optional, and reference-free prompt modes;
- candidate-identity blinding in the built-in pointwise prompt.

The provider-free `validate-judge` path now provides:

- a versioned protocol and exact contract fingerprint;
- independent and adjudicated human-label records;
- separate calibration and validation splits;
- raw agreement, Cohen's kappa, and Krippendorff's alpha;
- a binary confusion matrix with failure precision, failure recall, false
  approval, and Wilson intervals;
- bootstrap uncertainty for judge-human kappa;
- repeated-trial stability and prompt-paraphrase flip rates;
- named-versus-blinded identity checks and optional self-preference shifts;
- normalized pairwise A/B and B/A position flip and position-bias analysis;
- full-case coverage checks for every declared stability and bias condition;
- comparison of every saved judge variant against the same human labels;
- criterion, language, difficulty, risk, and other declared subgroup results;
- an explicit disagreement queue for human review;
- machine-readable gates and computed evidence levels;
- hash verification before a validation report can be attached to a run.

Validation analysis never calls a provider. Human annotations and judge
observations are generated and frozen separately, then analyzed repeatedly
without paying for the calls again.

The built-in ordinary judge command remains pointwise. Pairwise validation is
supported through saved, canonically labelled A/B and B/A observations. The
validator does not yet generate a pairwise tournament or decide which cases
deserve human annotation.

## Run the provider-free worked example

The bundled example is synthetic and is permanently capped at `exploratory`:

```bash
docker compose run --rm evalanche python -m evalanche.cli validate-judge --protocol examples/judge_validation/protocol.yaml
```

It produces a Markdown report, JSON report, gate CSV, disagreement CSV, and
subgroup CSV under `results/judge_validation/`. A failed declared target still
writes the diagnostic artifacts, then exits with status code 2.

Keep the protocol inputs and all generated CSV artifacts at the paths recorded
in the JSON report. When evidence is attached to an evaluation, Evalanche
rechecks their hashes, derives the level from the preserved gate table, and
rejects source drift or a manually raised status field.

The three input files have fixed, inspectable contracts.

### Cases

At minimum:

```text
case_id,split,<declared slice columns>
```

Keep the input, reference or source evidence, and candidate output in this
file as additional columns so every label remains auditable. Pointwise and
pairwise tasks may use different output columns.

### Human annotations

```text
case_id,criterion,reviewer_id,annotation_stage,label
```

`annotation_stage` is `independent` or `adjudicated`. Collect independent
labels before adjudication. Use reviewer pseudonyms where identity should not
be published, but retain the documented reviewer population and instructions.

### Judge observations

```text
case_id,criterion,judge_variant_id,trial_id,prompt_variant_id,identity_condition,presentation_order,label,status,uncertainty
```

- `identity_condition` is `blinded` or `named`;
- pointwise `presentation_order` is `pointwise`;
- pairwise orders are `AB` and `BA`;
- pairwise labels are canonical response identities, not first or second
  position, so an order reversal can be compared correctly;
- `status` is `success`, `error`, or `abstain`;
- `uncertainty` is optional from 0 to 1 and is reported diagnostically, never
  assumed to be calibrated probability.

## How evidence levels are enforced

The protocol declares a target level and stakeholder-approved gates. The
software computes the highest level actually achieved. A user cannot promote
evidence merely by writing `decision_grade` in a result file.

The calibrated gates cover held-out cases, human reference quality,
judge-human kappa, failure detection, false approval, baseline failures, and
identity blinding. Decision-grade additionally requires independent human
review controls, at least three uncached trials, prompt variants, identity-bias
testing, and required subgroup coverage. Every declared control condition must
cover every held-out validation case. Pairwise decision-grade protocols also
require both response orders. Publishable adds frozen-protocol and governance
gates.

Thresholds such as minimum kappa and maximum false approval have no universal
defaults. The protocol owner must choose and justify them based on error cost,
prevalence, task stakes, and intended use. Evalanche reports both the declared
threshold and observed value.

Synthetic fixtures can never exceed `exploratory`. `publishable` means the
evidence package passed its declared software-verifiable and documented
governance gates. Those gates also require either multiple saved judge variants
or a documented rationale for the frozen judge-selection protocol. It is not
peer review or external certification.

## Attach validated evidence to a judge run

The active evaluation config must repeat the exact validated identity and
attach the protocol and JSON report. The report's preserved input and artifact
files must also remain available for hash verification:

```yaml
judge:
  model: azure/replace-with-judge-deployment
  variant_id: primary
  provider_model_version: replace-with-exact-version
  temperature: 0
  prompt_id: evalanche.criteria_pointwise
  prompt_version: "1.0"
  rubric_id: my_task_quality
  rubric_version: "1.0"
  candidate_identity_blinded: true
  reference_mode: required
  protocol_path: configs/judge_protocols/my_task_quality_0_1_0.yaml
  validation_report_path: results/judge_validation/my_task_quality/0.1.0/my_task_quality_0.1.0_validation.json
  minimum_validation_level_for_selection: calibrated
```

The task description, measured construct, intended use, languages, scoring
range, threshold, and criteria must also match. Any prompt, rubric, model,
version, temperature, task, or criterion drift changes the contract hash and
is rejected before judge calls begin.

Without a matching report, judge rows remain `exploratory`. They can be
inspected and compared descriptively, but they cannot produce an
evidence-supported leader or an opt-in policy selection when the configured
minimum is `calibrated`.

Therefore:

> An Evalanche LLM-judge score is exploratory unless the project using it has
> completed the task-specific validation protocol and attached the exact
> matching report.

This limitation does not affect deterministic exact and JSON scoring.

## Claim language

Prefer:

> Under this rubric, calibration set, judge version, prompt, and run protocol,
> judge scores agreed with the adjudicated human labels at the reported level.

Avoid:

> The LLM judge objectively proved which model was best.

Last literature review: 2026-08-06.

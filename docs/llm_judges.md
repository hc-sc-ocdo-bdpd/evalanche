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
- false-positive rate, especially when the judge may approve bad outputs;
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

This is close to the current Evalanche judge capability.

### Decision-grade

Appropriate when the result may influence a material internal choice.

- everything in exploratory;
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
- deterministic routes kept separate from judge routes.

It does not yet provide a complete decision-grade protocol. In particular, it
does not automatically supply:

- a human calibration dataset and agreement analysis;
- repeated trials;
- prompt-paraphrase stability checks;
- pairwise A/B and B/A judging;
- multi-judge cross-validation;
- systematic bias analysis;
- automatic reference-aware versus reference-free comparison.

Therefore:

> An Evalanche LLM-judge score is exploratory unless the project using it has
> completed and documented the additional validation outside or on top of the
> current command.

This limitation does not affect deterministic exact and JSON scoring.

## Recommended implementation sequence

1. Add a versioned judge protocol record.
2. Add human calibration-case input and agreement reports.
3. Add repeat trials and stability summaries.
4. Add pairwise judging with A/B and B/A order control.
5. Add judge identity blinding and randomized presentation records.
6. Add cross-judge and subgroup validation.
7. Make decision-grade and publishable labels contingent on completed checks.

Until those features exist, document external calibration artifacts in the
decision brief and retain human review for important conclusions.

## Claim language

Prefer:

> Under this rubric, calibration set, judge version, prompt, and run protocol,
> judge scores agreed with the adjudicated human labels at the reported level.

Avoid:

> The LLM judge objectively proved which model was best.

Last literature review: 2026-08-06.

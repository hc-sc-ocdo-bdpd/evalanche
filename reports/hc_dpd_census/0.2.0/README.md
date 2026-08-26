# Health Canada DPD census model comparison

**Release status:** Published benchmark leaderboard. Generation,
deterministic evaluation, automated error analysis, grouped product-family
analysis, and the selected-case evidence audit are complete. Independent
human sign-off is not claimed.

This release compares four models on the same frozen bilingual DPD
structured-extraction census.

```text
Dataset: hc_dpd_structured_extraction_census
Dataset version: 0.2.0
Cases per model: 14,034
Product families: 7,017
Languages: English and French
Evaluation: deterministic canonical JSON match
Generation failures: 0
```

## Headline result

| Rank | Model | Strict passes | Pass rate | Average field score | p95 latency | Estimated cost |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | GPT-5.6 Sol | 14,021 / 14,034 | 99.91% | 0.99988 | 4.74 s | US$75.52 |
| 2 | GPT-5.6 Terra | 13,986 / 14,034 | 99.66% | 0.99952 | 4.63 s | US$37.76 |
| 3 | GPT-5.4 mini | 10,827 / 14,034 | 77.15% | 0.92344 | 4.69 s | US$11.40 |
| 4 | GPT-5.6 Luna | 10,662 / 14,034 | 75.97% | 0.95922 | 4.42 s | US$15.12 |

Strict pass requires every field in a case to match after the configured,
symmetric canonicalization rules. Average field score is diagnostic and does
not replace complete-record accuracy.

Sol and Terra are separated by 35 net strict passes. In the paired results,
Sol alone passed 42 cases, Terra alone passed 7, both passed 13,979, and both
failed 6.

## Interpretation

The result establishes that Sol and Terra nearly saturate this specific task,
while Mini and Luna exhibit materially different complete-record failure
rates. It does not establish a best model for Product Monograph extraction or
for unrelated Health Canada tasks.

The completed analysis confirms that English and French records from the same
product family are operationally dependent. Sol passed both languages for
7,007 of 7,017 products; Terra did so for 6,972. Mini is 12.10 percentage
points stronger in French than English on the strict contract, while Luna is
19.58 points weaker in French.

Failure diagnostics also show that 26 of Terra's 48 strict failures involve a
comma-bearing official label rendered as separate list values. The evidence
audit retained those failures because rendered multi-values use ` | ` while
the comma remains part of the official DPD label. The strict leaderboard
therefore remains unchanged, with Sol ahead by 35 complete records.

The [`analysis package`](analysis/README.md) contains the complete automated
findings, every Sol and Terra frontier case, and a deterministic 105-case
selected-case evidence set. The
[`evidence audit`](analysis/EVIDENCE_AUDIT.md) records the completed
source, expected-answer, and scoring checks.

## Included evidence

| File | Purpose |
| --- | --- |
| [`model_summary.csv`](model_summary.csv) | Complete aggregate operational and quality metrics |
| [`pairwise_comparisons.csv`](pairwise_comparisons.csv) | Exact paired pass and fail comparisons |
| [`evaluation_report.md`](evaluation_report.md) | Generated human-readable evaluation report |
| [`evaluation_metadata.json`](evaluation_metadata.json) | Evaluation configuration, hashes, methods, and summaries |
| [`generation/`](generation/) | Per-model generation metadata and pricing evidence |
| [`analysis/`](analysis/) | Reproducible slice, field, taxonomy, frontier, and manual-review artifacts |
| [`manifest.json`](manifest.json) | Published-file and retained raw-artifact integrity evidence |

The large case-level results and raw model outputs are intentionally not
tracked. Their repository-relative paths, byte sizes, and SHA-256 hashes are
recorded in the release manifest.

## Validation status

The audit rebuilt all 105 selected cases from the frozen DPD archive,
independently parsed all expected answers, and rescored all 420 selected model
outputs. It retained all results with no corrections, exclusions, or
scoring-rule changes.

This release is a benchmark leaderboard, not a formal Health Canada model
recommendation. Independent human validation is not claimed; the stated validation is the
completed automated source, expected-answer, and rescoring audit.

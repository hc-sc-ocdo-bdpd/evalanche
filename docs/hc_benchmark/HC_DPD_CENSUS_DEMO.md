# Health Canada DPD census and live demo

## Release

The full DPD-only structured-extraction release is:

```text
Dataset: hc_dpd_structured_extraction_census
Version: 0.2.0
Parent: hc_dpd_source_snapshot 2026.7.2
Source date: 2026-07-02
Product families: 7,017
Cases: 14,034
Languages: English and French
Manifest: configs/datasets/hc_dpd_structured_extraction_census_0.2.0_manifest.yaml
```

This release applies the same versioned parsing, normalization, bilingual case
generation, source-row traceability, and JSON answer contract used by slice
`0.1.0` to every product family that can be represented under that contract.
It is a full eligible-population release, not a sample.

The live-demo view contains 12 product families and 24 paired English and
French cases. It is deliberately bounded so a demonstration does not
accidentally make 14,034 model calls per candidate.

## Population accounting

| Stage | Drug rows or product families |
| --- | ---: |
| Marketed DPD drug rows | 13,538 |
| Human marketed rows satisfying row-level quality gates | 11,613 |
| Product families before family-level validation | 7,030 |
| Families excluded for conflicting French brand names | 13 |
| Product families in the census | 7,017 |
| English and French cases in the census | 14,034 |

The 7,017 included families cover:

| Complexity stratum | Product families |
| --- | ---: |
| Single ingredient, single variant | 3,123 |
| Multiple ingredients, single variant | 1,528 |
| Single ingredient, multiple variants | 2,197 |
| Multiple ingredients, multiple variants | 169 |

The 13 excluded families contain different French brand-name values across
their DPD variants. The current output contract has one scalar `brand_name`.
Choosing one source value automatically would hide a real source ambiguity, so
those families are excluded from this benchmark contract.

## Files

| File | Purpose | Records |
| --- | --- | ---: |
| `data/hc/benchmarks/dpd_structured_extraction_census/0.2.0/cases.csv.gz` | Complete generation-ready census | 14,034 |
| `data/hc/benchmarks/dpd_structured_extraction_census/0.2.0/products.csv.gz` | Complete product-family membership and source traceability | 7,017 |
| `data/hc/benchmarks/dpd_structured_extraction_census/0.2.0/demo_cases.csv` | Bounded live-demo view | 24 |
| `data/hc/benchmarks/dpd_structured_extraction_census/0.2.0/build_report.json` | Population accounting, coverage, size, and quality evidence | 1 |

The large CSV files use deterministic gzip encoding. Evalanche reads and
verifies compressed CSV files directly. Compression reduces the committed
artifacts from roughly 30 MB of CSV to about 3 MB without changing their
logical records.

## Demo composition

The fixed demo view uses seed `20260727` and contains three product families
from each of the four complexity strata. Each family contributes one English
and one French case.

The demo collectively covers:

- oral products
- injectable products
- topical products
- inhaled products
- ophthalmic products
- single and multiple ingredients
- single and multiple DPD variants

Active-ingredient overlap is avoided within the demo view to reduce redundant
examples. The demo is visible and intended for presentation and integration
testing. It is not a hidden test set.

## Verify the release

From the repository root:

```bash
docker compose run --rm evalanche python -m evalanche.cli verify-dataset \
  --manifest configs/datasets/hc_dpd_structured_extraction_census_0.2.0_manifest.yaml \
  --root .
```

Expected final lines:

```text
Verified files: 4/4
Status: VALID
```

This verifies file hashes, decompressed CSV record counts, all 7,017 unique
membership IDs, and their manifest membership. It makes no network or model
calls.

## Run the bounded demo

The candidate model or models are declared in
`configs/candidate_models.yaml`. With the current default, set the Azure
deployment through `CANDIDATE_MODEL` if needed.

This bounded demo calls every route listed in `configs/candidate_models.yaml`.
Keep only routes you have explicitly confirmed before running it. For an
access-scoped task comparison, use the
[local comparison workflow](../local_comparison.md).

Generate outputs:

```bash
docker compose run --rm evalanche python -m evalanche.cli generate \
  --config configs/generate_hc_dpd_census_demo.yaml
```

This makes 24 calls per configured candidate model and writes:

```text
data/generated/hc_dpd_census_demo_outputs.csv
data/generated/hc_dpd_census_demo_outputs_generation_metadata.json
```

Evaluate the saved outputs:

```bash
docker compose run --rm evalanche python -m evalanche.cli evaluate \
  --config configs/evaluate_hc_dpd_census_demo.yaml
```

Every demo case is routed to deterministic JSON scoring. The evaluation
command makes no LLM judge calls. It writes:

```text
results/evaluate_hc_dpd_census_demo_results.csv
results/evaluate_hc_dpd_census_demo_results_model_summary.csv
results/evaluate_hc_dpd_census_demo_results_pairwise_comparisons.csv
results/evaluate_hc_dpd_census_demo_results_model_selection.csv
results/evaluate_hc_dpd_census_demo_results_run_metadata.json
results/evaluate_hc_dpd_census_demo_results_comparison.md
```

The demo comparison profile follows the benchmark specification:

- object keys and scalar facts remain exact after configured text
  normalization;
- Unicode, casing, and insignificant whitespace are normalized;
- DIN strings are normalized to eight digits;
- numeric values and numeric strings at
  `/active_ingredients/*/strength` are compared by exact decimal value;
- `din`, `active_ingredients`, `dosage_forms`, `routes`, and `schedule` are
  compared without treating list order as meaningful;
- duplicate list members, missing fields, extra fields, and genuinely
  different values still fail a case.

The case-results CSV retains both `json_exact_match`, the raw parsed-JSON
comparison, and `json_canonical_match`, the authoritative benchmark
comparison. It also records `json_missing_fields`, `json_extra_fields`, and
`json_mismatched_fields`.

With one candidate, the artifacts report that no comparison is available.
With two or more candidates in `configs/candidate_models.yaml`, every model
receives the same 24 case IDs and the paired comparison becomes applicable.
Any policy selection remains opt-in and requires explicit access evidence.

## Interpret the demo result

The strict pass rate counts a case only when all eight expected top-level
fields match after configured canonicalization. The average score is the mean
top-level field-match rate, so it can be high even when a model makes at least
one mistake in most cases.

For example, an average score of `0.953` across 24 cases means 183 of 192
top-level fields matched. It does not mean that 95.3% of complete records were
correct. Use the mismatch columns in the case-results CSV to distinguish
schema violations from factual errors.

The saved generated outputs can be re-evaluated after a scoring-rule change
without making candidate-model calls:

```bash
docker compose run --rm evalanche python -m evalanche.cli evaluate \
  --config configs/evaluate_hc_dpd_census_demo.yaml
```

## Rebuild in a clean tree

The committed release is immutable. In a clean tree where version `0.2.0`
does not already exist:

```bash
docker compose run --rm evalanche python -m evalanche.cli build-dpd-census \
  --source-manifest configs/datasets/hc_dpd_2026-07-02_manifest.yaml \
  --root .
```

Expected summary:

```text
Products: 7017
Cases: 14034
Bounded demo: 12 products, 24 cases
Source-ambiguous families excluded: 13
Verified files: 4/4
Status: VALID
```

The builder reads only the frozen local DPD snapshot, refuses to overwrite an
existing release, writes through a staging directory, and produces
byte-identical output across rebuilds.

## What the demo proves

The demo proves that Evalanche can:

1. freeze and verify an official Health Canada source release;
2. transform the complete eligible DPD population reproducibly;
3. preserve bilingual reference values and exact source-row lineage;
4. give identical cases to one or more candidate models;
5. score valid JSON, canonical complete-case accuracy, and top-level field
   accuracy while retaining explicit mismatch diagnostics;
6. report failures, latency, token usage, available cost evidence, and model
   summaries; and
7. preserve the complete configuration, hashes, outputs, and report needed to
   reproduce the run.

## What it does not prove

The inputs are readable renderings of DPD relational records. They are not
Product Monograph PDFs or extracted monograph text. The release therefore
does not measure long-document Product Monograph extraction, and it is not a
general recommendation about which model Health Canada should use.

The DPD census remains useful as a complete extraction coverage and
operational stress suite. Completed analysis and the Product Monograph
benchmarks are summarized on the [Health Canada case-study landing
page](../case_studies/health_canada.md). They support the repository's main
model-selection guide rather than defining its primary user journey.

# Mission Evalanche

Evalanche is an extensible task-grounded language-model benchmark suite for
answering:

> How do candidate models compare on this specific task?

It combines deterministic metrics, rubric-based LLM judging, paired
statistics, operational evidence, cost accounting, and explicit selection
constraints. Its primary product is a versioned, reproducible leaderboard for
each benchmark. Model-selection policies and recommendation reports are
optional interpretation layers.

## Current status

Evalanche currently supports:

- candidate generation with separate, reusable output artifacts;
- exact, normalized string, and configurable canonical JSON evaluation;
- rubric-based criteria judging for open-ended tasks;
- case-level results, model summaries, paired comparisons, and recommendation
  reports;
- latency, retry, token, cost, and failure accounting;
- explicit endpoint price catalogs and local cost guardrails;
- durable checkpoints and safe resume for large generation runs;
- versioned dataset manifests with file hashes, record counts, provenance,
  sampling, and split membership;
- constraint-aware model selection based on quality, cost, latency,
  reliability, and declared capabilities;
- independent model and benchmark manifests with compatibility fingerprints;
- immutable compact result bundles and generated CSV, JSON, Markdown, and
  sortable HTML leaderboards;
- a reproducible bilingual Health Canada DPD benchmark with 14,034 cases;
- a frozen 40-product, 80-case Product Monograph evidence-window diagnostic;
- a separate draft native-PDF benchmark for end-to-end document retrieval,
  visual reading, and structured extraction.

The repository includes more than 280 focused tests, plus continuous checks
for lint, coverage, and committed benchmark integrity.

## Health Canada DPD benchmark

The current benchmark release contains:

| Property | Value |
| --- | ---: |
| Dataset | `hc_dpd_structured_extraction_census` |
| Version | `0.2.0` |
| Product families | 7,017 |
| English cases | 7,017 |
| French cases | 7,017 |
| Total cases per model | 14,034 |
| Scoring | Deterministic canonical JSON |

Four models completed the full census with no generation failures:

| Model | Strict passes | Pass rate | Average field score | Estimated run cost |
| --- | ---: | ---: | ---: | ---: |
| GPT-5.6 Sol | 14,021 | 99.91% | 0.99988 | US$75.52 |
| GPT-5.6 Terra | 13,986 | 99.66% | 0.99952 | US$37.76 |
| GPT-5.4 mini | 10,827 | 77.15% | 0.92344 | US$11.40 |
| GPT-5.6 Luna | 10,662 | 75.97% | 0.95922 | US$15.12 |

Automated error analysis, grouped product-family analysis, and the 105-case
frozen-source evidence audit are complete. The audit retained every strict
result and found no label or scoring corrections. Independent human sign-off
is not claimed.
See the
[`published result release`](reports/hc_dpd_census/0.2.0/README.md) for the
full scope, paired evidence, metadata, raw-artifact hashes, and interpretation
limits.

## Product Monograph benchmarks

Evalanche now keeps two Product Monograph questions separate:

| Benchmark | Model input | What it measures | Status |
| --- | --- | --- | --- |
| `hc_product_monograph_structured_extraction@0.1.0` | Label-selected text pages | Extraction and JSON fidelity after the relevant evidence is supplied | Frozen diagnostic |
| `hc_product_monograph_native_pdf_extraction@0.1.0` | Complete official PDF | End-to-end retrieval, visual reading, and extraction | Draft, blocked from ranked runs |

The evidence-window release contains:

| Property | Value |
| --- | ---: |
| Product families | 40 |
| Official English PDFs | 40 |
| Official French PDFs | 40 |
| Development products | 30 |
| Held-out products | 10 |
| Scored source-evidence items | 390 |
| Maximum evidence pages per case | 3 |

All 80 PDF hashes are distinct, every scored item has a recorded source page,
and the dataset manifest verifies 6 of 6 files. This diagnostic deliberately
uses pages selected with knowledge of the reference labels. It is useful for
isolating extraction quality, but it must not be described as full-document
retrieval or PDF understanding.

The active evidence-window runs show GPT-5.6 Sol at 64/80 strict passes and
GPT-5.6 Terra and Luna at 56/80 each. GPT-5.4 Mini's saved run contains 80 API
configuration failures and is visible but ineligible for rank. The corrected
model config omits unsupported temperature sampling so that model must be run
again before comparison.

The native-PDF release reuses the same cohort and provisional labels but sends
the complete hash-verified PDF through the Responses API. It remains draft
until the 390-item human label review, local source verification, provider
smoke test, and prompt lock are complete. See the
[`evidence-window benchmark card`](docs/hc_benchmark/HC_PRODUCT_MONOGRAPH_BENCHMARK.md)
and the
[`native-PDF benchmark plan`](docs/hc_benchmark/HC_PRODUCT_MONOGRAPH_NATIVE_PDF_BENCHMARK.md).

## Quick start

### Docker

Create a local environment file:

```bash
cp .env.example .env
```

On Windows Command Prompt:

```cmd
copy .env.example .env
```

Set the Azure resource base URL and credentials in `.env`, then build:

```bash
docker compose build
```

Show available commands:

```bash
docker compose run --rm evalanche python -m evalanche.cli --help
```

Run the complete local quality gate:

```bash
docker compose run --rm evalanche sh -lc \
  "python -m pip install --no-cache-dir -r requirements-dev.txt && ruff check . && coverage run -m pytest && coverage report"
```

Rebuild with `--no-cache` when dependency pins or the Dockerfile change.
The `.dockerignore` excludes credentials, Git metadata, caches, and large
local run artifacts from the build context.

### Local Python

Python 3.11 and 3.12 are supported by the current continuous integration
matrix.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m evalanche.cli --help
pytest
```

On Windows PowerShell, activate with:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Core workflow

For normal benchmark work, start with the registry:

```bash
python -m evalanche.cli registry-validate
python -m evalanche.cli build-leaderboard --all
```

Open `reports/benchmarks/index.html` for every versioned leaderboard.

Plan a new model run without provider calls:

```bash
python -m evalanche.cli run-benchmark \
  --benchmark hc_product_monograph_structured_extraction@0.1.0 \
  --model gpt_5_6_sol \
  --plan-only
```

Omit `--plan-only` after reviewing the 80-call plan. Generation, evaluation,
result registration, and leaderboard rebuilding then run as one workflow.
Existing models are not rerun. See
[`docs/benchmark_registry.md`](docs/benchmark_registry.md).

After a deterministic scorer or canonicalization rule changes, rescore saved
outputs without repeating provider generation calls:

```bash
python -m evalanche.cli rescore-benchmark \
  --benchmark hc_product_monograph_structured_extraction@0.1.0 \
  --model gpt_5_6_sol
```

Offline rescoring rejects benchmarks containing LLM-judge cases so the command
always keeps its zero-generation-call contract explicit.

### 1. Define representative cases

Generation inputs require:

```text
case_id
input
expected_output
evaluation_type
```

Supported evaluation routes are:

- `exact`, normalized string comparison;
- `json`, parsed and optionally canonicalized JSON comparison;
- `judge`, rubric-based LLM evaluation.

Additional columns such as language, domain, risk, source, and difficulty are
preserved through generation and evaluation.

### 2. Configure candidates and pricing

Candidate profiles define the readable model name, provider route, supported
request parameters, retries, output limit, provider model version, and
optional pricing ID.

```yaml
models:
  - name: example_model
    model: ${CANDIDATE_MODEL:-azure/example-deployment}
    temperature: 0
    max_retries: 3
    max_completion_tokens: 900
    pricing_id: example_price_2026_01_01
```

Reasoning models can omit unsupported sampling parameters and declare an
explicit reasoning baseline:

```yaml
models:
  - name: example_reasoning_model
    model: ${CANDIDATE_MODEL:-azure/example-reasoning-deployment}
    temperature:
    reasoning_effort: none
    max_retries: 3
    max_completion_tokens: 900
```

See [`configs/README.md`](configs/README.md) for the configuration inventory
and [`docs/endpoint_pricing.md`](docs/endpoint_pricing.md) for pricing rules.

### 3. Generate once

```bash
python -m evalanche.cli generate --config configs/generate_generic.yaml
```

Large runs can declare concurrency, rate limits, checkpoints, resume behavior,
a model-specific cost sample, a safety multiplier, and a maximum projected
cost. Inspect a guarded run without making model calls:

```bash
python -m evalanche.cli generate \
  --config configs/generate_hc_dpd_census_gpt_5_6_luna.yaml \
  --preflight-only
```

### 4. Evaluate saved outputs

Run the combined evaluator:

```bash
python -m evalanche.cli evaluate \
  --config configs/evaluate_generated_generic.yaml
```

Standalone deterministic and judge commands remain available for focused
workflows:

```bash
python -m evalanche.cli metrics \
  --config configs/metrics_generated_generic.yaml

python -m evalanche.cli judge \
  --config configs/judge_generated_generic.yaml
```

Generation and evaluation stay separate. A saved candidate response can be
inspected and evaluated again after a scoring change without another provider
call.

### 5. Review evidence

A combined evaluation can produce:

- case-level scores and failure reasons;
- model-level quality and operational summaries;
- exact paired pass and fail comparisons with multiplicity correction;
- constraint-aware selection output;
- complete run metadata and hashes;
- a plain-language recommendation report.

Missing latency, token, or cost evidence remains unknown. It is never silently
treated as zero.

For the completed DPD census, generate the compact analysis and selected-case
review package from the saved case-level results:

```bash
python -m evalanche.cli analyze-dpd-census
```

This command makes no model calls. It validates four-model case coverage,
analyzes fields, languages, complexity strata, bilingual product families,
failure mechanisms, costs, and paired frontier cases, then writes a
deterministic 105-case review worksheet.

Audit that worksheet against the frozen source archive and rescore all
selected outputs:

```bash
python -m evalanche.cli audit-dpd-review
```

The audit is deterministic, makes no model calls, and records its decisions
separately from the immutable review evidence. With the default paths, it also
refreshes the analysis and release integrity manifests.

## Dataset releases

Versioned manifests freeze dataset identity and provenance. Verify the generic
example:

```bash
python -m evalanche.cli verify-dataset \
  --manifest configs/datasets/generic_example_manifest.yaml \
  --root .
```

Verify the committed Health Canada census:

```bash
python -m evalanche.cli verify-dataset \
  --manifest configs/datasets/hc_dpd_structured_extraction_census_0.2.0_manifest.yaml \
  --root .
```

Dataset verification makes no model calls.

Acquire, rebuild, and verify the Product Monograph release:

```bash
python -m evalanche.cli acquire-product-monographs
python -m evalanche.cli build-product-monograph-benchmark
python -m evalanche.cli verify-dataset \
  --manifest configs/datasets/hc_product_monograph_structured_extraction_0.1.0_manifest.yaml \
  --root .
```

The 80 source PDFs are verified locally but not committed. The tracked source
lock preserves their official URLs, hashes, byte sizes, page counts, and
alignment checks.

Build and inspect the separate native-PDF draft:

```bash
python -m evalanche.cli build-product-monograph-native-pdf-benchmark
python -m evalanche.cli verify-dataset \
  --manifest configs/datasets/hc_product_monograph_native_pdf_extraction_0.1.0_manifest.yaml \
  --root .
python -m evalanche.cli run-benchmark \
  --benchmark hc_product_monograph_native_pdf_extraction@0.1.0 \
  --model gpt_5_6_sol \
  --plan-only
```

The plan is allowed, but execution is intentionally blocked while the
benchmark status is `draft`. The field-level review queue is
`data/hc/benchmarks/product_monograph_native_pdf_extraction/0.1.0/label_review.csv`.

## Full DPD workflow

The complete full-census workflow, including cost preflight, model-specific
generation, resume behavior, assembly, evaluation, and artifact retention, is
documented in the
[`Health Canada DPD census runbook`](docs/hc_benchmark/HC_DPD_CENSUS_RUNBOOK.md).

The 500-case visual comparison notebook remains available at
[`notebooks/HC_DPD_Model_Comparison_Demo.ipynb`](notebooks/HC_DPD_Model_Comparison_Demo.ipynb).
The full-census analysis notebook is
[`notebooks/HC_DPD_Census_Analysis.ipynb`](notebooks/HC_DPD_Census_Analysis.ipynb).
Both use the dependencies in `notebooks/requirements.txt` and make no model
calls when run with their default settings.

## Repository layout

```text
configs/      Model, benchmark, dataset, pricing, and legacy run configs
data/         Tracked examples and immutable benchmark releases
docs/         Method definitions, benchmark specifications, and runbooks
evalanche/    Application code
notebooks/    Reproducible demonstration notebook and builder
reports/      Compact, versioned, source-controlled result releases
tests/        Unit and integration tests
```

Local model outputs and case-level evaluations are written to
`data/generated/` and `results/`. Both paths are ignored by Git.

## Documentation

- [Dataset manifests](docs/dataset_manifests.md)
- [Benchmark registry and generated leaderboards](docs/benchmark_registry.md)
- [Endpoint pricing](docs/endpoint_pricing.md)
- [Operational metrics](docs/operational_metrics.md)
- [Statistical methods](docs/statistical_methods.md)
- [Constraint-aware model selection](docs/model_selection.md)
- [DPD source snapshot](docs/hc_benchmark/HC_DPD_SOURCE_SNAPSHOT.md)
- [Initial DPD benchmark slice](docs/hc_benchmark/HC_DPD_BENCHMARK_SLICE.md)
- [Full DPD census and bounded demo](docs/hc_benchmark/HC_DPD_CENSUS_DEMO.md)
- [DPD full-census runbook](docs/hc_benchmark/HC_DPD_CENSUS_RUNBOOK.md)
- [DPD census analysis and validation](docs/hc_benchmark/HC_DPD_CENSUS_ANALYSIS.md)
- [DPD model-comparison demo](docs/hc_benchmark/HC_DPD_MODEL_COMPARISON_DEMO.md)
- [DPD and Product Monograph benchmark specification](docs/hc_benchmark/HC_DPD_PM_BENCHMARK_SPEC.md)
- [Product Monograph benchmark card and runbook](docs/hc_benchmark/HC_PRODUCT_MONOGRAPH_BENCHMARK.md)
- [Native-PDF Product Monograph benchmark plan](docs/hc_benchmark/HC_PRODUCT_MONOGRAPH_NATIVE_PDF_BENCHMARK.md)
- [Product Monograph architecture review and decision record](docs/hc_benchmark/PRODUCT_MONOGRAPH_ARCHITECTURE_REVIEW.md)
- [Health Canada dataset inventory](docs/hc_benchmark/HC_DATASET_INVENTORY.md)
- [Contribution and release rules](CONTRIBUTING.md)

## Data and artifact policy

- Generic examples are synthetic.
- Frozen Health Canada source and benchmark releases retain official
  provenance, licensing, file hashes, and transformation lineage.
- Credentials and local endpoint URLs belong only in `.env`.
- Raw model outputs, checkpoints, and case-level results are not committed.
- Compact result releases under `reports/` retain hashes for excluded raw
  artifacts.
- Source snapshots and published releases are immutable. Corrections require a
  new version or an explicit superseding release.

## Interpretation limits

Evalanche results apply to the evaluated task, dataset, prompts, model
versions, provider configuration, and scoring policy.

- Public benchmarks help identify candidates but do not replace local
  task-specific evaluation.
- LLM judges are measurement instruments and require human calibration for
  important use.
- Case-level confidence intervals do not cover prompt sensitivity, provider
  drift, judge variability, or clustered cases unless those are modeled.
- Configured costs are estimates from declared rates and recorded token usage,
  not reconciled invoices.
- Latency reflects the observed run environment, not a service-level
  guarantee.
- Weighted selection policies encode stakeholder preferences and must not be
  presented as empirical facts.

## Next milestones

1. Complete and sign off the 390-item native-PDF label review queue.
2. Verify all 80 local PDFs, smoke-test the Azure Responses path, and lock the
   native-PDF prompt before promoting the draft.
3. Rerun GPT-5.4 Mini with the corrected request settings.
4. Run all compatible models on the promoted native-PDF benchmark and compare
   the paired score change against the evidence-window diagnostic.
5. Add broader monograph tasks only as separate benchmark contracts.

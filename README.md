# Evalanche

Evalanche is a practical resource for answering:

> Which model should I use for my problem, and what evidence supports that
> choice?

It does not claim that one model is best in general. It connects three kinds
of evidence:

1. task requirements and hard deployment constraints;
2. curated public benchmark evidence that helps create a shortlist;
3. reproducible local comparisons when public evidence is not specific enough.

Health Canada benchmarks are optional task packs and case studies. They prove
that the evaluation system can handle large structured datasets, bilingual
work, source-locked documents, and complete PDFs. They are not Evalanche's
main product or the only kind of benchmark it is intended to support.

## Start here

| What you need | Where to start |
| --- | --- |
| I do not know which model to consider | [Choosing a model with Evalanche](docs/choosing_a_model.md) |
| I want relevant public benchmark evidence | [Public benchmark evidence guide](docs/public_benchmark_evidence.md) |
| I have a shortlist and representative cases | [Core workflow](#core-workflow) |
| I need hard cost, latency, reliability, or capability rules | [Constraint-aware selection](docs/model_selection.md) |
| I want to see complete benchmark examples | [Optional Health Canada case studies](#optional-health-canada-case-studies) |

The normal decision path is:

```text
Define the task and constraints
        ↓
Use relevant public evidence
        ↓
Create a small credible shortlist
        ↓
Run a local comparison only if needed
        ↓
Choose using quality, cost, latency, reliability, and risk
```

## What Evalanche supports today

- independent model, dataset, benchmark, and result contracts;
- exact text, canonical JSON, and rubric-based LLM-judge evaluation;
- complete case results, field and slice summaries, paired statistics, and
  recommendation reports;
- token, cost, latency, retry, failure, and evidence-coverage accounting;
- explicit price catalogs, cost guardrails, checkpoints, and safe resume;
- compatibility fingerprints that prevent unlike runs from being mixed;
- local draft experiments that cannot alter published leaderboards;
- registry-discovered summaries that automatically include every completed
  compatible model, including models added later;
- immutable result bundles and CSV, JSON, Markdown, and sortable HTML
  leaderboards for ready or frozen benchmarks;
- optional Health Canada DPD and Product Monograph task packs.

The largest remaining product gap is deeper task-based guidance backed by a
maintained public-evidence catalog. The first useful catalog and model-choice
workflow now exist in `docs/`, but they do not yet ingest current public scores
or generate an interactive shortlist.

## Extensibility rule

The set of models is data, not application code. Adding a model means adding
one manifest under `configs/models/`, then running it by ID. Benchmark analysis
must never contain a fixed list of current model names.

The generic experiment summary discovers compatible completed results from
their run plans and rebuilds:

- overall quality and confidence intervals;
- total cost, cost per request, cost coverage, tokens, and latency;
- arbitrary benchmark slices;
- required JSON-field accuracy;
- all-model case outcomes;
- every pairwise comparison for however many models are present.

No existing model output must be regenerated when a new model is added.

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

Plan every currently compatible model without provider calls:

```bash
python -m evalanche.cli run-benchmark \
  --benchmark <benchmark_id>@<version> \
  --all-compatible \
  --plan-only
```

Use one or more repeated `--model <model_id>` arguments when only a shortlist
should run. For a `ready` or `frozen` benchmark, omitting `--plan-only` runs,
registers, and publishes each selected model. For a draft benchmark, use the
explicit local experiment mode instead:

```bash
python -m evalanche.cli run-benchmark \
  --benchmark <benchmark_id>@<version> \
  --model <model_id> \
  --experiment
```

Draft experiments save outputs and evaluations locally but cannot register a
result or alter a leaderboard. Review the generated plans and pricing before
execution. The multi-model command does not yet calculate one aggregate
projected budget.

Rebuild a comparison from every completed compatible local evaluation:

```bash
python -m evalanche.cli summarize-benchmark \
  --benchmark <benchmark_id>@<version>
```

This command makes no provider calls. A newly registered and completed model
appears automatically the next time the command runs. See
[`docs/benchmark_registry.md`](docs/benchmark_registry.md).

After a deterministic scorer or canonicalization rule changes, rescore saved
outputs without repeating provider generation calls:

```bash
python -m evalanche.cli rescore-benchmark \
  --benchmark hc_product_monograph_structured_extraction@0.1.0 \
  --model <model_id>
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

The terminal summary includes total cost, average cost per request, cost
coverage, token use, and p95 latency alongside quality.

## Optional Health Canada case studies

The following workflows are complete task-specific examples and research
assets. They are useful demonstrations of Evalanche, but completing or
expanding them is not required for the main model-selection product.

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
  --all-compatible \
  --plan-only
```

Run a selected model as an unregistered local experiment by replacing
`--all-compatible --plan-only` with `--model <model_id> --experiment`.
Summarize existing native-PDF evaluations, including runs made before this
command existed, with:

```bash
python -m evalanche.cli summarize-benchmark \
  --benchmark hc_product_monograph_native_pdf_extraction@0.1.0
```

Registered publication remains blocked while the benchmark status is `draft`.
The optional publication review queue is
`data/hc/benchmarks/product_monograph_native_pdf_extraction/0.1.0/label_review.csv`.
It contains 390 scored facts across 80 PDFs, not 390 documents. You only need
to complete it if you decide that this HC task pack should become a published,
ranked benchmark. See the native-PDF benchmark plan for exact review steps.

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

- [Choosing a model with Evalanche](docs/choosing_a_model.md)
- [Public benchmark evidence for model selection](docs/public_benchmark_evidence.md)
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

1. Expand the task-based public-evidence catalog and keep every source dated,
   scoped, and linked to the model-choice question it can answer.
2. Turn the decision worksheet into a simple shortlist workflow that applies
   hard capability and deployment filters before showing benchmark evidence.
3. Add smoke, screen, and standard local-run tiers with aggregate cost
   preflight, a maximum budget, and safe multi-model resume.
4. Add worked model-choice examples that combine public evidence, local
   results, cost, latency, reliability, and an explicit decision record.
5. Treat HC publication as an optional parallel track. Complete its label
   review and promote a new frozen release only if a public HC leaderboard or
   research contribution is actually worth the review and run cost.

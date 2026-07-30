# Mission Evalanche

Evalanche is a task-grounded language-model evaluation toolkit for answering:

> Which model should I use for this specific task?

It combines deterministic metrics, rubric-based LLM judging, paired
statistics, operational evidence, cost accounting, and explicit selection
constraints. The goal is a defensible task-specific recommendation, not a
generic model leaderboard.

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
- a reproducible bilingual Health Canada DPD benchmark with 14,034 cases.

The repository includes more than 250 focused tests, plus continuous checks
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

These results are provisional pending manual label review, grouped
product-family analysis, and error adjudication. See the
[`published result release`](reports/hc_dpd_census/0.2.0/README.md) for the
full scope, paired evidence, metadata, raw-artifact hashes, and interpretation
limits.

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

## Full DPD workflow

The complete full-census workflow, including cost preflight, model-specific
generation, resume behavior, assembly, evaluation, and artifact retention, is
documented in the
[`Health Canada DPD census runbook`](docs/hc_benchmark/HC_DPD_CENSUS_RUNBOOK.md).

The 500-case visual comparison notebook remains available at
[`notebooks/HC_DPD_Model_Comparison_Demo.ipynb`](notebooks/HC_DPD_Model_Comparison_Demo.ipynb).
Its dependencies are isolated in `notebooks/requirements.txt`.

## Repository layout

```text
configs/      Candidate, pricing, dataset, generation, and evaluation configs
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
- [Endpoint pricing](docs/endpoint_pricing.md)
- [Operational metrics](docs/operational_metrics.md)
- [Statistical methods](docs/statistical_methods.md)
- [Constraint-aware model selection](docs/model_selection.md)
- [DPD source snapshot](docs/hc_benchmark/HC_DPD_SOURCE_SNAPSHOT.md)
- [Initial DPD benchmark slice](docs/hc_benchmark/HC_DPD_BENCHMARK_SLICE.md)
- [Full DPD census and bounded demo](docs/hc_benchmark/HC_DPD_CENSUS_DEMO.md)
- [DPD full-census runbook](docs/hc_benchmark/HC_DPD_CENSUS_RUNBOOK.md)
- [DPD model-comparison demo](docs/hc_benchmark/HC_DPD_MODEL_COMPARISON_DEMO.md)
- [DPD and Product Monograph benchmark specification](docs/hc_benchmark/HC_DPD_PM_BENCHMARK_SPEC.md)
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

1. Analyze and manually validate the full DPD census result.
2. Publish the validated model comparison and operational decision report.
3. Implement the bilingual Product Monograph extraction benchmark.
4. Expand task families while keeping their datasets, metrics, and
   recommendations separate.
5. Calibrate LLM-judge tasks against human review.

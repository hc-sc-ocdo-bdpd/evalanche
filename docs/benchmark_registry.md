# Benchmark registry and generated leaderboards

The registry is the normal Evalanche software workflow for adding models,
datasets, benchmarks, and completed runs. It replaces benchmark-specific
assembly code and one-off notebooks when a local comparison is needed. The
handbook and evidence directory remain the repository's primary entry point.

## Contracts

The registry composes independent artifacts:

| Artifact | Responsibility |
| --- | --- |
| Model manifest | Provider route, request settings, version metadata, capabilities, and pricing reference |
| Access set | Dated confirmation of deployment routes available to the user |
| Dataset manifest | Source provenance, immutable files, membership, sampling, and splits |
| Benchmark manifest | Dataset binding, prompt, scoring rules, slices, grouping, runtime settings, and optional run tiers |
| Result bundle | Compatibility fingerprint, compact case scores, operational evidence, and source-result hash |

A model can be added without editing a benchmark. A benchmark can be added
without editing a model. Generic leaderboard code discovers compatible active
result bundles and never contains model names or dataset-specific branches.

## Registry locations

```text
configs/
  models/
    <model_id>.yaml
  benchmarks/
    <benchmark_id>_<version>.yaml
  access_sets/
    <access_set_id>.yaml
  datasets/
    <dataset release manifest>.yaml
```

Model IDs and benchmark references are stable identifiers. A benchmark
reference uses:

```text
<benchmark_id>@<major.minor.patch>
```

Validate every manifest and its case contract:

```bash
python -m evalanche.cli registry-validate
```

## Compatibility fingerprint

Every result bundle records a SHA-256 compatibility fingerprint over:

- benchmark ID and version;
- dataset ID and version;
- exact case-file hash;
- dataset-manifest hash;
- prompt contract;
- scoring and canonicalization contract;
- Evalanche evaluator version.

A changed case, prompt, scorer, or dataset release cannot silently enter an
existing leaderboard. The leaderboard build fails with the expected and
observed compatibility hashes.

## Plan models without calls

Resolve every access-confirmed model that declares the required capabilities:

```bash
python -m evalanche.cli run-benchmark \
  --benchmark <benchmark_id>@<version> \
  --all-compatible \
  --access-set configs/access_sets/<access_set_id>.yaml \
  --plan-only
```

Use repeated `--model` arguments to select a shortlist instead:

```bash
python -m evalanche.cli run-benchmark \
  --benchmark <benchmark_id>@<version> \
  --model <first_model_id> \
  --model <second_model_id> \
  --plan-only
```

The plan materializes candidate, generation, and evaluation configurations
under ignored local run directories. It reports the exact case and model-call
count. Planning also refuses a model that does not declare every capability
required by the benchmark.

`--all-compatible` without `--access-set` is rejected. A registered and
compatible route is not proof that the user can access it. Repeated `--model`
arguments explicitly confirm access for that command.

## Run a publishable benchmark

Only `ready` and `frozen` benchmarks can register results and publish
leaderboards. After reviewing the plan, run selected models:

```bash
python -m evalanche.cli run-benchmark \
  --benchmark <benchmark_id>@<version> \
  --model <model_id>
```

The command generates responses, evaluates them, registers an immutable
compact result bundle, rebuilds that benchmark's leaderboard, and refreshes
the benchmark index. Existing models are not rerun.

## Run a local experiment

Use `--experiment` when the evidence is still exploratory or the benchmark is
draft:

```bash
python -m evalanche.cli run-benchmark \
  --benchmark <benchmark_id>@<version> \
  --model <model_id> \
  --experiment
```

The command performs generation and evaluation, then writes a generic local
comparison. It never registers a result, rebuilds a leaderboard, or changes
the benchmark's status. This is the only execution mode allowed for a draft
benchmark. `--all-compatible --access-set <path> --experiment` and repeated
`--model` arguments are supported.

## Run a cost-aware tier campaign

An optional `tiers` mapping in the benchmark manifest defines cumulative
cohorts without putting dataset or model names in the engine. A tier declares:

- deterministic `all` or `balanced` sampling;
- whether the sampling unit is a case or the benchmark grouping key;
- a cumulative unit count, seed, and arbitrary stratification columns;
- an inherited parent tier;
- an observed cost-sample tier or explicit first-run token assumptions;
- safety and per-request reserve multipliers;
- optional quality and generation-failure promotion gates.

Inherited tiers are incremental. If smoke has 2 cases, screen has 20
cumulative cases, and standard has 80, the execution files contain 2, 18, and
60 cases. Summaries contain 2, 20, and 80. A case and model pair is called at
most once across the chain.

Preflight every compatible model inside the access set before provider calls:

```bash
python -m evalanche.cli run-benchmark \
  --benchmark <benchmark_id>@<version> \
  --tier smoke \
  --all-compatible \
  --access-set configs/access_sets/<access_set_id>.yaml \
  --preflight-only
```

The preflight materializes the deterministic cohort, checks prerequisites and
checkpoint state, prices pending calls for every selected model, applies each
tier's safety multiplier, and prints one aggregate projection. Execute only
after choosing a campaign limit:

```bash
python -m evalanche.cli run-benchmark \
  --benchmark <benchmark_id>@<version> \
  --tier smoke \
  --all-compatible \
  --access-set configs/access_sets/<access_set_id>.yaml \
  --experiment \
  --max-cost-usd <budget>
```

The full campaign is blocked before its first call when the aggregate
projection exceeds the cap. During execution, Evalanche runs models
sequentially, reserves conservative retry cost before starting work, uses the
existing case and model checkpoint ledger, and refuses to start more work when
the remaining local budget is insufficient. One provider call can still cost
more than its local estimate, so this is a hard local start gate, not a cloud
billing limit.

After the screen completes, select models through manifest data rather than a
fixed analysis list:

```bash
python -m evalanche.cli run-benchmark \
  --benchmark <benchmark_id>@<version> \
  --tier standard \
  --promote-from screen \
  --access-set configs/access_sets/<access_set_id>.yaml \
  --preflight-only
```

Repeat with `--experiment --max-cost-usd <budget>` to execute. A cumulative
tier summary is rebuilt after every model completes. Tier campaigns are local
and unregistered, including a tier named `standard`.

Summarize completed local runs for access-confirmed candidates:

```bash
python -m evalanche.cli summarize-benchmark \
  --benchmark <benchmark_id>@<version> \
  --access-set configs/access_sets/<access_set_id>.yaml
```

For a cumulative tier, add:

```bash
python -m evalanche.cli summarize-benchmark \
  --benchmark <benchmark_id>@<version> \
  --tier <tier_name> \
  --access-set configs/access_sets/<access_set_id>.yaml
```

This makes no model calls and writes under:

```text
results/benchmark_experiments/<benchmark_id>/<version>/
  model_summary.csv
  pairwise_comparisons.csv
  slice_summary.csv
  field_summary.csv
  case_outcomes.csv
  model_comparison.md
  model_comparison.json
```

Tier summaries use the same files one directory lower, under
`<version>/<tier_name>/`.

The command discovers evaluations through compatibility-checked run plans and
then applies the requested access scope. Use `--all-results` to build a
historical evidence summary of every compatible completion. That report states
that current access was not confirmed. Use repeated `--model` arguments for an
explicit one-command candidate scope.

To apply an updated deterministic scorer to the saved model outputs, without
making generation calls, run:

```bash
python -m evalanche.cli rescore-benchmark \
  --benchmark <benchmark_id>@<version> \
  --model <model_id>
```

The command writes a fresh evaluation artifact, registers a new compatible
result bundle, and rebuilds the benchmark leaderboard. It refuses judge cases
because those would require a new provider call.

## Register an existing evaluation

Completed single-model evaluation results can be registered separately:

```bash
python -m evalanche.cli register-result \
  --benchmark hc_dpd_structured_extraction@0.2.0 \
  --model <model_id> \
  --results results/example_single_model_evaluation.csv
```

For a migration view extracted from a larger historical result, preserve the
original artifact as provenance:

```bash
python -m evalanche.cli register-result \
  --benchmark hc_dpd_structured_extraction@0.2.0 \
  --model <model_id> \
  --results results/temporary_single_model_view.csv \
  --provenance results/evaluate_hc_dpd_census_all_models_results.csv
```

Registration requires exact case membership and exactly one declared model.
The source evaluation remains local. The tracked bundle retains its path,
hash, size, compact case evidence, and aggregate metrics.

## Result layout

```text
reports/benchmarks/
  index.html
  <benchmark_id>/
    <version>/
      active_runs/
        <model_id>.json
      runs/
        <model_id>/
          <run_id>/
            run.json
            case_scores.csv.gz
      leaderboard.csv
      leaderboard.json
      leaderboard.md
      leaderboard.html
      pairwise.csv
```

Run IDs are content-derived. Registering the same evidence again reproduces
the same bundle bytes. Active pointers select one compatible run per model
without deleting historical bundles.

## Leaderboards

Build one leaderboard:

```bash
python -m evalanche.cli build-leaderboard \
  --benchmark hc_dpd_structured_extraction@0.2.0
```

Build every registered leaderboard:

```bash
python -m evalanche.cli build-leaderboard --all
```

Open `reports/benchmarks/index.html` in a browser. Each standalone HTML table
is sortable and also has CSV, JSON, and Markdown equivalents.

Rank is strict case pass rate among complete runs. A run with any generation
failure or unscored case remains visible for diagnosis but is ineligible for
rank and excluded from paired comparisons. Field score, arbitrary dataset
slices, generation reliability, cost, and latency remain separate. Evalanche
does not collapse those dimensions into an unexplained universal
recommendation.

## Add another model

1. Add one YAML file under `configs/models/`.
2. Confirm and add the route to the relevant access set.
3. Validate the registry.
4. Plan or run any compatible benchmark by model ID.

No benchmark manifest, Python module, notebook, combined-output file, existing
result, or list of model names needs to change. Re-run `summarize-benchmark`
after the new evaluation completes and every aggregate, slice, field, case,
and pairwise output is rebuilt for the discovered set.

Tests add a synthetic model through a manifest and register it without a
Python edit to the engine.

## Add another dataset and benchmark

1. Freeze source and derived files with a dataset manifest.
2. Create a case table with `case_id`, `input`, `expected_output`, and
   `evaluation_type`.
3. Preserve any columns needed for slices and a grouping key.
4. Add a benchmark manifest that binds the dataset to its prompt and scorer.
5. Validate, plan, and run.

The Product Monograph work proves both text and native-file paths. The frozen
evidence-window diagnostic uses label-selected page text. The separate draft
native-PDF benchmark uses hash-verified complete files through the Responses
API and declares `pdf_input`, `responses_api`, and `vision` as required model
capabilities. Their input contracts and leaderboards remain separate.

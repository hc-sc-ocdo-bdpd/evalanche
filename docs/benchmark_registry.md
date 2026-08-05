# Benchmark registry and generated leaderboards

The registry is the normal Evalanche workflow for adding models, datasets,
benchmarks, and completed runs. It replaces benchmark-specific assembly code
and notebooks as the product surface.

## Contracts

The registry composes four independent artifacts:

| Artifact | Responsibility |
| --- | --- |
| Model manifest | Provider route, request settings, version metadata, capabilities, and pricing reference |
| Dataset manifest | Source provenance, immutable files, membership, sampling, and splits |
| Benchmark manifest | Dataset binding, prompt, scoring rules, slices, grouping, and runtime settings |
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

## Run a registered benchmark

Resolve a run without making provider calls:

```bash
python -m evalanche.cli run-benchmark \
  --benchmark hc_product_monograph_structured_extraction@0.1.0 \
  --model gpt_5_6_sol \
  --plan-only
```

The plan materializes candidate, generation, and evaluation configurations
under ignored local run directories. It reports the exact case and model-call
count. Planning also refuses a model that does not declare every capability
required by the benchmark.

Only `ready` and `frozen` benchmarks can execute. A `draft` benchmark can be
planned for review, but an execution attempt is blocked until its documented
promotion gates pass.

After reviewing the plan, omit `--plan-only`:

```bash
python -m evalanche.cli run-benchmark \
  --benchmark hc_product_monograph_structured_extraction@0.1.0 \
  --model gpt_5_6_sol
```

The command generates responses, evaluates them, registers an immutable
compact result bundle, rebuilds that benchmark's leaderboard, and refreshes
the benchmark index. Existing models are not rerun.

To apply an updated deterministic scorer to the saved model outputs, without
making generation calls, run:

```bash
python -m evalanche.cli rescore-benchmark \
  --benchmark hc_product_monograph_structured_extraction@0.1.0 \
  --model gpt_5_6_sol
```

The command writes a fresh evaluation artifact, registers a new compatible
result bundle, and rebuilds the benchmark leaderboard. It refuses judge cases
because those would require a new provider call.

## Register an existing evaluation

Completed single-model evaluation results can be registered separately:

```bash
python -m evalanche.cli register-result \
  --benchmark hc_dpd_structured_extraction@0.2.0 \
  --model gpt_5_6_sol \
  --results results/example_single_model_evaluation.csv
```

For a migration view extracted from a larger historical result, preserve the
original artifact as provenance:

```bash
python -m evalanche.cli register-result \
  --benchmark hc_dpd_structured_extraction@0.2.0 \
  --model gpt_5_6_sol \
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
2. Validate the registry.
3. Plan or run any compatible benchmark by model ID.

No benchmark manifest, Python module, notebook, combined-output file, or
existing result needs to change.

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

# Configuration inventory

Evalanche keeps run definitions explicit so a result can be traced to the
exact candidate, price card, prompt, dataset, and evaluation policy used.
Configuration filenames are therefore stable provenance, not disposable
examples.

## File groups

| Pattern | Purpose |
| --- | --- |
| `models/*.yaml` | Independently reusable registered model manifests |
| `benchmarks/*.yaml` | Dataset, prompt, scoring, required capability, input API, slice, and runtime benchmark contracts |
| `product_monograph/<version>/*.yaml` | Source-locked Product Monograph cohort and label-construction overrides |
| `candidate_models.yaml` | Small generic multi-model example |
| `candidate_gpt_*.yaml` | One pinned candidate profile per deployed model |
| `endpoint_pricing*.yaml` | Versioned public or organization-specific token rates |
| `datasets/*.yaml` | Dataset inventory and immutable release manifests |
| `generate_generic.yaml` | Minimal generation example |
| `generate_hc_dpd_census_demo*.yaml` | Bounded 24-case connectivity and cost pilots |
| `generate_hc_dpd_comparison_500*.yaml` | Frozen 500-case comparison runs |
| `generate_hc_dpd_census_gpt_*.yaml` | Full 14,034-case model runs |
| `evaluate_*.yaml` | Deterministic, judge, individual-model, or combined evaluations |
| `judge_*.yaml` | Standalone criteria-judge examples |
| `metrics_*.yaml` | Standalone deterministic-metric examples |

The historical run configurations remain flat because completed metadata
records their paths. New extensible work uses `models/`, `benchmarks/`, and
versioned dataset-specific configuration directories.

## Adding a model

The active model inventory is the set of valid YAML manifests in `models/`.
There is no central model list in Python and no benchmark analysis should
encode the currently available IDs.

1. Copy an existing manifest whose provider route is similar.
2. Give it a new stable `model_id`, provider deployment, declared capabilities,
   request settings, version metadata, and pricing reference.
3. Run `python -m evalanche.cli registry-validate`.
4. Plan a compatible benchmark by ID, or use `--all-compatible --plan-only`.
5. After evaluation, run `summarize-benchmark` to rebuild the discovered local
   comparison.

Do not add a new model to `candidate_models.yaml` or create a new combined
analysis configuration merely to make the registry workflow see it. Those
flat files are retained for historical provenance and the small generic
example.

## Naming

Use lowercase snake case and order the identifying parts from broad to
specific:

```text
<operation>_<dataset-or-task>_<scope>_<model>.yaml
```

Examples:

```text
generate_hc_dpd_census_gpt_5_6_terra.yaml
evaluate_hc_dpd_census_all_models.yaml
```

## Reproducibility rules

- Pin provider model versions in candidate profiles when the provider exposes
  a stable version identifier.
- Keep secrets and resource-specific base URLs in `.env`, never in YAML.
- Reference an explicit pricing catalog for any run that reports or limits
  cost.
- Use a new config when changing a prompt, dataset version, model version, or
  scoring policy for a published result.
- Keep pilot, comparison, and full-census outputs isolated by filename.
- Run `--preflight-only` before a large generation whenever the config declares
  a local cost guard.
- Run `registry-validate` before any registered benchmark plan or run.
- Change a benchmark version whenever case membership, prompt, or scoring
  semantics change. Compatibility fingerprints reject silent mixing.

See [`docs/benchmark_registry.md`](../docs/benchmark_registry.md) for the
extensible workflow,
[`docs/hc_benchmark/HC_DPD_CENSUS_RUNBOOK.md`](../docs/hc_benchmark/HC_DPD_CENSUS_RUNBOOK.md)
for the historical DPD workflow, and
[`docs/hc_benchmark/HC_PRODUCT_MONOGRAPH_BENCHMARK.md`](../docs/hc_benchmark/HC_PRODUCT_MONOGRAPH_BENCHMARK.md)
for the evidence-window release, and see
[`docs/hc_benchmark/HC_PRODUCT_MONOGRAPH_NATIVE_PDF_BENCHMARK.md`](../docs/hc_benchmark/HC_PRODUCT_MONOGRAPH_NATIVE_PDF_BENCHMARK.md)
for the draft native-PDF path.

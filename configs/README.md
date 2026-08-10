# Configuration inventory

Evalanche keeps run definitions explicit so a result can be traced to the
exact candidate, price card, prompt, dataset, and evaluation policy used.
Configuration filenames are therefore stable provenance, not disposable
examples.

## File groups

| Pattern | Purpose |
| --- | --- |
| `models/*.yaml` | Independently reusable registered model manifests |
| `access_sets/*.yaml` | Dated user confirmation of available deployment routes |
| `benchmarks/*.yaml` | Dataset, prompt, scoring, capabilities, input API, slices, runtime, and optional tier contracts |
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
| `judge_protocols/*.yaml` | Versioned task-specific judge validation contracts |
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
4. Add the confirmed route to a dated access set.
5. Plan by model ID, or use `--all-compatible --access-set <path> --plan-only`.
6. After evaluation, run `summarize-benchmark --access-set <path>` to rebuild
   the access-confirmed local comparison.

Do not add a new model to `candidate_models.yaml` or create a new combined
analysis configuration merely to make the registry workflow see it. Those
flat files are retained for historical provenance and the small generic
example.

## Adding benchmark tiers

Tiers belong in the benchmark manifest because sampling and promotion are
properties of a task, not properties of the current model inventory. A compact
example is:

```yaml
tiers:
  smoke:
    description: Verify the route and cost evidence.
    sampling:
      method: balanced
      unit: group
      count: 1
      seed: 20260805
      stratify_by: [language, difficulty]
    cost:
      initial_prompt_tokens: 10000
      initial_completion_tokens: 500
      safety_multiplier: 2.0
      request_ceiling_multiplier: 2.0
  screen:
    description: Screen a cumulative ten groups.
    inherits: smoke
    sampling:
      method: balanced
      unit: group
      count: 10
      seed: 20260805
      stratify_by: [language, difficulty]
    cost:
      sample_from: smoke
      initial_prompt_tokens: 10000
      initial_completion_tokens: 500
      safety_multiplier: 1.5
      request_ceiling_multiplier: 2.0
    promotion:
      minimum_pass_rate: 0.70
      maximum_generation_failure_rate: 0.05
```

`count` is cumulative and counts the selected unit. With `unit: group`, the
benchmark's `group_key` defines that unit. A child tier must contain its parent
and executes only its new cases. `registry-validate` checks the tier graph,
columns, membership, nonempty deltas, and cost-sample ancestry.

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
- Give every judge run stable task, prompt, rubric, model-version, and settings
  identifiers. A changed identity requires a new validation contract.
- Keep LLM-judge evidence exploratory unless the exact protocol, JSON report,
  source data, and generated validation artifacts remain hash-verifiable.
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

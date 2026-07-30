# Configuration inventory

Evalanche keeps run definitions explicit so a result can be traced to the
exact candidate, price card, prompt, dataset, and evaluation policy used.
Configuration filenames are therefore stable provenance, not disposable
examples.

## File groups

| Pattern | Purpose |
| --- | --- |
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

The root is intentionally flat for command-line discoverability and because
completed run metadata records these paths. New benchmark families should use
a clear family prefix instead of moving historical files.

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

See [`docs/hc_benchmark/HC_DPD_CENSUS_RUNBOOK.md`](../docs/hc_benchmark/HC_DPD_CENSUS_RUNBOOK.md)
for the complete Health Canada DPD workflow.

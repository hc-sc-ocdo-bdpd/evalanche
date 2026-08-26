# Configuration inventory

Evalanche keeps model, dataset, benchmark, pricing, and run definitions explicit
so every result can be traced to the exact contract that produced it.

## Main configuration groups

| Pattern | Purpose |
| --- | --- |
| `models/*.yaml` | Registered model configurations and declared capabilities |
| `access_sets/*.yaml` | Dated user confirmation of available model routes |
| `benchmarks/*.yaml` | Dataset, prompt, scoring, reporting, capability, runtime, and tier contracts |
| `datasets/*_manifest.yaml` | Versioned source and benchmark dataset manifests |
| `product_monograph/<version>/*.yaml` | Product Monograph cohort and reference-construction rules |
| `endpoint_pricing*.yaml` | Versioned pricing evidence |
| `judge_protocols/*.yaml` | Task-specific LLM-judge validation contracts |
| `generate_*.yaml`, `evaluate_*.yaml`, `judge_*.yaml`, `metrics_*.yaml` | Reproducible run definitions and examples |

Generated bring-your-own-task bundles live under ignored
`local_tasks/<task_id>/` directories and remain isolated from the main
registry unless deliberately reviewed and promoted.

## Add a model

A normal model addition should require data, not Python changes:

1. Add one manifest under `models/`.
2. Record the exact provider route, version, request settings, and verified capabilities.
3. Reference current pricing if cost will be measured.
4. Run `python -m evalanche.cli registry-validate`.
5. Confirm access for the run with repeated `--model` arguments or a dated access set.
6. Preflight before making calls.

The repository's existing model manifests are recorded configurations, not a
universal or automatically accessible model inventory.

## Benchmark reporting

A benchmark declares how complete results should be presented:

```yaml
reporting:
  mode: ranked
```

Use descriptive reporting when measurements are useful but the evidence
contract does not justify an official ordering:

```yaml
reporting:
  mode: descriptive
  reason: Reference labels lack independent human sign-off.
```

Descriptive runs retain pass rate, field score, slices, cost, latency,
reliability, and pairwise evidence, but receive no numeric rank. Incomplete or
failed runs remain ineligible.

## Benchmark tiers

Tiers are benchmark data because sampling and promotion belong to the task,
not the model inventory. A tier can define:

- cumulative case or group sampling;
- a deterministic seed and stratification columns;
- inheritance from a smaller tier;
- cost-estimation assumptions and safety multipliers;
- optional quality and generation-failure promotion gates.

Inherited tiers execute only new cases. `registry-validate` checks the tier
graph, membership, required columns, and cost-sample ancestry.

## Reproducibility rules

- Keep secrets and resource-specific endpoint URLs in `.env`, never YAML.
- Pin provider model versions when a stable identifier is available.
- Use explicit pricing evidence for any run that reports or limits cost.
- Create a new benchmark version for material changes to case membership,
  prompt, input representation, scorer, or evaluation semantics.
- Keep exact judge task, prompt, rubric, model, settings, and validation
  identities together.
- Run `--preflight-only` before paid multi-model work.
- Preserve completed raw outputs locally so reporting or rescoring does not
  repeat provider calls.
- Run `registry-validate` before a registered benchmark plan or run.

See [benchmark registry and results](../docs/benchmark_registry.md),
[local comparison](../docs/local_comparison.md), and the
[Product Monograph benchmark](../docs/hc_benchmark/HC_PRODUCT_MONOGRAPH_BENCHMARK.md).

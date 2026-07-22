# Constraint-aware model selection

Evalanche can apply explicit client requirements and weighted preferences after
it calculates model quality and operational metrics. The selection policy is
configured under `selection` in a combined evaluation YAML file.

## Decision order

The decision process has three stages:

1. Check that the evaluation is complete and includes at least two models.
2. Apply hard requirements to each model.
3. Rank eligible models using the configured weighted decision score.

A model is `ineligible` when available evidence proves that it failed a hard
requirement. A model is `unknown` when evidence needed by the policy is missing.
Unknown is not treated as either passing or failing. Any unknown candidate
blocks a recommendation because the missing evidence could change the result.
Unknown selection-policy fields are rejected during configuration loading so a
misspelled requirement cannot be silently ignored.

## Example configuration

```yaml
selection:
  enabled: true
  minimum_score_margin: 0.02

  weights:
    quality: 0.55
    cost: 0.15
    latency: 0.15
    reliability: 0.15

  constraints:
    require_model_profile: true
    minimum_pass_rate: 0.80
    minimum_pass_rate_ci_low: 0.60
    maximum_average_cost_usd: 0.01
    maximum_p95_latency_seconds: 8
    maximum_generation_failure_rate: 0.05
    required_capabilities:
      - bilingual
      - approved-hosting

  model_profiles:
    - name: model_a
      available: true
      capabilities:
        - bilingual
        - approved-hosting

    - name: model_b
      available: false
      capabilities:
        - bilingual
```

Model profile names must match the `model_name` values in the evaluation data.
Capability names are normalized to lowercase. A required capability should
represent a client-verified fact, such as deployment approval, hosting location,
language support, or an internal availability requirement.

When `require_model_profile` is false, a missing profile does not by itself
block quantitative selection. In that case, availability is shown as not
declared. Set the field to true whenever explicit deployment declarations are a
client requirement.

## Hard requirements

The supported hard requirements are:

- `minimum_pass_rate`: minimum observed pass rate.
- `require_model_profile`: require an explicit profile for every evaluated
  model. A missing profile produces an `unknown` status.
- `minimum_pass_rate_ci_low`: minimum lower endpoint of the reported 95%
  Wilson pass-rate interval.
- `maximum_average_cost_usd`: maximum average candidate-generation cost per
  request. Cost coverage must be complete.
- `maximum_p95_latency_seconds`: maximum p95 candidate-generation latency.
  Latency coverage must be complete.
- `maximum_generation_failure_rate`: maximum candidate-generation failure rate.
- `required_capabilities`: capabilities that must appear in the model profile.
- `model_profiles[].available`: whether the model is available for the intended
  deployment.

All limits are optional. A missing metric required by an enabled limit produces
an `unknown` status rather than a false pass.

## Weighted decision score

Weights must be nonnegative and must sum to a positive number. Evalanche
normalizes them by their total, so they do not need to add up to exactly one.

The four components are bounded between zero and one:

- Quality is the observed overall pass rate.
- Cost is `1 - average_cost / maximum_average_cost`, bounded to zero through
  one.
- Latency is `1 - p95_latency / maximum_p95_latency`, bounded to zero through
  one.
- Reliability is `1 - generation_failure_rate`.

A positive cost weight requires `maximum_average_cost_usd`, which supplies both
the hard limit and the scale for the cost component. A positive latency weight
similarly requires `maximum_p95_latency_seconds`.

Candidate cost uses the selected per-request cost recorded during generation.
An explicit endpoint-pricing estimate is preferred when complete, with complete
LiteLLM response-cost metadata used as fallback. The selection CSV and report
retain the cost source. See [`endpoint_pricing.md`](endpoint_pricing.md) for the
catalog and precedence rules.

The model with the highest score is selected only when its lead is at least
`minimum_score_margin`. This is a practical policy margin, not a statistical
confidence test.

When quality is the only positive weight and multiple models remain eligible,
Evalanche retains its paired statistical safeguard. The quality leader must
clearly outperform every other eligible model in the corrected paired tests.

## Artifacts

An enabled combined evaluation writes a model-selection CSV containing:

- eligibility status and reasons;
- missing evidence;
- quality, cost, latency, and reliability components;
- weighted decision score and selection rank;
- the recommended-model flag.

The same policy, decision, and model table are included in the Markdown report
and run metadata. Evaluation metadata schema version `0.6` includes a hash of
the model-selection CSV.

## Limits

The weighted score implements the configured policy. It does not prove that the
weights or thresholds are correct. Client owners should approve them before a
decision is used.

Latency, cost, and reliability values are point estimates from the current run.
They do not include operational uncertainty intervals and are not production
service-level guarantees. Capability and availability values are declarations
from configuration and must be independently verified.
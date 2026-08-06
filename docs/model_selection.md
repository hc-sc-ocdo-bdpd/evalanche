# Optional decision policy

Evalanche reports comparison evidence by default. It selects one model only
when a user deliberately enables a policy after confirming candidate access,
hard requirements, and decision preferences.

Start with the [model selection handbook](choosing_a_model.md). This page
documents the optional policy engine, not the primary user journey.

## When a policy is appropriate

Use it when:

- every candidate route is confirmed available;
- hard requirements have named owners and evidence;
- the task and cases are representative;
- the quality and operational measures are complete enough;
- the decision owner has approved thresholds and weights;
- selecting one model is actually useful.

Do not enable it merely to turn a comparison into a winner. A shortlist or
tradeoff table is often the honest output.

## Decision order

1. Confirm that at least two candidates were evaluated completely.
2. Require an explicit profile and availability declaration for every
   candidate.
3. Apply hard requirements.
4. Mark missing required evidence as `unknown`.
5. Exclude only candidates whose evidence shows a hard failure.
6. Score eligible candidates under the approved weighted policy.
7. Select one only when the configured margin and quality safeguards pass.

An unknown candidate blocks a policy selection because the missing evidence
could change the result.

## Configuration

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

Profile names must match `model_name` values in the evaluation data.
Availability must be written explicitly. An omitted availability value remains
unknown and cannot silently become eligible.

The profile is a decision record, not independent proof. The owner should
verify availability, hosting, region, approval, and capabilities through the
actual account and official provider documentation.

## Hard requirements

| Field | Meaning |
| --- | --- |
| `require_model_profile` | A missing profile produces an unknown status |
| `minimum_pass_rate` | Minimum observed strict pass rate |
| `minimum_pass_rate_ci_low` | Minimum lower endpoint of the 95 percent Wilson interval |
| `maximum_average_cost_usd` | Maximum observed generation cost per request, with complete coverage required |
| `maximum_p95_latency_seconds` | Maximum observed p95 generation latency, with complete coverage required |
| `maximum_generation_failure_rate` | Maximum provider-generation failure rate |
| `required_capabilities` | Verified capabilities required in the profile |
| `model_profiles[].available` | Explicit availability for the intended deployment |

A failed requirement produces `ineligible`. Missing evidence produces
`unknown`. Missing cost is never treated as zero.

## Weighted score

Weights are nonnegative and normalized by their total.

- Quality is observed overall pass rate.
- Cost is `1 - average_cost / maximum_average_cost`, bounded from zero to one.
- Latency is `1 - p95_latency / maximum_p95_latency`, bounded from zero to one.
- Reliability is `1 - generation_failure_rate`.

A positive cost weight requires `maximum_average_cost_usd`, which supplies a
hard limit and scale. A positive latency weight similarly requires
`maximum_p95_latency_seconds`.

The score is a policy calculation. It is not a statistically learned utility
function and it does not prove the preferences are correct.

## Selection safeguards

The highest policy score is selected only when its lead reaches
`minimum_score_margin`. This is a practical policy margin, not a confidence
test.

When quality is the only positive weight, the observed leader must also clearly
outperform every other eligible candidate in corrected paired tests. A rank
alone is not enough.

Run a sensitivity check when modest changes to weights or thresholds could
reverse the selected model. An unstable policy result should be reported as a
tradeoff, not hidden behind extra decimal places.

## Artifacts

An enabled evaluation writes:

- `*_model_selection.csv` with eligibility, evidence, components, score, and
  policy-selection flag;
- `*_comparison.md` with the comparison and optional policy outcome;
- metadata schema `1.0` with `decision`, `selection_policy`, access declarations,
  artifact paths, and hashes.

When the policy is disabled, the comparison report presents observed evidence
without a selected model.

The CSV field name `recommended` remains an internal compatibility field for
the policy flag. User-facing artifacts call it `Selected by policy` so a
configured preference rule is not confused with a universal recommendation.

## Limitations

- The policy can only compare candidates represented in the evaluated data.
- Access declarations can become stale.
- Capability declarations must be verified.
- Quality applies only to the task, cases, prompt, and scorer.
- Operational measurements are point estimates from the run environment, not
  production service-level guarantees.
- Cost estimates use declared prices and recorded usage, not reconciled bills.
- A weighted score can hide a severe subgroup failure unless that failure is a
  hard requirement.
- A policy does not replace stakeholder accountability.

Use the [decision record template](decision_record.md) to preserve who approved
the policy and what should trigger review.

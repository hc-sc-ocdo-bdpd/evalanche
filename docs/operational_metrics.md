# Operational metrics

Evalanche records operational evidence alongside output quality so a user can
compare speed, cost, and reliability. An optional policy can use those measures
only when it is explicitly enabled.

## Request latency

`generation_seconds` and `judge_seconds` measure end-to-end elapsed time for one
case. They include provider calls, validation retries, fallback calls, and retry
waits. The model summary reports the arithmetic mean and the 95th percentile
(p95) across observed requests.

These values describe the environment and provider state during the benchmark.
A sequential benchmark run is not a production load test, and its latency is
not a service-level guarantee.

## Token usage

Input, output, and total token fields use the usage returned with model
responses. When a response is rejected and retried, the tokens from every
observed response are included. A total is left blank when usage is missing for
one or more requests in that total. Coverage fields show how many requests had
usable metadata.

## Response cost

Cost values are in US dollars. Evalanche can calculate an estimate from an
explicit, versioned endpoint-pricing entry and recorded input/output tokens.
It does not infer a rate from a deployment name. Configured estimates are
preferred over LiteLLM response-cost metadata when both are complete.

The configured estimate, LiteLLM-reported value, selected cost, and selected
source remain separate in the evidence. LiteLLM cost is a fallback when a
configured estimate is unavailable. **Missing evidence is never treated as
zero.**

Summary cost is reported only when every request in that summary has a known
cost. `cost_coverage` shows the proportion of requests with cost metadata. For
example, a coverage value of `0.75` means three quarters of request costs were
available, and the total remains unknown.

`configured_cost_coverage` and `provider_reported_cost_coverage` describe each
underlying source independently. See
[`endpoint_pricing.md`](endpoint_pricing.md) for catalog fields, precedence,
retry accounting, and limitations.

## Reliability

Generation and judge requests have separate status and failure-rate fields.

- A candidate generation error is a candidate-model failure. It receives a
  score of zero because the model did not produce an answer for the case.
- A judge error is an evaluation-system failure. It remains unscored and is not
  counted as a candidate-model failure.
- **Any unscored judge row blocks a complete comparative conclusion until the
  judge call is recovered or rerun.**

Set `judge.continue_on_error: true` to preserve judge failures in combined run
artifacts. With the default value of `false`, a judge error stops the run.

## Retry accounting

`attempts` counts provider calls, including a fallback call made without a JSON
response-format parameter when needed. `failed_attempts` counts provider calls
that raised an exception. Token and cost totals include all responses observed
during retry handling, not only the final accepted response.

## Interpretation limits

Operational evidence can vary with region, network path, provider load, cold
starts, concurrency, rate limits, and retry policy. Benchmark reports should
retain the run metadata and should not generalize one run into a production
availability or latency commitment.

See [`model_selection.md`](model_selection.md) for the optional policy rules.
# Operational metrics

Evalanche records operational evidence alongside output quality so model
comparisons can eventually account for speed, cost, and reliability.

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

Cost values are in US dollars. Evalanche records cost only when LiteLLM includes
response-cost metadata. It does not invent a price from a deployment name or
treat missing cost as zero.

Summary cost is reported only when every request in that summary has a known
cost. `cost_coverage` shows the proportion of requests with cost metadata. For
example, a coverage value of `0.75` means three quarters of request costs were
available, and the total remains unknown.

## Reliability

Generation and judge requests have separate status and failure-rate fields.

- A candidate generation error is a candidate-model failure. It receives a
  score of zero because the model did not produce an answer for the case.
- A judge error is an evaluation-system failure. It remains unscored and is not
  counted as a candidate-model failure.
- Any unscored judge row blocks a comparative recommendation until the judge
  call is recovered or rerun.

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
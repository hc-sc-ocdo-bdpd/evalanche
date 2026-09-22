# Endpoint pricing

Evalanche can estimate token cost for custom deployments whose names are not
recognized by LiteLLM. The configured price is explicit and versioned. It is
never inferred from a deployment name.

## Pricing catalog

Start from `configs/endpoint_pricing.example.yaml`, but **replace every
example value before using it in an evaluation decision.**

```yaml
schema_version: "1.0"
catalog_version: "2026-07-22"

endpoints:
  - pricing_id: gpt_5_4_mini_global
    model: azure/gpt-5.4-mini
    currency: USD
    input_per_million_tokens: 0.00
    cached_input_per_million_tokens: 0.00
    output_per_million_tokens: 0.00
    effective_date: 2026-07-22
    source: replace with the approved public or internal rate source
    notes: replace zero placeholders before enabling this entry
```

**The zero values above are placeholders, not model prices.** The schema rejects an
entry whose input and output rates are both zero. Use the rate for the actual
provider, deployment type, region, billing arrangement, and effective date.
Rates are expressed in US dollars per one million tokens because current
selection thresholds are denominated in US dollars.

Each `pricing_id` must be unique. More than one ID may declare the same model
route when separate resources, deployment types, or contracts have different
rates.

## Explicit references

Generation config points to the catalog:

```yaml
endpoint_pricing_path: configs/endpoint_pricing.yaml
candidate_models_path: configs/candidate_models.yaml
```

Each priced candidate explicitly references its entry:

```yaml
models:
  - name: gpt_5_4_mini
    model: azure/gpt-5.4-mini
    pricing_id: gpt_5_4_mini_global
```

Judge and combined evaluation configs use the same pattern:

```yaml
endpoint_pricing_path: configs/endpoint_pricing.yaml

judge:
  model: azure/gpt-5.4-mini
  pricing_id: gpt_5_4_mini_global
  temperature: 0
  max_retries: 3
```

Evalanche rejects a `pricing_id` when the catalog is missing, the ID does not
exist, or the catalog entry's `model` is not an exact match for the configured
LiteLLM model route. This prevents a similarly named endpoint from silently
receiving the wrong price.

## Calculation and precedence

For each request, configured cost is calculated as:

```text
uncached input tokens * input rate
+ cached input tokens * cached input rate
+ output tokens * output rate
```

The total is divided by one million. If no cached-input rate is declared,
cached tokens use the regular input rate. If the provider does not expose a
cached-token count, all prompt tokens use the regular input rate.

The selected `cost_usd` follows this order:

1. A configured estimate, when an explicit price and complete input/output
   token evidence are available.
2. Complete LiteLLM response-cost metadata, as a fallback.
3. Unknown, when neither source is complete.

The configured estimate and provider-reported value are retained separately.
Evalanche does not treat missing usage, missing rates, or missing provider cost
as zero.

## Retries

Token usage and cost include every successful response observed inside retry
handling, including responses rejected by JSON parsing or judge validation.
Provider calls that fail without returning usage cannot be priced from token
evidence. Attempt and failure counts remain available alongside the cost.

## Reproducibility evidence

Generated CSV rows retain:

- selected cost and cost source
- configured estimate
- provider-reported cost
- pricing ID and exact model route
- input, cached-input, and output rates
- effective date and declared source
- pricing catalog version and SHA-256 hash

Generation and evaluation metadata also record the pricing catalog path,
SHA-256 hash, schema and catalog versions, resolved entries, and unpriced model
references. Old run metadata therefore retains the rates used even after the
catalog changes.

When a rate changes, update the catalog version, effective date, source, and
rate together. **Do not edit old run artifacts to apply a newer rate.** Re-run or
explicitly re-price the evaluation in a future workflow if a decision needs to
use current prices.

## Interpretation limits

Configured values estimate token charges. **They are not reconciled invoices
and do not currently model non-token charges** such as provisioned throughput,
reserved capacity, web search, file search, tools, storage, images, audio, or
currency conversion. A rate source may also exclude negotiated discounts or
taxes. Use a catalog entry only when its scope matches the endpoint and workload
being evaluated.

## Long-run cost guards

Generation configs can require a read-only preflight and a conservative local
stop:

```yaml
generation:
  cost_preflight_sample_path: data/generated/calibration_outputs.csv
  maximum_estimated_cost_usd: 30.00
  cost_safety_multiplier: 1.50
  request_cost_ceiling_usd: 0.25
```

The preflight prices the sample's observed input and output tokens with the
configured endpoint rate, assumes zero cached input tokens, scales the average
to the pending requests, and applies the safety multiplier. It aborts before
model calls when the projection exceeds the limit.

During generation, the same limit is checked before new work is submitted.
The runner includes completed configured cost, a worst-case reserve for
in-flight requests, and a reserve for failed attempts whose token usage was not
returned. Completed work is checkpointed before a budget stop.

`request_cost_ceiling_usd` is an explicit local reserve used before each
request. It is especially important for native PDFs because their provider
tokenization includes extracted text and rendered page images, so file bytes
cannot provide a defensible ceiling. Tier campaigns calculate this reserve
from observed earlier-tier token use, or from the benchmark's explicit
first-run token assumptions, and then apply the declared reserve multiplier.

Multi-model tier preflight sums every selected model after its safety
multiplier and blocks the campaign before the first call when the aggregate
exceeds `--max-cost-usd`. During execution, completed work consumes the shared
campaign budget before another model can start.

Run a preflight without model calls:

```bash
python -m evalanche.cli generate --config configs/generate.yaml --preflight-only
```

**This remains a local estimate and start gate, not a provider billing hard
stop.** A request may cost more than its reserve, and provider calls without
usage evidence, negotiated rates, taxes, foreign exchange, and non-token
charges can differ from the configured calculation.
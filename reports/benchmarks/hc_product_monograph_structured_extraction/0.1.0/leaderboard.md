# Health Canada Product Monograph evidence-window extraction

Benchmark: `hc_product_monograph_structured_extraction@0.1.0`

**Descriptive comparison.** Complete results are shown without an official rank.

Reason: Reference labels have complete automated source-page evidence but no independent human sign-off. Results are useful for method and tradeoff analysis, but Evalanche does not assign an official rank.

| Model | Status | Passed | Pass rate | Field score | Generation failures | Cost (USD) | p95 latency (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| GPT-5.6 Luna | descriptive | 56/80 | 70.00% | 91.25% | 0.00% | 0.0863 | 5.010 |
| GPT-5.6 Sol | descriptive | 64/80 | 80.00% | 93.75% | 0.00% | 0.4365 | 5.190 |
| GPT-5.6 Terra | descriptive | 56/80 | 70.00% | 90.00% | 0.00% | 0.2154 | 5.632 |
| GPT-5.4 Mini | ineligible | 0/80 | 0.00% | 0.00% | 100.00% |  | 12.979 |

No official rank is assigned. Pass rate, field score, cost, latency, reliability, and slices remain descriptive.

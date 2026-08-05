# Health Canada Product Monograph evidence-window extraction

Benchmark: `hc_product_monograph_structured_extraction@0.1.0`

| Rank | Model | Status | Passed | Pass rate | Field score | Generation failures | Cost (USD) | p95 latency (s) |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | GPT-5.6 Sol | eligible | 64/80 | 80.00% | 93.75% | 0.00% | 0.4365 | 5.190 |
| 2 | GPT-5.6 Luna | eligible | 56/80 | 70.00% | 91.25% | 0.00% | 0.0863 | 5.010 |
| 2 | GPT-5.6 Terra | eligible | 56/80 | 70.00% | 90.00% | 0.00% | 0.2154 | 5.632 |
|  | GPT-5.4 Mini | ineligible | 0/80 | 0.00% | 0.00% | 100.00% |  | 12.979 |

Ranks use strict pass rate. Cost and latency are displayed separately and are not collapsed into a recommendation score. Runs with any generation failures or unscored cases are shown but are not ranked.

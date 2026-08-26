# Health Canada case studies

These case studies show how Evalanche handles bilingual, structured,
source-grounded evaluation on public Health Canada data. They are supporting
examples, not a model recommendation for Health Canada generally.

## At a glance

| Benchmark | Input contract | Cases | Reporting |
| --- | --- | ---: | --- |
| DPD structured extraction 0.2.0 | Deterministic bilingual DPD records | 14,034 | Ranked |
| Product Monograph evidence-window extraction 0.1.0 | Official text windows containing the required evidence | 80 | Descriptive |
| Product Monograph full-PDF extraction 0.1.0 | Complete official English and French PDFs | 80 | Descriptive |

Every result applies only to its exact dataset, input contract, prompt, scorer,
model route, and recorded operating conditions.

## Drug Product Database census

The DPD task asks a model to reproduce selected structured fields from a
deterministic rendering of frozen DPD records. Strict pass requires the full
JSON record to match after the declared symmetric canonicalization. It tests
structured extraction and schema fidelity, not unstructured document reading.

| Model route in the release | Strict passes | Pass rate | Field score | Observed cost |
| --- | ---: | ---: | ---: | ---: |
| GPT-5.6 Sol | 14,021 / 14,034 | 99.91% | 99.99% | US$75.52 |
| GPT-5.6 Terra | 13,986 / 14,034 | 99.66% | 99.95% | US$37.76 |
| GPT-5.4 Mini | 10,827 / 14,034 | 77.15% | 92.34% | US$11.40 |
| GPT-5.6 Luna | 10,662 / 14,034 | 75.97% | 95.92% | US$15.12 |

The reference values come directly from the frozen DPD source. The benchmark
also includes product-family analysis, English and French slices, field-level
diagnostics, paired disagreements, and a deterministic 105-case evidence
audit. None of this turns the result into a universal model recommendation.

Open the [release overview](../../reports/hc_dpd_census/0.2.0/README.md),
[analysis and validation](../../reports/hc_dpd_census/0.2.0/analysis/README.md),
or [generated benchmark results](../../reports/benchmarks/hc_dpd_structured_extraction/0.2.0/leaderboard.html).

## Product Monograph evidence-window extraction

This condition supplies text from official Product Monograph pages selected to
contain the required facts. It isolates extraction and JSON fidelity after the
relevant evidence has already been located.

| Model route in the release | Strict passes | Pass rate | Field score | Observed cost | Result status |
| --- | ---: | ---: | ---: | ---: | --- |
| GPT-5.6 Sol | 64 / 80 | 80.00% | 93.75% | US$0.4365 | Descriptive |
| GPT-5.6 Luna | 56 / 80 | 70.00% | 91.25% | US$0.0863 | Descriptive |
| GPT-5.6 Terra | 56 / 80 | 70.00% | 90.00% | US$0.2154 | Descriptive |
| GPT-5.4 Mini | 0 / 80 | 0.00% | 0.00% | Unknown | Ineligible, generation failed |

The 390 scored reference facts are traceable to exact source pages through
automated evidence checks. Independent human sign-off is not claimed, so
Evalanche reports the measurements without assigning an official rank.

## Product Monograph full-PDF extraction

This condition sends the complete hash-locked official PDF through a native
file-input route. It adds document retrieval, layout interpretation, long
context, provider PDF processing, and transport behavior to the same structured
extraction problem.

| Model route in the release | Strict passes | Pass rate | Field score | Observed cost | Result status |
| --- | ---: | ---: | ---: | ---: | --- |
| GPT-5.6 Terra | 56 / 80 | 70.00% | 91.25% | US$7.8579 | Descriptive |
| GPT-5.6 Sol | 55 / 80 | 68.75% | 90.62% | US$15.7344 | Descriptive |
| GPT-5.6 Luna | 52 / 80 | 65.00% | 88.44% | US$3.1440 | Descriptive |
| GPT-5.4 Mini | 42 / 80 | 52.50% | 85.94% | US$2.3589 | Descriptive |

All four saved runs generated 80 of 80 responses. The compact result bundles
retain bilingual slices, latency, cost, field diagnostics, and paired
comparisons. As with the evidence-window condition, reference facts have
complete automated source-page evidence but no independent human sign-off, so
the results remain descriptive.

Open the [Product Monograph benchmark card](../hc_benchmark/HC_PRODUCT_MONOGRAPH_BENCHMARK.md),
[evidence-window results](../../reports/benchmarks/hc_product_monograph_structured_extraction/0.1.0/leaderboard.html), or
[full-PDF results](../../reports/benchmarks/hc_product_monograph_native_pdf_extraction/0.1.0/leaderboard.html).

## Why both monograph conditions matter

The two conditions deliberately answer different questions:

- **Evidence window:** how well does the model extract and normalize the facts
  once relevant pages have been supplied?
- **Full PDF:** how well does the complete document-processing path work when
  the model must also find and interpret those facts in the full file?

Their difference is diagnostically useful, but the scores should not be merged
into one benchmark or treated as interchangeable samples.

## Interpretation boundaries

- DPD results do not predict Product Monograph, summarization, RAG, coding,
  agent, safety, or clinical performance.
- Product Monograph results do not establish a best model for Health Canada.
- A registry entry or recorded result does not establish current model access,
  approval, price, or availability.
- Configured costs are benchmark estimates based on recorded usage and price
  cards, not invoices.
- Observed latency is benchmark evidence, not a service-level guarantee.
- Missing evidence remains unknown.

Detailed DPD material is available in the
[full-census runbook](../hc_benchmark/HC_DPD_CENSUS_RUNBOOK.md) and
[analysis guide](../hc_benchmark/HC_DPD_CENSUS_ANALYSIS.md).

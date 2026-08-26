# Health Canada case studies

Evalanche includes one publication-quality census benchmark and two Product
Monograph pilots. They demonstrate bilingual, structured, source-grounded
evaluation. They do not define which models a new user can access.

## At a glance

| Benchmark | Input | Cases | Lifecycle | Ranking |
| --- | --- | ---: | --- | --- |
| DPD structured extraction 0.2.0 | Deterministic bilingual Drug Product Database records | 14,034 | Frozen | Enabled |
| Product Monograph evidence-window extraction 0.1.0 | Label-selected official monograph text | 80 | Frozen pilot | Disabled, permanently provisional |
| Product Monograph native-PDF extraction 0.1.0 | Complete official English and French PDFs | 80 | Retired local pilot | Disabled, permanently provisional |

Every result applies only to its named input contract, prompt, dataset, model
route, and scoring policy. The unfinished Product Monograph 1.0.0 expansion
and its proposed audit campaign were retired on 2026-08-17. They are not
part of the supported resource and carry no future audit commitment.

## Drug Product Database census

### What it measures

The DPD task asks a model to reproduce selected structured fields from a
deterministic bilingual case rendering. Strict pass requires the complete JSON
record to match after symmetric canonicalization. It measures structured
copying and schema following, not unstructured document understanding.

### Frozen result

| Model route in the release | Strict passes | Pass rate | Average field score | Observed cost |
| --- | ---: | ---: | ---: | ---: |
| GPT-5.6 Sol | 14,021 / 14,034 | 99.91% | 99.99% | US$75.52 |
| GPT-5.6 Terra | 13,986 / 14,034 | 99.66% | 99.95% | US$37.76 |
| GPT-5.4 Mini | 10,827 / 14,034 | 77.15% | 92.34% | US$11.40 |
| GPT-5.6 Luna | 10,662 / 14,034 | 75.97% | 95.92% | US$15.12 |

Cost and quality are deliberately separate. This table is benchmark evidence,
not a universal product recommendation.

The analysis covers product-family dependence, English and French slices,
field accuracy, failure mechanisms, every frontier disagreement, and a
deterministic 105-case evidence set. The evidence audit rebuilt sources,
parsed expected answers independently, and rescored 420 selected outputs.
Independent human sign-off is not claimed or needed for the stated automated
validation claim.

Open:

- [release overview](../../reports/hc_dpd_census/0.2.0/README.md)
- [analysis and validation](../../reports/hc_dpd_census/0.2.0/analysis/README.md)
- [selected-case evidence audit](../../reports/hc_dpd_census/0.2.0/analysis/EVIDENCE_AUDIT.md)
- [sortable leaderboard](../../reports/benchmarks/hc_dpd_structured_extraction/0.2.0/leaderboard.html)

## Product Monograph evidence-window pilot

This task supplies only text windows selected to contain expected evidence. It
isolates extraction and JSON fidelity after relevant pages have already been
found. It does not measure native PDF transport, page layout, long-document
retrieval, or complete-PDF context behavior.

The saved measurements are descriptive because the labels were not
independently validated. Evalanche therefore assigns no official ranks:

| Model route in the release | Strict passes | Pass rate | Average field score | Observed cost | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| GPT-5.4 Mini | 0 / 80 | 0.00% | 0.00% | Unknown | Ineligible, generation failed |
| GPT-5.6 Luna | 56 / 80 | 70.00% | 91.25% | US$0.0863 | Provisional |
| GPT-5.6 Sol | 64 / 80 | 80.00% | 93.75% | US$0.4365 | Provisional |
| GPT-5.6 Terra | 56 / 80 | 70.00% | 90.00% | US$0.2154 | Provisional |

The evidence-window release is frozen to preserve reproducibility. It will not
be promoted to an official ranked benchmark, and no label audit is planned.

Open:

- [benchmark card](../hc_benchmark/HC_PRODUCT_MONOGRAPH_BENCHMARK.md)
- [descriptive result table](../../reports/benchmarks/hc_product_monograph_structured_extraction/0.1.0/leaderboard.md)
- [interactive descriptive report](../../reports/benchmarks/hc_product_monograph_structured_extraction/0.1.0/leaderboard.html)

## Product Monograph native-PDF pilot

The retired native-PDF pilot sends a complete official PDF through a native
file-input route. It adds retrieval, layout, context, transport, and provider
PDF processing to the extraction problem.

Its retained assets include:

- 80 source-locked PDF descriptors representing 40 bilingual product pairs;
- deterministic JSON scoring for brand, ingredients, strengths, dosage forms,
  and routes;
- URL, SHA-256, byte-size, page-count, and language checks;
- historical local outputs that remain outside the public ranked reports.

Its expected labels inherit DPD alignment and automated page-evidence checks,
not independent human labeling. The pilot is therefore permanently
provisional and unranked. The old `label_review.csv` is retained only to
reproduce the historical snapshot. It is not an outstanding work queue.

See the [retired native-PDF pilot card](../hc_benchmark/HC_PRODUCT_MONOGRAPH_NATIVE_PDF_BENCHMARK.md).

## Interpretation boundaries

- Do not compare scores across the DPD, evidence-window, and native-PDF input
  contracts as if they measured one task.
- Do not treat a registry entry or historical result as proof of current model
  access.
- Do not convert missing cost, latency, or reliability evidence to zero.
- Use the DPD census for the committed ranked case study.
- Use Product Monograph measurements only as explicitly provisional examples.

## Detailed documentation

- [Health Canada dataset inventory](../hc_benchmark/HC_DATASET_INVENTORY.md)
- [DPD full-census runbook](../hc_benchmark/HC_DPD_CENSUS_RUNBOOK.md)
- [DPD analysis and validation](../hc_benchmark/HC_DPD_CENSUS_ANALYSIS.md)
- [DPD and Product Monograph specification](../hc_benchmark/HC_DPD_PM_BENCHMARK_SPEC.md)
- [Product Monograph evidence-window card](../hc_benchmark/HC_PRODUCT_MONOGRAPH_BENCHMARK.md)
- [Product Monograph native-PDF pilot](../hc_benchmark/HC_PRODUCT_MONOGRAPH_NATIVE_PDF_BENCHMARK.md)
- [Architecture review](../hc_benchmark/PRODUCT_MONOGRAPH_ARCHITECTURE_REVIEW.md)

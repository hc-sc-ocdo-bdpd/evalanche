# Evalanche Combined Evaluation Report

## Result

**Evidence-supported leader:** `gpt_5_6_sol`

It passed 14021 of 14034 cases (99.9%) and clearly outperformed each other evaluated model in the paired pass/fail comparisons. This supports choosing it for this tested task, subject to the operational constraints below; it is not the best model in general.

## Evaluation Scope

- **Run:** `evaluate_hc_dpd_full_census_all_models`
- **Task:** `hc_dpd_bilingual_structured_extraction_full_census_comparison`
- **Rows evaluated:** 56136
- **Unique cases:** 14034
- **Models:** 4
- **Deterministic rows:** 56136
- **LLM-judged rows:** 0
- **Generation errors:** 0
- **Judge errors:** 0
- **Judge model:** `azure/gpt-5.4-mini`
- **Judge pass threshold:** 75.0%

## Model Leaderboard

The primary rank is based on overall case pass rate.

| Rank | Model | Passed | Unscored | Pass rate | 95% pass-rate interval | Average score | Deterministic pass rate | Judge pass rate | Generation errors | Judge errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | gpt_5_6_sol | 14021/14034 | 0 | 99.9% | 99.8%-99.9% | 1.000 | 99.9% | N/A | 0 | 0 |
| 2 | gpt_5_6_terra | 13986/14034 | 0 | 99.7% | 99.5%-99.7% | 1.000 | 99.7% | N/A | 0 | 0 |
| 3 | gpt_5_4_mini | 10827/14034 | 0 | 77.1% | 76.4%-77.8% | 0.923 | 77.1% | N/A | 0 | 0 |
| 4 | gpt_5_6_luna | 10662/14034 | 0 | 76.0% | 75.3%-76.7% | 0.959 | 76.0% | N/A | 0 | 0 |

The intervals show uncertainty in each pass rate. They should not be used
alone to compare models because every model answered the same cases.

## Pairwise Evidence

The exact paired test compares where one model passed and the other failed.
Holm correction limits false positives when several model pairs are tested.

| Model A | Model B | Paired cases | Excluded cases | Pass-rate difference (A - B) | A-only passes | B-only passes | Holm-adjusted p-value | Clear winner |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt_5_4_mini | gpt_5_6_luna | 14034 | 0 | +1.2% | 2868 | 2703 | 0.028 | gpt_5_4_mini |
| gpt_5_4_mini | gpt_5_6_sol | 14034 | 0 | -22.8% | 8 | 3202 | <0.001 | gpt_5_6_sol |
| gpt_5_4_mini | gpt_5_6_terra | 14034 | 0 | -22.5% | 17 | 3176 | <0.001 | gpt_5_6_terra |
| gpt_5_6_luna | gpt_5_6_sol | 14034 | 0 | -23.9% | 6 | 3365 | <0.001 | gpt_5_6_sol |
| gpt_5_6_luna | gpt_5_6_terra | 14034 | 0 | -23.7% | 18 | 3342 | <0.001 | gpt_5_6_terra |
| gpt_5_6_sol | gpt_5_6_terra | 14034 | 0 | +0.2% | 42 | 7 | <0.001 | gpt_5_6_sol |

## Model Selection Policy

_Constraint-aware selection is disabled. The result uses the quality-only statistical recommendation._

### Eligibility and Decision Scores

_No constraint-aware model-selection table was produced._

## Operational Performance

Latency is measured end to end for each request, including retries and retry
waits. Average and p95 values describe this run, not guaranteed production
performance.

### Candidate Generation

| Model | Requests | Failures | Failure rate | Average latency | p95 latency | Input tokens | Output tokens | Total tokens | Cost | Cost source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt_5_6_sol | 14034 | 0 | 0.0% | 4.18 s | 4.74 s | 4,995,185 | 1,684,774 | 6,679,959 | $75.519145 USD | configured_endpoint_pricing |
| gpt_5_6_terra | 14034 | 0 | 0.0% | 4.07 s | 4.63 s | 4,995,185 | 1,684,814 | 6,679,999 | $37.760172 USD | configured_endpoint_pricing |
| gpt_5_4_mini | 14034 | 0 | 0.0% | 3.97 s | 4.69 s | 4,995,185 | 1,700,357 | 6,695,542 | $11.397995 USD | configured_endpoint_pricing |
| gpt_5_6_luna | 14034 | 0 | 0.0% | 3.97 s | 4.42 s | 4,995,185 | 1,687,338 | 6,682,523 | $15.119213 USD | configured_endpoint_pricing |

### LLM Judge

_No judge requests were recorded._

Token totals include every response observed during retries. Configured
endpoint rates are the primary cost source when the required token usage is
complete. Complete LiteLLM response-cost metadata is retained separately and
used as a fallback. Otherwise the cost remains unknown.

## Failed Cases

| Case | Model | Type | Source | Reason |
| --- | --- | --- | --- | --- |
| hc_dpd_0010861d91472f57_fr | gpt_5_4_mini | json | deterministic | JSON did not exactly match; 5 of 8 expected fields matched. mismatched=["brand_name","company","product_status"]. |
| hc_dpd_009fde59c56d9ad2_en | gpt_5_4_mini | json | deterministic | JSON did not exactly match; 6 of 8 expected fields matched. mismatched=["brand_name","company"]. |
| hc_dpd_00b669c807290cf4_en | gpt_5_4_mini | json | deterministic | JSON did not exactly match; 5 of 8 expected fields matched. mismatched=["brand_name","company","product_status"]. |
| hc_dpd_00b669c807290cf4_fr | gpt_5_4_mini | json | deterministic | JSON did not exactly match; 5 of 8 expected fields matched. mismatched=["brand_name","company","product_status"]. |
| hc_dpd_00dd3df4ffb70e30_en | gpt_5_4_mini | json | deterministic | JSON did not exactly match; 6 of 8 expected fields matched. mismatched=["brand_name","company"]. |
| hc_dpd_00dfef2acde9050d_en | gpt_5_4_mini | json | deterministic | JSON did not exactly match; 5 of 8 expected fields matched. mismatched=["brand_name","company","product_status"]. |
| hc_dpd_011be56bd2842fd8_fr | gpt_5_4_mini | json | deterministic | JSON did not exactly match; 5 of 8 expected fields matched. mismatched=["brand_name","company","product_status"]. |
| hc_dpd_0120ec4d95f3dcf4_en | gpt_5_4_mini | json | deterministic | JSON did not exactly match; 6 of 8 expected fields matched. mismatched=["brand_name","company"]. |
| hc_dpd_013eb29835b36571_en | gpt_5_4_mini | json | deterministic | JSON did not exactly match; 5 of 8 expected fields matched. mismatched=["brand_name","company","product_status"]. |
| hc_dpd_017f95a303b4ba69_en | gpt_5_4_mini | json | deterministic | JSON did not exactly match; 5 of 8 expected fields matched. mismatched=["brand_name","company","product_status"]. |

_Showing 10 of 6640 failed rows._

## How Scores Were Selected

- Exact and JSON cases use deterministic evaluation as the authoritative result.
- Open-ended judge cases use the configured LLM judge.
- Generation errors receive a score of zero and are not sent to the judge.
- Judge errors remain unscored and prevent a comparative recommendation.
- JSON partial-field scores are diagnostic; only a full match after the
  configured deterministic canonicalization passes.

## Evidence Files

- Case results: `results/evaluate_hc_dpd_census_all_models_results.csv`
- Model summary: `results/evaluate_hc_dpd_census_all_models_results_model_summary.csv`
- Pairwise comparisons: `results/evaluate_hc_dpd_census_all_models_results_pairwise_comparisons.csv`
- Model selection: `results/evaluate_hc_dpd_census_all_models_results_model_selection.csv`
- Run metadata: `results/evaluate_hc_dpd_census_all_models_results_run_metadata.json`

## Limitations

- Results apply only to this dataset, task, prompts, models, and configuration.
- LLM-judge outputs are evaluation signals and should be calibrated against
  human review for important uses.
- Statistical intervals cover case-sampling uncertainty only; they do not
  cover prompt, generation, or judge variability.
- The statistical methods assume cases are representative and independent.
  Related or repeated cases require grouped analysis.
- Operational measurements reflect this run's network path, provider state,
  retries, and sequential execution. They are not production service-level
  guarantees.
- Missing token or cost metadata is reported as unknown, not zero.
- Configured token costs are estimates from declared rates and recorded usage,
  not reconciled provider invoices.
- Selection weights, thresholds, and capabilities express configured policy
  choices. They are not empirical facts and should be reviewed by the client.
- Weighted operational components use point estimates from this run and do not
  include uncertainty intervals.
- A small or unrepresentative test set can produce unstable rankings.

# DPD census analysis and validation package

**Status:** Automated analysis complete. The selected-case evidence audit is
published separately in `EVIDENCE_AUDIT.md`. Independent human sign-off is
not claimed.

This package analyzes the four completed model runs on the frozen 14,034-case
Health Canada DPD census. It makes no model calls and does not replace the
strict deterministic scores.

## Published leaderboard

| Model | Strict passes | Case pass rate | Average field score | Both-language product pass rate | Run cost |
| --- | --- | --- | --- | --- | --- |
| gpt_5_4_mini | 10827 | 77.15% | 0.92344 | 60.70% | $11.40 |
| gpt_5_6_luna | 10662 | 75.97% | 0.95922 | 56.59% | $15.12 |
| gpt_5_6_terra | 13986 | 99.66% | 0.99952 | 99.36% | $37.76 |
| gpt_5_6_sol | 14021 | 99.91% | 0.99988 | 99.86% | $75.52 |

The frozen benchmark is a census of the eligible DPD snapshot, not a sample
from that snapshot. The rates above are descriptive for this benchmark.
They do not estimate performance on Product Monographs or future tasks.

## Main findings

1. `gpt_5_6_sol` and `gpt_5_6_terra` nearly saturate this
   structured-copying task. They failed 13 and
   48 cases respectively.
2. Their errors are not nested. `gpt_5_6_sol` alone passed
   42 cases, `gpt_5_6_terra` alone passed
   7, and both failed 6.
3. More than half of `gpt_5_6_terra` failures are attributable to the
   diagnostic category `compound_label_split_only`
   (26 of 48). These cases contain an
   official DPD label such as `SPRAY, BAG-ON-VALVE`. The published evidence
   audit retains these failures because rendered multi-values use ` | `,
   while the comma remains part of the official source label.
4. `gpt_5_6_sol` has 1 failure in the same diagnostic
   category. A hypothetical repair would shrink the frontier gap from
   35 to 10 cases, but the evidence
   audit rejected that alternate scoring interpretation. The tracked strict
   leaderboard is unchanged.
5. Mini and Luna have similar strict pass rates for different reasons.
   Their dominant failures are scalar/list type-shape violations, so average
   field score remains high even when the complete JSON record fails.

## Language and product-family dependence

The English and French records for each product are paired. Reporting whether
both language cases pass gives a more operationally meaningful product-level
view than treating all 14,034 cases as independent.

| Model | French minus English pass-rate gap | Both-language products passed |
| --- | ---: | ---: |
| gpt_5_4_mini | +12.10% | 4259 / 7017 |
| gpt_5_6_luna | -19.58% | 3971 / 7017 |
| gpt_5_6_terra | +0.34% | 6972 / 7017 |
| gpt_5_6_sol | +0.04% | 7007 / 7017 |


Mini is substantially stronger in French than English on the strict contract,
while Luna shows the opposite pattern. The error taxonomy shows that these
gaps are driven mainly by different schema-shape mistakes, not a uniform
loss of factual extraction quality.

## Diagnostic sensitivity, not rescoring

The analysis classifies failed cases by asking whether one narrowly defined,
mechanical transformation would recover the strict expected answer:

- `schema_type_only`, a singleton scalar/list type was reversed;
- `duplicate_items_only`, every expected item was present but at least one
  list item was repeated;
- `compound_label_split_only`, one official comma-bearing label was returned
  as multiple list entries;
- `text_or_value_error`, the remaining output changes source text or values.

These transformations are diagnostics only. They are not applied to the
published pass rate because doing so would weaken the tested contract.

| Model | Strict rate | Type-shape sensitivity | All structural sensitivity | Remaining text/value failures |
| --- | ---: | ---: | ---: | ---: |
| gpt_5_4_mini | 77.15% | 97.80% | 99.49% | 71 |
| gpt_5_6_luna | 75.97% | 98.38% | 99.23% | 108 |
| gpt_5_6_terra | 99.66% | 99.68% | 99.90% | 14 |
| gpt_5_6_sol | 99.91% | 99.91% | 99.95% | 7 |


The sensitivity columns are upper bounds under hypothetical deterministic
repair. They do not show what a model would produce under API-enforced
structured outputs. A controlled structured-output rerun is required to test
that production design.

## Every `gpt_5_6_sol` strict failure

| Case | Language | Stratum | Mismatched fields | Diagnostic type | gpt_5_6_terra passed |
| --- | --- | --- | --- | --- | --- |
| hc_dpd_043c39ecda680f30_en | en | multi_ingredient | ["active_ingredients"] | text_or_value_error | yes |
| hc_dpd_3276d80002eaedbb_en | en | multi_ingredient_multi_variant | ["active_ingredients"] | duplicate_items_only | no |
| hc_dpd_3276d80002eaedbb_fr | fr | multi_ingredient_multi_variant | ["active_ingredients"] | duplicate_items_only | no |
| hc_dpd_3e3c972983718ecd_en | en | single_ingredient | ["company"] | text_or_value_error | yes |
| hc_dpd_3e3c972983718ecd_fr | fr | single_ingredient | ["company"] | text_or_value_error | yes |
| hc_dpd_464e4967d555d027_en | en | multi_ingredient | ["active_ingredients"] | text_or_value_error | yes |
| hc_dpd_54012ce9033dd2c6_en | en | multi_ingredient | ["dosage_forms"] | compound_label_split_only | no |
| hc_dpd_73d9e3c207a7580f_en | en | multi_variant | ["active_ingredients"] | text_or_value_error | yes |
| hc_dpd_97fc1d09ecf78eaa_fr | fr | single_ingredient | ["routes"] | text_or_value_error | yes |
| hc_dpd_c4f2ab05670620ad_en | en | multi_ingredient_multi_variant | ["active_ingredients"] | duplicate_items_only | no |
| hc_dpd_c4f2ab05670620ad_fr | fr | multi_ingredient_multi_variant | ["active_ingredients"] | duplicate_items_only | no |
| hc_dpd_df86ea9ed8e7d06b_fr | fr | single_ingredient | ["active_ingredients"] | text_or_value_error | no |
| hc_dpd_e59e4150dafc95bb_en | en | multi_ingredient_multi_variant | ["active_ingredients"] | duplicate_items_only | yes |

The automated evidence audit rebuilt these cases from the frozen source,
independently parsed the expected answers, and recomputed their strict scores.
It retained every selected result, including comma-bearing labels and the
administrative `(FR)` prefix. That is automated benchmark validation, not
independent human sign-off.

## Selected-case evidence set

The tracked `manual_review.csv` contains 105 unique cases:

- all 13 `gpt_5_6_sol` failures;
- all 42 cases passed only by `gpt_5_6_sol` within the
  frontier pair;
- a deterministic, language-and-complexity-stratified sample of shared
  lower-tier failures;
- a deterministic, language-and-complexity-stratified sample of all-model
  passes.

`manual_review.csv` preserves the immutable selection and model evidence.
`evidence_audit.csv` records the completed automated decisions separately, so
the source worksheet does not need to be edited or attributed to a person.

## Leaderboard interpretation

The published evidence supports these conclusions:

- `gpt_5_6_sol` is the strict-score leader for this exact task.
- `gpt_5_6_terra` is 35 complete-record passes behind and has a lower
  observed run cost. The leaderboard exposes that tradeoff without declaring
  a universal winner.
- Evalanche reports benchmark measurements. It does not make a formal product
  recommendation unless a downstream user explicitly defines decision
  priorities and constraints.
- Mini and Luna should not be dismissed solely from this condition. Their
  large schema-shape component justifies a separate structured-output
  experiment if a low-cost production route matters.
- this benchmark is now saturated for frontier models and should be followed
  by the harder Product Monograph extraction benchmark.

## Files

| File | Purpose |
| --- | --- |
| `model_overview.csv` | Quality, bilingual product outcomes, latency, cost, and diagnostic sensitivities |
| `slice_performance.csv` | Overall, language, stratum, and language-by-stratum results |
| `field_accuracy.csv` | Field-level accuracy overall and by language |
| `error_taxonomy.csv` | Diagnostic failure mechanisms overall and by language |
| `pairwise_tradeoffs.csv` | Paired outcomes with cost increments |
| `frontier_cases.csv` | Every case failed by either frontier model |
| `manual_review.csv` | Immutable selected-case evidence worksheet |
| `evidence_audit.csv` | Completed automated decisions for all selected cases |
| `evidence_audit_summary.json` | Audit inputs, hashes, checks, and outcomes |
| `EVIDENCE_AUDIT.md` | Human-readable audit result and interpretation |
| `analysis_manifest.json` | Input and output hashes, methods, and review-set counts |

## Limitations

- The benchmark inputs are deterministic renderings of structured DPD rows,
  not unstructured Product Monographs.
- The audit verifies lineage, expected-answer construction, and strict scoring
  independently in code, but it does not claim completed human validation.
- Diagnostic repairs intentionally do not alter the primary score.
- Run cost is estimated from configured token prices and recorded usage, not
  reconciled billing.
- One generation per case does not measure output variability across seeds.

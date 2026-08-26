# Health Canada DPD census analysis and validation

This workflow turns the retained four-model case results into a compact,
source-controlled analysis release, a deterministic selected-case set, and a
reproducible evidence audit. It makes no model calls and does not alter the
deterministic benchmark score.

## Scope

The analysis requires the completed result table:

```text
results/evaluate_hc_dpd_census_all_models_results.csv
```

Before analysis, the command verifies:

- all four required models are present;
- every model covers the same case IDs exactly once;
- case metadata is identical across models;
- every product has one English and one French case;
- expected answers use the frozen eight-field DPD contract.

The default analysis compares GPT-5.6 Sol with GPT-5.6 Terra as the frontier
pair and uses GPT-5.4 mini and GPT-5.6 Luna as the lower-cost pair.

## Generate the analysis release

From the repository root:

```bash
python -m evalanche.cli analyze-dpd-census
```

In Docker:

```bash
docker compose run --rm evalanche \
  python -m evalanche.cli analyze-dpd-census
```

The command reads saved results, writes the compact release under
`reports/hc_dpd_census/0.2.0/analysis/`, and prints
`ANALYSIS COMPLETE`. Repeating it with unchanged inputs produces the same
selected cases and output hashes.

Useful optional controls are:

```text
--results
--config
--output-dir
--primary-model
--comparison-model
--shared-failure-sample
--all-pass-sample
--audit-seed
```

Changing models or sample settings creates a different analysis protocol and
must be recorded with its output release.

## Audit the selected evidence

Run the independent deterministic audit after generating the selected-case set:

```bash
python -m evalanche.cli audit-dpd-evidence
```

In Docker:

```bash
docker compose run --rm evalanche \
  python -m evalanche.cli audit-dpd-evidence
```

The audit rebuilds the selected cases from the frozen marketed DPD archive,
parses each expected answer independently from the rendered source, and
rescores all 420 selected model outputs with the configured canonical JSON
contract. It records the completed verification separately from the selected-case
evidence and refreshes the analysis and release integrity manifests. No model
calls are made.

## Outputs

| File | Purpose |
| --- | --- |
| `model_overview.csv` | Strict quality, field score, bilingual products, latency, cost, and diagnostic sensitivities |
| `slice_performance.csv` | Overall, language, complexity, and language-by-complexity results |
| `field_accuracy.csv` | Field accuracy overall and by language |
| `error_taxonomy.csv` | Strict failures grouped by diagnostic mechanism |
| `pairwise_tradeoffs.csv` | Paired outcomes and observed incremental cost |
| `frontier_cases.csv` | Every case failed by Sol, Terra, or both |
| `selected_cases.csv` | Deterministically selected cases and model evidence |
| `evidence_audit.csv` | Completed automated verification for all selected cases |
| `evidence_audit_summary.json` | Audit inputs, hashes, verification counts, and outcomes |
| `EVIDENCE_AUDIT.md` | Human-readable audit result and interpretation |
| `analysis_manifest.json` | Input, configuration, method, sampling, and output integrity evidence |
| `README.md` | Human-readable analysis and limitations |

The tracked notebook
`notebooks/HC_DPD_Census_Analysis.ipynb` visualizes these compact files. Install
the notebook dependencies and run all cells in VS Code or Jupyter:

```bash
python -m pip install -r requirements.txt -r notebooks/requirements.txt
```

The notebook does not need the 190 MB case-level result file unless
`AUTO_REFRESH_LOCAL_ANALYSIS` is deliberately changed to `True`.

## Diagnostic categories

The primary metric remains the configured canonical JSON equality check. A
failed case is additionally tested against narrow, deterministic
transformations:

- `schema_type_only`, a singleton scalar or list shape is reversed;
- `duplicate_items_only`, expected list items are repeated;
- `compound_label_split_only`, one official comma-bearing label is returned
  as multiple list items;
- combined structural categories;
- `text_or_value_error`, output text or values still differ;
- missing fields, extra fields, invalid JSON, or generation errors.

The resulting sensitivity rates are upper bounds under hypothetical repair.
They do not change the strict result and do not predict what an
API-enforced structured-output run would produce.

## Selected-case validation protocol

The default 105-case audit set includes:

1. all 13 Sol strict failures;
2. all 42 cases passed by Sol and failed by Terra;
3. 25 deterministic, language-and-complexity-stratified cases failed by both
   lower-cost models;
4. 25 deterministic, language-and-complexity-stratified all-model passes.

Model evidence in `selected_cases.csv` records the deterministic selection.
The automated audit writes verification results to `evidence_audit.csv`, keyed
by `audit_case_id` and `case_id`.

The audit checks:

- exact regeneration from the archived DPD rows and source provenance;
- an independent parse of each rendered input and expected JSON object;
- a fresh canonical score for every selected model output;
- comma-bearing labels using ` | ` as the rendered multi-value delimiter;
- scalar/list contract errors, duplicate facts, and changed source text;
- error ownership and a conservative operational-severity category.

The completed audit retained all 105 selected results. It found no expected
answer corrections, case exclusions, scoring-rule changes, or strict-score
disagreements. Independent human sign-off is not claimed. The stated
validation claim is limited to the completed automated checks above.

## Current interpretation

Sol is the strict-score leader. Terra costs about half as much and trails by
35 complete records. The evidence audit retained the comma-bearing label
failures because rendered multi-values use ` | ` while the comma remains part
of the official DPD label. The strict leaderboard is therefore unchanged.

Mini and Luna have similar strict accuracy but different language,
complexity, and field patterns. Most of their failures are structural JSON
shape errors. The structural-error pattern shows why strict and field-level results should
remain visible together.

This structured DPD benchmark is saturated for the frontier models. The
Product Monograph case studies provide the repository's harder unstructured-document conditions. Evalanche reports these task-specific
measurements without turning them into a universal model recommendation.

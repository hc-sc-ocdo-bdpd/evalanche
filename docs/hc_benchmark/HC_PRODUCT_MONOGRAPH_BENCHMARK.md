# Health Canada Product Monograph evidence-window benchmark 0.1.0

- **Status:** Frozen and ready for registered model runs
- **Dataset:** `hc_product_monograph_structured_extraction@0.1.0`
- **Cases:** 80 label-selected text-window instances
- **Model input:** Extracted text, not PDF files
- **Scoring:** Deterministic canonical JSON
- **Human sign-off:** Not claimed

## Purpose

This release measures bilingual structured extraction after the relevant
Product Monograph pages have already been selected. It is the first real
dataset added through the generic benchmark registry after the DPD census.
It is a strong diagnostic for extraction and JSON fidelity, but it is not an
end-to-end document-understanding benchmark.

It proves two independent extension paths:

- a new model joins through one model manifest;
- a new source, dataset, task contract, and leaderboard join without changing
  registry, bundle, or leaderboard code.

## Cohort

| Property | Count |
| --- | ---: |
| Product families | 40 |
| English PDFs | 40 |
| French PDFs | 40 |
| Development products | 30 |
| Held-out products | 10 |
| Single-ingredient products | 20 |
| Multi-ingredient products | 10 |
| Multi-variant products | 10 |
| Short documents, up to 30 pages | 24 cases |
| Medium documents, 31 to 60 pages | 40 cases |
| Long documents, over 60 pages | 16 cases |

Related active-ingredient groups do not cross the development and held-out
boundary.

The cohort started from the frozen 40-product DPD pilot. Screening required
distinct, text-extractable, language-correct English and French PDFs whose
scope matched the DPD product family. Missing or misaligned products were
replaced within the same split and stratum using
`SHA-256("20260730|product_id")` order.

The release records 23 exclusions:

- 10 missing bilingual document pairs;
- 12 unresolved product-scope mismatches;
- 1 posted French PDF that contains the English document.

See `screening_exclusions.csv` in the dataset release for every product and
reason.

## Frozen source evidence

`source_documents.csv` records, for every PDF:

- product and language;
- official PDF URL and DPD information URL;
- Product Monograph identifier;
- retrieval timestamp;
- SHA-256;
- byte size and page count;
- extraction, language, and scope checks.

All 80 hashes are distinct. The PDFs total roughly 67 MB and are not committed
because individual sponsor copyright may apply. The source lock and extracted
evidence are committed. Acquisition refuses a changed hash.

Download and verify the exact files:

```bash
python -m evalanche.cli acquire-product-monographs
```

Verify an existing local cache without downloading:

```bash
python -m evalanche.cli verify-product-monograph-sources
```

## Task contract

Each case asks for exactly:

```json
{
  "brand_name": "EXAMPLE",
  "active_ingredients": [
    {
      "name": "EXAMPLE INGREDIENT",
      "strength": 10,
      "unit": "MG"
    }
  ],
  "dosage_forms": ["TABLET"],
  "routes": ["ORAL"]
}
```

Lists are compared as sets. Strengths are numeric and path-scoped numeric
equivalence is enabled. Text comparison is case-insensitive, whitespace
normalized, punctuation-insensitive, and diacritic-insensitive. Explicit
aliases cover defensible unit and dosage-form variants.

DIN, regulatory schedule, product status, and sponsor remain in case
alignment metadata but are not scored. Screening showed that DINs were not
stated in any of the 80 PDFs, and the other fields were not represented
consistently enough to make a fair source-only extraction target.

## Evidence windows

This release evaluates extraction, not full-document retrieval.

The builder finds a source page for every expected brand, ingredient and
strength pair, dosage form, and route. It then places page 1 plus the minimum
ordered evidence-page set into the model input.

| Evidence property | Value |
| --- | ---: |
| Scored reference items | 390 |
| Normalized exact items | 240 |
| Ingredient name and strength items | 146 |
| Declared alias items | 4 |
| Maximum pages in one case | 3 |

Every scored item has a recorded PDF page. The evidence table is
`field_evidence.csv.gz`. This page selection is label-aware and must not be
interpreted as an evaluation of a model's ability to locate facts across an
entire monograph.

## Rebuild

After acquiring the PDFs:

```bash
python -m evalanche.cli build-product-monograph-benchmark
```

Expected:

```text
Products: 40
Cases: 80
Evidence items: 390
Verified files: 6/6
Status: VALID
Independent human sign-off: NOT CLAIMED
```

Verify the frozen release directly:

```bash
python -m evalanche.cli verify-dataset \
  --manifest configs/datasets/hc_product_monograph_structured_extraction_0.1.0_manifest.yaml \
  --root .
```

## Plan and run a model

Check the exact call count without provider access:

```bash
python -m evalanche.cli run-benchmark \
  --benchmark hc_product_monograph_structured_extraction@0.1.0 \
  --model gpt_5_6_sol \
  --plan-only
```

Expected: 80 cases, 80 planned model calls, and no calls made.

Run after the plan and endpoint configuration are reviewed:

```bash
python -m evalanche.cli run-benchmark \
  --benchmark hc_product_monograph_structured_extraction@0.1.0 \
  --model gpt_5_6_sol
```

Repeat with another model ID. Existing runs are not repeated. Each completed
run registers itself and refreshes the Product Monograph leaderboard.

If deterministic canonicalization changes after generation, reuse the saved
80 outputs and rebuild the score and leaderboard with no model calls:

```bash
python -m evalanche.cli rescore-benchmark \
  --benchmark hc_product_monograph_structured_extraction@0.1.0 \
  --model gpt_5_6_sol
```

The active saved runs are:

| Model | Strict passes | Field score | Generation failures | Ranking status |
| --- | ---: | ---: | ---: | --- |
| GPT-5.6 Sol | 64/80 | 93.75% | 0 | Eligible |
| GPT-5.6 Terra | 56/80 | 90.00% | 0 | Eligible |
| GPT-5.6 Luna | 56/80 | 91.25% | 0 | Eligible |
| GPT-5.4 Mini | 0/80 | 0.00% | 80 | Ineligible, request configuration failed |

GPT-5.4 Mini received an unsupported `temperature` parameter. Its corrected
manifest now omits temperature and declares `reasoning_effort: none`, but the
saved failed run must not be treated as model-quality evidence. It remains
visible and unranked until replaced by a valid run.

## Interpretation

This 40-product purposive sample is a pilot. It supports comparisons on this
exact evidence-grounded task, prompt, scorer, and source lock. It does not
represent all marketed products, full-document retrieval, clinical
reasoning, or general model quality.

The automated source-evidence audit is complete. Independent human label
sign-off is not claimed. Any later label correction requires a new dataset
version because the release is frozen.

For the separate complete-document path, see
[`HC_PRODUCT_MONOGRAPH_NATIVE_PDF_BENCHMARK.md`](HC_PRODUCT_MONOGRAPH_NATIVE_PDF_BENCHMARK.md).

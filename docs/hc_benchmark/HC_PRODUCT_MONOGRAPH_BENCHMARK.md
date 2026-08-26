# Health Canada Product Monograph evidence-window pilot 0.1.0

- **Status:** Frozen historical pilot
- **Dataset:** `hc_product_monograph_structured_extraction@0.1.0`
- **Cases:** 80 label-selected text-window instances
- **Model input:** Extracted text, not PDF files
- **Scoring:** Deterministic canonical JSON
- **Ranking:** Disabled, permanently provisional
- **Human audit:** Not planned

## Purpose and boundary

This release measures bilingual structured extraction after relevant Product
Monograph pages have already been selected. It is a useful diagnostic for
extraction and JSON fidelity, not an end-to-end document-understanding test.

The builder uses expected labels to locate evidence pages. That design is
valid for an oracle-window diagnostic but cannot support claims about finding
facts in a complete monograph. Scores must not be compared directly with a
native-PDF task.

The labels have automated source-page evidence but no independent human
validation. The benchmark is frozen for reproducibility, permanently
provisional, and unranked. No audit or promotion campaign is planned.

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

Related active-ingredient groups do not cross the development and held-out
boundary. Screening recorded 23 exclusions: 10 missing bilingual pairs, 12
unresolved product-scope mismatches, and one French URL containing an English
document.

## Source evidence

`source_documents.csv` records the official URL, Product Monograph identifier,
retrieval time, SHA-256, byte size, page count, language check, and scope check
for every document. All 80 hashes are distinct. PDF bytes are not committed
because individual sponsor copyright may apply.

The release has 390 scored reference items. Every item has a recorded source
page in `field_evidence.csv.gz`. The evidence windows include page 1 and the
minimum ordered page set needed for the expected fields.

Acquire and verify the source lock:

```bash
python -m evalanche.cli acquire-product-monographs
python -m evalanche.cli verify-product-monograph-sources
```

## Task and scorer

Each case requests exactly four fields:

```json
{
  "brand_name": "EXAMPLE",
  "active_ingredients": [
    {"name": "EXAMPLE INGREDIENT", "strength": 10, "unit": "MG"}
  ],
  "dosage_forms": ["TABLET"],
  "routes": ["ORAL"]
}
```

Lists are compared as sets. Strengths use path-scoped numeric equivalence.
Text comparison is case-insensitive, whitespace-normalized,
punctuation-insensitive, and diacritic-insensitive. Versioned aliases cover
defensible unit and dosage-form equivalents. DIN, schedule, status, and
sponsor remain alignment metadata and are not scored.

## Rebuild and verify

After acquiring the source PDFs:

```bash
python -m evalanche.cli build-product-monograph-benchmark
python -m evalanche.cli verify-dataset --manifest configs/datasets/hc_product_monograph_structured_extraction_0.1.0_manifest.yaml --root .
```

The expected build has 40 products, 80 cases, 390 evidence items, and six
verified release files.

## Retained descriptive measurements

| Model | Strict passes | Field score | Generation failures | Status |
| --- | ---: | ---: | ---: | --- |
| GPT-5.4 Mini | 0/80 | 0.00% | 80 | Ineligible, request configuration failed |
| GPT-5.6 Luna | 56/80 | 91.25% | 0 | Provisional |
| GPT-5.6 Sol | 64/80 | 93.75% | 0 | Provisional |
| GPT-5.6 Terra | 56/80 | 90.00% | 0 | Provisional |

GPT-5.4 Mini received an unsupported `temperature` parameter in the retained
run. Its manifest is corrected, but the failed run remains visible as
infrastructure evidence and must not be interpreted as model quality.

No row receives an official rank. Pass rate, field score, cost, latency,
language slices, and pairwise diagnostics remain visible as descriptive data.

## Interpretation

This purposive pilot does not represent all marketed products, full-document
retrieval, clinical reasoning, or general model quality. Any label correction
would require a new dataset version because the release is frozen. The
supported resource does not include a revised version of this pilot.

For the retired complete-document pilot, see
[`HC_PRODUCT_MONOGRAPH_NATIVE_PDF_BENCHMARK.md`](HC_PRODUCT_MONOGRAPH_NATIVE_PDF_BENCHMARK.md).

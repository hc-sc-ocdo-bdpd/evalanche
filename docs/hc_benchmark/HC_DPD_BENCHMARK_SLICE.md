# Health Canada DPD benchmark slice

## Release

The first executable Health Canada benchmark release is:

```text
Dataset: hc_dpd_structured_extraction_slice
Version: 0.1.0
Parent: hc_dpd_source_snapshot 2026.7.2
Products: 40
Cases: 80
Languages: English and French
Manifest: configs/datasets/hc_dpd_structured_extraction_0.1.0_manifest.yaml
```

This release turns the frozen July 2, 2026 marketed Drug Product Database
(DPD) archive into a small, reproducible bilingual structured-extraction
benchmark. It establishes the data pipeline and case contract before Product
Monograph documents are acquired.

This release is deliberately DPD-only. Its inputs are deterministic text
renderings of DPD relational records, not Product Monographs. It must not be
used as evidence of production monograph-extraction performance.

## Files

| File | Purpose | Records |
| --- | --- | ---: |
| `data/hc/benchmarks/dpd_structured_extraction/0.1.0/cases.csv` | Generation-ready English and French extraction cases | 80 |
| `data/hc/benchmarks/dpd_structured_extraction/0.1.0/products.csv` | Sample membership, strata, splits, and source-row traceability | 40 |
| `data/hc/benchmarks/dpd_structured_extraction/0.1.0/build_report.json` | Population, sampling, coverage, and quality-gate evidence | 1 |

The release manifest records the exact byte size, SHA-256 hash, record count,
parser version, sampling plan, and split membership for every file.

## Eligibility and normalization

The builder:

1. verifies the frozen parent manifest and all three parent files;
2. revalidates the marketed ZIP structure and each relational table;
3. parses the extract with explicit schemas and no inferred header row;
4. joins drug, company, ingredient, form, route, schedule, and current-status
   rows by `DRUG_CODE`;
5. retains marketed human products with an eight-digit DIN, exactly one
   `DIN_OWNER`, numeric strengths, and complete English and French reference
   labels;
6. groups related variants by DIN owner, normalized brand, and active
   ingredient codes; and
7. records the exact source member and one-based row numbers used by every
   case.

Veterinary, disinfectant, and radiopharmaceutical classes are excluded.
Approved-only, cancelled, dormant, incomplete, and ambiguous records are also
excluded. Multi-ingredient families that also contain multiple DPD variants
are deferred from version `0.1.0`.

## Sampling

Sampling uses stable SHA-256 ranking with seed `20260702`. The selected product
families are:

- 20 single-ingredient, single-variant products
- 10 multi-ingredient, single-variant products
- 10 single-ingredient, multi-variant products

The release also requires collective coverage of oral, injectable, topical,
inhaled, and ophthalmic routes. The materialized sample contains additional
routes and dosage forms.

Every selected active-ingredient code is unique within the 40-product sample.
This creates a stronger leakage boundary than merely grouping identical
product families.

## Splits

The product-level split is:

- 30 development products, producing 60 language cases
- 10 held-out products, producing 20 language cases

The held-out allocation is stratified as five single-ingredient, two
multi-ingredient, and three multi-variant products. No selected active
ingredient code appears in both splits.

Prompt development may use only the development cases. The held-out cases
remain reserved for later blind comparisons.

## Case contract

Each product produces one English and one French case. The CSV preserves
metadata columns while remaining directly compatible with Evalanche
generation because it includes:

```text
case_id
input
expected_output
evaluation_type
```

`evaluation_type` is `json`. The expected object contains exactly:

```json
{
  "din": ["00000000"],
  "brand_name": "EXAMPLE",
  "active_ingredients": [
    {
      "name": "EXAMPLE INGREDIENT",
      "strength": 10,
      "unit": "MG"
    }
  ],
  "dosage_forms": ["TABLET"],
  "routes": ["ORAL"],
  "schedule": ["PRESCRIPTION"],
  "product_status": "MARKETED",
  "company": "EXAMPLE COMPANY"
}
```

French cases use the official French DPD labels where the source provides
them. JSON keys remain stable across languages.

## Rebuild

The committed release is immutable. To prove reproducibility in a clean tree,
remove no existing release and run the builder only where version `0.1.0` has
not already been materialized:

```bash
python -m evalanche.cli build-dpd-benchmark \
  --source-manifest configs/datasets/hc_dpd_2026-07-02_manifest.yaml \
  --root . \
  --version 0.1.0 \
  --seed 20260702
```

The builder refuses to overwrite an existing output directory or manifest. A
successful clean build reports:

```text
Products: 40
Cases: 80
Product splits: 30 development, 10 heldout
Verified files: 3/3
Status: VALID
```

Verify committed bytes independently:

```bash
python -m evalanche.cli verify-dataset \
  --manifest configs/datasets/hc_dpd_structured_extraction_0.1.0_manifest.yaml \
  --root .
```

Expected final lines:

```text
Verified files: 3/3
Status: VALID
```

Neither command makes model calls. The rebuild command reads only the frozen
local source release and makes no network requests.

## Boundary for the next data step

Manual validation of the completed DPD census result is the immediate
analytical boundary.

Product Monograph acquisition remains incomplete. The next release must
resolve authorized English and French monographs for candidate products,
freeze and hash the PDFs, extract document text, audit DPD-to-document
alignment, apply document-quality exclusions, and replace the rendered DPD
inputs with bounded monograph text.

Until that audit is complete, this release is an integration benchmark and
reference-label slice, not the final Product Monograph benchmark.

The DPD-only population expansion is frozen as census version `0.2.0`. It
contains all 7,017 product families representable by the current contract,
14,034 bilingual cases, and a fixed 24-case live-demo view. See
[`HC_DPD_CENSUS_DEMO.md`](HC_DPD_CENSUS_DEMO.md). The census expands DPD
coverage but does not change the Product Monograph boundary described above.

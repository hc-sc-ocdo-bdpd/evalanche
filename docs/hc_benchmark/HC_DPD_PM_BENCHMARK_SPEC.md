# Evalanche HC Benchmark 1: DPD and Product Monograph Extraction

- **Specification status:** DPD census and Product Monograph evidence-window releases complete; native-PDF release draft
- **Benchmark type:** Bilingual structured extraction
- **Primary scoring:** Deterministic JSON and field-level metrics

## 1. Research question

How reliably can candidate language models extract auditable drug-product facts from Health Canada-authorized English and French Product Monographs?

The benchmark produces task-specific evidence and a versioned leaderboard. It
does not declare a universal model recommendation, test general medical
knowledge, or evaluate independent regulatory or clinical judgment.

## 2. Why this benchmark comes first

This design has the strongest contribution-to-effort ratio among the identified HC sources:

- authentic Health Canada documents
- bilingual task coverage
- objective reference fields already available in DPD
- deterministic evaluation for most outputs
- limited need for expert manual annotation
- direct relevance to document extraction work performed across HC
- a clear path from a small internal pilot to a publishable public benchmark

## 3. Pilot composition and staged releases

The DPD source input is frozen as `hc_dpd_source_snapshot` version `2026.7.2`.
It contains the official marketed and approved archives published 2026-07-02.
See [`HC_DPD_SOURCE_SNAPSHOT.md`](HC_DPD_SOURCE_SNAPSHOT.md) for exact hashes,
row counts, validation evidence, and reproduction instructions.

The executable DPD-only foundation is frozen as
`hc_dpd_structured_extraction_slice` version `0.1.0`. It contains 40 product
families and 80 paired English and French cases rendered from the DPD
relational records. It proves normalization, sampling, split assignment,
source-row traceability, and generation compatibility. See
[`HC_DPD_BENCHMARK_SLICE.md`](HC_DPD_BENCHMARK_SLICE.md).

Version `0.1.0` is not the final Product Monograph benchmark and must not be
used as evidence of monograph-extraction performance.

The full DPD-only population is frozen separately as
`hc_dpd_structured_extraction_census` version `0.2.0`. It contains all 7,017
product families representable by the current bilingual JSON contract and
14,034 English and French cases. It adds the 169 multi-ingredient,
multi-variant families deferred from slice `0.1.0` and records 13 remaining
source-ambiguous French-brand families as exclusions. See
[`HC_DPD_CENSUS_DEMO.md`](HC_DPD_CENSUS_DEMO.md).

The census includes a fixed 24-case presentation view. It is a visible,
bounded operational demo, not a hidden test split. Version `0.2.0` remains a
DPD-rendering benchmark and is not evidence of Product Monograph performance.

The Product Monograph evidence-window diagnostic is frozen as
`hc_product_monograph_structured_extraction` version `0.1.0`. It contains
**40 marketed human-drug products** after document availability, document
quality, language, and scope-alignment checks. Its inputs are extracted text
from label-selected source pages, not PDF files, so its scores isolate
extraction and JSON fidelity rather than end-to-end document retrieval.

- 20 single-ingredient products
- 10 multi-ingredient products
- 10 products with multiple strengths, dosage forms, routes, or DINs
- English and French monographs where both are available
- 80 document-language cases after documented replacements

A separate `hc_product_monograph_native_pdf_extraction` version `0.1.0`
release sends one complete, hash-verified PDF per case through the Responses
API. It reuses the same cohort and provisional labels, but it is deliberately
`draft` until human label review, provider smoke testing, and prompt locking
are complete. The two input contracts must have separate leaderboards.

The materialized DPD slice includes varied oral, injectable, topical, inhaled,
ophthalmic, and other routes and dosage forms. Sampling is reproducible from
the frozen DPD snapshot using seed `20260702`. Product Monograph acquisition
may require documented replacements or exclusions in a new dataset version.

### Exclusions for the pilot

- veterinary products
- disinfectants
- radiopharmaceutical products
- cancelled or dormant products
- products without a usable DPD reference record
- products without a posted English product monograph
- image-only or severely malformed PDFs
- cases where DPD-to-monograph alignment cannot be resolved during audit

These exclusions reduce infrastructure and label-noise risk in the first release. Harder document types can be added deliberately in later versions.

## 4. Unit of evaluation

In Product Monograph versions `0.1.0` and `1.0.0`, one case represents one
Product Monograph language instance.

Each case contains:

```text
case_id
benchmark_version
snapshot_date
drug_code
din_list
brand_name
language
monograph_sha256
input
input_files
expected_output
evaluation_type
source_metadata
```

`evaluation_type` is `json`. In the evidence-window contract, `input` contains
the extraction instruction and bounded page text and `input_files` is absent.
In the native-PDF contract, `input` contains only the task instruction and
`input_files` contains the repository-relative path, filename, media type, and
frozen SHA-256 for the complete PDF. The expected output is canonical JSON in
both contracts.

In DPD-only versions `0.1.0` and `0.2.0`, one case represents one rendered
DPD product-family language instance. Its metadata uses `product_id`,
`drug_codes`, exact archive-member row numbers, and parent hashes instead of
`monograph_sha256`. The expected-output contract is otherwise the same.

## 5. Expected-output contract

```json
{
  "brand_name": "EXAMPLE",
  "active_ingredients": [
    {
      "name": "EXAMPLE INGREDIENT",
      "strength": 10,
      "unit": "mg"
    }
  ],
  "dosage_forms": ["TABLET"],
  "routes": ["ORAL"]
}
```

Arrays are used for fields that may legitimately contain more than one value. The schema does not force a single answer where the product record is one-to-many.

## 6. Canonicalization rules

Canonicalization is applied identically to reference and model outputs.

- trim leading and trailing whitespace
- collapse repeated internal whitespace
- compare controlled text fields case-insensitively
- Unicode-normalize text
- normalize DINs to eight digits
- normalize numeric strengths without changing their value
- map units through a versioned unit vocabulary
- compare ingredient, route, dosage-form, and schedule arrays as sets unless order is explicitly meaningful
- retain the original raw output for audit

Synonym mapping is versioned and limited to defensible equivalences. Fuzzy matching is not used for primary pass/fail decisions.

For the DPD census demo, these rules are declared explicitly under
`metrics.json_comparison` in
`configs/evaluate_hc_dpd_census_demo.yaml`. The run metadata preserves the
complete comparison profile. Numeric equivalence is path-scoped to
`/active_ingredients/*/strength`; it does not coerce other fields or structural
types. Raw parsed-JSON equality is retained as a diagnostic, while pass/fail
uses the configured canonical comparison.

## 7. Reference-label construction

1. Use frozen DPD marketed and approved source release `2026.7.2`.
2. Load each table with an explicit schema and `header=None`.
3. Validate column counts and join coverage.
4. Join relevant tables using `DRUG_CODE`.
5. Restrict the backbone to eligible marketed human drugs.
6. Identify the authorized monograph URLs through the DPD Online Query.
7. Download English and French documents separately and hash every file.
8. Match each document to its DPD product/DIN set.
9. Audit document language and product scope before case generation.
10. Record a PDF page for every scored reference item and exclude unresolved cases.

The audit records whether each candidate field is:

- confirmed in both DPD and the monograph
- present in DPD but not found in the monograph
- present in the monograph but represented differently in DPD
- ambiguous
- excluded from scoring

DIN, schedule, product status, and sponsor are retained as alignment metadata
but are not scored in version `0.1.0`. They are not stated consistently in the
source documents. Independent human sign-off is not claimed.

## 8. Scoring

### Required validity measures

- valid JSON rate
- schema-valid JSON rate
- required-field completion rate

### Primary effectiveness measures

- exact case pass rate
- macro-average field accuracy
- micro-average field accuracy
- set precision, recall, and F1 for multi-valued fields
- per-field accuracy

### Secondary measures

- English versus French performance gap
- cross-language consistency for the same product
- generation failure rate
- latency and token use, when available

A case passes only when the output is valid, schema-compliant JSON and every required scored field matches after canonicalization. Partial field scores are retained for diagnosis and model comparison.

## 9. Leakage and independence controls

- Freeze the benchmark before final model comparison.
- Treat the evidence-window benchmark as an oracle-page extraction diagnostic:
  its page selection used the reference labels and cannot support a retrieval
  claim.
- Do not include selected evidence text, page numbers, expected outputs, or
  source metadata in the native-PDF model prompt.
- Keep a 10-product held-out set that is not used for prompt development.
- In 1.0.0, treat all 40 exposed pilot families as development data and keep
  160 newly selected families held-out.
- In 1.0.0, represent every ingredient group only once across all 200
  families.
- Group related products by active ingredient and reference-product family when dividing development and held-out cases.
- Do not tune prompts against held-out outputs.
- Record model identifier, deployment, parameters, prompt version, evaluator version, dataset version, and run timestamp.

## 10. Quality gates

The evidence-window diagnostic is ready to run only when:

- the dataset manifest uses schema `1.0` and passes `verify-dataset`
- all source files have hashes and retrieval metadata
- all DPD tables pass schema and join validation
- every case has a traceable source document and DPD record
- every scored expected item has a recorded source-evidence page
- English and French pairing has been verified where applicable
- no held-out product family appears in the development subset
- deterministic metrics pass unit tests on representative one-to-one and one-to-many cases

The historical 0.1.0 native-PDF benchmark requires all 390 field-level labels
to receive human approval. The 1.0.0 expansion requires all 2,162 fact checks
plus 600 product identity, scope, and bilingual checks. Both also require all
source PDFs to pass local verification, an English and French development case
to succeed through every intended provider route, token and cost reporting to
be checked, and the prompt to be locked before any held-out run. Until then,
registration and leaderboard publication remain blocked. Explicit local
experiments may run, but stay provisional and unranked.

## 11. Expansion status

The major corpus expansion is complete as draft version 1.0.0:

1. Preserve the 80-document 0.1.0 release and results as historical pilot
   evidence.
2. Human-audit the 400-document 1.0.0 release before any ranked use.
3. Use smoke and screen tiers before a full 400-case model run.
4. Run compatible models against identical frozen 1.0.0 cases and keep
   native-PDF and evidence-window rankings separate.
5. Add harder scanned documents only if OCR is evaluated as a separate factor.
6. Develop Recalls and Summary Reports as separate optional task packs rather
   than mixing unrelated task types into one score.

## 12. Explicit non-goals

The first benchmark does not:

- assess clinical advice quality
- infer drug safety or efficacy
- rank products
- use Canada Vigilance reports as causal truth
- score open-ended warnings, indications, or contraindications without separate labels
- collapse every HC task into a single generic model score

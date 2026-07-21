# Evalanche HC Benchmark 1: DPD and Product Monograph Extraction

**Specification status:** Ready for implementation after core stabilization  
**Benchmark type:** Bilingual structured extraction  
**Primary scoring:** Deterministic JSON and field-level metrics

## 1. Research question

How reliably can candidate language models extract auditable drug-product facts from Health Canada-authorized English and French product monographs?

The benchmark is intended to support a task-specific model-selection decision. It is not a general medical-knowledge test and does not evaluate whether a model can independently make regulatory or clinical judgments.

## 2. Why this benchmark comes first

This design has the strongest contribution-to-effort ratio among the identified HC sources:

- authentic Health Canada documents
- bilingual task coverage
- objective reference fields already available in DPD
- deterministic evaluation for most outputs
- limited need for expert manual annotation
- direct relevance to document extraction work performed across HC
- a clear path from a small internal pilot to a publishable public benchmark

## 3. Pilot composition

The first release contains **40 marketed human-drug products**.

- 20 single-ingredient products
- 10 multi-ingredient products
- 10 products with multiple strengths, dosage forms, routes, or DINs
- English and French monographs where both are available
- target size of 80 document-language cases before exclusions

The pilot should include varied oral, injectable, topical, inhaled, ophthalmic, and other dosage forms. Sampling is reproducible from a frozen DPD snapshot using a recorded random seed.

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

One case represents one product-monograph language instance.

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
expected_output
evaluation_type
source_metadata
```

`evaluation_type` is `json`. The input contains the extraction instruction and monograph text or an explicitly bounded monograph section. The expected output is canonical JSON.

## 5. Expected-output contract

```json
{
  "din": ["00000000"],
  "brand_name": "EXAMPLE",
  "active_ingredients": [
    {
      "name": "EXAMPLE INGREDIENT",
      "strength": 10,
      "unit": "mg"
    }
  ],
  "dosage_forms": ["TABLET"],
  "routes": ["ORAL"],
  "schedule": ["PRESCRIPTION"],
  "product_status": "MARKETED",
  "company": "EXAMPLE COMPANY"
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

## 7. Reference-label construction

1. Freeze DPD marketed and approved extracts to a dated, hashed snapshot.
2. Load each table with an explicit schema and `header=None`.
3. Validate column counts and join coverage.
4. Join relevant tables using `DRUG_CODE`.
5. Restrict the backbone to eligible marketed human drugs.
6. Identify the authorized monograph URLs through the DPD Online Query.
7. Download English and French documents separately and hash every file.
8. Match each document to its DPD product/DIN set.
9. Manually audit all 40 pilot products against the source document.
10. Exclude or adjudicate any unresolved source disagreement before case generation.

The audit records whether each field is:

- confirmed in both DPD and the monograph
- present in DPD but not found in the monograph
- present in the monograph but represented differently in DPD
- ambiguous
- excluded from scoring

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
- Keep a 10-product held-out set that is not used for prompt development.
- Group related products by active ingredient and reference-product family when dividing development and held-out cases.
- Do not tune prompts against held-out outputs.
- Record model identifier, deployment, parameters, prompt version, evaluator version, dataset version, and run timestamp.

## 10. Quality gates

The pilot is ready to run only when:

- all source files have hashes and retrieval metadata
- all DPD tables pass schema and join validation
- every case has a traceable source document and DPD record
- every expected output has been manually audited
- English and French pairing has been verified where applicable
- no held-out product family appears in the development subset
- deterministic metrics pass unit tests on representative one-to-one and one-to-many cases

## 11. Expansion plan

After the 40-product pilot:

1. Review whether scores meaningfully separate candidate models.
2. Quantify label disagreements and exclusion rates.
3. Add harder scanned documents only if OCR is evaluated as a separate factor.
4. Expand to at least 100 products with broader therapeutic and document coverage.
5. Publish a benchmark card describing collection, intended use, limitations, licensing, and update policy.
6. Develop Recalls and Summary Reports as separate task packs rather than mixing unrelated task types into one score.

## 12. Explicit non-goals

The first benchmark does not:

- assess clinical advice quality
- infer drug safety or efficacy
- rank products
- use Canada Vigilance reports as causal truth
- score open-ended warnings, indications, or contraindications without separate labels
- collapse every HC task into a single generic model score

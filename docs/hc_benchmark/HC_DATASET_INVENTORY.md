# Evalanche Health Canada Dataset Inventory

**Status:** Dataset discovery, DPD releases, four-model census evaluation, and Product Monograph pilot disposition are complete.
**Ranking boundary:** DPD 0.2.0 is ranked. Product Monograph pilots are permanently provisional and unranked.

## Decision

The first Health Canada benchmark will use the **Drug Product Database (DPD) plus Health Canada-authorized product monographs**.

- Product monographs provide the source documents given to candidate models.
- DPD provides objective reference values for deterministic scoring.
- The first task is bilingual structured extraction, not open-ended medical judgment.
- Recalls and Summary Reports remain separate later task packs.
- Canada Vigilance is explicitly excluded from causal safety ranking.

This order minimizes manual annotation, makes scoring reproducible, and keeps the first benchmark aligned with Evalanche's exact and JSON evaluation routes.

## Source inventory

| Priority | Source | Access and format | Benchmark role | Readiness decision |
| --- | --- | --- | --- | --- |
| Primary | Drug Product Database (DPD) | Nightly database, bulk compressed UTF-8 text extracts, and JSON/XML API | Sampling frame and reference labels for DIN, brand name, active ingredient, strength, dosage form, route, schedule, status, and company | Selected. The marketed and approved 2026-07-02 archives are frozen and validated as source release `2026.7.2`. |
| Primary | Health Canada-authorized product monographs | PDF documents accessed through the DPD Online Query, with English and French availability varying by product | Document input for bilingual structured extraction | Selected. Eligibility rules, pilot size, manifest fields, and matching procedure are defined. |
| Secondary | Recalls and Safety Alerts | Daily English/French CSV and JSON feeds, with records linkable by NID | Bilingual classification, structured extraction, and constrained summarization | Scoped as a separate later task pack. It will not be mixed into the DPD/monograph benchmark. |
| Secondary | Summary Reports API, including Summary Safety Reviews, Summary Basis of Decision, and Regulatory Decision Summaries | English/French JSON or XML API | Grounded summarization and extraction | Scoped as a later task pack after the primary benchmark. |
| Backlog | Canada Vigilance adverse reaction data | Full database extracts as compressed ASCII text files | Structured extraction or signal-triage experiments only | Access and limitations assessed. Not suitable as causal truth, incidence data, or a simple product-safety leaderboard. |
| Supporting | Notice of Compliance database | Nightly public database covering human-drug authorizations from 1994 onward | Authorization dates, NOC status, and supporting regulatory metadata | Supporting source only. Not required for the first benchmark. |

## Primary benchmark reference fields

The initial extraction target is limited to fields for which DPD can provide auditable reference values:

1. DIN
2. Brand name
3. Active ingredient name
4. Active ingredient strength and unit
5. Dosage form
6. Route of administration
7. Schedule
8. Product status
9. Company or DIN owner

Indications, contraindications, warnings, interactions, adverse reactions, and patient instructions are not treated as DPD-derived ground truth. Those fields require separate human annotation, carefully defined silver labels, or rubric-based evaluation.

## Known source limitations

### DPD and product monographs

- DPD is relational, so tables must be joined by `DRUG_CODE` and normalized before case generation.
- One monograph can cover multiple DINs, strengths, or formulations.
- Not every drug has a posted monograph.
- Older products may lack a French monograph, and monograph templates vary by age.
- DPD and monograph values can differ because of timing, scope, naming, or normalization rather than model error.
- The benchmark therefore freezes both sources to a dated snapshot and records automated source evidence and unresolved label limitations explicitly.
- The official read-me and published archive differ in some structural details, so source-specific validation records the observed file names and column counts and fails on an unreviewed change.

### Recalls and Safety Alerts

- The source is updated daily, so benchmark releases require immutable snapshots.
- English and French records must be paired and validated by NID.
- Product categories and record templates vary, so task-specific eligibility filters are required.

### Summary Reports

- Report templates vary over time.
- The API contains several document families that must remain distinct task types.
- Summarization cases need a defined evidence scope and human validation before release.

### Canada Vigilance

- Reports describe suspected associations, not confirmed causality.
- Under-reporting, reporting bias, incomplete reports, and possible duplicates are known limitations.
- Report counts cannot be used to estimate incidence or directly compare product safety.

## Reproducibility requirements

Every materialized source snapshot will have a manifest containing:

- source identifier
- source URL
- retrieval timestamp in UTC
- source-reported update date, where available
- local relative path
- byte size
- SHA-256 hash
- language
- record count
- parser/schema version
- validation status and error count

The versioned contract, working example, and verification command are
documented in
[`docs/dataset_manifests.md`](../dataset_manifests.md). The verifier rejects
unknown fields, unsafe paths, inconsistent sampling or split records, missing
files, and mismatched sizes, hashes, or supported record counts.

Benchmark releases are versioned and immutable. Updating a source creates a new benchmark version rather than silently changing existing cases.

The first source release is documented in
[`HC_DPD_SOURCE_SNAPSHOT.md`](HC_DPD_SOURCE_SNAPSHOT.md). Its schema `1.0`
manifest verifies two unmodified official archives and the generated validation
report. The release contains 24 relational tables across marketed and approved
cohorts, with 210,632 table rows in total. This total is not a unique-product
count.

The first executable benchmark release is documented in
[`HC_DPD_BENCHMARK_SLICE.md`](HC_DPD_BENCHMARK_SLICE.md). Version `0.1.0`
contains 40 sampled product families and 80 paired English and French DPD
record-extraction cases. It records exact source rows, complexity strata, and
30-product development and 10-product held-out assignments. No selected active
ingredient code crosses those splits.

The DPD-only slice proves the normalization, sampling, traceability, and case
generation path. It does not replace Product Monographs as the intended source
documents and is not evidence of monograph-extraction performance.

The full eligible-population DPD release is documented in
[`HC_DPD_CENSUS_DEMO.md`](HC_DPD_CENSUS_DEMO.md). Version `0.2.0` contains
7,017 product families and 14,034 paired English and French cases. It includes
all four supported complexity strata, including 169 multi-ingredient,
multi-variant families. Thirteen source-ambiguous families with conflicting
French brand names remain explicitly excluded pending manual adjudication.

The census also contains a fixed 24-case live-demo view. It demonstrates the
complete generation, deterministic evaluation, operational measurement, and
reporting workflow without triggering a full 14,034-call run per model. Like
the slice, it uses DPD renderings rather than Product Monograph documents.

## Available artifacts

- Frozen DPD source release `2026.7.2` with a verified schema `1.0` manifest.
- DPD benchmark slice `0.1.0` with deterministic sampling, bilingual cases,
  source-row traceability, and fixed splits.
- Full-population DPD census `0.2.0` with 14,034 cases and a bounded 24-case
  demonstration view.
- Compatible census results for GPT-5.4 Mini and GPT-5.6 Luna, Terra, and Sol,
  with independent source parsing and grouped product-family analysis.
- Frozen Product Monograph evidence-window pilot `0.1.0`, preserved as
  permanently provisional descriptive evidence.
- Retired Product Monograph native-PDF pilot `0.1.0`, preserved without
  published model results.
- The unfinished Product Monograph 1.0.0 expansion and proposed audit program
  are intentionally absent from the supported resource.

There is no remaining Product Monograph audit, promotion, or expansion stream.
The DPD census is the committed ranked Health Canada case study.

## Official references

- [DPD data extracts](https://www.canada.ca/en/health-canada/services/drugs-health-products/drug-products/drug-product-database/what-data-extract-drug-product-database.html)
- [DPD API guide](https://health-products.canada.ca/api/documentation/dpd-documentation-en.html)
- [Product monograph access and limitations](https://www.canada.ca/en/health-canada/services/drugs-health-products/drug-products/applications-submissions/guidance-documents/product-monograph/frequently-asked-questions-product-monographs-posted-health-canada-website.html)
- [Recalls and Safety Alerts](https://recalls-rappels.canada.ca/en)
- [Summary Reports API guide](https://health-products.canada.ca/api/documentation/summary-report-documentation-en.html)
- [Canada Vigilance adverse reaction database](https://www.canada.ca/en/health-canada/services/drugs-health-products/medeffect-canada/adverse-reaction-database.html)
- [Interpretation of suspected adverse reaction data](https://www.canada.ca/en/health-canada/services/drugs-health-products/medeffect-canada/adverse-reaction-database/interpretation-suspected-adverse-reaction-data.html)
- [Health Canada drug and medical device databases](https://www.canada.ca/en/health-canada/services/drugs-health-products/drug-products/applications-submissions/guidance-documents/drug-and-medical-device-databases.html)

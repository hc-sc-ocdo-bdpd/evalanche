# Evalanche Health Canada Dataset Inventory

**Status:** Dataset discovery and benchmark scoping complete
**Implementation gate:** Stabilize Evalanche's case schema, combined evaluator, and result contract before materializing the HC benchmark datasets.

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
| Primary | Drug Product Database (DPD) | Nightly database, bulk compressed UTF-8 text extracts, and JSON/XML API | Sampling frame and reference labels for DIN, brand name, active ingredient, strength, dosage form, route, schedule, status, and company | Selected. Source, fields, joins, normalization approach, and evaluation use are defined. |
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
- The benchmark therefore freezes both sources to a dated snapshot and manually audits every pilot reference record.

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

Benchmark releases are versioned and immutable. Updating a source creates a new benchmark version rather than silently changing existing cases.

## Sequencing

1. Stabilize the generic Evalanche case schema and combined evaluation result contract.
2. Normalize a frozen DPD snapshot into validated tables.
3. Build the eligible human-marketed product backbone.
4. Select the pilot sample and acquire matched English/French monographs.
5. Audit document-to-DPD alignment and construct deterministic JSON cases.
6. Run the pilot across candidate models and review error patterns.
7. Expand only after the pilot demonstrates reliable labels and useful model separation.

The remaining activities are implementation of this defined plan, not unresolved dataset discovery or an unowned preparation stream.

## Official references

- [DPD data extracts](https://www.canada.ca/en/health-canada/services/drugs-health-products/drug-products/drug-product-database/what-data-extract-drug-product-database.html)
- [DPD API guide](https://health-products.canada.ca/api/documentation/dpd-documentation-en.html)
- [Product monograph access and limitations](https://www.canada.ca/en/health-canada/services/drugs-health-products/drug-products/applications-submissions/guidance-documents/product-monograph/frequently-asked-questions-product-monographs-posted-health-canada-website.html)
- [Recalls and Safety Alerts](https://recalls-rappels.canada.ca/en)
- [Summary Reports API guide](https://health-products.canada.ca/api/documentation/summary-report-documentation-en.html)
- [Canada Vigilance adverse reaction database](https://www.canada.ca/en/health-canada/services/drugs-health-products/medeffect-canada/adverse-reaction-database.html)
- [Interpretation of suspected adverse reaction data](https://www.canada.ca/en/health-canada/services/drugs-health-products/medeffect-canada/adverse-reaction-database/interpretation-suspected-adverse-reaction-data.html)
- [Health Canada drug and medical device databases](https://www.canada.ca/en/health-canada/services/drugs-health-products/drug-products/applications-submissions/guidance-documents/drug-and-medical-device-databases.html)

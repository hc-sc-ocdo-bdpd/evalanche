# Health Canada Product Monograph benchmark expansion 1.0.0

## Status and decision

- Dataset families: `hc_product_monograph_structured_extraction@1.0.0` and
  `hc_product_monograph_native_pdf_extraction@1.0.0`
- Status: draft, human audit required, registered ranking blocked
- Product families: 200
- Official PDFs and cases: 400, 200 English and 200 French
- Provisional scored facts: 2,162
- Required human approvals: 2,762
- Model calls made while building the release: 0

This is one material expansion rather than a series of small pilot versions.
It increases the cohort from 40 to 200 product families, keeps the useful
evidence-window and complete-PDF questions separate, and adds source, selection,
label, and publication controls needed for a defensible future benchmark.

The release is not gold-labeled yet. It is suitable for source inspection,
audit work, local engineering, and cost preflight. It must not produce a
ranked or published model comparison until the human gates pass.

## What changed from 0.1.0

| Property | 0.1.0 | 1.0.0 draft |
| --- | ---: | ---: |
| Product families | 40 | 200 |
| English cases | 40 | 200 |
| French cases | 40 | 200 |
| Total cases | 80 | 400 |
| Development families | 30 | 40 |
| Held-out families | 10 | 160 |
| Exact ingredient groups | 40 | 200 |
| Provisional fact rows | 390 | 2,162 |
| Product identity and scope audit | No | Yes, 600 checks |
| Full fact audit | 390 facts | 2,162 facts |
| Full source screening log | Pilot exclusions | 535 retained and excluded records |

All 40 pilot families are development data in 1.0.0 because prior model results
exposed every pilot case. None remains held-out. The 160 held-out families are
new, were selected without model outputs, and share no ingredient group with
development or with one another.

## Cohort contract

The builder starts from the frozen DPD 0.2.0 census of 7,017 marketed
human-drug families. It applies exact quotas:

| Stratum | Families |
| --- | ---: |
| Single ingredient | 80 |
| Multiple ingredients | 50 |
| Multiple variants | 50 |
| Multiple ingredients and multiple variants | 20 |
| Total | 200 |

The selected route distribution is:

| Route group | Families |
| --- | ---: |
| Oral | 91 |
| Parenteral | 39 |
| Ophthalmic or otic | 20 |
| Topical | 18 |
| Other | 16 |
| Inhaled or nasal | 16 |

The route mix is created by a declared deterministic coverage cycle within
each stratum, not by looking at model performance. It is intentionally
purposive, not a probability sample of all marketed products.

Every retained family must pass all automated screening conditions:

1. Every DPD drug-code variant resolves to one Product Monograph per language.
2. English and French resolve to distinct official PDFs.
3. The files begin as PDFs, contain extractable text, and pass a source-language
   marker check.
4. The English and French file hashes are distinct.
5. No PDF hash appears anywhere else in the selected cohort.
6. No ingredient group appears in more than one selected family.
7. Every provisional brand, ingredient and strength, dosage form, and route
   has a cited source page.
8. Exact stratum quotas and the 40/160 split remain satisfied.

The final screening log contains 535 records: 200 retained and 335 excluded.
Exclusions are retained with stage and reason, including missing documents,
fact-alignment failures, duplicate ingredient groups, language or PDF defects,
and inconsistent variant links.

## Source and label locks

`source_documents.csv` records the official URL, DPD information URL,
monograph ID, retrieval time, SHA-256, byte size, page count, extraction status,
and automated language status for all 400 PDFs.

All 400 SHA-256 values are unique. The locked files total approximately 370 MB
and 17,733 pages. PDFs are downloaded locally for verification but are not
committed because individual sponsor copyright may apply.

The evidence-window release contains 400 cases with 2,162 source-linked facts:

| Automated support method | Facts |
| --- | ---: |
| Normalized exact | 1,245 |
| Ingredient name and strength | 859 |
| Ingredient name or strength only | 51 |
| Declared alias | 7 |

The 51 name-or-strength rows and 7 alias rows are priority items for human
scrutiny, but every row must be reviewed. Each case contains at most four
label-selected pages.

Exact duplicate expected values are removed while preserving source order.
Different strengths, units, ingredients, forms, and routes remain separate.

## Two benchmark questions

The two releases share product membership, provisional expected JSON, source
hashes, audit queues, and deterministic scoring. They differ only in what the
model receives.

| Benchmark | Model input | Primary question |
| --- | --- | --- |
| Evidence window | Label-selected extracted pages | Can the model extract and serialize the supported facts? |
| Native PDF | Complete official PDF through file input | Can the model locate, interpret, extract, and serialize the facts from the full document? |

Do not merge their scores into one leaderboard. The paired difference is a
diagnostic for retrieval and document handling versus extraction and schema
fidelity.

## Build and verification workflow

Build or deterministically rebuild the full expansion:

```bash
python -m evalanche.cli expand-product-monograph-benchmark
```

The builder uses a local content cache, retries public-source requests, refuses
changed inherited hashes, preserves unchanged review rows, resets changed rows
to pending, writes both release families, and verifies their manifests.

Acquire or verify the 400 selected PDFs from the finished source lock:

```bash
python -m evalanche.cli acquire-product-monographs \
  --sources data/hc/benchmarks/product_monograph_structured_extraction/1.0.0/source_documents.csv \
  --raw-dir data/hc/product_monographs/1.0.0/raw

python -m evalanche.cli verify-product-monograph-sources \
  --sources data/hc/benchmarks/product_monograph_structured_extraction/1.0.0/source_documents.csv \
  --raw-dir data/hc/product_monographs/1.0.0/raw
```

After editing only the permitted human-review columns, refresh derived reports
and manifest hashes without network or model calls:

```bash
python -m evalanche.cli refresh-product-monograph-expansion
python -m evalanche.cli check-product-monograph-expansion-audit
```

The release-local
[`HUMAN_AUDIT_GUIDE.md`](../../data/hc/benchmarks/product_monograph_structured_extraction/1.0.0/HUMAN_AUDIT_GUIDE.md)
defines every product, fact, correction, adjudication, and publication check.

## Human promotion gate

Promotion requires 2,762 approvals:

- 2,162 fact approvals in `fact_audit.csv`;
- 200 product identity approvals;
- 200 product-scope approvals;
- 200 bilingual-equivalence approvals.

The checker also verifies that immutable audit columns still match cases,
evidence, and source locks, reviewer timestamps are UTC, correction metadata is
complete, and both dataset manifests match the current audit files.

Corrections belong in the upstream override or selection logic, followed by a
rebuild and review of reset rows. Never repair a reference label only inside an
audit CSV.

Even after all checks pass, promotion from `draft` must be an explicit release
decision with peer review. The offline refresh intentionally does not change
benchmark status.

## Cost-controlled model workflow after audit

Both manifests define cumulative group-preserving tiers:

| Tier | Product pairs | Cases | Purpose |
| --- | ---: | ---: | --- |
| Smoke | 2 | 4 | Transport, parsing, token, and cost validation |
| Screen | 25 | 50 | Broad model screening |
| Standard | 200 | 400 | Complete comparison for promoted models |

Earlier tier outputs are reused. A model promoted through all tiers makes at
most 400 calls, not 454. Native-PDF preflight uses a deliberately large token
assumption because long PDFs are expensive and provider accounting can vary.

Inspect a plan without model calls:

```bash
python -m evalanche.cli run-benchmark \
  --benchmark hc_product_monograph_native_pdf_extraction@1.0.0 \
  --tier smoke \
  --all-compatible \
  --access-set configs/access_sets/my_available_models.yaml \
  --preflight-only
```

Normal registered execution remains blocked while the benchmark is draft.

## Interpretation limits

This benchmark supports claims only about the exact source lock, sample,
prompt, input contract, scorer, model route, and run configuration.

It does not establish clinical correctness, general medical reasoning,
representativeness of all marketed drugs, private Health Canada document
performance, current product status after the retrieval date, or universal
model quality. Bilingual availability, text extraction, and automated evidence
alignment are inclusion criteria and therefore sources of selection bias.

No 1.0.0 model results exist at release creation. The older 0.1.0 results are
historical pilot evidence and must not be presented as scores on this expanded
cohort.

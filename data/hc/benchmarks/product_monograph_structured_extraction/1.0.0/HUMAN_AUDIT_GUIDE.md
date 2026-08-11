# Product Monograph 1.0.0 human audit guide

## Release boundary

This 200-family, 400-document release is a draft. Its labels are provisional
and its benchmark results must remain unranked until every required check is
approved by a qualified human reviewer who did not create the automated
labels.

The audit has two complementary parts:

| Queue | Rows | Checks per row | Required approvals |
| --- | ---: | ---: | ---: |
| `fact_audit.csv` | 2,162 | 1 | 2,162 |
| `product_audit.csv` | 200 | 3 | 600 |
| Total | 2,362 rows | | 2,762 |

The 2,162 fact rows are facts, not documents. A row represents one brand
name, active ingredient with strength and unit, dosage form, or route in one
source language.

## Prepare the exact source files

Acquire the PDFs recorded in the source lock:

```bash
python -m evalanche.cli acquire-product-monographs \
  --sources data/hc/benchmarks/product_monograph_structured_extraction/1.0.0/source_documents.csv \
  --raw-dir data/hc/product_monographs/1.0.0/raw
```

Then verify every hash, byte size, page count, language pair, and unique file:

```bash
python -m evalanche.cli verify-product-monograph-sources \
  --sources data/hc/benchmarks/product_monograph_structured_extraction/1.0.0/source_documents.csv \
  --raw-dir data/hc/product_monographs/1.0.0/raw
```

Do not approve against a current web copy unless its SHA-256 is identical to
the source lock. The local hash-locked PDF is the source of truth. The excerpt
in the queue is a navigation aid, not a substitute for the full page and its
document context.

## Audit every product family first

Open `product_audit.csv`. For each row, open both PDFs and review the DPD drug
codes, DIN list, expected English JSON, and expected French JSON. Edit only:

```text
identity_review_status
scope_review_status
bilingual_review_status
reviewer
reviewed_at_utc
notes
```

Approve the three checks only when all of the following are true:

1. `identity_review_status`: each PDF is the intended Product Monograph for
   the named brand and sponsor, not a similarly named product, prescribing
   information for another market, or the wrong language.
2. `scope_review_status`: the monograph covers the exact DPD family used by
   the expected JSON. Confirm every included ingredient, strength, dosage
   form, and route. Look specifically for missing variants and for extra
   sibling products that the DPD family did not include.
3. `bilingual_review_status`: the English and French PDFs describe equivalent
   product scope. Their wording need not be literal translations, but neither
   side may omit or add a scored variant.

Complete all three statuses together. Use `needs_correction` when the intended
family can be repaired, and `rejected` when the source pair should not be in
the cohort. Give a precise note for either status. Every reviewed row requires
a reviewer identifier and an ISO 8601 UTC timestamp such as
`2026-08-11T18:30:00Z`.

## Audit every scored fact

Open `fact_audit.csv`. For each row, open the PDF identified by
`monograph_id` and `monograph_sha256`, go to `source_page`, and verify the fact
in the full page context. Edit only:

```text
human_review_status
reviewer
reviewed_at_utc
corrected_value
notes
```

Apply these field-specific rules:

1. Brand name: confirm the source-language brand is the product's marketed
   name, not only an ingredient name, heading from a referenced product, or a
   sponsor name.
2. Active ingredient: confirm the ingredient identity, strength, and unit as
   one fact. Check salt, hydrate, ester, base-equivalent, concentration, dose,
   and per-container semantics. Do not infer an unstated conversion.
3. Dosage form: confirm the source-language form and every distinct form in
   scope. Do not substitute packaging, route, release mechanism, or device
   wording unless the monograph treats it as the dosage form.
4. Route: confirm each route is authorized for the product family. Do not
   infer a route solely from a dosage form or administration instruction when
   the document's product scope says otherwise.

Rows whose `automated_support_method` is `normalized_name_or_strength` need
special scrutiny because automation found only the ingredient name or the
strength on the cited page. Rows marked `declared_alias` also need close review
because an explicit normalization alias connected the expected value to the
page wording. All rows still require review, including `normalized_exact`.

Use `approved` only when the exact expected item is supported. Use
`needs_correction` when the source is right but the expected value is wrong,
put the proposed replacement in `corrected_value`, and explain the issue in
`notes`. Use `rejected` when the cited evidence, source identity, or product
scope cannot support a repair. Every reviewed row requires the reviewer and
UTC timestamp.

## Corrections and adjudication

Do not edit IDs, expected values, source pages, excerpts, URLs, hashes, or
other immutable columns in either queue.

For a correction:

1. Record `needs_correction`, the proposed value, and the reason.
2. Have a second qualified person adjudicate every correction, rejection, or
   genuinely ambiguous source. Record the decision and both identities in the
   notes or the repository review record.
3. Apply the accepted correction in
   `configs/product_monograph/1.0.0/label_overrides.yaml`, or change the cohort
   logic if the product pair is invalid.
4. Rebuild the expansion. Any changed or new audit row resets to `pending`.
5. Re-audit the changed rows against the rebuilt source lock.

An approved queue must not contain unresolved `needs_correction` or `rejected`
rows.

## Refresh and check after each review batch

The offline refresh validates immutable columns, recomputes audit summaries
and build reports, and updates both dataset manifest hashes. It makes no
network or model calls:

```bash
python -m evalanche.cli refresh-product-monograph-expansion
python -m evalanche.cli check-product-monograph-expansion-audit
```

At final sign-off, require completeness:

```bash
python -m evalanche.cli check-product-monograph-expansion-audit --require-complete
```

Promotion is allowed only when the checker reports 2,762 of 2,762 approvals,
no metadata or immutable-column issues, and valid hashes for both the
evidence-window and native-PDF datasets. Promotion must still be an explicit
release decision. The refresh command does not change the draft status.

## Final publication review

Before any ranked run or public claim, a human release owner must also verify:

- the development prompt was locked before any held-out output was inspected;
- the 40 inherited families are treated only as development because earlier
  model results exposed them;
- all 160 held-out families are new and all 200 ingredient groups are unique;
- evidence-window and native-PDF results remain separate rankings;
- source restrictions and sponsor copyright are respected, with PDFs kept out
  of the repository;
- every correction and adjudication is traceable;
- dataset and benchmark manifests still verify after the final review commit;
- limitations, selection bias, retrieval date, model route, and exact scorer
  are stated with any result.

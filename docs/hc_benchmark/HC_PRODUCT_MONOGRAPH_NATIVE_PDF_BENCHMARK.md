# Health Canada Product Monograph native-PDF benchmark 0.1.0

> **Expanded draft available:** Version 1.0.0 contains 200 product families,
> 400 complete PDFs, unique ingredient groups, and fact plus product-scope
> audit gates. See
> [`HC_PRODUCT_MONOGRAPH_EXPANSION_1_0_0.md`](HC_PRODUCT_MONOGRAPH_EXPANSION_1_0_0.md).

- **Status:** Draft, local experiments allowed, publication blocked
- **Dataset:** `hc_product_monograph_native_pdf_extraction@0.1.0`
- **Cases:** 80 complete official PDFs, 40 English and 40 French
- **Scoring:** Deterministic canonical JSON
- **Human label review:** 0 of 390 items approved at release creation

## Decision

Keep the evidence-window benchmark and add this as a separate benchmark.

The evidence-window release is useful because it isolates extraction and JSON
fidelity after the relevant pages are supplied. It cannot measure whether a
model can find those pages, interpret document layout, or use visual content.
This release asks that broader and more valuable question by sending the
complete PDF through a native file-input API.

Scores from the two releases must not be placed in one ranking. Their paired
difference is useful diagnostic evidence:

| Result pattern | Likely interpretation |
| --- | --- |
| Strong on both | Good retrieval and good extraction |
| Strong window, weak PDF | Retrieval, layout, context, or PDF transport is the bottleneck |
| Weak on both | Extraction, schema following, or label alignment is the bottleneck |
| Weak window, strong PDF | Investigate prompt sensitivity, page-selection effects, or run validity |

## Input contract

Each case stores a repository-relative PDF descriptor containing:

- the expected local path;
- the official filename;
- `application/pdf` media type;
- the frozen SHA-256.

At request time Evalanche:

1. refuses absolute paths and paths that escape the repository;
2. verifies the file exists and matches the frozen SHA-256;
3. enforces the 50 MB per-file and combined request limits;
4. Base64-encodes the complete PDF;
5. sends it as an `input_file` through the Responses API;
6. records Responses token use, cost, latency, retries, and failures.

The benchmark requires model capabilities `pdf_input`, `responses_api`, and
`vision`. Its default concurrency is one because complete PDFs can use much
more context and memory than evidence windows.

The native request intentionally omits the optional PDF `detail` field. This
matches the documented Azure Base64 PDF request shape and lets the provider use
its default. The generic file-input layer accepts an explicit `detail` value
for providers that document support for it.

## Label provenance and draft boundary

The provisional expected JSON is inherited unchanged from the frozen
evidence-window release. Those labels originate in DPD alignment, include
targeted overrides, and have automated Product Monograph page-evidence checks.
That is meaningful evidence, but it is not independent human labeling.

The existing builder also used the expected labels to find the evidence pages.
That is appropriate for an oracle-page diagnostic. It would be leakage if the
same selected pages were presented as an end-to-end retrieval test. This
native-PDF release does not send those selected pages or page numbers to the
model.

The review queue is:

```text
data/hc/benchmarks/product_monograph_native_pdf_extraction/0.1.0/label_review.csv
```

It contains one row for each of 390 scored field items, including the exact PDF
hash, source page, automated support method, and human review fields. Allowed
review states are `pending`, `approved`, `rejected`, and `needs_correction`.
The builder preserves completed review columns when it refreshes artifacts.

The queue is 390 facts across 80 documents, not 390 PDFs. A fact might be one
active ingredient, one strength, one dosage form, or one route that appears in
the expected JSON.

### Do you need to complete this now?

No. Label review is required only if this optional HC task pack is going to be
promoted and published as a ranked benchmark. It is not required to keep the
current local runs as useful experimental evidence, test another model, or
continue building Evalanche's general model-selection product.

Until review is complete, describe results as local, provisional, and
unranked. Do not copy them into a public benchmark leaderboard.

### How to review the labels

Only these four CSV columns may be edited:

```text
human_review_status
reviewer
reviewed_at_utc
notes
```

For each row:

1. Open the exact source PDF identified by `monograph_id`, `monograph_sha256`,
   and `source_url`.
2. Go to `source_page` and confirm that `expected_item` is supported for the
   named `field`, product, and language.
3. Set `human_review_status` to `approved` only when the item is correct.
4. Use `needs_correction` when the intended source is right but the expected
   item needs a data correction. Use `rejected` when the cited evidence or
   document alignment does not support the item.
5. For every nonpending row, enter a reviewer identifier and an ISO 8601 UTC
   time such as `2026-08-05T18:30:00Z`.
6. Add a specific note for every `needs_correction` or `rejected` row. Notes
   are optional for an approval.

Do not edit `expected_item`, `source_page`, hashes, IDs, or source metadata in
the review CSV. The checker treats those as immutable. A correction belongs in
the upstream label construction or override, followed by a new dataset
version.

Check progress at any time without model calls:

```bash
docker compose run --rm evalanche python -m evalanche.cli check-product-monograph-label-review
```

When every item is expected to be approved, enforce completeness:

```bash
docker compose run --rm evalanche python -m evalanche.cli check-product-monograph-label-review --require-complete
```

Rebuild and verify the dataset after review edits so the release report and
manifest hashes are refreshed before promotion.

## Docker workflow

Build the container and download the exact source-locked PDFs:

```bash
docker compose build
docker compose run --rm evalanche python -m evalanche.cli acquire-product-monographs
docker compose run --rm evalanche python -m evalanche.cli verify-product-monograph-sources
```

Build and verify this draft dataset:

```bash
docker compose run --rm evalanche python -m evalanche.cli build-product-monograph-native-pdf-benchmark
docker compose run --rm evalanche python -m evalanche.cli verify-dataset --manifest configs/datasets/hc_product_monograph_native_pdf_extraction_0.1.0_manifest.yaml --root .
```

The benchmark defines three cumulative, deterministic tiers:

| Tier | Cumulative PDFs | New calls after its parent | Purpose |
| --- | ---: | ---: | --- |
| smoke | 2, one bilingual pair | 2 | Verify PDF transport, parsing, token use, price evidence, and resume |
| screen | 20, ten bilingual pairs | 18 | Eliminate models that should not consume the full pilot budget |
| standard | 80, forty bilingual pairs | 60 | Complete the current pilot for promoted models |

The selector keeps English and French product pairs together and balances the
declared split, extraction stratum, and document-complexity columns. Membership
is deterministic from the benchmark manifest. A model that advances through
all tiers uses exactly 80 calls.

Before any model has observations, smoke uses an explicit conservative
assumption of 150,000 input tokens and 900 completion tokens per PDF, then a
2.0 safety multiplier. Screen and standard reprice their pending calls from
the selected model's observed earlier-tier token use, with the manifest token
assumption retained only as a fallback. The output is a budget estimate and a
local request-start gate, not a reconciled Azure invoice limit.

Inspect every compatible smoke run in your confirmed access set and its
aggregate cost without provider calls:

```bash
docker compose run --rm evalanche python -m evalanche.cli run-benchmark --benchmark hc_product_monograph_native_pdf_extraction@0.1.0 --tier smoke --all-compatible --access-set configs/access_sets/my_available_models.yaml --preflight-only
```

Set a cap above the printed projection, then execute. This remains a local,
unregistered experiment:

```bash
docker compose run --rm evalanche python -m evalanche.cli run-benchmark --benchmark hc_product_monograph_native_pdf_extraction@0.1.0 --tier smoke --all-compatible --access-set configs/access_sets/my_available_models.yaml --experiment --max-cost-usd <budget>
```

Then preflight and execute the cumulative screen tier. It reuses smoke as its
cost sample and calls only the 18 new PDFs per model:

```bash
docker compose run --rm evalanche python -m evalanche.cli run-benchmark --benchmark hc_product_monograph_native_pdf_extraction@0.1.0 --tier screen --all-compatible --access-set configs/access_sets/my_available_models.yaml --preflight-only
```

After screen, preflight every model that passes the manifest's generic
promotion gates, currently at least 50 percent strict pass and no more than 5
percent generation failures:

```bash
docker compose run --rm evalanche python -m evalanche.cli run-benchmark --benchmark hc_product_monograph_native_pdf_extraction@0.1.0 --tier standard --promote-from screen --access-set configs/access_sets/my_available_models.yaml --preflight-only
```

Repeat the screen or standard command with
`--experiment --max-cost-usd <budget>` to execute it. Rerunning the same tier
resumes its checkpoint ledger and does not repeat completed case and model
pairs.

To combine every completed compatible standard evaluation, including the four
complete 80-PDF runs generated before tiers existed, run:

```bash
docker compose run --rm evalanche python -m evalanche.cli summarize-benchmark --benchmark hc_product_monograph_native_pdf_extraction@0.1.0 --tier standard --all-results
```

The summary discovers model IDs from compatible run plans. It includes a new
model automatically after that model completes the benchmark. Its terminal and
file outputs include quality, cost, cost per request, cost coverage, tokens,
latency, language and split slices, field accuracy, case outcomes, and every
model pair. It makes no model calls.

Earlier complete 80-case plans are treated as legacy standard completions.
They are subset safely for smoke or screen summaries and are never rerun only
to fit the newer tier layout.

Normal execution without `--experiment` remains rejected while the benchmark
is draft. This prevents provisional labels from creating a
publishable-looking leaderboard.

## Promotion gates

Promote a new release to `ready` only after all of these are recorded:

- all 390 label-review items are approved, with corrections released as a new
  dataset version where necessary;
- all 80 local PDFs pass hash, byte-size, page-count, uniqueness, and language
  checks;
- one development English case and one development French case succeed through
  every intended provider route;
- returned token and cost fields are present or explicitly documented as
  unknown;
- the development prompt is locked before any held-out run;
- the benchmark and dataset manifests are rehashed and verified;
- failed or partial infrastructure runs remain visible but unranked.

After promotion, run any selected model or all compatible models and let the
registered workflow score, bundle, and rebuild the leaderboard:

```bash
docker compose run --rm evalanche python -m evalanche.cli run-benchmark --benchmark <promoted_benchmark_id>@<version> --all-compatible --access-set configs/access_sets/my_available_models.yaml
```

## Comparison policy

Use a separate leaderboard for each input contract. Native provider PDF
processing can differ in extraction, rendering, tokenization, and supported
limits. A model that cannot accept the declared native-PDF contract is
incompatible with this benchmark, not a zero-scoring participant.

If broader provider coverage is later required, add separately named
benchmarks for full extracted text or explicitly rendered page images. Do not
silently substitute either representation inside this benchmark because that
would change what is being measured.

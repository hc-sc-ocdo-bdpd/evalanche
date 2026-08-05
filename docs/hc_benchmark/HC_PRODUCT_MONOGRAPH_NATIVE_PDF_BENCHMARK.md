# Health Canada Product Monograph native-PDF benchmark 0.1.0

- **Status:** Draft, execution blocked by the registry
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

Validate model compatibility and inspect the 80-call plan without provider
calls:

```bash
docker compose run --rm evalanche python -m evalanche.cli run-benchmark --benchmark hc_product_monograph_native_pdf_extraction@0.1.0 --model gpt_5_6_sol --plan-only
```

An execution attempt without `--plan-only` is rejected while the benchmark is
draft. This prevents provisional labels or an untested provider route from
creating a publishable-looking leaderboard.

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

After promotion, run one model at a time and let the registered workflow score,
bundle, and rebuild the leaderboard:

```bash
docker compose run --rm evalanche python -m evalanche.cli run-benchmark --benchmark hc_product_monograph_native_pdf_extraction@0.1.0 --model gpt_5_6_sol
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

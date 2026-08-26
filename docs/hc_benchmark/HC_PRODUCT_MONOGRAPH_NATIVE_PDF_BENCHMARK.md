# Health Canada Product Monograph native-PDF pilot 0.1.0

- **Status:** Retired, local-only, permanently provisional
- **Dataset:** `hc_product_monograph_native_pdf_extraction@0.1.0`
- **Cases:** 80 complete official PDFs, 40 English and 40 French
- **Scoring:** Deterministic canonical JSON
- **Ranking:** Disabled
- **Human audit:** Not planned

## Disposition

This pilot is retained for reproducibility and architecture evidence. It is not
an active benchmark-development program and must not produce an official model
ranking. The unfinished 1.0.0 expansion and proposed audit campaign were
retired from Evalanche on 2026-08-17.

The evidence-window pilot isolates extraction after the relevant text is
supplied. This native-PDF pilot asks a different question by sending the full
document. Its historical results can help diagnose retrieval, layout, context,
or transport failures, but only as provisional local evidence.

## Input contract

Each case stores a repository-relative PDF descriptor with the expected local
path, official filename, `application/pdf` media type, and frozen SHA-256.
When the legacy local workflow is reproduced, Evalanche:

1. rejects absolute paths and paths that escape the repository;
2. verifies the file and frozen SHA-256;
3. enforces file and combined-request size limits;
4. encodes the complete PDF;
5. sends it as an `input_file` through the Responses API;
6. records token use, cost, latency, retries, and failures.

The input contract requires `pdf_input`, `responses_api`, and `vision` model
capabilities. Native PDF rendering and tokenization may differ by provider, so
results apply only to the declared route and contract.

## Label provenance

The expected JSON comes from the frozen evidence-window pilot. Those labels
originate in DPD alignment, targeted overrides, and automated monograph
page-evidence checks. They do not have independent human sign-off.

The file below is part of the original reproducibility snapshot:

```text
data/hc/benchmarks/product_monograph_native_pdf_extraction/0.1.0/label_review.csv
```

Its 390 rows are scored facts across 80 documents. The human-review columns
are preserved to keep the original builder output stable. The file is inert:
there is no expected review action, completion gate, promotion, or publication
plan.

## Offline integrity workflow

The retired dataset can still be reconstructed and verified without making
model calls:

```bash
docker compose build
docker compose run --rm evalanche python -m evalanche.cli acquire-product-monographs
docker compose run --rm evalanche python -m evalanche.cli verify-product-monograph-sources
docker compose run --rm evalanche python -m evalanche.cli build-product-monograph-native-pdf-benchmark
docker compose run --rm evalanche python -m evalanche.cli verify-dataset --manifest configs/datasets/hc_product_monograph_native_pdf_extraction_0.1.0_manifest.yaml --root .
```

Registered execution remains blocked because the benchmark lifecycle is
`retired`. Existing local outputs may be summarized for historical analysis,
but must not be copied into a public leaderboard or described as validated
model rankings.

## Comparison policy

Keep a separate result surface for each input contract. Evidence windows,
native provider PDFs, extracted text, and rendered page images measure
different systems. A model that cannot accept a contract is incompatible, not
a zero-scoring participant.

A future validated native-PDF benchmark would require a new dataset version,
independently validated labels, an explicit input contract, reproducible
sources, and a new ranking decision. It is outside the current resource scope
and is not implied by the retained pilot.

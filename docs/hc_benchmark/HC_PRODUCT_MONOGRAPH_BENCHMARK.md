# Health Canada Product Monograph benchmark

Evalanche contains two frozen 0.1.0 input conditions built from the same 40
marketed product families and 80 official English and French Product
Monographs. They share the same four-field structured reference contract and
are reported descriptively.

## What is scored

Each case returns JSON containing:

- `brand_name`;
- `active_ingredients`, including strength and unit;
- `dosage_forms`;
- `routes`.

DIN, regulatory schedule, product status, and sponsor are retained as alignment
metadata where available but are not scored.

Strict pass requires the complete record to match after the benchmark's
versioned canonicalization. Field score is retained as a diagnostic for partial
matches.

## Reference evidence

The cohort contains 390 scored facts. Each fact is tied to exact official
Product Monograph source-page evidence through deterministic or declared-alias
checks. Source descriptors preserve the official URL, SHA-256 hash, byte size,
page count, language, and monograph identifier.

This establishes traceability and reproducibility. It is not independent human
label sign-off. For that reason both Product Monograph conditions use
`reporting.mode: descriptive`, which preserves all measurements without
assigning an official model rank.

## Condition 1, evidence-window extraction

Benchmark:
`hc_product_monograph_structured_extraction@0.1.0`

The model receives only official source text from pages selected to contain the
required evidence, with page 1 included. This condition isolates extraction,
normalization, bilingual reading, and JSON fidelity after relevant evidence has
already been located.

It does not measure full-document retrieval, PDF transport, page layout, or
provider file processing.

## Condition 2, full-PDF extraction

Benchmark:
`hc_product_monograph_native_pdf_extraction@0.1.0`

The model receives one complete hash-locked official PDF through the Responses
file-input route. The prompt contains no evidence-window text. The request path
verifies the file hash before use.

This condition adds:

- locating relevant facts anywhere in the document;
- PDF layout and page interpretation;
- long-document context management;
- provider file transport and PDF processing;
- the same extraction and JSON contract used by the evidence-window condition.

The raw PDFs are not committed because copyright may remain with individual
sponsors. The repository preserves the source locks needed to reacquire and
verify them.

## Results

The generated reports are authoritative for the compact recorded results:

- [evidence-window results](../../reports/benchmarks/hc_product_monograph_structured_extraction/0.1.0/leaderboard.md)
- [full-PDF results](../../reports/benchmarks/hc_product_monograph_native_pdf_extraction/0.1.0/leaderboard.md)

Both conditions preserve pass rate, field score, bilingual slices, failure
status, cost, latency, and paired comparisons where compatible evidence is
available.

Do not merge the two conditions into one score. The difference between them is
itself useful evidence about the added difficulty and operational behavior of
full-document input.

## Rebuild and verify

The evidence-window release can be rebuilt from the frozen cohort, source lock,
and locally acquired PDFs:

```bash
docker compose run --rm evalanche python -m evalanche.cli build-product-monograph
```

Build the full-PDF descriptors from the same source lock:

```bash
docker compose run --rm evalanche python -m evalanche.cli build-product-monograph-native-pdf
```

Verify either dataset manifest without model calls:

```bash
docker compose run --rm evalanche python -m evalanche.cli verify-dataset \
  --manifest configs/datasets/hc_product_monograph_structured_extraction_0.1.0_manifest.yaml \
  --root .

docker compose run --rm evalanche python -m evalanche.cli verify-dataset \
  --manifest configs/datasets/hc_product_monograph_native_pdf_extraction_0.1.0_manifest.yaml \
  --root .
```

A new model comparison should still begin from confirmed current access,
current provider behavior, and current pricing. Use the ordinary
[local comparison workflow](../local_comparison.md) rather than treating these
recorded configurations as a current model inventory.

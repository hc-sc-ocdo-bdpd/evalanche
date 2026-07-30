# Versioned dataset manifests

Evalanche dataset manifests make benchmark inputs, sampling decisions, split
assignments, and provenance inspectable and reproducible. The manifest is a
frozen record of a dataset release. It is not a download script and it does not
silently update when a source changes.

The current manifest schema version is `1.0`.

## What the manifest records

Every manifest records release identity, source provenance, files, and lineage.
Sampled benchmark releases also record their sampling and split decisions:

1. `release` identifies the release as a `source_snapshot` or `benchmark` and
   records its immutable semantic version, intended use, known limitations,
   languages, task types, and terms.
2. `sources` records where each source came from, when it was retrieved, and
   the source-reported modification date when one is available.
3. `files` records each repository-relative path, byte size, SHA-256 hash,
   record count, parser schema, and producer validation result.
4. `lineage` records the parent dataset version, transformation code version,
   and ordered transformation descriptions.
5. Benchmark-only `sampling` records the population, sampling method, unit,
   target size, membership file and ID column, random seed, criteria, and
   strata.
6. Benchmark-only `splits` records every selected member ID and its exact
   development or held-out assignment. Split members cannot overlap.

A `source_snapshot` cannot declare sampling or splits. A `benchmark` requires
both. This lets the same contract freeze raw DPD inputs before a pilot sample
exists, without inventing meaningless split assignments for the full source.

Unknown fields are rejected. This protects the release from misspelled fields
that would otherwise look like valid metadata.

## Working example

The repository includes a complete manifest for the six synthetic example
cases:

```text
configs/datasets/generic_example_manifest.yaml
```

Its file path is resolved from the repository root. Manifest paths must use
forward slashes, must be relative, and cannot contain `..`.

Manifest values do not expand environment variables. A frozen release must be
explicit and must load the same way in every environment.

The repository's `.gitattributes` pins tracked files under `data/` to LF line
endings. This prevents Windows Git settings from changing dataset bytes after
their byte sizes and SHA-256 hashes have been recorded. Do not convert a
frozen dataset file's line endings in place.

ZIP, PDF, and Parquet files under `data/` are explicitly marked as binary so
Git never applies line-ending conversion to their bytes.

## Materialized Health Canada source example

The first non-synthetic source release freezes the Health Canada DPD marketed
and approved archives published on 2026-07-02:

```text
configs/datasets/hc_dpd_2026-07-02_manifest.yaml
```

Its source-specific creation and validation process is documented in
[`docs/hc_benchmark/HC_DPD_SOURCE_SNAPSHOT.md`](hc_benchmark/HC_DPD_SOURCE_SNAPSHOT.md).
The release is a `source_snapshot`, so it correctly omits sampling and splits.
The manifest records the untouched archive hashes, while a hashed validation
report records every ZIP member's structure and row count.

## Materialized Health Canada benchmark example

The DPD source release is the parent of the first executable Health Canada
benchmark slice:

```text
configs/datasets/hc_dpd_structured_extraction_0.1.0_manifest.yaml
```

This `benchmark` release records a 40-product stratified sample, the
`products.csv` membership table, exact 30-product development and 10-product
held-out assignments, and 80 paired English and French cases. The verifier
checks the generated files and confirms that the product IDs in the membership
table exactly match the IDs declared across both splits.

The build, scope, source-row traceability, and limitations are documented in
[`docs/hc_benchmark/HC_DPD_BENCHMARK_SLICE.md`](hc_benchmark/HC_DPD_BENCHMARK_SLICE.md).

The complete eligible-population DPD benchmark is frozen separately as:

```text
configs/datasets/hc_dpd_structured_extraction_census_0.2.0_manifest.yaml
```

It records all 7,017 representable product families, 14,034 paired English and
French cases, and a bounded 24-case demo view. Its census cases and membership
tables use deterministic gzip compression, and the verifier checks their
decompressed CSV record counts and membership IDs. See
[`docs/hc_benchmark/HC_DPD_CENSUS_DEMO.md`](hc_benchmark/HC_DPD_CENSUS_DEMO.md).

## Verify a release

Run the verifier from the repository root:

```bash
docker compose run --rm evalanche python -m evalanche.cli verify-dataset --manifest configs/datasets/generic_example_manifest.yaml --root .
```

Expected status:

```text
Dataset: generic_instruction_following_example 1.0.0
Verified files: 1/1
Status: VALID
```

Optionally save an auditable JSON verification report:

```bash
docker compose run --rm evalanche python -m evalanche.cli verify-dataset --manifest configs/datasets/generic_example_manifest.yaml --root . --output results/generic_example_dataset_verification.json
```

The command exits with status 1 when the YAML contract is invalid, a file is
missing, or a recorded byte size, hash, or supported record count does not
match. For benchmarks, it also confirms that the membership file contains the
same unique member IDs recorded across the splits.

Verification is local and makes no model or provider API calls.

Verify the materialized DPD benchmark slice with:

```bash
docker compose run --rm evalanche python -m evalanche.cli verify-dataset --manifest configs/datasets/hc_dpd_structured_extraction_0.1.0_manifest.yaml --root .
```

Expected final lines:

```text
Verified files: 3/3
Status: VALID
```

Verify the full DPD census with:

```bash
docker compose run --rm evalanche python -m evalanche.cli verify-dataset --manifest configs/datasets/hc_dpd_structured_extraction_census_0.2.0_manifest.yaml --root .
```

Expected final lines:

```text
Verified files: 4/4
Status: VALID
```

## Record count methods

Each file declares how its record count is handled:

- `csv_rows` counts CSV records after the header with Python's CSV parser.
  Files ending in `.gz` are decompressed transparently.
- `jsonl_records` parses and counts nonblank JSON Lines records. Files ending
  in `.gz` are decompressed transparently.
- `declared` records a count that the built-in verifier cannot inspect. The
  byte size and hash are still verified.
- `not_applicable` requires a null record count.

Use an automatically verified method for materialized benchmark tables when
possible. PDF files must declare `page_count`; their file bytes and hash are
verified, but page counting is not currently automatic.

## Release workflow

1. Download or generate source files into a new versioned location.
2. Validate and normalize the files, including line endings for text formats,
   without changing an earlier release.
3. For a benchmark, select the sample with the recorded method and random seed.
4. For a benchmark, record exact split member IDs after grouping controls.
5. Calculate each file's byte size and SHA-256 hash.
6. Set `status: frozen` and `immutable: true` only when review is complete.
7. Run `verify-dataset` and retain the optional JSON verification report.
8. Commit the manifest with the release metadata allowed by the source terms.

If any source file or sampling decision changes, create a new dataset version.
Do not edit a frozen release in place.

## Reproducibility boundaries

A valid manifest proves that the referenced bytes match the recorded release
and that the manifest is internally consistent. It does not prove that:

- the source data is correct or complete;
- the sample is representative;
- the license permits redistribution of every source file;
- producer validation was methodologically sufficient;
- a declared-only record count or PDF page count is accurate.

Those decisions require source-specific validation and review. The Health
Canada benchmark specification adds those domain-specific quality gates.
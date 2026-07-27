# Health Canada DPD source snapshot

## Release

The first frozen Health Canada source release is:

```text
Dataset: hc_dpd_source_snapshot
Version: 2026.7.2
Source date: 2026-07-02
Manifest: configs/datasets/hc_dpd_2026-07-02_manifest.yaml
```

It contains the official marketed and approved Drug Product Database (DPD)
all-files archives. It does not contain cancelled, dormant, or product
monograph files.

The marketed cohort is the parent sampling population for the DPD-only
benchmark slice and the planned Product Monograph benchmark. The approved
cohort is frozen at the same time so later normalization can detect status
boundaries and avoid mixing records retrieved on different dates.

## Frozen files

| Cohort | Repository path | Bytes | Relational rows | SHA-256 |
| --- | --- | ---: | ---: | --- |
| Marketed | `data/hc/dpd/snapshots/2026-07-02/raw/allfiles.zip` | 1,513,470 | 170,236 | `94c8ef8639fd07eea9c134d8ede137d77574d4ae6761955f5aa582558ec38b25` |
| Approved | `data/hc/dpd/snapshots/2026-07-02/raw/allfiles_ap.zip` | 387,730 | 40,396 | `410fd4da8b01ccfa42a15a4d226830619dd91b777684f505975404ec075ac440` |

The row totals sum records across 12 relational tables. They are not unique
product counts.

`data/hc/dpd/snapshots/2026-07-02/validation.json` records each member's byte
size, SHA-256 hash, CRC32 value, ZIP timestamp, column count, and row count. It
also records the HTTP evidence observed during retrieval.

## Creation safeguards

The `snapshot-dpd` command fails without publishing a partial release unless
all checks pass:

1. Each response is HTTP 200 with an `application/zip` content type.
2. The final download remains on the trusted `www.canada.ca` host.
3. Each `Last-Modified` date exactly matches the requested source date.
4. The archive contains the exact expected 12 files for its cohort.
5. Member paths are safe, unencrypted, and dated on the source date.
6. Every member decodes as UTF-8 and parses as strict CSV.
7. Every nonblank record has the expected number of columns.
8. Downloaded bytes and validation-time bytes have identical hashes.
9. The generated schema `1.0` manifest passes local file verification.

Existing snapshot or manifest paths are never overwritten. A changed official
release must use a new source date and dataset version.

The official read-me page and the actual archive are not fully aligned in every
detail. For example, the archive uses `biosimilar*.txt`, and the therapeutic
class files currently contain four fields. Evalanche validates the observed
2026-07-02 archive contract and fails closed on a future structural change.
The parser schema must be deliberately reviewed and versioned before accepting
such a change.

## Verify the committed release

From the repository root:

```bash
docker compose run --rm evalanche python -m evalanche.cli verify-dataset --manifest configs/datasets/hc_dpd_2026-07-02_manifest.yaml --root .
```

Expected result:

```text
Dataset: hc_dpd_source_snapshot 2026.7.2
Verified files: 3/3
Status: VALID
```

The full test suite also reparses both archives with the source-specific DPD
validator. Verification and tests make no model calls.

## Create a later release

First confirm the date displayed on the official DPD extract page, then run:

```bash
docker compose run --rm evalanche python -m evalanche.cli snapshot-dpd --source-date YYYY-MM-DD --root .
```

This command downloads data from the official Health Canada URLs. It is not an
offline operation. Do not use an old date to retrieve a newer dynamic archive.
The `Last-Modified` equality check is intended to stop that mistake.

## Terms and attribution

The DPD dataset is listed under the Open Government Licence - Canada. Retain
the manifest attribution to Health Canada and the original source URLs when
redistributing the snapshot. The manifest records the licence URL and release
provenance.

## Official sources

- [DPD data extracts](https://www.canada.ca/en/health-canada/services/drugs-health-products/drug-products/drug-product-database/what-data-extract-drug-product-database.html)
- [DPD extract read-me and structure](https://www.canada.ca/en/health-canada/services/drugs-health-products/drug-products/drug-product-database/read-file-drug-product-database-data-extract.html)
- [Open Government Licence - Canada](https://open.canada.ca/en/open-government-licence-canada)

## Next boundary

This release continues to freeze source evidence only. Its marketed archive is
now consumed by the separate, immutable
[`HC_DPD_BENCHMARK_SLICE.md`](HC_DPD_BENCHMARK_SLICE.md) release, which records
normalization, product filtering, deterministic sampling, and benchmark
splits. Product Monograph acquisition and DPD-to-document audit remain the next
data boundary.
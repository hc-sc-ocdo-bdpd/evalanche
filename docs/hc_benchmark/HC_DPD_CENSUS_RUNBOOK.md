# Health Canada DPD census runbook

This runbook covers generation and evaluation for the frozen bilingual DPD
structured-extraction census.

```text
Dataset: hc_dpd_structured_extraction_census
Version: 0.2.0
Product families: 7,017
Cases: 14,034
Languages: English and French
```

All cases use deterministic JSON scoring. Evaluation and comparison commands
make no model or judge calls.

## Prerequisites

Create `.env` from `.env.example` and set the Azure API key, resource base URL,
and API version. Candidate routes may be overridden with the model-specific
environment variables documented in `.env.example`.

Build the container whenever the Dockerfile or dependency pins change:

```bash
docker compose build --no-cache
```

Run the repository checks:

```bash
docker compose run --rm evalanche sh -lc \
  "python -m pip install --no-cache-dir -r requirements-dev.txt && ruff check . && pytest"
```

Verify the frozen census:

```bash
docker compose run --rm evalanche python -m evalanche.cli verify-dataset \
  --manifest configs/datasets/hc_dpd_structured_extraction_census_0.2.0_manifest.yaml \
  --root .
```

Expected status:

```text
Verified files: 4/4
Status: VALID
```

## Execution controls

The full-census configurations default to four workers and 60 request starts
per minute. Each model has an independent local USD limit and an independent
checkpoint.

```dotenv
DPD_FULL_CENSUS_RPM=60
DPD_FULL_CENSUS_WORKERS=4
DPD_FULL_CENSUS_LUNA_MAX_USD=40
DPD_FULL_CENSUS_TERRA_MAX_USD=90
DPD_FULL_CENSUS_SOL_MAX_USD=170
```

The preflight estimates token cost from a completed model-specific sample,
applies a 1.5x safety multiplier, and refuses to start above the configured
limit. It is a local safeguard, not a provider billing cap.

## Run a model

Use the matching generation and evaluation configs:

| Model | Generation config | Evaluation config |
| --- | --- | --- |
| GPT-5.4 mini | `configs/generate_hc_dpd_census_gpt_5_4_mini.yaml` | `configs/evaluate_hc_dpd_census_gpt_5_4_mini.yaml` |
| GPT-5.6 Luna | `configs/generate_hc_dpd_census_gpt_5_6_luna.yaml` | `configs/evaluate_hc_dpd_census_gpt_5_6_luna.yaml` |
| GPT-5.6 Terra | `configs/generate_hc_dpd_census_gpt_5_6_terra.yaml` | `configs/evaluate_hc_dpd_census_gpt_5_6_terra.yaml` |
| GPT-5.6 Sol | `configs/generate_hc_dpd_census_gpt_5_6_sol.yaml` | `configs/evaluate_hc_dpd_census_gpt_5_6_sol.yaml` |

Run the read-only preflight first:

```bash
docker compose run --rm evalanche python -m evalanche.cli generate \
  --config configs/generate_hc_dpd_census_gpt_5_6_luna.yaml \
  --preflight-only
```

Proceed only when it reports `Status: READY`. Start or resume generation with
the same config:

```bash
docker compose run --rm evalanche python -m evalanche.cli generate \
  --config configs/generate_hc_dpd_census_gpt_5_6_luna.yaml
```

Evaluate the completed output:

```bash
docker compose run --rm evalanche python -m evalanche.cli evaluate \
  --config configs/evaluate_hc_dpd_census_gpt_5_6_luna.yaml
```

Replace `luna` with the desired model tier in all three commands. GPT-5.4 mini
uses its own corresponding filename.

## Resume behavior

Each completed response is appended to that model's JSONL checkpoint. If the
process stops because of `Ctrl+C`, a terminal closure, or a network failure,
repeat the identical generation command.

Do not run two instances of the same model config simultaneously. Do not
change the dataset, prompt, candidate profile, price catalog, API base, or API
version during a resumable run. These values are protected by the run
fingerprint.

## Assemble the full comparison

After the desired models complete, require them explicitly during assembly:

```bash
docker compose run --rm evalanche python -m evalanche.cli \
  assemble-dpd-census-comparison \
  --require-model gpt_5_4_mini \
  --require-model gpt_5_6_luna \
  --require-model gpt_5_6_terra \
  --require-model gpt_5_6_sol
```

Run the combined deterministic evaluation:

```bash
docker compose run --rm evalanche python -m evalanche.cli evaluate \
  --config configs/evaluate_hc_dpd_census_all_models.yaml
```

Assembly validates complete and identical case membership, inputs, reference
answers, and evaluation routes before combining models.

## Artifact policy

Generated outputs, checkpoints, and case-level evaluation results are written
under `data/generated/` and `results/`. They are intentionally ignored by Git
because a complete local run is hundreds of megabytes.

Retain these artifacts locally or in approved artifact storage until analysis
and validation are complete. Commit only compact, versioned result releases
under `reports/`, including raw-artifact hashes that preserve provenance.

The published aggregate result for the completed four-model run is:

[`reports/hc_dpd_census/0.2.0/README.md`](../../reports/hc_dpd_census/0.2.0/README.md)

## Interpretation boundary

The census measures extraction from rendered DPD relational records. It does
not measure extraction from Product Monograph documents and does not establish
a best model for unrelated tasks.

English and French cases from the same product family are related. Current
case-level confidence intervals do not account for that clustering. Manual
label review, grouped analysis, and error taxonomy work remain necessary
before treating the result as a formal benchmark recommendation.

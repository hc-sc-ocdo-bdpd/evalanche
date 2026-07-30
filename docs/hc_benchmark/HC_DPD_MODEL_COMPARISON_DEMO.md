# Health Canada DPD model-comparison demo

## Demo question

> How much quality do we gain, and how much more do we pay, when the same
> Health Canada extraction task is sent to different GPT models?

The comparison uses one frozen sample of 500 cases:

- 250 Health Canada DPD product families
- one English and one French case for every product
- proportional representation of all four benchmark complexity strata
- identical source data, instructions, output schema, and scorer for every
  model

The completed 14,034-case GPT-5.4 mini output is reused. It is filtered to
the exact 500 comparison case IDs, so no GPT-5.4 mini calls are repeated.

## Model run order

Run Luna, Terra, and Sol independently. Luna is the lowest-cost GPT-5.6 tier,
Terra provides the middle price and quality point, and Sol provides the
highest-cost tier in this comparison. A completed model can be assembled and
evaluated without waiting for the others.

At the default 60 request starts per minute, each 500-case run should take
roughly 9 to 15 minutes depending on latency. The runs are separate,
checkpointed, resumable, and can be stopped after any completed model.
The default assumes each deployment has at least 60 RPM. If a portal quota
is lower, set `DPD_COMPARISON_RPM` in `.env` to that lower value.

## Confirm the frozen sample

```bat
docker compose run --rm evalanche python -m evalanche.cli build-dpd-comparison
```

Expected:

```text
Products: 250
Cases: 500
Seed: 20260728
Status: VALID
```

The readable comparison CSV is:

```text
data/hc/demos/dpd_model_comparison/1.0.0/cases.csv
```

## Run GPT-5.6 Terra first

The 24-case pilot confirms the deployment and records Terra-specific token
usage. The preflight then estimates the 500-case charge without model calls.

```bat
docker compose run --rm evalanche python -m evalanche.cli generate --config configs/generate_hc_dpd_census_demo_gpt_5_6_terra.yaml
```

```bat
docker compose run --rm evalanche python -m evalanche.cli generate --config configs/generate_hc_dpd_comparison_500_gpt_5_6_terra.yaml --preflight-only
```

Only continue if the preflight reports `READY`:

```bat
docker compose run --rm evalanche python -m evalanche.cli generate --config configs/generate_hc_dpd_comparison_500_gpt_5_6_terra.yaml
```

## Add GPT-5.6 Luna

```bat
docker compose run --rm evalanche python -m evalanche.cli generate --config configs/generate_hc_dpd_census_demo_gpt_5_6_luna.yaml
```

```bat
docker compose run --rm evalanche python -m evalanche.cli generate --config configs/generate_hc_dpd_comparison_500_gpt_5_6_luna.yaml --preflight-only
```

```bat
docker compose run --rm evalanche python -m evalanche.cli generate --config configs/generate_hc_dpd_comparison_500_gpt_5_6_luna.yaml
```

## Add GPT-5.6 Sol

```bat
docker compose run --rm evalanche python -m evalanche.cli generate --config configs/generate_hc_dpd_census_demo_gpt_5_6_sol.yaml
```

```bat
docker compose run --rm evalanche python -m evalanche.cli generate --config configs/generate_hc_dpd_comparison_500_gpt_5_6_sol.yaml --preflight-only
```

```bat
docker compose run --rm evalanche python -m evalanche.cli generate --config configs/generate_hc_dpd_comparison_500_gpt_5_6_sol.yaml
```

Running the same command again resumes an interrupted model from its last
completed response.

## Refresh the comparison after any completed model

```bat
docker compose run --rm evalanche python -m evalanche.cli assemble-dpd-comparison
```

```bat
docker compose run --rm evalanche python -m evalanche.cli evaluate --config configs/evaluate_hc_dpd_comparison_500.yaml
```

The assembly command includes every complete model output it finds. It
requires all included models to cover the same 500 case IDs.

Open this notebook and select **Run All**:

```text
notebooks/HC_DPD_Model_Comparison_Demo.ipynb
```

If the selected Python environment does not already support notebooks and
charts, install the separate demo dependencies:

```bat
python -m pip install -r notebooks/requirements.txt
```

The notebook also performs the assembly and deterministic evaluation
locally when possible. It never makes model calls.

## Cost guardrails

Public Microsoft Global Standard rates recorded in the configuration are:

| Model | Input, USD per 1M tokens | Output, USD per 1M tokens | 500-case configured limit |
|---|---:|---:|---:|
| GPT-5.6 Luna | $1.00 | $6.00 | $5 |
| GPT-5.6 Terra | $2.50 | $15.00 | $10 |
| GPT-5.6 Sol | $5.00 | $30.00 | $20 |

Using the GPT-5.4 mini run's observed token profile only as a planning
estimate, the three 500-case runs together are roughly $6.37 USD before
the safety multiplier and $9.56 USD after the configured 1.5x multiplier.
The exact preflight values come from each model's own 24-case pilot.

The limits are local safeguards based on recorded token usage and public
rates. They are not Azure invoice hard caps. Actual organizational rates,
tax, foreign exchange, or provider billing adjustments can differ.

## Deployment names

The defaults assume the deployment names are exactly:

```text
gpt-5.6-luna
gpt-5.6-terra
gpt-5.6-sol
```

If a deployment name differs, add only the matching route to `.env`, for
example:

```text
GPT_5_6_TERRA_MODEL=azure/my-terra-deployment
```

Do not put an API key into a YAML file or notebook.

## Honest interpretation

- The full GPT-5.4 mini census result is evidence across all 14,034 cases.
- The cross-model result is a paired comparison on a representative
  500-case sample.
- A strict pass requires the complete eight-field JSON contract.
- Average field score explains near misses, but does not replace strict
  pass rate.
- These inputs are rendered DPD records, not Product Monograph documents.
- Report cost and quality separately. Do not force them into an arbitrary
  single winner.

The completed full-census result is published separately at
[`reports/hc_dpd_census/0.2.0/README.md`](../../reports/hc_dpd_census/0.2.0/README.md).

## Public sources used in the notebook

- Health Canada DPD access:
  https://www.canada.ca/en/health-canada/services/drugs-health-products/drug-products/drug-product-database.html
- Health Canada DPD open data:
  https://open.canada.ca/data/en/dataset/bf55e42a-63cb-4556-bfd8-44f26e5a36fe
- Microsoft GPT-5.6 pricing:
  https://azure.microsoft.com/en-us/blog/gpt-5-6-now-available-in-microsoft-foundry/
- Azure OpenAI pricing:
  https://azure.microsoft.com/en-us/pricing/details/azure-openai/

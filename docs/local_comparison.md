# Compare models available to you

Use this workflow when public evidence leaves a decision-relevant question
unanswered and you can access two or more candidate model deployments.

If you only want to understand model selection or public benchmarks, no setup
is required. Return to the [handbook](choosing_a_model.md) or the
[evidence directory](evidence/README.md).

## What the workflow produces

- saved model outputs that can be rescored without new provider calls;
- deterministic case and field results where possible;
- optional exploratory LLM-judge results;
- total and average cost with coverage information;
- token, latency, retry, and failure summaries;
- paired disagreements and subgroup results;
- a comparison report that does not select a model automatically;
- exact run plans, configuration, versions, and hashes.

## Before you begin

You need:

- Docker;
- the Evalanche repository;
- access to each intended endpoint or self-hosted route;
- provider credentials stored in `.env`;
- representative task cases and expected outputs;
- current pricing records if cost preflight or cost comparison matters.

Evalanche does not require all model providers to be the same. A new transport
that LiteLLM does not already support may require a provider adapter.

## 1. Build and validate the environment

Create the local environment file:

```bash
cp .env.example .env
```

Windows Command Prompt:

```cmd
copy .env.example .env
```

Enter only the credentials and routes you need, then build:

```bash
docker compose build
docker compose run --rm evalanche python -m evalanche.cli --help
```

See [setup.md](setup.md) for the full quality gate, rebuild guidance, and
JupyterLab command.

## 2. Confirm access to each candidate

An Evalanche model manifest describes a route. It does not prove that route is
available to you.

### Add or review model manifests

Each route is one YAML file under `configs/models/`:

```yaml
schema_version: "1.0"
model_id: my_first_deployment
display_name: My first deployment
provider_route: azure/my-first-deployment
request:
  temperature: 0
  max_retries: 3
  max_completion_tokens: 900
pricing_catalog_path: configs/endpoint_pricing.example.yaml
pricing_id: my_first_price
provider_model_version: replace_with_exact_version_if_known
deployment_type: api
resource_region: replace_with_region
capabilities:
  - text
  - json
```

Keep credentials and resource-specific base URLs in `.env`, never in a model
manifest. Use a separate model ID for materially different deployments,
versions, quantization, or serving configurations.

The four model manifests already present in the repository preserve prior
Health Canada work. They are historical examples, not your default candidate
list.

### Create a dated access set

Copy the template:

```bash
cp examples/local_comparison/access_set.example.yaml configs/access_sets/my_available_models.yaml
```

Replace every row and the date:

```yaml
schema_version: "1.0"
access_set_id: my_available_models
description: Routes available for the document-extraction comparison.
confirmed_on: 2026-08-06
models:
  - model_id: my_first_deployment
    access_confirmed: true
    confirmation_basis: I can call this deployment in the intended account.
  - model_id: my_second_deployment
    access_confirmed: true
    confirmation_basis: This self-hosted route is running in the approved environment.
```

`access_confirmed` can only be `true`. Omit a route when access is not
confirmed. Update `confirmed_on` when access, version, region, or operating
conditions change.

For a one-time run, repeated `--model` arguments are sufficient. They record
that those exact IDs were explicitly selected for that command.

## 3. Prepare representative cases

Start from the synthetic templates in
[`examples/local_comparison/`](../examples/local_comparison/):

- `classification_cases.csv`;
- `structured_extraction_cases.csv`;
- `factual_cases.csv`;
- `open_ended_cases.csv`.

Replace every placeholder. The minimum columns are:

```text
case_id,input,expected_output,evaluation_type
```

Useful additional columns include:

```text
group_id,language,split,risk,domain,difficulty,source
```

### Choose the evaluation route

| Route | Use when | Expected output |
| --- | --- | --- |
| `exact` | One normalized answer or closed label is authoritative | Accepted string or label |
| `json` | A structured object and fields are authoritative | Valid JSON object |
| `judge` | The real construct requires open-ended judgment | Human-reviewed reference or anchor plus rubric |

For JSON, define required fields and canonicalization before running the held
out set. For open-ended work, read [llm_judges.md](llm_judges.md). The current
judge path is exploratory unless calibrated against human-reviewed examples.

## 4. Register the local task

The fastest registry starting points are:

- [`exact_benchmark.example.yaml`](../examples/local_comparison/exact_benchmark.example.yaml)
  for classification or short factual answers;
- [`json_benchmark.example.yaml`](../examples/local_comparison/json_benchmark.example.yaml)
  for structured extraction.

Copy one into `configs/benchmarks/`, then update:

- benchmark and dataset IDs;
- case and manifest paths;
- task description;
- system prompt and output contract;
- scoring and canonicalization;
- slice columns and grouping;
- required capabilities;
- request API and limits;
- smoke, screen, and later tier sizes;
- token assumptions and safety factors;
- limitations.

For example:

```bash
cp examples/local_comparison/json_benchmark.example.yaml \
  configs/benchmarks/my_json_task_0.1.0.yaml
```

The sample dataset record is intentionally minimal for local development. A
published or reusable release should use the complete manifest contract shown
in `configs/datasets/generic_example_manifest.yaml` and should record source,
license, sampling, split, lineage, file size, row count, and SHA-256 hash.

Validate every registry record without model calls:

```bash
docker compose run --rm evalanche \
  python -m evalanche.cli registry-validate
```

Fix every reported issue before proceeding.

## 5. Plan or preflight before spending money

### One or more explicitly selected models

```bash
docker compose run --rm evalanche \
  python -m evalanche.cli run-benchmark \
  --benchmark my_json_task@0.1.0 \
  --model my_first_deployment \
  --model my_second_deployment \
  --plan-only
```

Each `--model` is an explicit access confirmation for this run.

### Every compatible model in an access set

For a tiered task, inspect route compatibility, resume state, and aggregate
projected cost:

```bash
docker compose run --rm evalanche \
  python -m evalanche.cli run-benchmark \
  --benchmark my_json_task@0.1.0 \
  --tier smoke \
  --all-compatible \
  --access-set configs/access_sets/my_available_models.yaml \
  --preflight-only
```

`--all-compatible` without `--access-set` is rejected. Capability compatibility
does not establish access.

The preflight lists:

- included deployments;
- incompatible deployments and missing capabilities;
- cumulative and newly executed cases;
- checkpointed and pending requests;
- cost sample source;
- per-model projection;
- aggregate budgeted projection;
- campaign plan path.

PDF and other large inputs use conservative token reserves and request cost
ceilings. A local gate can prevent the next request from starting, but it
cannot guarantee the provider's final invoice for an in-flight request.

## 6. Run smoke, then screen

Choose a cap at or above the reviewed projection:

```bash
docker compose run --rm evalanche \
  python -m evalanche.cli run-benchmark \
  --benchmark my_json_task@0.1.0 \
  --tier smoke \
  --all-compatible \
  --access-set configs/access_sets/my_available_models.yaml \
  --experiment \
  --max-cost-usd 5.00
```

Review transport, parse status, outputs, tokens, cost evidence, and errors.
Do not continue merely because the command completed.

Preflight and run the screen tier with the same access set:

```bash
docker compose run --rm evalanche \
  python -m evalanche.cli run-benchmark \
  --benchmark my_json_task@0.1.0 \
  --tier screen \
  --all-compatible \
  --access-set configs/access_sets/my_available_models.yaml \
  --preflight-only
```

Add `--experiment --max-cost-usd <reviewed_cap>` to execute. A child tier only
runs cases not already completed in its parent. Rerunning resumes checkpoints
and does not repeat a completed case and model pair.

Promotion is also access-scoped:

```bash
docker compose run --rm evalanche \
  python -m evalanche.cli run-benchmark \
  --benchmark my_json_task@0.1.0 \
  --tier standard \
  --promote-from screen \
  --access-set configs/access_sets/my_available_models.yaml \
  --preflight-only
```

Promotion gates come from the benchmark manifest, not model-name logic.

## 7. Build the comparison report

For access-confirmed candidates:

```bash
docker compose run --rm evalanche \
  python -m evalanche.cli summarize-benchmark \
  --benchmark my_json_task@0.1.0 \
  --tier screen \
  --access-set configs/access_sets/my_available_models.yaml
```

Or select exact models for the summary:

```bash
docker compose run --rm evalanche \
  python -m evalanche.cli summarize-benchmark \
  --benchmark my_json_task@0.1.0 \
  --tier screen \
  --model my_first_deployment \
  --model my_second_deployment
```

To study all past compatible results without claiming current access:

```bash
docker compose run --rm evalanche \
  python -m evalanche.cli summarize-benchmark \
  --benchmark my_json_task@0.1.0 \
  --tier screen \
  --all-results
```

The report labels this last mode as historical evidence only.

Generated comparison artifacts normally appear under:

```text
results/benchmark_experiments/<benchmark_id>/<version>/<tier>/
```

Open `model_comparison.md` first. CSV files retain model, pairwise, slice,
field, and case-outcome detail. `model_comparison.json` records the access scope
and artifact hashes.

## 8. Read the report correctly

Check in this order:

1. Candidate access boundary.
2. Case and model completeness.
3. Generation and scorer failures.
4. Strict pass rate and uncertainty.
5. Paired disagreements.
6. Important fields and slices.
7. Shared failures.
8. Total cost, cost coverage, tokens, latency, and retries.
9. Failure severity.
10. Limitations and missing evidence.

Observed order is descriptive. The report does not claim that its first row is
the best model in general or the right choice for every stakeholder.

Use the [decision record template](decision_record.md) to document the tradeoff.

## 9. Add a new model later

1. Add one valid model manifest.
2. Confirm and add the route to the access set.
3. Run `registry-validate`.
4. Preflight smoke.
5. Run smoke and screen.
6. Rerun the summary.

No Python analysis code changes. Existing model outputs are not regenerated.
Adding a model to a fixed benchmark requires only that new model's calls.

## 10. Open-ended tasks

The registry examples above focus on deterministic routes because they are the
most defensible and simplest to operate. Evalanche's general `generate` and
`evaluate` commands can mix exact, JSON, and judge cases using:

- `configs/generate_generic.yaml`;
- `configs/evaluate_generated_generic.yaml`;
- the `open_ended_cases.csv` template.

Before using the generic configuration:

- replace its candidate list with routes you have confirmed;
- replace its judge route and rubric;
- keep automatic selection disabled unless every candidate has an explicit
  availability profile and the policy was approved;
- create and evaluate a human calibration set;
- label results exploratory until the judge protocol is validated.

Start from the complete provider-free fixture:

```bash
docker compose run --rm evalanche python -m evalanche.cli validate-judge --protocol examples/judge_validation/protocol.yaml
```

Copy `examples/judge_validation/`, replace every synthetic case, human label,
and judge observation, then set a justified target and gates. The validation
command makes no provider calls. It compares the saved observations against
the adjudicated human reference, measures stability and bias controls, writes
the disagreement queue, and exits with status 2 if the declared target is not
met.

After validation, attach the exact protocol and generated JSON report to the
evaluation config. Evalanche verifies the task, prompt, rubric, judge model,
version, settings, and contract hash before accepting the evidence level. See
[llm_judges.md](llm_judges.md) for the full input contract and claim limits.

## Common problems

| Problem | Meaning or action |
| --- | --- |
| `--all-compatible requires --access-set` | Create or pass a dated access set |
| Access set references an unknown ID | Add or correct the model manifest |
| No compatible models | Review required capabilities and the actual provider route |
| Cost is unknown | Add current pricing and verify token reporting |
| Budget block | Review the projection and choose a justified higher cap or smaller tier |
| Child tier requires parent result | Complete the parent tier for that model first |
| Historical report says access not confirmed | Use `--access-set` or explicit `--model` for a candidate comparison |
| Judge rows are unscored | Recover or rerun judge calls; do not count judge failure as model failure |

For registry internals, versioning, and publication controls, read
[benchmark_registry.md](benchmark_registry.md).

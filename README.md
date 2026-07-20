# Mission Evalanche

Mission Evalanche is a lightweight, extensible model evaluation toolkit for task-grounded language model evaluation.

The project is designed to help answer:

> Which model should I use for this specific task?

The core principle is that there is no single best model in the abstract. Model choice depends on factors such as:

- task performance
- representative test data
- evaluation method
- cost
- latency
- reliability
- language requirements
- privacy and deployment constraints
- risk tolerance

Public benchmarks can help identify candidate models, but final recommendations should be based on evaluations that reflect the actual intended use case.

## Current capabilities

Evalanche currently supports:

1. Generating outputs from one or more candidate models.
2. Saving generated outputs as reusable evaluation artifacts.
3. Routing each test case to an appropriate evaluation method.
4. Running deterministic checks for constrained outputs.
5. Running rubric-based LLM judge evaluation for open-ended outputs.
6. Producing case-level results.
7. Producing model leaderboards.
8. Saving reproducibility metadata.
9. Producing a plain-language model recommendation report.

The current workflow keeps generation and evaluation separate so outputs can be inspected, reused, and evaluated multiple ways without calling candidate models again.

## Evaluation methods

### Deterministic evaluation

Deterministic evaluation uses code-based checks rather than another language model.

It is appropriate for tasks with constrained or objectively checkable answers, such as:

- classification labels
- true or false answers
- dates
- numbers
- exact extraction
- structured JSON output

Current deterministic checks include:

- exact string match
- normalized string match
- JSON validity
- JSON exact match
- JSON field match rate

### LLM judge evaluation

LLM judge evaluation uses a configurable rubric to evaluate outputs that require semantic judgment.

It is appropriate for tasks such as:

- rewriting
- summarization
- explanations
- question answering
- document drafting
- other open-ended generation tasks

The current criteria judge supports:

- configurable criteria
- configurable criterion weights
- criterion-level scores
- weighted overall scores
- pass or fail thresholds
- criterion-level reasoning
- overall reasoning

LLM judge results should be treated as evaluation signals rather than objective truth. Important or high-risk evaluations should eventually be calibrated against human or expert review.

## Evaluation routing

Each test case declares how its output should be evaluated using the `evaluation_type` column.

Supported values:

- `exact`: normalized string matching for constrained answers
- `json`: parsed JSON structure and value matching
- `judge`: rubric-based LLM evaluation for open-ended answers

Example:

```csv
case_id,input,expected_output,evaluation_type
case_001,"Return the sentiment label.","negative","exact"
case_002,"Return the extracted fields as JSON.","{""name"":""Jordan""}","json"
case_003,"Rewrite this sentence clearly.","A clear rewrite.","judge"
```

Deterministic metrics intentionally skip `judge` cases.

The LLM judge intentionally skips `exact` and `json` cases.

This prevents inappropriate evaluation methods, such as requiring an open-ended rewrite to exactly match one reference sentence.

## Repository structure

```text
evalanche/
  Dockerfile
  docker-compose.yml
  requirements.txt
  .env.example
  .gitignore
  README.md

  configs/
    candidate_models.yaml
    generate_generic.yaml
    judge_generated_generic.yaml
    judge_generic.yaml
    judge_generic_model_comparison.yaml
    metrics_generated_generic.yaml

  data/
    examples/
      generic_cases.csv
      generic_outputs.csv
      generic_model_comparison_outputs.csv

    generated/
      generated output files are written here

  evalanche/
    __init__.py
    cli.py
    config.py
    generation.py
    io.py
    llm.py
    metadata.py
    recommendation.py
    reporting.py
    routing.py

    judges/
      __init__.py
      criteria.py

    metrics/
      __init__.py
      deterministic.py

  results/
    evaluation outputs are written here
```

The `data/generated/` and `results/` directories are ignored by Git by default.

## Environment setup

Create a local `.env` file from the example:

```bash
cp .env.example .env
```

On Windows Command Prompt:

```cmd
copy .env.example .env
```

On PowerShell:

```powershell
Copy-Item .env.example .env
```

Do not commit the real `.env` file.

### Azure OpenAI example

```bash
AZURE_API_KEY=your-key-here
AZURE_API_BASE=https://your-resource-name.cognitiveservices.azure.com/
AZURE_API_VERSION=2024-12-01-preview

JUDGE_MODEL=azure/gpt-5.4-mini
CANDIDATE_MODEL=azure/gpt-5.4-mini
```

Use the Azure resource base URL only.

Do not use a full route such as:

```text
https://your-resource-name.cognitiveservices.azure.com/openai/responses?api-version=...
```

The deployment name is supplied separately through the LiteLLM model route.

## Docker setup

Build the image:

```bash
docker compose build
```

The repository is mounted into the container, so most Python and configuration changes do not require rebuilding the image.

Rebuild when dependencies or the Dockerfile change:

```bash
docker compose build --no-cache
```

## Candidate model configuration

Candidate models are defined in:

```text
configs/candidate_models.yaml
```

Example:

```yaml
models:
  - name: gpt_5_4_mini
    model: ${CANDIDATE_MODEL:-azure/gpt-5.4-mini}
    temperature: 0
    max_retries: 3
    max_completion_tokens:
```

The `name` field is the readable model name used in reports.

The `model` field is the LiteLLM model route used for API calls.

Additional models can be added to the list:

```yaml
models:
  - name: model_a
    model: azure/deployment-a
    temperature: 0
    max_retries: 3

  - name: model_b
    model: azure/deployment-b
    temperature: 0
    max_retries: 3
```

## Input dataset format

Generation input files must contain:

```text
case_id
input
expected_output
evaluation_type
```

Example:

```csv
case_id,input,expected_output,evaluation_type
case_001,"Classify the sentiment as positive, neutral, or negative: The shipment arrived late and damaged.","negative","exact"
case_002,"Return only the due date in YYYY-MM-DD format: Submit the form by 2026-03-15.","2026-03-15","exact"
case_003,"Rewrite in plain language: Applicants are required to provide documentation prior to the stated deadline.","Applicants must provide documents before the deadline.","judge"
case_004,"Return only the number: What is 8 multiplied by 6?","48","exact"
case_005,"Classify as refund, technical_support, or account_update: I was charged twice and need the extra charge reversed.","refund","exact"
case_006,"Extract the name and amount as JSON: Jordan paid 42.50 for the permit.","{""name"":""Jordan"",""amount"":42.50}","json"
```

Additional columns may be added later, such as:

```text
language
domain
difficulty
risk_level
benchmark
source
category
```

Generation preserves additional input columns in the generated output file.

## Generate candidate outputs

Run generation:

```bash
docker compose run --rm evalanche python -m evalanche.cli generate --config configs/generate_generic.yaml
```

The generation command:

1. Loads the input cases.
2. Loads the candidate model configuration.
3. Calls every candidate model for every test case.
4. Saves the model outputs.
5. Records latency and token usage.
6. Records generation failures without necessarily stopping the run.
7. Writes generation metadata.

Example generated files:

```text
data/generated/generic_generated_outputs.csv
data/generated/generic_generated_outputs_generation_metadata.json
```

Generated outputs should be inspected before important evaluations.

## Run deterministic metrics

Run deterministic evaluation:

```bash
docker compose run --rm evalanche python -m evalanche.cli metrics --config configs/metrics_generated_generic.yaml
```

The metrics command evaluates only cases routed to:

```text
exact
json
```

Cases routed to `judge` are skipped.

Example outputs:

```text
results/metrics_generated_generic_first_slice_results.csv
results/metrics_generated_generic_first_slice_results_metrics_summary.csv
results/metrics_generated_generic_first_slice_results_metrics_metadata.json
```

The case-level output includes fields such as:

```text
metric_applicable
metric_status
metric_passed
exact_match
normalized_exact_match
expected_is_json
output_is_json
json_exact_match
json_expected_field_count
json_matching_field_count
json_field_match_rate
```

## Run LLM judge evaluation

Run rubric-based judge evaluation:

```bash
docker compose run --rm evalanche python -m evalanche.cli judge --config configs/judge_generated_generic.yaml
```

The judge command evaluates only cases routed to:

```text
judge
```

Cases routed to `exact` or `json` are skipped.

Example outputs:

```text
results/judge_generated_generic_first_slice_results.csv
results/judge_generated_generic_first_slice_results_model_summary.csv
results/judge_generated_generic_first_slice_results_run_metadata.json
results/judge_generated_generic_first_slice_results_recommendation.md
```

The case-level judge output includes:

```text
weighted_score
passed
overall_reason
criterion scores
criterion reasons
judge token usage
raw judge result
```

## Judge configuration

Example rubric:

```yaml
run:
  name: judge_generated_generic_first_slice
  input_path: data/generated/generic_generated_outputs.csv
  output_path: results/judge_generated_generic_first_slice_results.csv

judge:
  model: ${JUDGE_MODEL:-azure/gpt-5.4-mini}
  temperature: 0
  max_retries: 3

task:
  name: generated_generic_instruction_following
  description: >
    Evaluate whether the candidate model output satisfies the user instruction
    and matches the expected output. The input is self-contained.

scoring:
  score_min: 0
  score_max: 5
  pass_threshold: 0.75

criteria:
  - name: correctness
    weight: 0.70
    description: >
      The output is semantically correct relative to the instruction and expected
      output. Formatting problems should not be heavily penalized here unless they
      change the meaning.

  - name: completeness
    weight: 0.20
    description: >
      The output includes all required information.

  - name: format_following
    weight: 0.10
    description: >
      The output follows the requested format, label set, or structure.
```

Criterion weights are normalized when the weighted score is calculated.

## Output artifacts

Evalanche writes multiple artifacts so evaluation runs can be inspected and reproduced.

### Generation artifacts

```text
generated outputs CSV
generation metadata JSON
```

### Deterministic evaluation artifacts

```text
case-level metric results CSV
model-level metric summary CSV
metrics metadata JSON
```

### Judge evaluation artifacts

```text
case-level judge results CSV
model-level judge summary CSV
run metadata JSON
recommendation Markdown report
```

Metadata artifacts include information such as:

```text
run name
timestamp
input path
output paths
file hashes
candidate models
judge model
prompt configuration
rubric
criterion weights
pass threshold
case counts
model counts
token usage
summary results
known limitations
```

## Model recommendation reports

Judge runs currently produce a recommendation report based on the judge leaderboard.

The recommendation is intentionally task-specific.

It should be interpreted as:

> Based on this dataset, rubric, judge model, and configuration, this was the strongest model among those evaluated.

It should not be interpreted as:

> This is the best model in general.

A later iteration will combine deterministic and judge evidence into one task-aware model recommendation.

## Current command summary

Generate candidate outputs:

```bash
docker compose run --rm evalanche python -m evalanche.cli generate --config configs/generate_generic.yaml
```

Run deterministic metrics:

```bash
docker compose run --rm evalanche python -m evalanche.cli metrics --config configs/metrics_generated_generic.yaml
```

Run the LLM judge:

```bash
docker compose run --rm evalanche python -m evalanche.cli judge --config configs/judge_generated_generic.yaml
```

Show CLI help:

```bash
docker compose run --rm evalanche python -m evalanche.cli --help
```

## Development approach

Mission Evalanche is being developed through small, working, tested iterations.

Each iteration should:

1. Add one clear capability.
2. Preserve inspectable intermediate artifacts.
3. Avoid coupling generation to one evaluation method.
4. Record enough metadata to reproduce the run.
5. Support the larger model-selection workflow.
6. Avoid unnecessary architecture until the need is demonstrated.
7. Be committed after successful testing.

## Planned development

Near-term planned work:

1. Add automated tests for routing, normalization, JSON comparison, configuration validation, and leaderboard calculations.
2. Combine deterministic and judge results into one task-aware case result.
3. Produce one combined model leaderboard and recommendation.
4. Add an end-to-end run command while preserving the separate commands.
5. Add cost, latency, token, and generation reliability comparisons.
6. Compare multiple real candidate models.
7. Add research-backed evaluation method documentation.

Longer-term work may include:

- classification accuracy and macro F1
- confusion matrices
- field-level extraction metrics
- numeric tolerance
- date equivalence
- list and set comparison
- bilingual evaluation
- human and judge agreement analysis
- pairwise model comparison
- RAG evaluation
- retrieval metrics
- agent and tool-use evaluation
- public-sector and Health Canada benchmark datasets

## Health Canada benchmark direction

The reusable Evalanche core is being built independently of any single benchmark.

A future Health Canada or public-sector benchmark can plug into the same generation, routing, evaluation, reporting, and recommendation workflow.

Potential benchmark tasks include:

- structured extraction from product monographs
- comparison against Drug Product Database fields
- bilingual regulatory document evaluation
- recall and safety alert classification
- grounded regulatory summarization
- long-document faithfulness evaluation

The benchmark is intended to extend the core framework, not replace it or block its practical use.

## Data use

Examples, test fixtures, and demonstration datasets should use synthetic or publicly available data unless another data source has been explicitly approved for the project.

Sensitive, protected, personal, confidential, or internal organizational information must not be committed to the repository.

Environment variables, credentials, generated outputs, and local evaluation artifacts should remain excluded from version control where appropriate.

## Limitations

Current limitations include:

- Deterministic exact matching can reject semantically equivalent answers.
- JSON comparison currently uses strict key and value equality.
- Open-ended judge results depend on the selected judge model and rubric.
- The current recommendation report is based only on judge-evaluated cases.
- Cost estimates are not yet implemented.
- Multiple-run variability is not yet measured.
- Human calibration is not yet implemented.
- Judge bias testing is not yet implemented.
- Combined task-aware recommendations are not yet implemented.

## Project status

The current implementation demonstrates the core evaluation loop:

```text
test cases
-> candidate model generation
-> saved outputs
-> task-aware evaluation routing
-> deterministic or LLM judge evaluation
-> model-level summaries
-> reproducibility metadata
-> recommendation artifacts
```

The next major capability will combine routed evaluation results into one scientifically defensible model comparison.
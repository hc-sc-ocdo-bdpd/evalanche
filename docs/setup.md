# Setup and command reference

You need this page only if you want to run Evalanche comparisons, rebuild case
studies, or contribute code. The handbook and research directory work directly
on GitHub without installation.

## Supported path

The supported repository setup uses Docker Compose. Install Docker using its
official instructions for your operating system, then run the commands below
from the repository root.

## 1. Create the local environment file

macOS, Linux, or a shell that supports `cp`:

```bash
cp .env.example .env
```

Windows Command Prompt:

```cmd
copy .env.example .env
```

Edit `.env` and fill only the routes and credentials you intend to use.

Rules:

- Never commit `.env`.
- Never put credentials in YAML, CSV, notebooks, or reports.
- Verify current API versions, regions, model availability, and limits against
  official provider documentation.
- The model names in `.env.example` preserve repository examples. They are not
  recommendations or proof of access.

## 2. Build

```bash
docker compose build
```

Rebuild without cache after changing the Dockerfile or pinned requirements:

```bash
docker compose build --no-cache
```

The `.dockerignore` excludes credentials, Git metadata, caches, and large local
run artifacts from the image build context.

## 3. Confirm the command interface

```bash
docker compose run --rm evalanche python -m evalanche.cli --help
```

No provider call is made by `--help`.

## 4. Validate the repository

Registry and manifest validation:

```bash
docker compose run --rm evalanche python -m evalanche.cli registry-validate
```

Full development quality gate:

```bash
docker compose run --rm evalanche sh -lc \
  "python -m pip install --no-cache-dir -r requirements-dev.txt && ruff check . && coverage run -m pytest && coverage report --fail-under=80"
```

Windows Command Prompt, as one line:

```cmd
docker compose run --rm evalanche sh -lc "python -m pip install --no-cache-dir -r requirements-dev.txt && ruff check . && coverage run -m pytest && coverage report --fail-under=80"
```

These checks make no provider calls.

## 5. Run a comparison

Follow [local_comparison.md](local_comparison.md). Its workflow requires an
explicit model list or a dated access set before a multi-model campaign can
run.

## 6. Launch the notebooks

The committed notebooks read saved artifacts and make no provider calls with
their default settings.

Launch JupyterLab in a temporary container:

```bash
docker compose run --rm -p 8888:8888 evalanche sh -lc \
  "python -m pip install --no-cache-dir -r notebooks/requirements.txt && jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
```

Open the local URL printed by JupyterLab. Stop the container when finished.

## Local artifact locations

| Path | Content | Git behavior |
| --- | --- | --- |
| `data/generated/` | Generated cases, plans, checkpoints, and provider outputs | Ignored |
| `results/` | Local evaluations and comparison artifacts | Ignored except committed release structure |
| `reports/` | Compact source-controlled releases and leaderboards | Tracked |
| `configs/` | Versioned route, benchmark, dataset, pricing, and run definitions | Tracked |

Raw provider output should normally remain local. Compact public releases
retain hashes and provenance without requiring provider calls to be repeated.

## Safe command categories

These are designed to make no provider calls:

- `--help`;
- `registry-validate`;
- `verify-dataset`;
- `--plan-only`;
- `--preflight-only`;
- `summarize-benchmark`;
- `rescore-benchmark`;
- Health Canada analysis, audit, and dataset build commands unless a runbook
  explicitly states otherwise.

Commands that generate or judge outputs can call providers. Review the command,
selected models, access scope, price data, and cost guard first.

## Troubleshooting

### A command cannot find `.env`

Confirm `.env` exists in the repository root and that Docker Compose is run
from that root.

### A route authenticates but the model call fails

Check provider route syntax, deployed model name, API version, region, account
permissions, modality support, quota, and rate limits. Do not assume that a
public model name equals the deployment name.

### Registry validation fails

Read every reported file and field. Common causes include an unknown pricing
reference, missing dataset file, incompatible evaluation type, duplicate case
ID, missing slice column, or invalid tier graph.

### Docker uses an old dependency

Rebuild with `--no-cache`.

### A prior run exists

Tiered campaigns and generation configs can resume from checkpoints. Review
the preflight before deleting or replacing any local artifact. Completed
provider outputs are intentionally reusable for rescoring.

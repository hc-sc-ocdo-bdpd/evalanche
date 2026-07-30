# Contributing

Evalanche changes should preserve three properties: task-specific evidence,
reproducible artifacts, and clear operational boundaries.

## Development setup

Python 3.11 is the runtime used by the Docker image. Python 3.12 is also tested
in continuous integration.

Create an environment and install the pinned development dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

On Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

Run the required checks before committing:

```bash
ruff check .
coverage run -m pytest
coverage report
python -m evalanche.cli verify-dataset \
  --manifest configs/datasets/hc_dpd_structured_extraction_census_0.2.0_manifest.yaml \
  --root .
```

The coverage configuration enforces a package-wide minimum of 80 percent.
New behavior should include focused tests, especially for invalid inputs,
partial failures, and reproducibility guards.

## Change boundaries

- Keep generation separate from evaluation. Saved model outputs must remain
  reusable without another provider call.
- Make benchmark and scoring behavior configuration driven where practical.
- Preserve exact prompts, model routes, pricing evidence, input hashes, and
  output hashes in run metadata.
- Treat frozen dataset versions and published result releases as immutable.
  Corrections require a new version or a clearly documented superseding
  release.
- Do not silently relax deterministic matching rules. Canonicalization must be
  symmetric, explicit, versioned, and retained in metadata.
- Do not mix unrelated task or source families into one headline score.
- Do not commit credentials, local endpoint URLs, personal paths, generated
  checkpoints, or case-level provider outputs.

## Configuration changes

Run configuration paths are part of historical metadata. Do not move or rename
a configuration after it has been used for a published run. Add a new file
when a model version, prompt, dataset version, price card, or execution policy
changes materially.

See [`configs/README.md`](configs/README.md) for the configuration inventory
and naming rules.

## Data and result releases

Tracked benchmark data belongs under a versioned `data/` path and must have a
validated dataset manifest. Large generated outputs and case-level results
remain under the ignored `data/generated/` and `results/` directories.

Compact, reviewable result releases belong under `reports/`. A release should
include:

- the aggregate model summary;
- paired comparisons where models answered the same cases;
- run and generation metadata;
- hashes and sizes for retained raw artifacts;
- an interpretation note that states validation status and limitations.

## Pull requests and commits

Keep commits focused and use an imperative summary. The body should explain
why the change is needed, identify any artifact or schema version changes, and
state the checks that passed. Generated files should be reviewed just as
carefully as source code.

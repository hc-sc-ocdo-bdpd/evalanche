# Contributing

Evalanche changes should preserve three properties: task-specific evidence,
reproducible artifacts, and clear operational boundaries.

The primary product is the access-first handbook and research directory. The
optional software compares only user-confirmed model routes, and the Health
Canada benchmarks remain supporting case studies.

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
python -m evalanche.cli evidence-status --root . --check
python -m evalanche.cli verify-dataset \
  --manifest configs/datasets/hc_dpd_structured_extraction_census_0.2.0_manifest.yaml \
  --root .
```

The coverage configuration enforces a package-wide minimum of 80 percent.
New behavior should include focused tests, especially for invalid inputs,
partial failures, and reproducibility guards.

Before a release, also run the supported clean-checkout Docker path:

```bash
cp .env.example .env
docker compose config --quiet
docker compose build
docker compose run --rm evalanche sh -lc \
  "python -m pip install --no-cache-dir -r requirements-dev.txt && ruff check . && coverage run -m pytest && coverage report --fail-under=80"
docker compose run --rm evalanche python -m evalanche.cli registry-validate --root .
docker compose run --rm evalanche python -m evalanche.cli evidence-status --root . --check
```

These commands do not call a model provider. Continuous integration repeats
the Docker build and runs the core offline commands in a container with
network access disabled, which protects the boundary between validation and
paid execution.

## Change boundaries

- Keep generation separate from evaluation. Saved model outputs must remain
  reusable without another provider call.
- Make benchmark and scoring behavior configuration driven where practical.
- Treat the active model set as registry data. Generic run, summary, and
  reporting code must not contain a fixed list of model IDs or require edits
  when another compatible model manifest is added.
- Never treat registry membership or declared compatibility as user access.
  Multi-model commands must require explicit model IDs or a dated access set.
- Keep public evidence separate from local candidate scope. Public sources can
  mention inaccessible models without adding them to a local comparison.
- Preserve comparison-first report semantics. A single selected model is
  allowed only through an explicit policy with availability profiles and
  approved thresholds or weights.
- Keep tier sampling, cost assumptions, inheritance, and promotion gates in
  benchmark manifests. Generic campaign code must not name a task, dataset, or
  current model.
- Preserve cumulative tier membership and incremental execution. A child tier
  must never repeat a case and model call already completed by its parent.
- Treat aggregate cost limits as local request-start gates. Retain the sample,
  price card, safety multiplier, request reserve, and checkpoint ledger needed
  to explain every estimate and stop.
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
- Keep generated bring-your-own-task bundles under the ignored `local_tasks/`
  path unless a deliberate review promotes a sanitized task into the main
  registry.

## Research documentation

- Prefer original papers, official benchmark sites, maintained repositories,
  and official provider documentation.
- Record a review date and publication status.
- Keep every curated entry's maintenance status, next review date, version
  scope, and change trigger current in `docs/evidence/catalog.yaml`.
- Mark replaced sources as superseded instead of deleting their history.
- Explain what the source measures, how it is scored, which setup matters, and
  where it should not transfer.
- Do not copy fast-changing leaderboard scores into evergreen guidance.
- Mark preprints as preprints and avoid presenting a recent result as settled
  consensus.
- Use model and system terminology precisely. Agent-harness results are not
  base-model results.

After reviewing evidence, update `status_report_as_of` and regenerate the
committed status page:

```bash
python -m evalanche.cli evidence-status \
  --root . --as-of YYYY-MM-DD --write
python -m evalanche.cli evidence-status --root . --check
```

The command validates metadata and documentation links but makes no network or
model calls. Source changes still require human interpretation.

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

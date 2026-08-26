# Technical maintenance

Evalanche is a versioned model-selection resource and optional evaluation
toolkit. This guide covers the technical work needed to keep its evidence,
benchmarks, software, and generated reports reproducible.

## Routine offline checks

Run these checks before relying on the repository after an update:

```bash
python -m evalanche.cli --version
python -m evalanche.cli registry-validate --root .
python -m evalanche.cli evidence-status --root . --check
ruff check .
coverage run -m pytest
coverage report --fail-under=80
```

Use the Docker commands in `docs/setup.md` for the supported clean
environment. Immediately before a new comparison, confirm the model route,
provider interface, exact model version, price, quota, and required modality.

## Update triggers

Review the affected artifacts when any of these changes:

- an evidence source reaches its `review_by` date or publishes a new version;
- a model, deployment, provider interface, or price changes;
- a source dataset, parser, or schema changes;
- a benchmark prompt, input representation, scorer, rubric, or judge contract
  changes;
- a dependency or container change affects execution or generated output;
- a result claim no longer matches the evidence available.

## Versioning rules

Create a new benchmark or dataset version when any of these changes
materially:

- case membership or source snapshot;
- expected output or label;
- prompt or input representation;
- scorer, normalization, rubric, or judge contract;
- measured construct or intended interpretation;
- serving setup when it affects comparability.

Do not edit a frozen benchmark release in place. Do not merge results with
different compatibility fingerprints. A software-only release may keep the
evaluator compatibility version stable when dataset, prompt, scorer, and
evaluation semantics are unchanged.

Use `ranking.enabled: false` with a clear reason when a benchmark supports
descriptive measurements but not an ordered claim. `frozen` means
reproducible, not automatically validated.

## Evidence catalog

After reviewing a source, update `docs/evidence/catalog.yaml` with the review
date, next review date, version scope, trigger, and status. Mark replaced
sources `superseded` instead of deleting their history. Regenerate the status
page with:

```bash
python -m evalanche.cli evidence-status --root . --as-of YYYY-MM-DD --write
python -m evalanche.cli evidence-status --root . --check
```

These commands validate committed metadata and links. They do not browse the
web or make provider calls, so the source itself must still be reviewed.

## Adding a model or task

A normal model route is one manifest under `configs/models/`. Confirm actual
access separately through explicit model selection or a dated access set.
Never add a model to generic Python logic.

Start a task with `init-task`, replace all placeholders, confirm the task
review fields, validate it, preflight the smallest tier, and review projected
cost before any provider call. Add a sanitized reusable task to the main
registry only after its data, prompt, scorer, and intended interpretation are
documented.

## Release checks

1. Review the diff for credentials, endpoint URLs, local paths, raw provider
   outputs, restricted data, or unintended generated files.
2. Run lint, tests, coverage, registry validation, evidence validation,
   relevant dataset verification, and repository-relative link checks.
3. Build the Docker image from a clean tracked-file export and repeat the
   offline gates inside it.
4. Rebuild leaderboards and inspect their HTML, Markdown, CSV, and JSON
   surfaces.
5. Confirm that eligible, provisional, incomplete, failed, and retired results
   have the intended rank behavior.
6. Update `CHANGELOG.md`, the package version, affected benchmark
   documentation, and the compatibility version only when evaluation
   semantics changed.
7. Tag or otherwise identify the exact source revision used for the release.

## Security and failure handling

- Stop a run if credentials, restricted data, unexpected provider routing,
  excessive cost, or unexplained source drift is observed.
- Preserve sanitized logs and hashes needed to diagnose the failure, but do
  not commit secrets or restricted content.
- Rotate exposed credentials through the platform that issued them.
- Treat dependency, container, and provider-interface changes as versioned
  maintenance, not silent updates.

The repository does not currently assert a public open-source license. Confirm
appropriate terms before external distribution or accepting outside
contributions.

## Retiring artifacts

Retire a benchmark when its evidence is no longer maintained, its source or
route cannot be reproduced, or its claim is no longer appropriate. Preserve
the minimal manifests, fingerprints, summaries, and limitation notes needed
to interpret historical results. Remove unfinished data and active work queues
that are not part of the supported resource.

If the repository itself is archived, preserve a tagged source release, the
container definition, dependency files, evidence catalog, compact result
bundles, generated reports, and changelog.

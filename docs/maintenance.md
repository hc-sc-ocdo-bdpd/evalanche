# Technical maintenance

Evalanche is a versioned model-selection resource and optional evaluation
toolkit. Maintenance should preserve current guidance, reproducible evidence,
and safe execution without turning the repository into a project log.

## Routine checks

Use the supported Docker environment from `docs/setup.md`:

```bash
docker compose run --rm evalanche python -m evalanche.cli --version
docker compose run --rm evalanche python -m evalanche.cli registry-validate --root .
docker compose run --rm evalanche python -m evalanche.cli evidence-status --root . --check
docker compose run --rm evalanche sh -lc "python -m pip install --no-cache-dir -r requirements-dev.txt && ruff check . && coverage run -m pytest && coverage report --fail-under=80"
```

Before a new comparison, separately confirm the current model route, provider
interface, model version, capability support, price, quota, and access.

## Update triggers

Review affected material when:

- a curated evidence source reaches its review date or changes materially;
- a model, deployment, provider interface, capability, or price changes;
- a source dataset, parser, or schema changes;
- a benchmark prompt, input contract, scorer, rubric, or judge contract changes;
- a dependency or container change affects execution or generated output;
- a published claim no longer matches its evidence.

## Versioning and immutability

Create a new dataset or benchmark version when a material part of the measured
contract changes, including case membership, reference outputs, prompt, input
representation, scorer, rubric, judge contract, or intended construct.

**Do not mix incompatible result fingerprints.** A documentation or
reporting-only software change can preserve evaluator compatibility when
evaluation semantics are unchanged.

`frozen` means the release contract is reproducible. It does not by itself say
whether results should be ranked. Use `reporting.mode: descriptive` when the
measurement is useful but the evidence does not justify an official ordering.

## Evidence catalog

After reviewing a public source, update `docs/evidence/catalog.yaml` with its
review date, next review date, version scope, trigger, and current status.
Supersession is appropriate for an external source that has genuinely been
replaced, because the catalog records evidence lineage rather than project
history.

```bash
docker compose run --rm evalanche python -m evalanche.cli evidence-status --root . --as-of YYYY-MM-DD --write
docker compose run --rm evalanche python -m evalanche.cli evidence-status --root . --check
```

These commands validate local metadata and links. They do not substitute for
reviewing the external source itself.

## Models and tasks

A normal model route is one manifest under `configs/models/`. Confirm actual
access separately, never encode current model IDs into generic analysis code.

For a new task, use `init-task`, replace placeholders, validate the bundle,
preflight the smallest tier, and inspect projected cost before provider calls.
Only move a reusable task into the main registry after its data, prompt,
scorer, limitations, and intended interpretation are documented.

## Release checks

1. Review the diff for credentials, private endpoints, local paths, raw provider
   outputs, restricted data, and accidental generated files.
2. Run lint, tests, coverage, registry validation, evidence validation, and
   relevant dataset verification.
3. Check repository-relative Markdown links.
4. Build benchmark result surfaces and inspect HTML, Markdown, CSV, and JSON.
5. Confirm ranked, descriptive, and incomplete results display as intended.
6. Verify copy-and-paste commands from a clean checkout.
7. Record the exact source revision used for any externally shared release.

## Security and failure handling

**Stop a run if credentials, restricted data, unexpected provider routing,
excessive cost, or unexplained source drift is observed.** Preserve only the
sanitized logs and hashes needed for diagnosis. Rotate exposed credentials
through the issuing platform.

**The repository does not assert a public open-source licence.** Confirm the
appropriate distribution terms before external release or outside
contributions.
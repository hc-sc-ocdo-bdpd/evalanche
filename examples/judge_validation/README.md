# Judge validation worked example

This directory contains a complete, provider-free example of the
`validate-judge` workflow. The cases, human labels, and judge observations are
synthetic fixtures. They test the artifact contract and reporting path, but
they are not evidence that any real judge is calibrated.

Run it from the repository root:

```bash
docker compose run --rm evalanche python -m evalanche.cli validate-judge --protocol examples/judge_validation/protocol.yaml
```

The command makes no provider calls. It creates:

- a Markdown validation report;
- a machine-readable JSON report;
- a gate table;
- a disagreement review file;
- subgroup results.

The example target is deliberately `exploratory`. Evalanche caps every
protocol with `synthetic_fixture: true` at that level, even if the sample
metrics happen to pass stronger numeric gates.

All stability, prompt, identity, position, and subgroup controls are computed
on the held-out validation split. Decision-grade control gates also require
every declared condition to cover every validation case, so a few repeated
cases cannot stand in for the full evidence set.

To validate a real judge, copy all four data files, replace every synthetic
row with task-specific evidence, set `synthetic_fixture: false`, document the
human-review protocol, choose stakeholder-approved gates, and preserve a
held-out validation split.

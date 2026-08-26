# Inspect AI adapter

This directory contains the optional Inspect AI adapter for Evalanche's frozen
Health Canada DPD benchmark.

The maintained setup, validation, execution, import, retry, reporting, and
interpretation guide is:

[`../../docs/integrations/inspect_ai_dpd.md`](../../docs/integrations/inspect_ai_dpd.md)

Keep operational guidance in that document so the integration has one
canonical runbook.

## Offline validation

From the repository root:

```bash
docker compose --profile inspect build inspect
docker compose --profile inspect run --rm inspect python -m integrations.inspect_ai.cli doctor
docker compose --profile inspect run --rm inspect python -m integrations.inspect_ai.cli parity
docker compose --profile inspect run --rm inspect python -m integrations.inspect_ai.cli replay --display plain
```

These commands make no provider calls. Paid execution requires an explicit
`run` command, credentials, model selection, scope, and campaign name.

The adapter reuses the frozen dataset, benchmark fingerprint, prompt, and
canonical Evalanche scorer. It writes Inspect logs and derived reports under
the ignored `results/inspect_ai/` directory and never changes the standard
Evalanche leaderboard automatically.

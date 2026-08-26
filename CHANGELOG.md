# Changelog

All notable Evalanche repository releases are documented here.

## 0.4.0, 2026-08-17

### Added

- Explicit benchmark ranking policy with validated reasons for disabled ranks.
- Provisional result status in CSV, JSON, Markdown, HTML, and benchmark index
  surfaces.
- Package `--version` command.
- Technical maintenance guidance for evidence, benchmark, dependency, and
  report updates.
- Tests for disabled ranking and permanent provisional treatment.

### Changed

- Product Monograph 0.1.0 measurements are permanently provisional and
  unranked because their reference labels lack independent validation.
- The native-PDF 0.1.0 benchmark is a retired local pilot.
- Package version is separated from the evaluator compatibility version so
  reporting and documentation releases do not invalidate frozen results.
- Active DPD and Product Monograph bundles now use the current scorer
  fingerprint. Full offline rescoring preserved every compact case-score hash
  and the published DPD model totals.
- Primary user commands are one-line and Windows Command Prompt compatible.
- Inspect AI operational guidance has one canonical runbook.

### Removed

- The unfinished Product Monograph 1.0.0 expansion, its 200-family draft data,
  audit package, expansion code, commands, and tests.
- The Product Monograph review command and active human-audit workflow.
- Duplicate Inspect AI runbook content.

### Compatibility

The evaluator compatibility version remains 0.3.0. Existing result bundles
remain compatible because dataset, prompt, scorer, and evaluator semantics are
unchanged.

## 0.3.0

- Added configuration-driven model and benchmark registry workflows.
- Added compact immutable result bundles, generated leaderboards, access sets,
  tiered execution, cost preflight, checkpoints, and resume.
- Added the frozen DPD census release, Product Monograph pilots, guided task
  initialization, judge-validation controls, and optional Inspect AI support.

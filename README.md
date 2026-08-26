# Evalanche

Evalanche is a practical, research-grounded resource for answering:

> What model should I use for this task, and what evidence should I trust?

There is no best model for every problem. The useful question is which model
configuration fits a particular task, set of constraints, and decision
context.

You can use Evalanche without installing anything. Start with the handbook and
public evidence. If those sources leave a decision-relevant question
unanswered, use the optional local workflow to compare only models you can
actually access.

## Choose your path

| I want to... | Start here |
| --- | --- |
| Understand how careful model selection works | [Model-selection handbook](docs/choosing_a_model.md) |
| Find credible papers, benchmarks, and evaluation methods | [Research and evidence directory](docs/evidence/README.md) |
| Compare model deployments available to me | [Local comparison guide](docs/local_comparison.md) |
| See the method applied to real public data | [Health Canada case studies](docs/case_studies/health_canada.md) |

The first two paths require no code.

## The method in one minute

1. Define the real task, input, output, and unacceptable failures.
2. List only model configurations you can actually use in the intended environment.
3. Remove candidates that fail hard requirements such as modality, privacy,
   region, language, context, or output format.
4. Review public evidence that measures something meaningfully similar.
5. Identify what remains unknown and whether it matters enough to test locally.
6. If needed, run a small representative comparison before expanding the test.
7. Compare quality, failure patterns, cost, latency, and reliability, then
   record the tradeoff you accepted and when to revisit it.

Public evidence can discuss any relevant model. A local candidate list cannot.
A registry entry or compatible API route is never treated as proof that a model
is available or approved for a user.

## What Evalanche helps you avoid

- Treating one broad leaderboard as a universal answer.
- Averaging unrelated benchmark scores into a false universal score.
- Testing models that cannot be used in the intended environment.
- Paying for a large run before checking transport, parsing, token reporting,
  and cost.
- Treating an LLM judge as an unquestioned source of truth.
- Hiding uncertainty, severe failures, or operational tradeoffs behind one
  aggregate score.

## What the optional software does

The evaluation engine supports:

- independent model, access, dataset, benchmark, pricing, and result records;
- exact text and canonical JSON scoring;
- rubric-based LLM judging when deterministic evaluation is not appropriate;
- offline task-specific judge validation against human annotations;
- agreement, stability, prompt, identity, order, and subgroup diagnostics;
- case, field, language, slice, cost, token, latency, retry, and failure summaries;
- paired comparisons and confidence intervals;
- smoke, screen, and standard tiers with aggregate cost preflight;
- budget gates, checkpoints, and safe resume;
- immutable compact result bundles and compatibility fingerprints;
- ranked or descriptive benchmark reporting, depending on the evidence contract;
- comparison reports that preserve tradeoffs instead of selecting a model by default.

Unvalidated judge results remain exploratory. A matching validation contract
and report are required before judge evidence can support stronger comparative
claims. See [LLM judges](docs/llm_judges.md).

Adding a normal model deployment is data-only. Its manifest records the route,
request settings, declared capabilities, version, and pricing reference. The
generic analysis code contains no fixed model list.

An optional [Inspect AI adapter](docs/integrations/inspect_ai_dpd.md) is kept
separate from the default runtime for teams that want to reuse Evalanche DPD
artifacts with Inspect-native logs.

## Access comes before comparison

There are two safe ways to select models for a run:

- repeat `--model <model_id>` for routes you confirm for that run;
- use a dated access-set YAML file with `--all-compatible`.

`--all-compatible` means all compatible deployments inside that confirmed
access set, not every model in the repository. `--all-results` is a separate
way to inspect previously recorded evidence and does not imply current access.

The shortest bring-your-own-task path is offline:

```bash
docker compose run --rm evalanche python -m evalanche.cli init-task
```

It creates an isolated starter bundle under `local_tasks/`, makes no provider
calls, and gives exact validation, preflight, tier, and reporting commands.
See the [local comparison guide](docs/local_comparison.md) for the complete
workflow.

## Evidence, not a universal ranking

The evidence directory is organized by the question a source can help answer,
including broad capability, coding, tool use, long context, document and
multimodal work, retrieval, factuality, multilingual behavior, safety, human
preference, and operational performance.

Each entry explains what was measured, how it was scored, which setup matters,
and where transfer to another task becomes weak. Fast-changing sources are
linked and dated rather than copied into evergreen ranking tables.

Explore the [research and evidence directory](docs/evidence/README.md). Its
[status page](docs/evidence/status.md) records review dates and maintenance
state. The offline check is:

```bash
docker compose run --rm evalanche python -m evalanche.cli evidence-status --check
```

## Health Canada case studies

The repository includes three supporting benchmark conditions built from
public Health Canada data:

- a full Drug Product Database structured-extraction census;
- Product Monograph extraction from evidence-selected text windows;
- Product Monograph extraction from complete official PDFs.

The DPD census supports ranked reporting within its exact DPD contract. The two
Product Monograph conditions are descriptive because their reference labels
have complete automated source-page evidence but no independent human sign-off.
They remain useful for studying extraction behavior and the effect of changing
from supplied evidence to full-document input, without implying an official
model ranking.

Open the [Health Canada case-study page](docs/case_studies/health_canada.md) for
results and interpretation boundaries.

## Install only if you want to run comparisons

Docker is the supported execution path. Start with
[setup and command reference](docs/setup.md), then follow the
[local comparison guide](docs/local_comparison.md).

```bash
cp .env.example .env
docker compose build
docker compose run --rm evalanche python -m evalanche.cli --help
```

Windows Command Prompt uses `copy .env.example .env` for the first command.

## Repository map

```text
docs/         Handbook, evidence, methods, and case studies
configs/      Model, access, benchmark, dataset, pricing, and run definitions
data/         Synthetic examples and versioned benchmark releases
evalanche/    Optional evaluation software
integrations/ Optional adapters with isolated dependencies
reports/      Compact result releases and generated benchmark reports
notebooks/    Reproducible case-study analysis
tests/        Unit, integration, and repository-contract tests
```

Local provider outputs and large case-level results are ignored by Git.
Credentials and resource-specific endpoint URLs belong only in `.env`.

## Principles

- Access first.
- Task relevance over broad rank.
- Public evidence before unnecessary local calls.
- Deterministic or executable scoring when a defensible answer exists.
- Human judgment when the construct genuinely requires it.
- LLM judges treated as fallible measurement instruments.
- Missing evidence stays missing, never zero.
- Small tests before expensive tests.
- Quality, cost, latency, reliability, and severe failures stay visible.
- Comparison by default, automatic selection only through an explicit policy.
- Exact versions, sources, prompts, settings, and limitations are preserved.
- Health Canada case studies support the resource without defining it.

## Development and maintenance

Evalanche has been developed with substantial assistance from OpenAI's GPT-5.6
Sol. AI-assisted contributions are reviewed and validated by maintainers, who
remain accountable for the repository and its outputs.

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution rules and
[technical maintenance](docs/maintenance.md) for upkeep and validation.

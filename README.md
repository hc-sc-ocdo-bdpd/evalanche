# Evalanche

Evalanche is a practical, research-grounded guide to answering:

> What model should I use for this task, and what evidence should I trust?

There is no best model for every problem. The useful question is which model
fits a particular task, set of constraints, and decision context.

You can use Evalanche without installing anything. Start with the guide and
follow the evidence as far as your decision requires. If public evidence does
not answer the important questions, Evalanche also provides an optional local
comparison workflow for models you can actually access.

## Choose your path

| I want to... | Start here |
| --- | --- |
| Understand how careful model selection works | [The model selection handbook](docs/choosing_a_model.md) |
| Find credible papers, leaderboards, and evaluation methods | [Research and evidence directory](docs/evidence/README.md) |
| Compare model deployments available to me | [Local comparison guide](docs/local_comparison.md) |
| Browse complete public examples | [Health Canada case studies](docs/case_studies/health_canada.md) |

The first two paths require no code.

## The method in one minute

1. Define the real task, required input, and acceptable output.
2. List only models you can actually use in the intended environment.
3. Remove candidates that fail a hard requirement, such as modality,
   privacy, region, language, or output format.
4. Read public evidence that measures something meaningfully similar.
5. Identify what remains unknown for the decision.
6. Run a small representative comparison only if that missing evidence could
   change the choice.
7. Review failures, quality, cost, latency, and reliability, then record the
   tradeoff you accepted.

Public evidence can discuss any relevant model. A local candidate list cannot.
Evalanche never treats a registered model, a compatible API route, or a public
leaderboard entry as proof that a model is available to you.

## What Evalanche helps you avoid

- Treating one broad leaderboard as a universal answer.
- Comparing scores produced by different tasks and harnesses as if they shared
  one scale.
- Testing models that cannot be purchased, approved, deployed, or operated in
  the intended environment.
- Paying for large runs before checking transport, parsing, token reporting,
  and cost.
- Using an LLM judge as an unquestioned source of truth.
- Selecting a model from a tiny score difference while hiding uncertainty or
  operational tradeoffs.

## What the optional software does

The evaluation engine supports:

- independent model, dataset, benchmark, price, and result records;
- exact text and canonical JSON scoring;
- rubric-based LLM judging for properties that need human-like judgment;
- offline task-specific judge validation against independent and adjudicated
  human labels;
- chance-corrected agreement, stability, prompt, identity, order, and subgroup
  diagnostics for saved judge observations;
- case, field, language, slice, cost, token, latency, retry, and failure
  summaries;
- paired comparisons with confidence intervals and multiplicity correction;
- smoke, screen, and standard tiers with aggregate cost preflight;
- local budget gates, checkpoints, and safe resume;
- immutable result bundles and reproducible artifact hashes;
- comparison reports that preserve tradeoffs instead of selecting a model by
  default.

Unvalidated judge results remain exploratory. A matching validation contract
and report are required before judge evidence can support a comparative leader
or an opt-in policy selection. See [LLM judges](docs/llm_judges.md).

Adding a normal model deployment is data-only. Its manifest records the route,
request settings, declared capabilities, version, and pricing reference. The
analysis code contains no fixed list of model names.

## Access comes before comparison

There are two safe ways to start a run:

- Repeat `--model <model_id>` for the deployments you confirm for that run.
- Create a dated access-set YAML file and use it with `--all-compatible`.

`--all-compatible` means all compatible deployments inside the access set. It
does not mean all models in the repository.

Historical summaries use `--all-results`. They are clearly labelled as past
evidence and do not claim that the included models remain available.

See the [local comparison guide](docs/local_comparison.md) for the complete,
copy-and-paste Docker workflow, input templates, cost preflight, and report
interpretation.

The shortest bring-your-own-task path starts with one offline command:

```bash
docker compose run --rm evalanche python -m evalanche.cli init-task
```

It asks only for task-owned information, creates an isolated bundle under
`local_tasks/`, adds model stubs only for routes you explicitly name, and makes
no provider calls. The generated README contains the exact validation, plan,
preflight, tier, and report commands for that task. See
[guided task initialization](docs/local_comparison.md#guided-task-initialization-recommended)
for the non-interactive form and safety boundaries.

## Evidence, not a universal ranking

Evalanche organizes public evidence by the question it can answer:

- broad capability orientation;
- coding and software agents;
- tool and function calling;
- long-context work;
- documents and multimodal reasoning;
- retrieval and embeddings;
- factuality and grounded generation;
- multilingual behavior;
- safety and risk;
- human preference;
- cost, latency, and throughput.

Each source note explains what was measured, how it was scored, what setup
matters, and where the result should not be transferred. The directory links
to source papers, official benchmark sites, and maintained frameworks rather
than copying fast-changing leaderboard scores into permanent advice.

Explore the [research and evidence directory](docs/evidence/README.md).
Its [maintenance status page](docs/evidence/status.md) shows when each source
was last reviewed and whether it is current, stale, or superseded. The offline
status command can evaluate the catalog for today's date without contacting a
source site or model provider:

```bash
docker compose run --rm evalanche python -m evalanche.cli evidence-status
```

## Health Canada case studies

Evalanche includes two substantial Health Canada examples:

- Drug Product Database structured extraction;
- Product Monograph evidence-window and native-PDF extraction.

They demonstrate bilingual, structured, source-grounded evaluation and remain
useful benchmark assets. They do not define the repository's model universe or
its main navigation. Native-PDF quality results remain provisional until the
reference labels receive the documented human review.

Open the [Health Canada landing page](docs/case_studies/health_canada.md) for
results, status, limitations, and detailed runbooks.

## Install only if you want to run comparisons

Evalanche uses Docker for the supported setup. The repository explains its own
configuration and commands, but does not attempt to teach general software
installation.

Start with [setup and command reference](docs/setup.md), then follow the
[local comparison guide](docs/local_comparison.md).

The shortest validation path is:

```bash
cp .env.example .env
docker compose build
docker compose run --rm evalanche python -m evalanche.cli --help
```

Windows Command Prompt uses `copy .env.example .env` for the first command.

## Repository map

```text
docs/         Handbook, research directory, methods, and case studies
configs/      Model, access, benchmark, dataset, pricing, and run definitions
data/         Synthetic examples and versioned benchmark releases
evalanche/    Optional evaluation software
reports/      Compact result releases and leaderboards
notebooks/    Reproducible case-study analysis
tests/        Unit, integration, and repository-contract tests
```

Local provider outputs and case-level results are ignored by Git. Credentials
and resource-specific endpoint URLs belong only in `.env`.

## Project principles

- Access first.
- Task relevance over broad rank.
- Deterministic scoring when a defensible answer exists.
- Human judgment when the construct truly requires it.
- Calibrated LLM judging only as a fallible measurement instrument.
- Missing evidence reported as missing, never converted to zero.
- Small tests before expensive tests.
- Comparison by default, automatic selection only through an explicit policy.
- Exact versions, source dates, prompts, settings, and limitations preserved.
- Health Canada examples kept useful without dominating the general resource.

For contribution and release rules, see [CONTRIBUTING.md](CONTRIBUTING.md).

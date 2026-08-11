# Evaluation frameworks

Evalanche should not recreate every established evaluation tool. Choose the
framework whose abstraction matches the work, then preserve enough metadata to
interpret the result.

Framework review dates, version scope, and maintenance status are tracked on
the [evidence maintenance status page](status.md).

## Quick chooser

| Need | Consider first |
| --- | --- |
| Safety, capability, agent, or tool evaluations with reusable solvers, scorers, logs, and sandboxes | Inspect AI |
| Transparent multi-scenario language-model evaluation and maintained leaderboards | HELM |
| Broad academic language-model task harness with many community tasks | lm-evaluation-harness |
| Embedding, retrieval, reranking, clustering, and multilingual representation | MTEB |
| Custom eval definitions and model-output grading in the OpenAI ecosystem | OpenAI Evals |
| Small access-aware comparison with deterministic JSON scoring, operational metrics, cost gates, and reproducible local artifacts | Evalanche |

These tools can complement one another. A public framework result can become
one evidence source in an Evalanche decision brief.

## Inspect AI

- **Official sources:** [Documentation](https://inspect.aisi.org.uk/),
  [evaluation catalog](https://inspect.aisi.org.uk/evals/), and
  [repository](https://github.com/UKGovernmentBEIS/inspect_ai).
- **Maintainer:** UK AI Security Institute.
- **Core abstraction:** Dataset, solver, scorer, model, and evaluation log.
- **Strengths:** Reusable agent and tool patterns, sandboxing, structured logs,
  model-provider support, and a maintained catalog with more than 200
  prebuilt evaluations at the time of review.
- **Use it when:** The evaluation requires agents, tools, complex solvers,
  safety or capability suites, or established Inspect tasks.
- **Important checks:** Exact Inspect version, task revision, solver, scorer,
  sandbox, model adapter, token or action budget, and log retention.
- **Relationship to Evalanche:** Inspect is documented as an external
  framework. Evalanche does not import or wrap Inspect in its current runtime,
  so `inspect-ai` is not a required dependency.

## HELM

- **Official sources:** [Documentation and leaderboards](https://crfm.stanford.edu/helm/)
  and [repository](https://github.com/stanford-crfm/helm).
- **Maintainer:** Stanford Center for Research on Foundation Models.
- **Core abstraction:** Scenario, adaptation method, model, and multiple
  metrics.
- **Strengths:** Transparent scenario definitions, standardized adapters,
  multi-metric reporting, and maintained benchmark suites.
- **Use it when:** A HELM scenario or leaderboard already covers the broad
  capability or risk question.
- **Important checks:** Scenario and release, model adapter, prompt and
  adaptation, metric, model version, and any missing result coverage.
- **Relationship to Evalanche:** HELM evidence can orient a shortlist. A local
  Evalanche test may still be needed for the actual deployment and task.

## lm-evaluation-harness

- **Official source:** [EleutherAI repository](https://github.com/EleutherAI/lm-evaluation-harness).
- **Core abstraction:** Task configuration, model adapter, request, and metric.
- **Strengths:** Large community task collection, broad model-backend support,
  few-shot and log-likelihood evaluation, and common use in model reports.
- **Use it when:** The desired language-model benchmark already exists in the
  harness or fits its evaluation model.
- **Important checks:** Harness commit, task YAML and dataset revision, few-shot
  count, chat template, generation arguments, model revision, batching, and
  decontamination treatment.
- **Important limit:** Community task availability does not prove that every
  task is current, uncontaminated, or valid for a local decision. Review each
  task definition.

## MTEB

- **Official sources:** [Repository](https://github.com/embeddings-benchmark/mteb)
  and [paper](https://aclanthology.org/2023.eacl-main.148/).
- **Core abstraction:** Embedding model, task, dataset, split, and task-specific
  metric.
- **Strengths:** Broad task coverage for embeddings, retrieval, reranking,
  clustering, classification, similarity, multilingual, and multimodal work.
- **Use it when:** The object of selection is an embedding or retrieval model,
  not primarily a generative model.
- **Important checks:** Task and language, query or document instructions,
  embedding dimension, normalization, pooling, model revision, and whether a
  reranker is part of the system.
- **Relationship to Evalanche:** MTEB is the stronger specialized framework
  for its domain. Evalanche can document access, operational constraints, and
  a decision that uses MTEB evidence.

## OpenAI Evals

- **Official source:** [Repository](https://github.com/openai/evals).
- **Core abstraction:** Evaluation specification, samples, completion
  function, and grader.
- **Strengths:** Custom evaluation definitions and a public registry of eval
  patterns, with integrations centered on the OpenAI evaluation ecosystem.
- **Use it when:** Its graders and execution workflow match the provider and
  evaluation design.
- **Important checks:** Repository commit, eval definition, model and endpoint,
  grader, prompt, sampling, retry behavior, and result artifacts.
- **Important limit:** Using a framework does not validate a custom dataset,
  rubric, or grader automatically.

## Evalanche

- **Core abstraction:** Access-confirmed deployment, versioned dataset,
  benchmark manifest, saved generation, scorer, tier, and comparison artifact.
- **Strengths:** Access-first candidate scoping, deterministic structured
  scoring, provider-cost preflight, safe resume, paired evidence, field and
  slice summaries, and Health Canada case studies.
- **Use it when:** A user has a small set of available model routes and needs a
  representative, reproducible comparison whose operational evidence matters.
- **Do not use it as:** A universal public model catalog, a replacement for
  established task suites, or an automatically authoritative LLM judge.

## Framework selection checklist

Before choosing a tool, answer:

1. What is the unit being evaluated: model, endpoint, agent, retrieval system,
   or complete product?
2. What input, output, tools, and environment must be represented?
3. Can success be checked deterministically or by executable state?
4. Does a maintained task already exist?
5. Are model adapters and required providers supported?
6. Can exact prompts, versions, logs, and artifacts be preserved?
7. Can the framework report the operational measures the decision needs?
8. Does the team have the expertise to audit the task and scorer?

The best framework is not the one with the most tasks. It is the one that can
support the intended claim with the least unnecessary custom machinery.

Last source review: 2026-08-06.

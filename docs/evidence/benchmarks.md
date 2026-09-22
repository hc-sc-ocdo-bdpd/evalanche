# Benchmarks and living evidence sources

This directory gathers public evidence relevant to a task. **It is not a list
of models Evalanche recommends, and it is not proof that any model is
available to a user.**

**Open the live source before making a decision.** Record the model version,
evaluation date, harness, settings, and score you actually used.

Review dates, version scope, and current, stale, or superseded status are
tracked on the [evidence maintenance status page](status.md).

## Quick task map

| Task or question | Useful starting points |
| --- | --- |
| Broad capability orientation | HELM, LiveBench |
| Open-ended human preference | LM Arena |
| Capability with price and speed context | Artificial Analysis |
| Repository issue resolution | SWE-bench Verified |
| Current code generation | LiveCodeBench |
| Function and tool calling | BFCL |
| Web and computer-like agents | WebArena, GAIA |
| Policy-constrained tool agents | tau-bench |
| Long-context reasoning | LongBench v2 |
| Document images and layout | DocVQA |
| Broad multimodal reasoning | MMMU-Pro |
| Embeddings and retrieval | MTEB and MMTEB |
| Retrieval-augmented generation | ARES |
| Grounded long-form factuality | FACTS Grounding |
| Short fact questions | SimpleQA |
| Multilingual knowledge transfer | Global-MMLU |
| General-purpose AI safety | MLCommons AILuminate |

## Broad capability and preference

### HELM

- **Status:** Research framework and maintained benchmark collection.
- **Official sources:** [HELM site](https://crfm.stanford.edu/helm/),
  [paper](https://arxiv.org/abs/2211.09110), and
  [repository](https://github.com/stanford-crfm/helm).
- **Helps answer:** How do models compare across a transparent set of
  scenarios and multiple metrics?
- **Measures:** Results depend on the selected HELM leaderboard and scenario.
  HELM explicitly represents scenario, adaptation method, and metrics.
- **Scoring:** Mixes task-appropriate metrics rather than one universal
  measure.
- **Setup to verify:** Exact scenario, model adapter, prompt, model version,
  token budget, and metric.
- **Important limit:** **A broad average does not prove performance on a local
  task, and it does not establish that every listed model is accessible.**
- **Reviewed:** 2026-08-06.

### LiveBench

- **Status:** Research benchmark and maintained benchmark suite.
- **Official sources:** [Repository](https://github.com/LiveBench/LiveBench)
  and [paper](https://arxiv.org/abs/2406.19314).
- **Helps answer:** How do current models compare on refreshed, objectively
  scored broad tasks?
- **Measures:** Broad categories including reasoning, mathematics, coding,
  language, data analysis, and instruction following, subject to the current
  release.
- **Scoring:** Ground-truth or objective scoring designed to avoid an LLM judge
  for the primary score.
- **Setup to verify:** Release date, category, question source, contamination
  policy, model version, prompt, and any reasoning or sampling configuration.
- **Important limit:** **Freshness reduces one contamination risk, but category
  performance still does not equal performance on a specialized document,
  agent, language, or output contract.**
- **Reviewed:** 2026-08-06.

### LM Arena

- **Status:** Maintained public platform and ICML 2024 research paper.
- **Official sources:** [Current site](https://lmarena.ai/),
  [how it works](https://lmarena.ai/how-it-works),
  [FAQ](https://lmarena.ai/faq), and
  [ICML paper](https://arxiv.org/abs/2403.04132).
- **Helps answer:** Which anonymous responses people prefer for organic prompts
  under the arena protocol?
- **Measures:** Pairwise human preference. Current categories and ranking
  methodology should be checked on the site.
- **Scoring:** Blinded votes are aggregated with a paired-comparison rating
  method. The current FAQ describes a Bradley-Terry approach.
- **Setup to verify:** Leaderboard category, date window, model alias and
  version, voting filters, style controls, and confidence or uncertainty
  presentation.
- **Important limit:** **Preference is not the same construct as factual
  correctness, structured-output reliability, policy compliance, or task
  success in a specialized workflow.**
- **Reviewed:** 2026-08-06.

### Artificial Analysis

- **Status:** Maintained comparison service with changing methodology.
- **Official sources:** [Model comparison site](https://artificialanalysis.ai/),
  [intelligence methodology](https://artificialanalysis.ai/methodology/intelligence-benchmarking),
  and [performance methodology](https://artificialanalysis.ai/methodology/performance-benchmarking).
- **Helps answer:** How do current model APIs compare across selected
  capability evaluations, price, output speed, and latency measurements?
- **Measures:** A maintained composite capability index plus separate
  operational measures. The included evaluations and weights can change.
- **Scoring:** The capability index combines named evaluations. Operational
  methodology defines measurements such as time to first token and output
  speed.
- **Setup to verify:** Index version and weights, reasoning setting, provider
  route, price assumptions, cache assumptions, date, and geographic test
  conditions.
- **Important limit:** **A composite index is not a decision itself. It does not
  establish local compliance, account availability, or quality on a specialized
  task.**
- **Reviewed:** 2026-08-06.

## Coding and software work

### SWE-bench Verified

- **Status:** Human-validated software benchmark subset with maintained harness.
- **Official sources:** [Verified benchmark](https://www.swebench.com/verified.html),
  [main site](https://www.swebench.com/), and
  [repository](https://github.com/SWE-bench/SWE-bench).
- **Helps answer:** Can a software agent resolve real repository issues and
  produce patches that pass repository tests?
- **Measures:** A human-validated subset of 500 SWE-bench tasks from open
  source repositories.
- **Scoring:** Repository-specific tests and the evaluation harness determine
  whether the issue is resolved.
- **Setup to verify:** Agent harness, model version, tools, retrieval,
  trajectory budget, retry or sampling policy, container version, and
  submission status.
- **Important limit:** **A score reflects the agent system and repository
  distribution, not raw base-model coding ability in a different environment.**
- **Reviewed:** 2026-08-06.

### LiveCodeBench

- **Status:** Continuously refreshed executable coding benchmark.
- **Official sources:** [Official site](https://livecodebench.github.io/),
  [repository](https://github.com/LiveCodeBench/LiveCodeBench), and
  [paper](https://arxiv.org/abs/2403.07974).
- **Helps answer:** How do models perform on recently released coding problems
  with executable evaluation?
- **Measures:** Code-related capabilities using continuously collected
  contest problems, with current scenarios defined by the benchmark release.
- **Scoring:** Execution against tests, with task-specific metrics.
- **Setup to verify:** Release cutoff, scenario, language, prompt, model date,
  sampling, and pass-at-k or other reported metric.
- **Important limit:** **It does not cover repository navigation, long-horizon
  software engineering, or a particular coding agent interface unless that is
  explicitly part of the setup.**
- **Reviewed:** 2026-08-06.

## Function calling and agents

### Berkeley Function Calling Leaderboard, BFCL

- **Status:** Versioned function-calling benchmark and leaderboard.
- **Official sources:** [Current leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)
  and [BFCL paper](https://proceedings.mlr.press/v267/patil25a.html).
- **Helps answer:** Can a model select and format function calls across simple,
  parallel, multiple, and agentic tool-use settings?
- **Measures:** Function-calling behavior across a versioned suite. The live
  site should be checked for the current BFCL version and categories.
- **Scoring:** Structured and executable checks appropriate to each category.
- **Setup to verify:** BFCL version, native function-calling interface,
  provider mode, prompt, model version, and whether results are official or
  self-submitted.
- **Important limit:** **A function-calling score does not establish reliability
  with the user's tool schemas, permissions, side effects, recovery logic, or
  orchestration design.**
- **Reviewed:** 2026-08-06.

### WebArena

- **Status:** End-to-end agent benchmark in realistic simulated web environments.
- **Official sources:** [WebArena site](https://webarena.dev/) and
  [paper](https://arxiv.org/abs/2307.13854).
- **Helps answer:** Can an autonomous agent complete realistic web tasks in
  reproducible self-hosted environments?
- **Measures:** End-to-end tasks across simulated websites with realistic
  structure and state.
- **Scoring:** Functional correctness based on the resulting environment state
  and task-specific evaluators.
- **Setup to verify:** Agent scaffolding, browser tools, model version,
  demonstrations, action budget, observation format, and environment version.
- **Important limit:** **Base-model quality is not the same as agent quality, and
  the safety profile on a simulated site does not transfer to live websites with
  real permissions.**
- **Reviewed:** 2026-08-06.

### GAIA

- **Status:** General-assistant benchmark with varying task difficulty and tracks.
- **Official sources:** [Paper](https://arxiv.org/abs/2311.12983) and
  [official leaderboard](https://huggingface.co/spaces/gaia-benchmark/leaderboard).
- **Helps answer:** Can a general assistant solve questions that require
  reasoning, multimodal inputs, browsing, coding, or tool use?
- **Measures:** Real-world questions divided by difficulty, with short
  unambiguous answers.
- **Scoring:** Answer correctness under the benchmark protocol.
- **Setup to verify:** Agent and tool stack, model version, access to web and
  files, action budget, submission track, and potential data exposure.
- **Important limit:** **The underlying model is not necessarily the sole source
  of the result, and the benchmark does not prove the system is safe or
  authorized for a specific workflow.**
- **Reviewed:** 2026-08-06.

### tau-bench

- **Status:** Tool-agent benchmark with domain policy and user-simulator tasks.
- **Official sources:** [Current project site](https://taubench.com/),
  [repository](https://github.com/sierra-research/tau2-bench), and
  [original paper](https://arxiv.org/abs/2406.12045).
- **Helps answer:** Can a tool agent interact with a simulated user while
  following domain policy and reaching the correct task state?
- **Measures:** Multi-turn tool-agent-user behavior in domain environments.
- **Scoring:** Task and policy outcomes in the simulated environment, subject
  to the current benchmark version.
- **Setup to verify:** Benchmark generation, domain, user simulator, agent
  prompt, tools, model versions for both agent and simulator, and pass-k
  treatment.
- **Important limit:** **A simulated benchmark does not prove production
  reliability with real users, unseen policies, or irreversible tools.**
- **Reviewed:** 2026-08-06.

## Long context

### LongBench v2

- **Status:** Long-context multi-task benchmark with published protocol.
- **Official sources:** [Official site](https://longbench2.github.io/) and
  [ACL 2025 paper](https://aclanthology.org/2025.acl-long.183/).
- **Helps answer:** Can a model reason over long single documents,
  multi-document collections, dialogue, code repositories, and structured
  data?
- **Measures:** 503 multiple-choice questions across six long-context task
  categories, with context lengths spanning thousands to very large inputs.
- **Scoring:** Multiple-choice accuracy, with protocols described in the
  paper.
- **Setup to verify:** Direct or retrieval setting, context truncation,
  reasoning prompt, model context limit, and model version.
- **Important limit:** **A long-context score does not guarantee reliable
  citation, complete PDF transport, custom extraction, or performance at one
  exact production length.**
- **Reviewed:** 2026-08-06.

## Documents and multimodal reasoning

### DocVQA

- **Status:** Document visual question answering benchmark and challenge set.
- **Official sources:** [DocVQA site](https://www.docvqa.org/) and
  [dataset paper](https://arxiv.org/abs/2007.00398).
- **Helps answer:** Can a system answer questions from document images using
  text and layout information?
- **Measures:** Visual question answering on document images. The original
  dataset contains roughly 50,000 questions over about 12,000 images.
- **Scoring:** Answer matching using the challenge protocol and metrics.
- **Setup to verify:** Task track, OCR policy, image resolution, prompt, model
  version, and any external text extraction.
- **Important limit:** **Document-image accuracy does not cover native multi-page
  PDF support, long-document retrieval, complete field extraction, or a custom
  schema.**
- **Reviewed:** 2026-08-06.

### MMMU-Pro

- **Status:** Challenging multimodal benchmark with multiple settings.
- **Official sources:** [ACL 2025 paper](https://aclanthology.org/2025.acl-long.736/)
  and [official repository](https://github.com/MMMU-Benchmark/MMMU).
- **Helps answer:** Can a multimodal model solve discipline-specific questions
  where visual information is intended to matter?
- **Measures:** A strengthened version of MMMU that filters questions solvable
  from text alone and adds more challenging answer formats and vision-only
  variants.
- **Scoring:** Accuracy under the selected MMMU-Pro setting.
- **Setup to verify:** Standard, vision-only, or other setting; prompt;
  response format; image preprocessing; and model version.
- **Important limit:** **It does not establish operational document extraction,
  OCR completeness, specialist image safety, or PDF API behavior.**
- **Reviewed:** 2026-08-06.

## Embeddings and retrieval

### MTEB and MMTEB

- **Status:** Major benchmark collection for embedding and representation quality.
- **Official sources:** [MTEB repository](https://github.com/embeddings-benchmark/mteb)
  and [MTEB paper](https://aclanthology.org/2023.eacl-main.148/).
- **Helps answer:** How do embedding models compare for classification,
  clustering, retrieval, reranking, similarity, summarization, and related
  representation tasks across datasets and languages?
- **Measures:** A broad, maintained collection of embedding evaluation tasks.
  The current repository includes the expanded multilingual and multimodal
  benchmark landscape.
- **Scoring:** Task-specific measures such as retrieval ranking, correlation,
  classification, or clustering metrics.
- **Setup to verify:** Task, dataset, language, query and document instruction,
  embedding dimensions, pooling, normalization, model revision, and use of a
  reranker.
- **Important limit:** **Embedding quality does not equal end-to-end retrieval
  quality or a generative model's reasoning ability.**
- **Reviewed:** 2026-08-06.

## Retrieval-augmented generation and factuality

### ARES

- **Status:** Research method for component-level retrieval-augmented evaluation.
- **Official sources:** [NAACL 2024 paper](https://aclanthology.org/2024.naacl-long.20/)
  and [repository](https://github.com/stanford-futuredata/ARES).
- **Helps answer:** How can a retrieval-augmented generation system be assessed
  separately for context relevance, answer faithfulness, and answer relevance?
- **Measures:** RAG evaluation using synthetic data, lightweight human labels,
  learned judges, and prediction-powered inference.
- **Scoring:** Component-level judgments with statistical estimation described
  in the paper.
- **Setup to verify:** Domain, generator, retriever, synthetic-data process,
  human label sample, judge model, and confidence procedure.
- **Important limit:** **Automated judges are not automatically calibrated for a
  new domain, and local validation is still required.**
- **Reviewed:** 2026-08-06.

### FACTS Grounding

- **Status:** Grounded factuality benchmark for long-form answers.
- **Official sources:** [Google DeepMind overview](https://deepmind.google/blog/facts-grounding-a-new-benchmark-for-evaluating-the-factuality-of-large-language-models/)
  and [paper](https://arxiv.org/abs/2501.03200).
- **Helps answer:** Does a long-form response remain factually supported by a
  supplied source document?
- **Measures:** Grounded long-form generation over public documents and user
  requests.
- **Scoring:** Automated evaluation protocol described in the paper, including
  eligibility and factuality assessment.
- **Setup to verify:** Dataset version, judge ensemble or evaluator version,
  response eligibility rules, model version, and prompt.
- **Important limit:** **Groundedness does not equal correctness without a source,
  retrieval quality, or performance on a specialized regulated corpus.**
- **Reviewed:** 2026-08-06.

### SimpleQA

- **Status:** Short factuality benchmark with clear-answer questions.
- **Official sources:** [OpenAI release](https://openai.com/index/introducing-simpleqa/)
  and [paper](https://arxiv.org/abs/2411.04368).
- **Helps answer:** How often does a model answer short, fact-seeking questions
  correctly, incorrectly, or without an answer?
- **Measures:** Short factual questions designed to have one clear answer.
- **Scoring:** Correct, incorrect, or not attempted under the published
  grader protocol.
- **Setup to verify:** Grader, model version, prompt, browsing or tool access,
  and contamination concerns.
- **Important limit:** **Short factuality scores do not cover long-form
  groundedness, domain expertise, or current facts beyond the dataset.**
- **Reviewed:** 2026-08-06.

## Multilingual evidence

### Global-MMLU

- **Status:** Multilingual knowledge benchmark with language-sensitive design.
- **Official source:** [Paper](https://arxiv.org/abs/2412.03304).
- **Helps answer:** How do multilingual knowledge evaluations change when
  translation quality and culturally sensitive content are treated explicitly?
- **Measures:** Multilingual MMLU-style questions with attention to translation
  artifacts and culturally sensitive versus culturally agnostic content.
- **Scoring:** Multiple-choice accuracy, with language and content categories.
- **Setup to verify:** Language, native or translated source, model version,
  prompt, and culturally sensitive subset.
- **Important limit:** **A multilingual average can hide the single language that
  matters, and it does not prove fluency or local task performance in every
  language variety.**
- **Reviewed:** 2026-08-06.

For multilingual retrieval and embeddings, also use MTEB or MMTEB. For any
required language, inspect that language separately. **A multilingual average can
hide the only language that matters.**

## Safety and risk

### MLCommons AILuminate

- **Status:** General-purpose AI safety benchmark with standardized hazard coverage.
- **Official sources:** [AILuminate](https://mlcommons.org/ailuminate/) and
  [method paper](https://arxiv.org/abs/2503.05731).
- **Helps answer:** How does a general-purpose AI system respond across a
  defined set of hazard categories under a standardized assessment?
- **Measures:** System responses across the benchmark's taxonomy and supported
  languages and modalities, subject to the current release.
- **Scoring:** Safety assessment protocol and grading defined by MLCommons.
- **Setup to verify:** Benchmark version, model and system settings, hazard
  category, language, grader, and deployment mode.
- **Important limit:** **Safety evaluation must be tied to the actual risks,
  permissions, and deployment context; it does not prove complete safety for a
  particular product, population, or regulatory setting.**
- **Reviewed:** 2026-08-06.

## How to use a live leaderboard

For a decision record, capture:

1. The URL and access date.
2. The exact leaderboard, category, or track.
3. Model and system identity.
4. Result and uncertainty.
5. Harness, prompt, tools, and budget.
6. Submission provenance.
7. Why the source is relevant.
8. Why it may not transfer.

If any of these are unavailable, lower the evidence-strength label.
**Missing information is not a reason to invent precision.**

Last source review: 2026-08-06.

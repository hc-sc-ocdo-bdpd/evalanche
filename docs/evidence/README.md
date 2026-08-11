# Research and evidence directory

This directory helps answer two questions:

1. What methods produce credible model-selection evidence?
2. Where can I find current evidence for the task I care about?

It is a curated guide, not a universal model database. It does not copy a
snapshot of every leaderboard or automatically turn public scores into a
candidate list.

Public evidence may cover any relevant model. Only models whose access has
been explicitly confirmed can enter an Evalanche local comparison or decision
policy.

## Start by need

| Need | Read |
| --- | --- |
| Understand validity, task transfer, contamination, uncertainty, and human evaluation | [Foundational papers and guidance](foundations.md) |
| Find a benchmark or leaderboard for a task | [Benchmarks and living evidence sources](benchmarks.md) |
| Choose an evaluation framework | [Evaluation frameworks](frameworks.md) |
| Check whether a curated source is current, stale, or superseded | [Evidence maintenance status](status.md) |
| Decide whether and how to use an LLM judge | [Evalanche LLM judge method](../llm_judges.md) |
| Apply the evidence to a decision | [Model selection handbook](../choosing_a_model.md) |

## Evidence map

No one source covers the full decision.

| Evidence type | Helps answer | Does not establish by itself |
| --- | --- | --- |
| Official provider documentation | Current route, modality, context, price, region, limits, and terms | Independent task quality |
| Public benchmark | Performance on a defined task and harness | Fit for a different workflow or current user access |
| Human-preference arena | Which responses raters prefer under that protocol | Factual correctness or task success in every domain |
| Operational measurement service | Observed price, latency, speed, and selected capability results | Compliance, availability, or local task quality |
| Research paper | Method, dataset, analysis, and limitations | Current production behavior after model or harness changes |
| Local representative evaluation | Fit for a specific task, route, prompt, and dataset | Universal performance outside that scope |
| Human review | Judgment under stated criteria and reviewer population | Objective truth without a valid protocol and agreement evidence |

## Source review standard

Every curated entry should make these fields recoverable:

- source and official URL;
- publication or release status;
- last reviewed date;
- current, stale, or superseded maintenance status;
- next review date and change that should trigger an earlier review;
- exact version scope or living-source scope covered by the note;
- question the source can help answer;
- construct and task actually measured;
- dataset and version;
- model or system unit being compared;
- prompt, tools, budget, and inference conditions that matter;
- scoring method;
- important limitation;
- conditions under which not to rely on the result.

When a leaderboard changes frequently, Evalanche links to the current source
and preserves the interpretation of its method. It does not treat one scraped
score table as evergreen guidance.

The machine-readable source of this maintenance metadata is
[`catalog.yaml`](catalog.yaml). Its generated [status page](status.md) puts
stale and superseded sources first.

## Publication status labels

| Label | Use |
| --- | --- |
| Peer-reviewed | Published through a peer-reviewed venue |
| Official standard or guidance | Issued by a standards or public body |
| Maintained official resource | Benchmark owner site, repository, or methodology |
| Preprint | Public research that has not completed peer review |
| Provider-declared | Claim or result published by a model or service provider |
| Local Evalanche result | Reproducible result tied to an Evalanche task and run record |

Status does not determine relevance. A peer-reviewed benchmark can still be a
poor match for the intended task. A current provider page can be the right
source for route availability while remaining a weak source for independent
quality claims.

## Maintenance status labels

| Label | Meaning |
| --- | --- |
| Current | Reviewed for the recorded version scope and not yet due for routine review |
| Stale | Its review date has passed or a known change requires reassessment |
| Superseded | Retained for history, but replaced or withdrawn and not suitable as current guidance |

Maintenance status is not an evidence-strength rating. A current source can
still be weak for a particular decision.

## A practical evidence stack

For most decisions, use the smallest set that covers the important questions:

1. Official documentation for availability and hard service constraints.
2. One or more task-relevant public sources for orientation.
3. A source that reports operational properties when those matter.
4. A small local comparison for unresolved task-specific questions.
5. Human or expert review when the output construct or consequences require it.

Adding more benchmarks does not automatically strengthen a claim. Additional
evidence helps only when it measures a relevant and distinct part of the
decision.

## Maintenance rules

- Prefer original papers, official benchmark sites, and maintained
  repositories.
- Mark preprints as preprints.
- Distinguish a base model from the full agent, retrieval system, or provider
  product that was evaluated.
- Never infer current model availability from a leaderboard entry.
- Record the exact date when live scores, prices, or provider limits are used
  in a decision.
- Preserve conflicting results and investigate setup differences.
- Treat a model alias as unstable unless the provider guarantees a fixed
  version.
- Recheck a source before a consequential decision.
- Add a source only when it answers a question not already covered well.

Run the offline status check before using the directory for a new decision:

```bash
python -m evalanche.cli evidence-status --root .
```

To update the committed snapshot after a human review:

1. Update the entry in `catalog.yaml`, including `last_reviewed`, `review_by`,
   `status`, `version_scope`, and `review_trigger`.
2. Update `status_report_as_of` to the review date.
3. Regenerate and check the snapshot:

```bash
python -m evalanche.cli evidence-status \
  --root . --as-of YYYY-MM-DD --write
python -m evalanche.cli evidence-status --root . --check
```

These commands read local files only. They do not contact source sites or
model providers. A person must still inspect changed sources and decide what
the change means.

Last directory review: 2026-08-06.

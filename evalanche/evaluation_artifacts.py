from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from evalanche import __version__
from evalanche.config import EvaluationConfig
from evalanche.metadata import sha256_file
from evalanche.judges.protocol import (
    judge_claim_block_reason,
    resolve_judge_evidence,
)
from evalanche.operational import summarize_operations
from evalanche.pricing import build_pricing_snapshot
from evalanche.selection import (
    build_recommendation_decision,
)
from evalanche.statistics import (
    CONFIDENCE_LEVEL,
    SIGNIFICANCE_LEVEL,
)


def _format_percent(value: Any) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.1%}"


def _format_decimal(value: Any) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.3f}"


def _format_p_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    numeric = float(value)
    if numeric < 0.001:
        return "<0.001"
    return f"{numeric:.3f}"


def _format_seconds(value: Any) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.2f} s"


def _format_integer(value: Any) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{int(value):,}"


def _format_cost(
    value: Any,
    coverage: Any,
    requests: Any,
) -> str:
    request_count = int(requests or 0)
    if request_count == 0:
        return "N/A"
    if value is None or pd.isna(value):
        coverage_text = _format_percent(coverage)
        return f"Unknown ({coverage_text} coverage)"
    return f"${float(value):.6f} USD"


def _format_usd(value: Any) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"${float(value):.6f} USD"


def _format_declared_bool(value: Any) -> str:
    if value is None or pd.isna(value):
        return "Not declared"
    return "Yes" if bool(value) else "No"


def _numeric_mean_or_none(series: pd.Series) -> float | None:
    value = pd.to_numeric(series, errors="coerce").mean()
    return None if pd.isna(value) else float(value)


def _observed_pricing_entries(
    results: pd.DataFrame,
    prefix: str,
) -> list[dict[str, Any]]:
    source_columns = {
        "pricing_id": f"{prefix}_pricing_id",
        "model": f"{prefix}_pricing_model",
        "currency": f"{prefix}_pricing_currency",
        "input_per_million_tokens": (
            f"{prefix}_pricing_input_per_million_tokens"
        ),
        "cached_input_per_million_tokens": (
            f"{prefix}_pricing_cached_input_per_million_tokens"
        ),
        "output_per_million_tokens": (
            f"{prefix}_pricing_output_per_million_tokens"
        ),
        "effective_date": f"{prefix}_pricing_effective_date",
        "source": f"{prefix}_pricing_source",
        "catalog_version": f"{prefix}_pricing_catalog_version",
        "catalog_sha256": f"{prefix}_pricing_catalog_sha256",
    }
    if source_columns["pricing_id"] not in results.columns:
        return []

    available = {
        key: column
        for key, column in source_columns.items()
        if column in results.columns
    }
    observed = results[list(available.values())].copy()
    observed = observed[
        observed[source_columns["pricing_id"]].notna()
    ].drop_duplicates()

    records: list[dict[str, Any]] = []
    for raw in observed.to_dict(orient="records"):
        record = {
            key: raw[column]
            for key, column in available.items()
        }
        records.append(
            {
                key: None if pd.isna(value) else value
                for key, value in record.items()
            }
        )
    return records


def _escape_markdown(
    value: Any,
    *,
    max_length: int | None = None,
) -> str:
    if value is None or pd.isna(value):
        text = ""
    else:
        text = str(value)

    text = " ".join(text.split()).replace("|", "\\|")

    if max_length is not None and len(text) > max_length:
        return text[: max_length - 3] + "..."

    return text


def _leaderboard_markdown(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "_No model summary available._"

    columns = [
        "Rank",
        "Model",
        "Passed",
        "Unscored",
        "Pass rate",
        "95% pass-rate interval",
        "Average score",
        "Deterministic pass rate",
        "Judge pass rate",
        "Generation errors",
        "Judge errors",
    ]
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]

    for _, row in summary.iterrows():
        rows.append(
            "| "
            + " | ".join(
                [
                    str(row["rank"]),
                    _escape_markdown(row["model_name"]),
                    f"{row['passed_cases']}/{row['scored_cases']}",
                    str(row["unscored_cases"]),
                    _format_percent(row["pass_rate"]),
                    (
                        f"{_format_percent(row['pass_rate_ci_low'])}-"
                        f"{_format_percent(row['pass_rate_ci_high'])}"
                    ),
                    _format_decimal(row["average_score"]),
                    _format_percent(
                        row["deterministic_pass_rate"]
                    ),
                    _format_percent(row["judge_pass_rate"]),
                    str(row["generation_errors"]),
                    str(row["judge_errors"]),
                ]
            )
            + " |"
        )

    return "\n".join(rows)


def _pairwise_markdown(comparisons: pd.DataFrame) -> str:
    if comparisons.empty:
        return (
            "_At least two models are required for paired model "
            "comparisons._"
        )

    columns = [
        "Model A",
        "Model B",
        "Paired cases",
        "Excluded cases",
        "Pass-rate difference (A - B)",
        "A-only passes",
        "B-only passes",
        "Holm-adjusted p-value",
        "Clear winner",
    ]
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]

    for _, row in comparisons.iterrows():
        winner = row["clear_winner"]
        winner_text = (
            _escape_markdown(winner)
            if winner is not None and not pd.isna(winner)
            else "No clear difference"
        )
        rows.append(
            "| "
            + " | ".join(
                [
                    _escape_markdown(row["model_a"]),
                    _escape_markdown(row["model_b"]),
                    str(row["paired_cases"]),
                    str(row["excluded_cases"]),
                    (
                        f"{float(row['pass_rate_difference']):+.1%}"
                        if pd.notna(row["pass_rate_difference"])
                        else "N/A"
                    ),
                    str(row["model_a_only_passed"]),
                    str(row["model_b_only_passed"]),
                    _format_p_value(
                        row["holm_adjusted_p_value"]
                    ),
                    winner_text,
                ]
            )
            + " |"
        )

    return "\n".join(rows)


def _stage_operations_markdown(
    summary: pd.DataFrame,
    prefix: str,
) -> str:
    if (
        summary.empty
        or f"{prefix}_requests" not in summary.columns
        or int(summary[f"{prefix}_requests"].sum()) == 0
    ):
        return f"_No {prefix} requests were recorded._"

    columns = [
        "Model",
        "Requests",
        "Failures",
        "Failure rate",
        "Average latency",
        "p95 latency",
        "Input tokens",
        "Output tokens",
        "Total tokens",
        "Cost",
        "Cost source",
    ]
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]

    for _, row in summary.iterrows():
        requests = row[f"{prefix}_requests"]
        rows.append(
            "| "
            + " | ".join(
                [
                    _escape_markdown(row["model_name"]),
                    str(requests),
                    str(row[f"{prefix}_errors"]),
                    _format_percent(
                        row[f"{prefix}_failure_rate"]
                    ),
                    _format_seconds(
                        row[f"{prefix}_average_seconds"]
                    ),
                    _format_seconds(
                        row[f"{prefix}_p95_seconds"]
                    ),
                    _format_integer(
                        row[f"{prefix}_prompt_tokens"]
                    ),
                    _format_integer(
                        row[f"{prefix}_completion_tokens"]
                    ),
                    _format_integer(
                        row[f"{prefix}_total_tokens"]
                    ),
                    _format_cost(
                        row[f"{prefix}_cost_usd"],
                        row[f"{prefix}_cost_coverage"],
                        requests,
                    ),
                    _escape_markdown(
                        row[f"{prefix}_cost_sources"] or "Unknown"
                    ),
                ]
            )
            + " |"
        )

    return "\n".join(rows)


def _selection_policy_markdown(config: EvaluationConfig) -> str:
    selection = config.selection
    if not selection.enabled:
        return (
            "_The optional decision policy is disabled. The report presents "
            "comparison evidence without selecting a model._"
        )

    constraints = selection.constraints
    required_capabilities = (
        ", ".join(constraints.required_capabilities)
        if constraints.required_capabilities
        else "None"
    )

    def optional_percent(value: Any) -> str:
        return _format_percent(value) if value is not None else "Not set"

    def optional_seconds(value: Any) -> str:
        return _format_seconds(value) if value is not None else "Not set"

    def optional_cost(value: Any) -> str:
        return (
            f"${float(value):.6f} USD"
            if value is not None
            else "Not set"
        )

    return "\n".join(
        [
            "This optional policy was explicitly enabled. Hard requirements "
            "are applied before weighted scoring. A model "
            "with missing required evidence remains unknown rather than "
            "passing or failing the requirement.",
            "",
            f"- Minimum pass rate: "
            f"{optional_percent(constraints.minimum_pass_rate)}",
            f"- Minimum 95% pass-rate lower bound: "
            f"{optional_percent(constraints.minimum_pass_rate_ci_low)}",
            f"- Maximum average request cost: "
            f"{optional_cost(constraints.maximum_average_cost_usd)}",
            f"- Maximum p95 generation latency: "
            f"{optional_seconds(constraints.maximum_p95_latency_seconds)}",
            f"- Maximum generation failure rate: "
            f"{optional_percent(constraints.maximum_generation_failure_rate)}",
            f"- Required capabilities: {required_capabilities}",
            "- Require a model profile for every candidate: "
            f"{selection.constraints.require_model_profile}",
            f"- Minimum decision-score margin: "
            f"{selection.minimum_score_margin:.3f}",
            "- Weights: "
            f"quality={selection.weights.quality:g}, "
            f"cost={selection.weights.cost:g}, "
            f"latency={selection.weights.latency:g}, "
            f"reliability={selection.weights.reliability:g}",
        ]
    )


def _selection_markdown(selection: pd.DataFrame) -> str:
    if selection.empty:
        return "_No constraint-aware model-selection table was produced._"

    columns = [
        "Model",
        "Status",
        "Rank",
        "Decision score",
        "Pass rate",
        "Average cost/request",
        "Cost source",
        "p95 latency",
        "Generation failure rate",
        "Available",
        "Capabilities",
        "Selected by policy",
        "Reasons",
    ]
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]

    for _, row in selection.iterrows():
        reasons = "; ".join(
            value
            for value in (
                str(row["constraint_failures"]).strip(),
                str(row["missing_evidence"]).strip(),
            )
            if value and value.lower() != "nan"
        )
        rows.append(
            "| "
            + " | ".join(
                [
                    _escape_markdown(row["model_name"]),
                    _escape_markdown(row["selection_status"]),
                    _format_integer(row["selection_rank"]),
                    _format_decimal(row["decision_score"]),
                    _format_percent(row["pass_rate"]),
                    _format_usd(row["generation_average_cost_usd"]),
                    _escape_markdown(
                        row["generation_cost_sources"] or "Unknown"
                    ),
                    _format_seconds(row["generation_p95_seconds"]),
                    _format_percent(row["generation_failure_rate"]),
                    _format_declared_bool(row["available"]),
                    _escape_markdown(
                        row["capabilities"] or "None declared"
                    ),
                    "Yes" if bool(row["recommended"]) else "No",
                    _escape_markdown(reasons or "None"),
                ]
            )
            + " |"
        )

    return "\n".join(rows)


def _failures_markdown(
    results: pd.DataFrame,
    max_rows: int = 10,
) -> str:
    failures = results[results["final_passed"] == False]

    if failures.empty:
        return "_No failed cases._"

    columns = ["Case", "Model", "Type", "Source", "Reason"]
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]

    for _, row in failures.head(max_rows).iterrows():
        rows.append(
            "| "
            + " | ".join(
                [
                    _escape_markdown(row["case_id"]),
                    _escape_markdown(row["model_name"]),
                    _escape_markdown(row["evaluation_type"]),
                    _escape_markdown(row["evaluation_source"]),
                    _escape_markdown(
                        row["evaluation_reason"],
                        max_length=160,
                    ),
                ]
            )
            + " |"
        )

    if len(failures) > max_rows:
        rows.append(
            f"\n_Showing {max_rows} of {len(failures)} failed rows._"
        )

    return "\n".join(rows)


def _quality_recommendation_text(
    summary: pd.DataFrame,
    details: dict[str, Any],
) -> str:
    if summary.empty:
        return (
            "**Comparison outcome:** Not available. "
            "No model results were produced."
        )

    if details["status"] == "incomplete_evaluation":
        return (
            "**Comparison outcome:** Not available.\n\n"
            f"{details['unscored_cases']} model result(s) could not be "
            "scored because the judge failed. Those rows were kept as "
            "operational failures and were not counted as candidate-model "
            "failures. Rerun or recover the failed judge calls before "
            "selecting a model."
        )

    if details["status"] == "insufficient_judge_validation":
        leader = details.get("observed_leader")
        observed = (
            f" `{leader}` had the highest observed pass rate."
            if leader is not None
            else ""
        )
        return (
            "**Comparison outcome:** Observed results only.\n\n"
            f"{details['evidence_block_reason']}{observed} The report does "
            "not treat this as an evidence-supported leader or select a "
            "model."
        )

    if len(summary) == 1:
        model = summary.iloc[0]
        return (
            "**Comparison outcome:** Not available.\n\n"
            f"Only `{model['model_name']}` was evaluated. It "
            f"passed {model['passed_cases']} of "
            f"{model['cases']} cases "
            f"({_format_percent(model['pass_rate'])}; "
            f"95% interval "
            f"{_format_percent(model['pass_rate_ci_low'])}-"
            f"{_format_percent(model['pass_rate_ci_high'])}), "
            "but at least two candidate models are required "
            "for a model comparison."
        )

    if details["status"] == "observed_tie":
        names = ", ".join(
            f"`{name}`"
            for name in details["top_ranked_models"]
        )
        return (
            "**Comparison outcome:** "
            "No clear winner yet.\n\n"
            f"The top observed pass rate was tied by {names}. "
            "These models are tied on the primary ranking "
            "measure, overall pass rate. The evaluation does "
            "not support selecting one of them without "
            "additional evidence."
        )

    observed_leader = str(details["observed_leader"])
    leader = summary[
        summary["model_name"] == observed_leader
    ].iloc[0]

    if details["status"] == "insufficient_evidence":
        unclear_names = ", ".join(
            f"`{name}`"
            for name in details["not_distinguished_from"]
        )
        return (
            "**Comparison outcome:** "
            "No clear winner yet.\n\n"
            f"`{observed_leader}` had the highest observed "
            f"result, passing {leader['passed_cases']} of "
            f"{leader['cases']} cases "
            f"({_format_percent(leader['pass_rate'])}). "
            "However, the paired evidence did not clearly "
            f"distinguish it from {unclear_names} after "
            "correcting for multiple comparisons. More "
            "representative cases or operational constraints "
            "are needed to choose between them."
        )

    return (
        f"**Evidence-supported result:** "
        f"`{observed_leader}`\n\n"
        f"It passed {leader['passed_cases']} of "
        f"{leader['cases']} cases "
        f"({_format_percent(leader['pass_rate'])}) and "
        "clearly outperformed each other evaluated model in "
        "the paired pass/fail comparisons. This is strong task-specific "
        "evidence, but it does not select a model automatically. Access, "
        "hard requirements, and operational tradeoffs still apply."
    )


def _selection_reasons(
    selection: pd.DataFrame,
    model_names: list[str],
) -> str:
    selected = selection[
        selection["model_name"].astype(str).isin(model_names)
    ]
    reasons: list[str] = []
    for _, row in selected.iterrows():
        detail = str(row["constraint_failures"]).strip()
        missing = str(row["missing_evidence"]).strip()
        combined = "; ".join(
            value
            for value in (detail, missing)
            if value and value.lower() != "nan"
        )
        reasons.append(
            f"`{row['model_name']}`: {combined or 'no reason recorded'}"
        )
    return "; ".join(reasons)


def _constraint_recommendation_text(
    config: EvaluationConfig,
    summary: pd.DataFrame,
    selection: pd.DataFrame,
    decision: dict[str, Any],
) -> str:
    status = decision["status"]

    if status in {"no_results", "incomplete_evaluation"}:
        return _quality_recommendation_text(summary, decision)

    if status == "insufficient_judge_validation":
        return (
            "**Policy decision:** Not available.\n\n"
            f"{decision['evidence_block_reason']} The configured policy "
            "cannot select a model from judge evidence below that level."
        )

    if status == "single_model":
        model_name = decision.get("observed_leader")
        matching = selection[
            selection["model_name"].astype(str) == str(model_name)
        ]
        policy_status = (
            str(matching.iloc[0]["selection_status"])
            if not matching.empty
            else "not assessed"
        )
        return (
            "**Policy decision:** Not available.\n\n"
            f"Only `{model_name}` was evaluated. Its policy status was "
            f"`{policy_status}`, but at least two candidate models are "
            "required for a comparative selection."
        )

    if status == "insufficient_operational_data":
        unknown = decision["unknown_models"]
        return (
            "**Policy decision:** Not available.\n\n"
            "Required evidence is missing for one or more candidates: "
            f"{_selection_reasons(selection, unknown)}. Missing evidence "
            "is not treated as a passed requirement."
        )

    if status == "no_eligible_models":
        ineligible = decision["ineligible_models"]
        return (
            "**Policy decision:** No eligible model.\n\n"
            "Every evaluated candidate failed at least one configured "
            f"requirement: {_selection_reasons(selection, ineligible)}."
        )

    if status == "policy_tie":
        names = ", ".join(
            f"`{name}`" for name in decision["top_ranked_models"]
        )
        return (
            "**Policy decision:** No clear selection.\n\n"
            f"{names} tied on the configured weighted decision score."
        )

    if status == "insufficient_policy_margin":
        leader = decision["top_ranked_models"][0]
        return (
            "**Policy decision:** No clear selection.\n\n"
            f"`{leader}` had the highest decision score, but its margin "
            f"of {decision['score_margin']:.3f} was below the configured "
            f"minimum of {config.selection.minimum_score_margin:.3f}."
        )

    if status == "insufficient_quality_evidence":
        leader = decision["top_ranked_models"][0]
        return (
            "**Policy decision:** No clear selection.\n\n"
            f"`{leader}` had the highest quality-only decision score, but "
            "the corrected paired evidence did not clearly distinguish it "
            "from every other eligible model."
        )

    recommended = str(decision["recommended_model"])
    score = float(decision["top_selection_score"])
    if status == "sole_eligible_model":
        explanation = (
            "It was the only evaluated candidate that satisfied every "
            "configured requirement with complete required evidence."
        )
    elif status == "quality_leader":
        explanation = (
            "It satisfied every configured requirement and the corrected "
            "paired quality evidence distinguished it from every other "
            "eligible model."
        )
    else:
        explanation = (
            "It satisfied every configured requirement and had the highest "
            "weighted decision score among eligible models"
            f", with a margin of {decision['score_margin']:.3f}."
        )

    return (
        f"**Policy-selected model:** `{recommended}`\n\n"
        f"Its configured decision score was {score:.3f}. {explanation} "
        "This is an opt-in, task-specific policy result over explicitly "
        "profiled candidates, not a claim that it is best in general."
    )


def _recommendation_text(
    config: EvaluationConfig,
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    selection: pd.DataFrame,
    *,
    evidence_block_reason: str | None = None,
) -> str:
    decision = build_recommendation_decision(
        summary,
        comparisons,
        selection,
        config.selection,
        evidence_block_reason=evidence_block_reason,
    )
    if decision["mode"] == "quality_only":
        return _quality_recommendation_text(summary, decision)
    return _constraint_recommendation_text(
        config,
        summary,
        selection,
        decision,
    )


def build_evaluation_report(
    *,
    config: EvaluationConfig,
    results: pd.DataFrame,
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    selection: pd.DataFrame,
    case_results_path: str | Path,
    summary_path: str | Path,
    comparison_path: str | Path,
    selection_path: str | Path,
    metadata_path: str | Path,
) -> str:
    source_counts = results[
        "evaluation_source"
    ].value_counts()
    deterministic_rows = int(
        source_counts.get("deterministic", 0)
    )
    judge_rows = int(
        source_counts.get("llm_judge", 0)
    )
    generation_errors = int(
        source_counts.get("generation_error", 0)
    )
    judge_errors = int(
        source_counts.get("judge_error", 0)
    )
    judge_evidence = resolve_judge_evidence(config)
    evidence_block_reason = judge_claim_block_reason(results, config)
    validation_report = (
        f"`{judge_evidence.validation_report_path}`"
        if judge_evidence.validation_report_path is not None
        else "Not attached"
    )

    return f"""# Evalanche Model Comparison

## Comparison outcome

{_recommendation_text(config, summary, comparisons, selection, evidence_block_reason=evidence_block_reason)}

## Evaluation Scope

- **Run:** `{config.run.name}`
- **Task:** `{config.task.name}`
- **Rows evaluated:** {len(results)}
- **Unique cases:** {results['case_id'].nunique()}
- **Models:** {results['model_name'].nunique()}
- **Deterministic rows:** {deterministic_rows}
- **LLM-judged rows:** {judge_rows}
- **Generation errors:** {generation_errors}
- **Judge errors:** {judge_errors}
- **Judge model:** `{config.judge.model}`
- **Judge pass threshold:** {_format_percent(config.scoring.pass_threshold)}
- **Judge evidence level:** `{judge_evidence.level}`
- **Minimum judge level for selection:** `{config.judge.minimum_validation_level_for_selection}`
- **Judge validation report:** {validation_report}

## Model Leaderboard

The primary rank is based on overall case pass rate.

{_leaderboard_markdown(summary)}

The intervals show uncertainty in each pass rate. They should not be used
alone to compare models because every model answered the same cases.

## Pairwise Evidence

The exact paired test compares where one model passed and the other failed.
Holm correction limits false positives when several model pairs are tested.

{_pairwise_markdown(comparisons)}

## Model Selection Policy

{_selection_policy_markdown(config)}

### Eligibility and Decision Scores

{_selection_markdown(selection)}

## Operational Performance

Latency is measured end to end for each request, including retries and retry
waits. Average and p95 values describe this run, not guaranteed production
performance.

### Candidate Generation

{_stage_operations_markdown(summary, "generation")}

### LLM Judge

{_stage_operations_markdown(summary, "judge")}

Token totals include every response observed during retries. Configured
endpoint rates are the primary cost source when the required token usage is
complete. Complete LiteLLM response-cost metadata is retained separately and
used as a fallback. Otherwise the cost remains unknown.

## Failed Cases

{_failures_markdown(results)}

## How Scores Were Selected

- Exact and JSON cases use deterministic evaluation as the authoritative result.
- Open-ended judge cases use the configured LLM judge.
- Judge evidence below the configured validation level remains descriptive and
  cannot produce an evidence-supported leader or policy-selected model.
- Generation errors receive a score of zero and are not sent to the judge.
- Judge errors remain unscored and prevent a complete comparative conclusion.
- JSON partial-field scores are diagnostic; only a full match after the
  configured deterministic canonicalization passes.

## Evidence Files

- Case results: `{case_results_path}`
- Model summary: `{summary_path}`
- Pairwise comparisons: `{comparison_path}`
- Model selection: `{selection_path}`
- Run metadata: `{metadata_path}`

## Limitations

- Results apply only to this dataset, task, prompts, models, and configuration.
- LLM-judge outputs are evaluation signals and should be calibrated against
  human review for important uses.
- A judge validation level applies only to the exact task, rubric, prompt,
  judge model, version, settings, and validation data in its matched contract.
- Statistical intervals cover case-sampling uncertainty only; they do not
  cover prompt, generation, or judge variability.
- The statistical methods assume cases are representative and independent.
  Related or repeated cases require grouped analysis.
- Operational measurements reflect this run's network path, provider state,
  retries, and sequential execution. They are not production service-level
  guarantees.
- Missing token or cost metadata is reported as unknown, not zero.
- Configured token costs are estimates from declared rates and recorded usage,
  not reconciled provider invoices.
- Selection weights, thresholds, and capabilities express configured policy
  choices. They are not empirical facts and should be reviewed by the client.
- Weighted operational components use point estimates from this run and do not
  include uncertainty intervals.
- A small or unrepresentative test set can produce unstable rankings.
"""


def save_evaluation_report(
    *,
    config: EvaluationConfig,
    results: pd.DataFrame,
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    selection: pd.DataFrame,
    case_results_path: str | Path,
    summary_path: str | Path,
    comparison_path: str | Path,
    selection_path: str | Path,
    metadata_path: str | Path,
) -> Path:
    case_results_path = Path(case_results_path)
    report_path = case_results_path.with_name(
        case_results_path.stem
        + "_comparison.md"
    )

    report = build_evaluation_report(
        config=config,
        results=results,
        summary=summary,
        comparisons=comparisons,
        selection=selection,
        case_results_path=case_results_path,
        summary_path=summary_path,
        comparison_path=comparison_path,
        selection_path=selection_path,
        metadata_path=metadata_path,
    )
    report_path.write_text(
        report,
        encoding="utf-8",
    )

    return report_path


def build_evaluation_metadata(
    *,
    config_path: str | Path | None,
    config: EvaluationConfig,
    cases: pd.DataFrame,
    results: pd.DataFrame,
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    selection: pd.DataFrame,
    case_results_path: str | Path,
    summary_path: str | Path,
    comparison_path: str | Path,
    selection_path: str | Path,
    report_path: str | Path,
) -> dict[str, Any]:
    source_counts = results[
        "evaluation_source"
    ].value_counts()
    decision = build_recommendation_decision(
        summary,
        comparisons,
        selection,
        config.selection,
        evidence_block_reason=judge_claim_block_reason(results, config),
    )
    judge_evidence = resolve_judge_evidence(config)
    operations = summarize_operations(results)
    pricing_snapshot = build_pricing_snapshot(
        path=config.endpoint_pricing_path,
        references=[
            {
                "role": "judge",
                "model": config.judge.model,
                "pricing_id": config.judge.pricing_id,
            }
        ],
    )
    pricing_snapshot["observed_generation_endpoints"] = (
        _observed_pricing_entries(results, "generation")
    )
    pricing_snapshot["observed_judge_endpoints"] = (
        _observed_pricing_entries(results, "judge")
    )

    return {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "evalanche_version": __version__,
        "runtime": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "run": {
            "name": config.run.name,
            "config_path": (
                str(config_path)
                if config_path is not None
                else None
            ),
            "input_path": str(config.run.input_path),
            "case_results_path": str(
                case_results_path
            ),
            "model_summary_path": str(summary_path),
            "pairwise_comparisons_path": str(
                comparison_path
            ),
            "model_selection_path": str(selection_path),
            "comparison_report_path": str(report_path),
            "endpoint_pricing_path": (
                str(config.endpoint_pricing_path)
                if config.endpoint_pricing_path is not None
                else None
            ),
        },
        "hashes": {
            "config_sha256": (
                sha256_file(config_path)
                if config_path is not None
                else None
            ),
            "input_sha256": sha256_file(
                config.run.input_path
            ),
            "case_results_sha256": sha256_file(
                case_results_path
            ),
            "model_summary_sha256": sha256_file(
                summary_path
            ),
            "pairwise_comparisons_sha256": sha256_file(
                comparison_path
            ),
            "model_selection_sha256": sha256_file(
                selection_path
            ),
            "comparison_report_sha256": sha256_file(
                report_path
            ),
            "endpoint_pricing_sha256": pricing_snapshot[
                "catalog_sha256"
            ],
        },
        "task": {
            "name": config.task.name,
            "description": config.task.description,
            "measured_construct": config.task.measured_construct,
            "intended_use": config.task.intended_use,
            "languages": config.task.languages,
        },
        "routing": {
            "exact_and_json": "deterministic",
            "judge": "llm_judge",
            "generation_error": "automatic_failure",
        },
        "metrics": config.metrics.model_dump(mode="json"),
        "judge": {
            "model": config.judge.model,
            "variant_id": config.judge.variant_id,
            "provider_model_version": (
                config.judge.provider_model_version
            ),
            "temperature": config.judge.temperature,
            "max_retries": config.judge.max_retries,
            "continue_on_error": (
                config.judge.continue_on_error
            ),
            "pricing_id": config.judge.pricing_id,
            "prompt_id": config.judge.prompt_id,
            "prompt_version": config.judge.prompt_version,
            "rubric_id": config.judge.rubric_id,
            "rubric_version": config.judge.rubric_version,
            "candidate_identity_blinded": (
                config.judge.candidate_identity_blinded
            ),
            "reference_mode": config.judge.reference_mode,
            "minimum_validation_level_for_selection": (
                config.judge.minimum_validation_level_for_selection
            ),
            "score_min": config.scoring.score_min,
            "score_max": config.scoring.score_max,
            "pass_threshold": (
                config.scoring.pass_threshold
            ),
            "criteria": [
                {
                    "name": criterion.name,
                    "weight": criterion.weight,
                    "description": (
                        criterion.description
                    ),
                }
                for criterion in config.criteria
            ],
            "validation_evidence": judge_evidence.as_dict(),
        },
        "input_data": {
            "rows": int(len(cases)),
            "unique_cases": int(
                cases["case_id"].nunique()
            ),
            "models": sorted(
                cases["model_name"]
                .astype(str)
                .unique()
                .tolist()
            ),
            "evaluation_types": sorted(
                cases["evaluation_type"]
                .astype(str)
                .unique()
                .tolist()
            ),
        },
        "results": {
            "rows": int(len(results)),
            "deterministic_rows": int(
                source_counts.get("deterministic", 0)
            ),
            "judge_rows": int(
                source_counts.get("llm_judge", 0)
            ),
            "generation_errors": int(
                source_counts.get(
                    "generation_error",
                    0,
                )
            ),
            "judge_errors": int(
                source_counts.get(
                    "judge_error",
                    0,
                )
            ),
            "overall_pass_rate": _numeric_mean_or_none(
                results["final_passed"]
            ),
            "average_score": _numeric_mean_or_none(
                results["final_score"]
            ),
            "generation_total_tokens": operations[
                "generation"
            ]["total_tokens"],
            "judge_total_tokens": operations["judge"][
                "total_tokens"
            ],
        },
        "operations": operations,
        "endpoint_pricing": pricing_snapshot,
        "selection_policy": config.selection.model_dump(
            mode="json"
        ),
        "statistics": {
            "confidence_level": CONFIDENCE_LEVEL,
            "pass_rate_interval": "wilson_score",
            "paired_test": (
                "two_sided_exact_mcnemar"
            ),
            "multiple_comparison_correction": (
                "holm"
            ),
            "familywise_significance_level": (
                SIGNIFICANCE_LEVEL
            ),
            "scope": "case_sampling_only",
        },
        "decision": {
            "comparative": (
                int(
                    summary[
                        "model_name"
                    ].nunique()
                )
                >= 2
            ),
            **decision,
        },
        "model_summary": json.loads(
            summary.to_json(orient="records")
        ),
        "pairwise_comparisons": json.loads(
            comparisons.to_json(
                orient="records"
            )
        ),
        "model_selection": json.loads(
            selection.to_json(orient="records")
        ),
        "limitations": [
            (
                "Results are specific to the "
                "evaluated task and dataset."
            ),
            (
                "LLM-judge outputs are evaluation "
                "signals, not objective truth."
            ),
            (
                "Intervals cover case-sampling "
                "uncertainty only."
            ),
            (
                "The statistical methods assume "
                "representative, independent cases."
            ),
            (
                "Operational metrics describe this run and are not "
                "production service-level guarantees."
            ),
            (
                "Missing token and cost metadata is unknown, not zero."
            ),
            (
                "Configured token costs are estimates from declared rates "
                "and recorded usage, not reconciled provider invoices."
            ),
            (
                "Selection weights, thresholds, and capabilities are "
                "configured policy choices."
            ),
            (
                "Weighted operational components use point estimates "
                "without uncertainty intervals."
            ),
        ],
    }


def save_evaluation_metadata(
    *,
    config_path: str | Path | None,
    config: EvaluationConfig,
    cases: pd.DataFrame,
    results: pd.DataFrame,
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    selection: pd.DataFrame,
    case_results_path: str | Path,
    summary_path: str | Path,
    comparison_path: str | Path,
    selection_path: str | Path,
    report_path: str | Path,
) -> Path:
    case_results_path = Path(case_results_path)
    metadata_path = case_results_path.with_name(
        case_results_path.stem
        + "_run_metadata.json"
    )

    metadata = build_evaluation_metadata(
        config_path=config_path,
        config=config,
        cases=cases,
        results=results,
        summary=summary,
        comparisons=comparisons,
        selection=selection,
        case_results_path=case_results_path,
        summary_path=summary_path,
        comparison_path=comparison_path,
        selection_path=selection_path,
        report_path=report_path,
    )
    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return metadata_path

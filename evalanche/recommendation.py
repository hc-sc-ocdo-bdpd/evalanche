from __future__ import annotations

from pathlib import Path

import pandas as pd

from evalanche.config import EvalConfig
from evalanche.judges.protocol import resolve_judge_evidence
from evalanche.reporting import build_model_summary


def _format_percent(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.1%}"


def _format_decimal(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.3f}"


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No model summary available._"

    display = df.copy()

    for col in display.columns:
        if "score" in col:
            display[col] = display[col].apply(_format_decimal)
        elif col == "pass_rate":
            display[col] = display[col].apply(_format_percent)

    columns = list(display.columns)

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"

    rows = []
    for _, row in display.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in columns) + " |")

    return "\n".join([header, separator] + rows)


def build_comparison_report(
    *,
    config: EvalConfig,
    results: pd.DataFrame,
    model_summary_path: str | Path,
    metadata_path: str | Path,
) -> str:
    summary = build_model_summary(results)
    judge_evidence = resolve_judge_evidence(config)

    if summary.empty:
        return _build_empty_report(config=config)

    top_rank = summary["rank"].min()
    winners = summary[summary["rank"] == top_rank].copy()

    total_models = int(summary["model_name"].nunique())
    total_rows = int(len(results))
    total_cases = (
        int(results["case_id"].nunique())
        if "case_id" in results.columns
        else total_rows
    )

    if len(winners) == 1:
        winner = winners.iloc[0]
        observed_leader = str(winner["model_name"])

        outcome_text = (
            f"**Highest observed score:** `{observed_leader}`\n\n"
            f"`{observed_leader}` had the highest average weighted score "
            f"({_format_decimal(winner['average_weighted_score'])}) "
            "in this evaluation run "
            f"and a pass rate of {_format_percent(winner['pass_rate'])}. "
            "This comparison does not select a model automatically."
        )
    else:
        winner_names = [str(name) for name in winners["model_name"].tolist()]
        observed_leaders = ", ".join(
            f"`{name}`" for name in winner_names
        )

        outcome_text = (
            f"**Highest observed scores:** {observed_leaders}\n\n"
            "These models are tied for the top rank in this evaluation run. "
            "Use access, hard requirements, cost, latency, reliability, and "
            "manual review of failure cases to interpret the tradeoff."
        )

    report = f"""# Evalanche Model Comparison

## Observed outcome

{outcome_text}

## Evaluation Scope

- **Run name:** `{config.run.name}`
- **Task:** `{config.task.name}`
- **Judge model:** `{config.judge.model}`
- **Judge evidence level:** `{judge_evidence.level}`
- **Judge protocol:** `{judge_evidence.protocol_id or 'not attached'}`
- **Rows evaluated:** {total_rows}
- **Unique cases:** {total_cases}
- **Models compared:** {total_models}
- **Pass threshold:** {_format_percent(config.scoring.pass_threshold)}

## Model Leaderboard

{_markdown_table(summary)}

## Interpretation

This comparison is task-grounded. It should be read as:

> These are the observed results for this dataset, rubric, judge model, and configuration.

It should not be read as:

> The highest observed score is the best model in general or the right model for every user.

Only models the user has confirmed they can access should enter a decision
shortlist. A single selection requires an explicit decision policy that also
accounts for hard constraints and operational evidence.

## Evidence Files

- Model summary: `{model_summary_path}`
- Run metadata: `{metadata_path}`
- Case-level results: `{config.run.output_path}`

## Caveats

- This comparison is limited to the evaluated task, dataset, rubric, and judge model.
- LLM-as-judge scores are evaluation signals, not objective truth.
- The judge evidence level applies only to its exact matched validation
  contract. Exploratory evidence cannot support a model selection.
- For high-stakes use cases, automated judge results should be calibrated against human or expert review.
- If the test set is small or unrepresentative, the comparison should be treated as preliminary.
- Cost, latency, privacy, deployment availability, bilingual performance, and operational constraints should be considered before production use.
"""

    return report


def _build_empty_report(*, config: EvalConfig) -> str:
    return f"""# Evalanche Model Comparison

No comparison could be generated because no model summary was available.

Run name: `{config.run.name}`
Task: `{config.task.name}`
"""


def save_comparison_report(
    *,
    config: EvalConfig,
    results: pd.DataFrame,
    case_results_path: str | Path,
    model_summary_path: str | Path,
    metadata_path: str | Path,
) -> Path:
    case_results_path = Path(case_results_path)
    comparison_path = case_results_path.with_name(
        case_results_path.stem + "_comparison.md"
    )

    report = build_comparison_report(
        config=config,
        results=results,
        model_summary_path=model_summary_path,
        metadata_path=metadata_path,
    )

    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.write_text(report, encoding="utf-8")

    return comparison_path


# Backward-compatible Python aliases. New artifacts and CLI messages use
# comparison language and never select a model automatically.
build_recommendation_report = build_comparison_report
save_recommendation_report = save_comparison_report

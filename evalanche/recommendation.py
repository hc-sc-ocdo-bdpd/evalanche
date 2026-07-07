from __future__ import annotations

from pathlib import Path

import pandas as pd

from evalanche.config import EvalConfig
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


def build_recommendation_report(
    *,
    config: EvalConfig,
    results: pd.DataFrame,
    model_summary_path: str | Path,
    metadata_path: str | Path,
) -> str:
    summary = build_model_summary(results)

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
        recommendation = str(winner["model_name"])

        recommendation_text = (
            f"**Recommended model:** `{recommendation}`\n\n"
            f"`{recommendation}` is the top-ranked model for this evaluation run. "
            f"It had the highest average weighted score "
            f"({_format_decimal(winner['average_weighted_score'])}) "
            f"and a pass rate of {_format_percent(winner['pass_rate'])}."
        )
    else:
        winner_names = [str(name) for name in winners["model_name"].tolist()]
        recommendation = ", ".join(f"`{name}`" for name in winner_names)

        recommendation_text = (
            f"**Recommended models:** {recommendation}\n\n"
            f"These models are tied for the top rank in this evaluation run. "
            f"Use secondary considerations such as cost, latency, deployment constraints, "
            f"or manual review of failure cases to choose between them."
        )

    report = f"""# Evalanche Model Recommendation

## Recommendation

{recommendation_text}

## Evaluation Scope

- **Run name:** `{config.run.name}`
- **Task:** `{config.task.name}`
- **Judge model:** `{config.judge.model}`
- **Rows evaluated:** {total_rows}
- **Unique cases:** {total_cases}
- **Models compared:** {total_models}
- **Pass threshold:** {_format_percent(config.scoring.pass_threshold)}

## Model Leaderboard

{_markdown_table(summary)}

## Interpretation

This recommendation is task-grounded. It should be read as:

> Based on this dataset, rubric, judge model, and configuration, the recommended model is the strongest option among the models evaluated.

It should not be read as:

> This is the best model in general.

## Evidence Files

- Model summary: `{model_summary_path}`
- Run metadata: `{metadata_path}`
- Case-level results: `{config.run.output_path}`

## Caveats

- This recommendation is limited to the evaluated task, dataset, rubric, and judge model.
- LLM-as-judge scores are evaluation signals, not objective truth.
- For high-stakes use cases, automated judge results should be calibrated against human or expert review.
- If the test set is small or unrepresentative, the recommendation should be treated as preliminary.
- Cost, latency, privacy, deployment availability, bilingual performance, and operational constraints should be considered before production use.
"""

    return report


def _build_empty_report(*, config: EvalConfig) -> str:
    return f"""# Evalanche Model Recommendation

No recommendation could be generated because no model summary was available.

Run name: `{config.run.name}`
Task: `{config.task.name}`
"""


def save_recommendation_report(
    *,
    config: EvalConfig,
    results: pd.DataFrame,
    case_results_path: str | Path,
    model_summary_path: str | Path,
    metadata_path: str | Path,
) -> Path:
    case_results_path = Path(case_results_path)
    recommendation_path = case_results_path.with_name(
        case_results_path.stem + "_recommendation.md"
    )

    report = build_recommendation_report(
        config=config,
        results=results,
        model_summary_path=model_summary_path,
        metadata_path=metadata_path,
    )

    recommendation_path.parent.mkdir(parents=True, exist_ok=True)
    recommendation_path.write_text(report, encoding="utf-8")

    return recommendation_path
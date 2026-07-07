from __future__ import annotations

from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table


console = Console()


def build_model_summary(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()

    criterion_score_cols = [
        col for col in results.columns
        if col.endswith("_score") and col != "weighted_score"
    ]

    aggregations = {
        "case_id": "count",
        "weighted_score": "mean",
        "passed": "mean",
    }

    for col in criterion_score_cols:
        aggregations[col] = "mean"

    if "total_tokens" in results.columns:
        aggregations["total_tokens"] = "sum"

    summary = (
        results
        .groupby("model_name", dropna=False)
        .agg(aggregations)
        .reset_index()
    )

    summary = summary.rename(
        columns={
            "case_id": "cases",
            "weighted_score": "average_weighted_score",
            "passed": "pass_rate",
        }
    )

    summary["rank"] = (
        summary["average_weighted_score"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    ordered_cols = [
        "rank",
        "model_name",
        "cases",
        "average_weighted_score",
        "pass_rate",
    ]

    remaining_cols = [
        col for col in summary.columns
        if col not in ordered_cols
    ]

    summary = summary[ordered_cols + remaining_cols]
    summary = summary.sort_values(
        ["rank", "model_name"],
        ascending=[True, True],
    )

    return summary


def save_model_summary(results: pd.DataFrame, output_path: str | Path) -> Path:
    result_path = Path(output_path)
    summary_path = result_path.with_name(
        result_path.stem + "_model_summary.csv"
    )

    summary = build_model_summary(results)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)

    return summary_path


def print_summary(results: pd.DataFrame) -> None:
    total = len(results)

    if total == 0:
        console.print("[bold red]No results to summarize.[/bold red]")
        return

    pass_rate = results["passed"].mean()
    average_score = results["weighted_score"].mean()

    table = Table(title="Evalanche Judge Summary")
    table.add_column("Metric")
    table.add_column("Value")

    table.add_row("Cases", str(total))
    table.add_row("Average weighted score", f"{average_score:.3f}")
    table.add_row("Pass rate", f"{pass_rate:.1%}")

    console.print(table)


def print_model_leaderboard(results: pd.DataFrame) -> None:
    summary = build_model_summary(results)

    if summary.empty:
        console.print("[bold red]No model summary to display.[/bold red]")
        return

    table = Table(title="Model Leaderboard")
    table.add_column("Rank")
    table.add_column("Model")
    table.add_column("Cases")
    table.add_column("Avg score")
    table.add_column("Pass rate")

    for _, row in summary.iterrows():
        table.add_row(
            str(row["rank"]),
            str(row["model_name"]),
            str(row["cases"]),
            f"{row['average_weighted_score']:.3f}",
            f"{row['pass_rate']:.1%}",
        )

    console.print(table)


def print_failures(results: pd.DataFrame, max_rows: int = 10) -> None:
    failures = results[results["passed"] == False].copy()

    if failures.empty:
        console.print("[bold green]No failed cases.[/bold green]")
        return

    table = Table(title=f"Failed Cases, showing up to {max_rows}")
    table.add_column("case_id")
    table.add_column("model_name")
    table.add_column("score")
    table.add_column("reason")

    for _, row in failures.head(max_rows).iterrows():
        table.add_row(
            str(row["case_id"]),
            str(row["model_name"]),
            f"{row['weighted_score']:.3f}",
            str(row["overall_reason"]),
        )

    console.print(table)
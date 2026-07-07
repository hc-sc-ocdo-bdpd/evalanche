from __future__ import annotations

import pandas as pd
from rich.console import Console
from rich.table import Table


console = Console()


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

    if "model_name" in results.columns:
        for model_name, group in results.groupby("model_name"):
            table.add_row(
                f"{model_name} average score",
                f"{group['weighted_score'].mean():.3f}",
            )
            table.add_row(
                f"{model_name} pass rate",
                f"{group['passed'].mean():.1%}",
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
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


def _format_percent(value: Any) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.1%}"


def _format_decimal(value: Any) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.3f}"


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
        "Pass rate",
        "Average score",
        "Deterministic pass rate",
        "Judge pass rate",
        "Generation errors",
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
                    f"{row['passed_cases']}/{row['cases']}",
                    _format_percent(row["pass_rate"]),
                    _format_decimal(row["average_score"]),
                    _format_percent(row["deterministic_pass_rate"]),
                    _format_percent(row["judge_pass_rate"]),
                    str(row["generation_errors"]),
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


def _recommendation_text(summary: pd.DataFrame) -> str:
    if summary.empty:
        return (
            "**Comparative recommendation:** Not available. No model "
            "results were produced."
        )

    if len(summary) == 1:
        model = summary.iloc[0]
        return (
            "**Comparative recommendation:** Not available.\n\n"
            f"Only `{model['model_name']}` was evaluated. It passed "
            f"{model['passed_cases']} of {model['cases']} cases "
            f"({_format_percent(model['pass_rate'])}), but at least two "
            "candidate models are required for a model comparison."
        )

    top_rank = summary["rank"].min()
    winners = summary[summary["rank"] == top_rank]

    if len(winners) > 1:
        names = ", ".join(
            f"`{name}`" for name in winners["model_name"].tolist()
        )
        return (
            f"**Top-ranked models:** {names}\n\n"
            "These models are tied on the primary ranking measure, overall "
            "pass rate. The evaluation does not support selecting one of "
            "them without additional evidence."
        )

    winner = winners.iloc[0]
    return (
        f"**Top-ranked model:** `{winner['model_name']}`\n\n"
        f"It passed {winner['passed_cases']} of {winner['cases']} cases "
        f"({_format_percent(winner['pass_rate'])}). This means it performed "
        "best among the evaluated candidates on this test set, not that it "
        "is the best model in general."
    )


def build_evaluation_report(
    *,
    config: EvaluationConfig,
    results: pd.DataFrame,
    summary: pd.DataFrame,
    case_results_path: str | Path,
    summary_path: str | Path,
    metadata_path: str | Path,
) -> str:
    source_counts = results["evaluation_source"].value_counts()
    deterministic_rows = int(source_counts.get("deterministic", 0))
    judge_rows = int(source_counts.get("llm_judge", 0))
    generation_errors = int(source_counts.get("generation_error", 0))

    return f"""# Evalanche Combined Evaluation Report

## Result

{_recommendation_text(summary)}

## Evaluation Scope

- **Run:** `{config.run.name}`
- **Task:** `{config.task.name}`
- **Rows evaluated:** {len(results)}
- **Unique cases:** {results['case_id'].nunique()}
- **Models:** {results['model_name'].nunique()}
- **Deterministic rows:** {deterministic_rows}
- **LLM-judged rows:** {judge_rows}
- **Generation errors:** {generation_errors}
- **Judge model:** `{config.judge.model}`
- **Judge pass threshold:** {_format_percent(config.scoring.pass_threshold)}

## Model Leaderboard

The primary rank is based on overall case pass rate.

{_leaderboard_markdown(summary)}

## Failed Cases

{_failures_markdown(results)}

## How Scores Were Selected

- Exact and JSON cases use deterministic evaluation as the authoritative result.
- Open-ended judge cases use the configured LLM judge.
- Generation errors receive a score of zero and are not sent to the judge.
- JSON partial-field scores are diagnostic; only a full expected JSON match passes.

## Evidence Files

- Case results: `{case_results_path}`
- Model summary: `{summary_path}`
- Run metadata: `{metadata_path}`

## Limitations

- Results apply only to this dataset, task, prompts, models, and configuration.
- LLM-judge outputs are evaluation signals and should be calibrated against human review for important uses.
- The current ranking does not yet include statistical uncertainty.
- Cost, latency, privacy, deployment availability, and bilingual requirements are not yet decision constraints in the ranking.
- A small or unrepresentative test set can produce unstable rankings.
"""


def save_evaluation_report(
    *,
    config: EvaluationConfig,
    results: pd.DataFrame,
    summary: pd.DataFrame,
    case_results_path: str | Path,
    summary_path: str | Path,
    metadata_path: str | Path,
) -> Path:
    case_results_path = Path(case_results_path)
    report_path = case_results_path.with_name(
        case_results_path.stem + "_recommendation.md"
    )
    report = build_evaluation_report(
        config=config,
        results=results,
        summary=summary,
        case_results_path=case_results_path,
        summary_path=summary_path,
        metadata_path=metadata_path,
    )
    report_path.write_text(report, encoding="utf-8")
    return report_path


def build_evaluation_metadata(
    *,
    config_path: str | Path | None,
    config: EvaluationConfig,
    cases: pd.DataFrame,
    results: pd.DataFrame,
    summary: pd.DataFrame,
    case_results_path: str | Path,
    summary_path: str | Path,
    report_path: str | Path,
) -> dict[str, Any]:
    source_counts = results["evaluation_source"].value_counts()
    top_models = (
        summary.loc[
            summary["rank"] == summary["rank"].min(),
            "model_name",
        ]
        .astype(str)
        .tolist()
        if not summary.empty
        else []
    )

    return {
        "schema_version": "0.3",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evalanche_version": __version__,
        "runtime": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "run": {
            "name": config.run.name,
            "config_path": (
                str(config_path) if config_path is not None else None
            ),
            "input_path": str(config.run.input_path),
            "case_results_path": str(case_results_path),
            "model_summary_path": str(summary_path),
            "recommendation_path": str(report_path),
        },
        "hashes": {
            "config_sha256": (
                sha256_file(config_path)
                if config_path is not None
                else None
            ),
            "input_sha256": sha256_file(config.run.input_path),
            "case_results_sha256": sha256_file(case_results_path),
            "model_summary_sha256": sha256_file(summary_path),
            "recommendation_sha256": sha256_file(report_path),
        },
        "task": {
            "name": config.task.name,
            "description": config.task.description,
        },
        "routing": {
            "exact_and_json": "deterministic",
            "judge": "llm_judge",
            "generation_error": "automatic_failure",
        },
        "metrics": {
            "case_sensitive": config.metrics.case_sensitive,
            "trim_whitespace": config.metrics.trim_whitespace,
            "collapse_whitespace": config.metrics.collapse_whitespace,
        },
        "judge": {
            "model": config.judge.model,
            "temperature": config.judge.temperature,
            "max_retries": config.judge.max_retries,
            "score_min": config.scoring.score_min,
            "score_max": config.scoring.score_max,
            "pass_threshold": config.scoring.pass_threshold,
            "criteria": [
                {
                    "name": criterion.name,
                    "weight": criterion.weight,
                    "description": criterion.description,
                }
                for criterion in config.criteria
            ],
        },
        "input_data": {
            "rows": int(len(cases)),
            "unique_cases": int(cases["case_id"].nunique()),
            "models": sorted(
                cases["model_name"].astype(str).unique().tolist()
            ),
            "evaluation_types": sorted(
                cases["evaluation_type"].astype(str).unique().tolist()
            ),
        },
        "results": {
            "rows": int(len(results)),
            "deterministic_rows": int(
                source_counts.get("deterministic", 0)
            ),
            "judge_rows": int(source_counts.get("llm_judge", 0)),
            "generation_errors": int(
                source_counts.get("generation_error", 0)
            ),
            "overall_pass_rate": float(
                results["final_passed"].astype(float).mean()
            ),
            "average_score": float(results["final_score"].mean()),
            "generation_total_tokens": (
                int(
                    pd.to_numeric(
                        results["generation_total_tokens"],
                        errors="coerce",
                    ).sum()
                )
                if "generation_total_tokens" in results.columns
                else None
            ),
            "judge_total_tokens": int(
                pd.to_numeric(
                    results["judge_total_tokens"],
                    errors="coerce",
                ).sum()
            ),
        },
        "recommendation": {
            "comparative": int(summary["model_name"].nunique()) >= 2,
            "top_ranked_models": top_models,
            "ranking_measure": "overall_pass_rate",
        },
        "model_summary": json.loads(
            summary.to_json(orient="records")
        ),
        "limitations": [
            "Results are specific to the evaluated task and dataset.",
            "LLM-judge outputs are evaluation signals, not objective truth.",
            "The current ranking does not include statistical uncertainty.",
            "Operational constraints are not yet part of the ranking.",
        ],
    }


def save_evaluation_metadata(
    *,
    config_path: str | Path | None,
    config: EvaluationConfig,
    cases: pd.DataFrame,
    results: pd.DataFrame,
    summary: pd.DataFrame,
    case_results_path: str | Path,
    summary_path: str | Path,
    report_path: str | Path,
) -> Path:
    case_results_path = Path(case_results_path)
    metadata_path = case_results_path.with_name(
        case_results_path.stem + "_run_metadata.json"
    )
    metadata = build_evaluation_metadata(
        config_path=config_path,
        config=config,
        cases=cases,
        results=results,
        summary=summary,
        case_results_path=case_results_path,
        summary_path=summary_path,
        report_path=report_path,
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return metadata_path
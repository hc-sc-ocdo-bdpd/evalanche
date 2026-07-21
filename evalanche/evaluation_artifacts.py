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
        "95% pass-rate interval",
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
                    (
                        f"{_format_percent(row['pass_rate_ci_low'])}–"
                        f"{_format_percent(row['pass_rate_ci_high'])}"
                    ),
                    _format_decimal(row["average_score"]),
                    _format_percent(
                        row["deterministic_pass_rate"]
                    ),
                    _format_percent(row["judge_pass_rate"]),
                    str(row["generation_errors"]),
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
        "Pass-rate difference (A − B)",
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
                    f"{float(row['pass_rate_difference']):+.1%}",
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


def _recommendation_details(
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
) -> dict[str, Any]:
    if summary.empty:
        return {
            "status": "no_results",
            "top_ranked_models": [],
            "observed_leader": None,
            "recommended_model": None,
            "not_distinguished_from": [],
        }

    top_rank = summary["rank"].min()
    top_models = (
        summary.loc[
            summary["rank"] == top_rank,
            "model_name",
        ]
        .astype(str)
        .tolist()
    )

    if len(summary) == 1:
        return {
            "status": "single_model",
            "top_ranked_models": top_models,
            "observed_leader": top_models[0],
            "recommended_model": None,
            "not_distinguished_from": [],
        }

    if len(top_models) > 1:
        return {
            "status": "observed_tie",
            "top_ranked_models": top_models,
            "observed_leader": None,
            "recommended_model": None,
            "not_distinguished_from": top_models,
        }

    observed_leader = top_models[0]
    other_models = sorted(
        set(summary["model_name"].astype(str))
        - {observed_leader}
    )
    not_distinguished_from: list[str] = []

    for other_model in other_models:
        matching = comparisons[
            (
                (comparisons["model_a"] == observed_leader)
                & (comparisons["model_b"] == other_model)
            )
            | (
                (comparisons["model_a"] == other_model)
                & (comparisons["model_b"] == observed_leader)
            )
        ]

        if (
            len(matching) != 1
            or matching.iloc[0]["clear_winner"]
            != observed_leader
        ):
            not_distinguished_from.append(other_model)

    if not_distinguished_from:
        return {
            "status": "insufficient_evidence",
            "top_ranked_models": top_models,
            "observed_leader": observed_leader,
            "recommended_model": None,
            "not_distinguished_from": (
                not_distinguished_from
            ),
        }

    return {
        "status": "clear_leader",
        "top_ranked_models": top_models,
        "observed_leader": observed_leader,
        "recommended_model": observed_leader,
        "not_distinguished_from": [],
    }


def _recommendation_text(
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
) -> str:
    details = _recommendation_details(
        summary,
        comparisons,
    )

    if summary.empty:
        return (
            "**Comparative recommendation:** Not available. "
            "No model results were produced."
        )

    if len(summary) == 1:
        model = summary.iloc[0]
        return (
            "**Comparative recommendation:** Not available.\n\n"
            f"Only `{model['model_name']}` was evaluated. It "
            f"passed {model['passed_cases']} of "
            f"{model['cases']} cases "
            f"({_format_percent(model['pass_rate'])}; "
            f"95% interval "
            f"{_format_percent(model['pass_rate_ci_low'])}–"
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
            "**Comparative recommendation:** "
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
            "**Comparative recommendation:** "
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
        f"**Evidence-supported leader:** "
        f"`{observed_leader}`\n\n"
        f"It passed {leader['passed_cases']} of "
        f"{leader['cases']} cases "
        f"({_format_percent(leader['pass_rate'])}) and "
        "clearly outperformed each other evaluated model in "
        "the paired pass/fail comparisons. This supports "
        "choosing it for this tested task, subject to the "
        "operational constraints below; it is not the best "
        "model in general."
    )


def build_evaluation_report(
    *,
    config: EvaluationConfig,
    results: pd.DataFrame,
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    case_results_path: str | Path,
    summary_path: str | Path,
    comparison_path: str | Path,
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

    return f"""# Evalanche Combined Evaluation Report

## Result

{_recommendation_text(summary, comparisons)}

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

The intervals show uncertainty in each pass rate. They should not be used
alone to compare models because every model answered the same cases.

## Pairwise Evidence

The exact paired test compares where one model passed and the other failed.
Holm correction limits false positives when several model pairs are tested.

{_pairwise_markdown(comparisons)}

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
- Pairwise comparisons: `{comparison_path}`
- Run metadata: `{metadata_path}`

## Limitations

- Results apply only to this dataset, task, prompts, models, and configuration.
- LLM-judge outputs are evaluation signals and should be calibrated against
  human review for important uses.
- Statistical intervals cover case-sampling uncertainty only; they do not
  cover prompt, generation, or judge variability.
- The statistical methods assume cases are representative and independent.
  Related or repeated cases require grouped analysis.
- Cost, latency, privacy, deployment availability, and bilingual requirements
  are not yet decision constraints in the ranking.
- A small or unrepresentative test set can produce unstable rankings.
"""


def save_evaluation_report(
    *,
    config: EvaluationConfig,
    results: pd.DataFrame,
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    case_results_path: str | Path,
    summary_path: str | Path,
    comparison_path: str | Path,
    metadata_path: str | Path,
) -> Path:
    case_results_path = Path(case_results_path)
    report_path = case_results_path.with_name(
        case_results_path.stem
        + "_recommendation.md"
    )

    report = build_evaluation_report(
        config=config,
        results=results,
        summary=summary,
        comparisons=comparisons,
        case_results_path=case_results_path,
        summary_path=summary_path,
        comparison_path=comparison_path,
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
    case_results_path: str | Path,
    summary_path: str | Path,
    comparison_path: str | Path,
    report_path: str | Path,
) -> dict[str, Any]:
    source_counts = results[
        "evaluation_source"
    ].value_counts()
    recommendation = _recommendation_details(
        summary,
        comparisons,
    )

    return {
        "schema_version": "0.4",
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
            "recommendation_path": str(report_path),
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
            "recommendation_sha256": sha256_file(
                report_path
            ),
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
            "case_sensitive": (
                config.metrics.case_sensitive
            ),
            "trim_whitespace": (
                config.metrics.trim_whitespace
            ),
            "collapse_whitespace": (
                config.metrics.collapse_whitespace
            ),
        },
        "judge": {
            "model": config.judge.model,
            "temperature": config.judge.temperature,
            "max_retries": config.judge.max_retries,
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
            "overall_pass_rate": float(
                results["final_passed"]
                .astype(float)
                .mean()
            ),
            "average_score": float(
                results["final_score"].mean()
            ),
            "generation_total_tokens": (
                int(
                    pd.to_numeric(
                        results[
                            "generation_total_tokens"
                        ],
                        errors="coerce",
                    ).sum()
                )
                if "generation_total_tokens"
                in results.columns
                else None
            ),
            "judge_total_tokens": int(
                pd.to_numeric(
                    results["judge_total_tokens"],
                    errors="coerce",
                ).sum()
            ),
        },
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
        "recommendation": {
            "comparative": (
                int(
                    summary[
                        "model_name"
                    ].nunique()
                )
                >= 2
            ),
            **recommendation,
            "ranking_measure": "overall_pass_rate",
        },
        "model_summary": json.loads(
            summary.to_json(orient="records")
        ),
        "pairwise_comparisons": json.loads(
            comparisons.to_json(
                orient="records"
            )
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
                "Operational constraints are not "
                "yet part of the ranking."
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
    case_results_path: str | Path,
    summary_path: str | Path,
    comparison_path: str | Path,
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
        case_results_path=case_results_path,
        summary_path=summary_path,
        comparison_path=comparison_path,
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
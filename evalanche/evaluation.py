from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console
from rich.table import Table
from tqdm.auto import tqdm

from evalanche.config import EvaluationConfig, MetricsConfig
from evalanche.evaluation_artifacts import (
    save_evaluation_metadata,
    save_evaluation_report,
)
from evalanche.identity import (
    validate_balanced_model_coverage,
    validate_evaluation_keys,
)
from evalanche.io import load_eval_cases
from evalanche.judges import CriteriaJudge
from evalanche.metrics.deterministic import score_row
from evalanche.routing import EXACT, JSON, JUDGE


console = Console()


def _generation_failed(row: dict[str, Any]) -> bool:
    status = row.get("generation_status")

    if status is None or pd.isna(status):
        return False

    return str(status).strip().lower() != "success"


def _deterministic_reason(metric: dict[str, Any]) -> str:
    if metric["evaluation_type"] == EXACT:
        if metric["metric_passed"]:
            return "Output matched after configured text normalization."
        return "Output did not match after configured text normalization."

    if not metric["output_is_json"]:
        return "Model output was not valid JSON."

    if metric["metric_passed"]:
        return "Model output exactly matched the expected JSON."

    matching = metric["json_matching_field_count"]
    expected = metric["json_expected_field_count"]

    if matching is not None and expected is not None:
        return (
            "JSON did not exactly match; "
            f"{matching} of {expected} expected fields matched."
        )

    return "Model output did not match the expected JSON."


def _deterministic_score(metric: dict[str, Any]) -> float:
    if metric["evaluation_type"] == EXACT:
        return float(bool(metric["metric_passed"]))

    field_match_rate = metric["json_field_match_rate"]
    if field_match_rate is not None:
        return float(field_match_rate)

    return float(bool(metric["metric_passed"]))


def evaluate_cases(
    cases: pd.DataFrame,
    config: EvaluationConfig,
    *,
    judge: Any | None = None,
) -> pd.DataFrame:
    cases = validate_evaluation_keys(
        cases,
        source_name="evaluation input",
    )
    validate_balanced_model_coverage(
        cases,
        source_name="evaluation input",
    )

    metrics_config = MetricsConfig(
        run=config.run,
        metrics=config.metrics,
    )

    rows = cases.to_dict(orient="records")
    metrics_by_key = {
        (str(row["case_id"]), str(row["model_name"])): score_row(
            row,
            metrics_config,
        )
        for row in rows
    }

    judge_rows = [
        row
        for row in rows
        if row["evaluation_type"] == JUDGE
        and not _generation_failed(row)
    ]

    judge_results: dict[tuple[str, str], dict[str, Any]] = {}

    if judge_rows:
        active_judge = judge or CriteriaJudge(config)

        for row in tqdm(judge_rows, desc="Judging cases"):
            result = active_judge.judge_case(row)
            key = (str(row["case_id"]), str(row["model_name"]))
            judge_results[key] = result

    combined_records: list[dict[str, Any]] = []

    for row in rows:
        key = (str(row["case_id"]), str(row["model_name"]))
        metric = metrics_by_key[key]

        record = dict(row)
        record.update(metric)
        record.update(
            {
                "judge_weighted_score": None,
                "judge_passed": None,
                "judge_overall_reason": None,
                "judge_raw_result": None,
                "judge_prompt_tokens": None,
                "judge_completion_tokens": None,
                "judge_total_tokens": None,
            }
        )

        for criterion in config.criteria:
            record[f"judge_{criterion.name}_score"] = None
            record[f"judge_{criterion.name}_reason"] = None

        if _generation_failed(row):
            generation_error = row.get("generation_error")
            reason = (
                str(generation_error).strip()
                if generation_error is not None
                and not pd.isna(generation_error)
                and str(generation_error).strip()
                else "Candidate generation failed."
            )

            record.update(
                {
                    "evaluation_source": "generation_error",
                    "final_score": 0.0,
                    "final_passed": False,
                    "evaluation_reason": reason,
                }
            )

        elif row["evaluation_type"] in {EXACT, JSON}:
            record.update(
                {
                    "evaluation_source": "deterministic",
                    "final_score": _deterministic_score(metric),
                    "final_passed": bool(metric["metric_passed"]),
                    "evaluation_reason": _deterministic_reason(metric),
                }
            )

        else:
            judge_result = judge_results.get(key)
            if judge_result is None:
                raise ValueError(
                    "Missing judge result for "
                    f"case_id={row['case_id']!r}, "
                    f"model_name={row['model_name']!r}"
                )

            record.update(
                {
                    "evaluation_source": "llm_judge",
                    "judge_weighted_score": judge_result[
                        "weighted_score"
                    ],
                    "judge_passed": judge_result["passed"],
                    "judge_overall_reason": judge_result[
                        "overall_reason"
                    ],
                    "judge_raw_result": judge_result.get(
                        "raw_judge_result"
                    ),
                    "judge_prompt_tokens": judge_result.get(
                        "prompt_tokens"
                    ),
                    "judge_completion_tokens": judge_result.get(
                        "completion_tokens"
                    ),
                    "judge_total_tokens": judge_result.get(
                        "total_tokens"
                    ),
                    "final_score": float(
                        judge_result["weighted_score"]
                    ),
                    "final_passed": bool(judge_result["passed"]),
                    "evaluation_reason": judge_result[
                        "overall_reason"
                    ],
                }
            )

            for criterion in config.criteria:
                record[f"judge_{criterion.name}_score"] = (
                    judge_result.get(f"{criterion.name}_score")
                )
                record[f"judge_{criterion.name}_reason"] = (
                    judge_result.get(f"{criterion.name}_reason")
                )

        combined_records.append(record)

    return pd.DataFrame(combined_records)


def _safe_rate(series: pd.Series) -> float | None:
    if series.empty:
        return None
    return float(series.astype(float).mean())


def build_evaluation_summary(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []

    for model_name, group in results.groupby("model_name", dropna=False):
        deterministic = group[
            group["evaluation_source"] == "deterministic"
        ]
        judged = group[group["evaluation_source"] == "llm_judge"]
        generation_errors = group[
            group["evaluation_source"] == "generation_error"
        ]

        record: dict[str, Any] = {
            "model_name": model_name,
            "cases": int(len(group)),
            "passed_cases": int(group["final_passed"].sum()),
            "failed_cases": int((~group["final_passed"]).sum()),
            "pass_rate": _safe_rate(group["final_passed"]),
            "average_score": float(group["final_score"].mean()),
            "deterministic_cases": int(len(deterministic)),
            "deterministic_pass_rate": _safe_rate(
                deterministic["final_passed"]
            ),
            "judge_cases": int(len(judged)),
            "judge_pass_rate": _safe_rate(judged["final_passed"]),
            "average_judge_score": (
                float(judged["final_score"].mean())
                if not judged.empty
                else None
            ),
            "generation_errors": int(len(generation_errors)),
        }

        if "generation_total_tokens" in group.columns:
            record["generation_total_tokens"] = int(
                pd.to_numeric(
                    group["generation_total_tokens"],
                    errors="coerce",
                ).sum()
            )

        record["judge_total_tokens"] = int(
            pd.to_numeric(
                group["judge_total_tokens"],
                errors="coerce",
            ).sum()
        )

        records.append(record)

    summary = pd.DataFrame(records)
    summary["rank"] = (
        summary["pass_rate"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    ordered_columns = [
        "rank",
        "model_name",
        "cases",
        "passed_cases",
        "failed_cases",
        "pass_rate",
        "average_score",
        "deterministic_cases",
        "deterministic_pass_rate",
        "judge_cases",
        "judge_pass_rate",
        "average_judge_score",
        "generation_errors",
    ]
    remaining_columns = [
        column
        for column in summary.columns
        if column not in ordered_columns
    ]

    return summary[ordered_columns + remaining_columns].sort_values(
        ["rank", "model_name"],
        ascending=[True, True],
    )


def save_evaluation_summary(
    results: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    summary_path = output_path.with_name(
        output_path.stem + "_model_summary.csv"
    )
    summary = build_evaluation_summary(results)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    return summary_path


def print_evaluation_summary(results: pd.DataFrame) -> None:
    summary = build_evaluation_summary(results)

    if summary.empty:
        console.print("[bold red]No evaluation results to display.[/bold red]")
        return

    table = Table(title="Combined Model Evaluation")
    table.add_column("Rank")
    table.add_column("Model")
    table.add_column("Passed")
    table.add_column("Pass rate")
    table.add_column("Avg score")
    table.add_column("Generation errors")

    for _, row in summary.iterrows():
        table.add_row(
            str(row["rank"]),
            str(row["model_name"]),
            f"{row['passed_cases']}/{row['cases']}",
            f"{row['pass_rate']:.1%}",
            f"{row['average_score']:.3f}",
            str(row["generation_errors"]),
        )

    console.print(table)


def run_evaluation(
    config: EvaluationConfig,
    *,
    config_path: str | Path | None = None,
) -> tuple[pd.DataFrame, Path, Path, Path, Path]:
    cases = load_eval_cases(config.run.input_path)
    results = evaluate_cases(cases, config)

    output_path = Path(config.run.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    summary_path = save_evaluation_summary(results, output_path)
    summary = build_evaluation_summary(results)

    metadata_path = output_path.with_name(
        output_path.stem + "_run_metadata.json"
    )

    report_path = save_evaluation_report(
        config=config,
        results=results,
        summary=summary,
        case_results_path=output_path,
        summary_path=summary_path,
        metadata_path=metadata_path,
    )

    metadata_path = save_evaluation_metadata(
        config_path=config_path,
        config=config,
        cases=cases,
        results=results,
        summary=summary,
        case_results_path=output_path,
        summary_path=summary_path,
        report_path=report_path,
    )

    print_evaluation_summary(results)

    return (
        results,
        output_path,
        summary_path,
        metadata_path,
        report_path,
    )
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
from evalanche.llm import operational_from_error
from evalanche.metrics.deterministic import score_row
from evalanche.operational import summarize_stage
from evalanche.routing import EXACT, JSON, JUDGE
from evalanche.selection import (
    build_model_selection,
    build_recommendation_decision,
    mark_recommended_model,
    save_model_selection,
)
from evalanche.statistics import (
    build_pairwise_comparisons,
    save_pairwise_comparisons,
    wilson_score_interval,
)


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


def _judge_error_result(
    row: dict[str, Any],
    error: BaseException,
) -> dict[str, Any]:
    operational = operational_from_error(error)
    return {
        "case_id": row["case_id"],
        "model_name": row["model_name"],
        "_status": "error",
        "error": repr(error),
        "prompt_tokens": operational.get("prompt_tokens"),
        "cached_prompt_tokens": operational.get(
            "cached_prompt_tokens"
        ),
        "completion_tokens": operational.get(
            "completion_tokens"
        ),
        "total_tokens": operational.get("total_tokens"),
        "latency_seconds": operational.get(
            "latency_seconds"
        ),
        "api_seconds": operational.get("api_seconds"),
        "attempts": operational.get("attempts"),
        "failed_attempts": operational.get("failed_attempts"),
        "cost_usd": operational.get("cost_usd"),
        "cost_source": operational.get("cost_source"),
        "configured_cost_usd": operational.get(
            "configured_cost_usd"
        ),
        "provider_reported_cost_usd": operational.get(
            "provider_reported_cost_usd"
        ),
        "pricing_id": operational.get("pricing_id"),
        "pricing_model": operational.get("pricing_model"),
        "pricing_currency": operational.get("pricing_currency"),
        "pricing_input_per_million_tokens": operational.get(
            "pricing_input_per_million_tokens"
        ),
        "pricing_cached_input_per_million_tokens": operational.get(
            "pricing_cached_input_per_million_tokens"
        ),
        "pricing_output_per_million_tokens": operational.get(
            "pricing_output_per_million_tokens"
        ),
        "pricing_effective_date": operational.get(
            "pricing_effective_date"
        ),
        "pricing_source": operational.get("pricing_source"),
    }


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
            try:
                result = active_judge.judge_case(row)
                result["_status"] = "success"
            except Exception as error:
                if not config.judge.continue_on_error:
                    raise
                result = _judge_error_result(row, error)

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
                "judge_status": "not_requested",
                "judge_error": "",
                "judge_seconds": None,
                "judge_api_seconds": None,
                "judge_attempts": None,
                "judge_failed_attempts": None,
                "judge_prompt_tokens": None,
                "judge_cached_prompt_tokens": None,
                "judge_completion_tokens": None,
                "judge_total_tokens": None,
                "judge_cost_usd": None,
                "judge_cost_source": None,
                "judge_configured_cost_usd": None,
                "judge_provider_reported_cost_usd": None,
                "judge_pricing_id": None,
                "judge_pricing_model": None,
                "judge_pricing_currency": None,
                "judge_pricing_input_per_million_tokens": None,
                "judge_pricing_cached_input_per_million_tokens": None,
                "judge_pricing_output_per_million_tokens": None,
                "judge_pricing_effective_date": None,
                "judge_pricing_source": None,
                "judge_pricing_catalog_version": None,
                "judge_pricing_catalog_sha256": None,
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
                    "judge_status": (
                        "skipped_generation_error"
                        if row["evaluation_type"] == JUDGE
                        else "not_requested"
                    ),
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
                    "judge_status": judge_result.get(
                        "_status",
                        "success",
                    ),
                    "judge_error": judge_result.get("error", ""),
                    "judge_seconds": judge_result.get(
                        "latency_seconds"
                    ),
                    "judge_api_seconds": judge_result.get(
                        "api_seconds"
                    ),
                    "judge_attempts": judge_result.get("attempts"),
                    "judge_failed_attempts": judge_result.get(
                        "failed_attempts"
                    ),
                    "judge_prompt_tokens": judge_result.get(
                        "prompt_tokens"
                    ),
                    "judge_cached_prompt_tokens": judge_result.get(
                        "cached_prompt_tokens"
                    ),
                    "judge_completion_tokens": judge_result.get(
                        "completion_tokens"
                    ),
                    "judge_total_tokens": judge_result.get(
                        "total_tokens"
                    ),
                    "judge_cost_usd": judge_result.get("cost_usd"),
                    "judge_cost_source": judge_result.get(
                        "cost_source"
                    ),
                    "judge_configured_cost_usd": judge_result.get(
                        "configured_cost_usd"
                    ),
                    "judge_provider_reported_cost_usd": judge_result.get(
                        "provider_reported_cost_usd"
                    ),
                    "judge_pricing_id": judge_result.get(
                        "pricing_id"
                    ),
                    "judge_pricing_model": judge_result.get(
                        "pricing_model"
                    ),
                    "judge_pricing_currency": judge_result.get(
                        "pricing_currency"
                    ),
                    "judge_pricing_input_per_million_tokens": judge_result.get(
                        "pricing_input_per_million_tokens"
                    ),
                    "judge_pricing_cached_input_per_million_tokens": judge_result.get(
                        "pricing_cached_input_per_million_tokens"
                    ),
                    "judge_pricing_output_per_million_tokens": judge_result.get(
                        "pricing_output_per_million_tokens"
                    ),
                    "judge_pricing_effective_date": judge_result.get(
                        "pricing_effective_date"
                    ),
                    "judge_pricing_source": judge_result.get(
                        "pricing_source"
                    ),
                    "judge_pricing_catalog_version": judge_result.get(
                        "pricing_catalog_version"
                    ),
                    "judge_pricing_catalog_sha256": judge_result.get(
                        "pricing_catalog_sha256"
                    ),
                }
            )

            if record["judge_status"] == "error":
                record.update(
                    {
                        "evaluation_source": "judge_error",
                        "final_score": None,
                        "final_passed": None,
                        "evaluation_reason": (
                            "Judge failed: "
                            f"{judge_result.get('error', 'unknown error')}"
                        ),
                    }
                )
                combined_records.append(record)
                continue

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
        scored = group[group["final_passed"].notna()]
        deterministic = group[
            group["evaluation_source"] == "deterministic"
        ]
        judged = group[
            group["evaluation_source"] == "llm_judge"
        ]
        passed_cases = int(
            scored["final_passed"].astype(bool).sum()
        )
        scored_cases = int(len(scored))

        if scored_cases:
            (
                pass_rate_ci_low,
                pass_rate_ci_high,
            ) = wilson_score_interval(
                passed_cases,
                scored_cases,
            )
        else:
            pass_rate_ci_low = None
            pass_rate_ci_high = None

        generation_operations = summarize_stage(
            group,
            "generation",
        )
        judge_operations = summarize_stage(
            group,
            "judge",
        )

        record: dict[str, Any] = {
            "model_name": model_name,
            "cases": int(len(group)),
            "scored_cases": scored_cases,
            "unscored_cases": int(len(group) - scored_cases),
            "passed_cases": passed_cases,
            "failed_cases": int(scored_cases - passed_cases),
            "pass_rate": _safe_rate(scored["final_passed"]),
            "pass_rate_ci_low": pass_rate_ci_low,
            "pass_rate_ci_high": pass_rate_ci_high,
            "average_score": (
                float(scored["final_score"].mean())
                if scored_cases
                else None
            ),
            "deterministic_cases": int(len(deterministic)),
            "deterministic_pass_rate": _safe_rate(
                deterministic["final_passed"]
            ),
            "judge_cases": int(len(judged)),
            "judge_pass_rate": _safe_rate(
                judged["final_passed"]
            ),
            "average_judge_score": (
                float(judged["final_score"].mean())
                if not judged.empty
                else None
            ),
        }

        record.update(
            {
                f"generation_{key}": value
                for key, value in generation_operations.items()
            }
        )
        record.update(
            {
                f"judge_{key}": value
                for key, value in judge_operations.items()
            }
        )

        records.append(record)

    summary = pd.DataFrame(records)
    summary["rank"] = (
        summary["pass_rate"]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )

    ordered_columns = [
        "rank",
        "model_name",
        "cases",
        "scored_cases",
        "unscored_cases",
        "passed_cases",
        "failed_cases",
        "pass_rate",
        "pass_rate_ci_low",
        "pass_rate_ci_high",
        "average_score",
        "deterministic_cases",
        "deterministic_pass_rate",
        "judge_cases",
        "judge_pass_rate",
        "average_judge_score",
        "generation_requests",
        "generation_successes",
        "generation_errors",
        "generation_failure_rate",
        "generation_average_seconds",
        "generation_p95_seconds",
        "generation_prompt_tokens",
        "generation_cached_prompt_tokens",
        "generation_completion_tokens",
        "generation_total_tokens",
        "generation_cost_usd",
        "generation_cost_coverage",
        "generation_cost_sources",
        "generation_configured_cost_usd",
        "generation_configured_cost_coverage",
        "generation_provider_reported_cost_usd",
        "generation_provider_reported_cost_coverage",
        "judge_requests",
        "judge_successes",
        "judge_errors",
        "judge_failure_rate",
        "judge_average_seconds",
        "judge_p95_seconds",
        "judge_prompt_tokens",
        "judge_cached_prompt_tokens",
        "judge_completion_tokens",
        "judge_total_tokens",
        "judge_cost_usd",
        "judge_cost_coverage",
        "judge_cost_sources",
        "judge_configured_cost_usd",
        "judge_configured_cost_coverage",
        "judge_provider_reported_cost_usd",
        "judge_provider_reported_cost_coverage",
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
    table.add_column("95% range")
    table.add_column("Avg score")
    table.add_column("Gen failures")
    table.add_column("Judge failures")

    for _, row in summary.iterrows():
        pass_rate = row["pass_rate"]
        interval_low = row["pass_rate_ci_low"]
        interval_high = row["pass_rate_ci_high"]
        average_score = row["average_score"]
        interval = (
            f"{interval_low:.1%}-{interval_high:.1%}"
            if pd.notna(interval_low) and pd.notna(interval_high)
            else "N/A"
        )

        table.add_row(
            str(row["rank"]),
            str(row["model_name"]),
            f"{row['passed_cases']}/{row['scored_cases']}",
            f"{pass_rate:.1%}" if pd.notna(pass_rate) else "N/A",
            interval,
            (
                f"{average_score:.3f}"
                if pd.notna(average_score)
                else "N/A"
            ),
            (
                f"{row['generation_errors']}/"
                f"{row['generation_requests']}"
            ),
            f"{row['judge_errors']}/{row['judge_requests']}",
        )

    console.print(table)


def run_evaluation(
    config: EvaluationConfig,
    *,
    config_path: str | Path | None = None,
) -> tuple[pd.DataFrame, Path, Path, Path, Path, Path, Path]:
    cases = load_eval_cases(config.run.input_path)
    results = evaluate_cases(cases, config)
    output_path = Path(config.run.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    summary_path = save_evaluation_summary(results, output_path)
    summary = build_evaluation_summary(results)

    comparisons = build_pairwise_comparisons(results)
    comparison_path = save_pairwise_comparisons(
        comparisons,
        output_path,
    )

    selection = build_model_selection(
        summary,
        config.selection,
    )
    decision = build_recommendation_decision(
        summary,
        comparisons,
        selection,
        config.selection,
    )
    selection = mark_recommended_model(selection, decision)
    selection_path = save_model_selection(
        selection,
        output_path,
    )

    metadata_path = output_path.with_name(
        output_path.stem + "_run_metadata.json"
    )
    report_path = save_evaluation_report(
        config=config,
        results=results,
        summary=summary,
        comparisons=comparisons,
        selection=selection,
        case_results_path=output_path,
        summary_path=summary_path,
        comparison_path=comparison_path,
        selection_path=selection_path,
        metadata_path=metadata_path,
    )
    metadata_path = save_evaluation_metadata(
        config_path=config_path,
        config=config,
        cases=cases,
        results=results,
        summary=summary,
        comparisons=comparisons,
        selection=selection,
        case_results_path=output_path,
        summary_path=summary_path,
        comparison_path=comparison_path,
        selection_path=selection_path,
        report_path=report_path,
    )

    print_evaluation_summary(results)

    return (
        results,
        output_path,
        summary_path,
        comparison_path,
        selection_path,
        metadata_path,
        report_path,
    )
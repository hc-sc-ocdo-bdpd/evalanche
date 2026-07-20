from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console
from rich.table import Table

from evalanche import __version__
from evalanche.config import MetricsConfig
from evalanche.identity import validate_evaluation_keys
from evalanche.routing import (
    DETERMINISTIC_EVALUATION_TYPES,
    EXACT,
    JSON,
    validate_evaluation_types,
)


console = Console()


REQUIRED_METRIC_COLUMNS = {
    "case_id",
    "input",
    "expected_output",
    "evaluation_type",
    "model_name",
    "model_output",
}


def sha256_file(path: str | Path) -> str | None:
    file_path = Path(path)

    if not file_path.exists() or not file_path.is_file():
        return None

    digest = hashlib.sha256()

    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_metric_cases(path: str | Path) -> pd.DataFrame:
    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(f"Metrics input file not found: {input_path}")

    df = pd.read_csv(input_path)

    missing = REQUIRED_METRIC_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Metrics input is missing required columns: {sorted(missing)}"
        )

    df = validate_evaluation_types(
        df,
        source_name=str(input_path),
    )

    return validate_evaluation_keys(
        df,
        source_name=str(input_path),
    )


def normalize_text(
    value: Any,
    *,
    case_sensitive: bool,
    trim_whitespace: bool,
    collapse_whitespace: bool,
) -> str:
    if pd.isna(value):
        text = ""
    else:
        text = str(value)

    if trim_whitespace:
        text = text.strip()

    if collapse_whitespace:
        text = re.sub(r"\s+", " ", text)

    if not case_sensitive:
        text = text.lower()

    return text


def try_parse_json(value: Any) -> tuple[bool, Any]:
    if pd.isna(value) or not isinstance(value, str):
        return False, None

    text = value.strip()

    if text.startswith("```json"):
        text = text.removeprefix("```json").strip()

    if text.startswith("```"):
        text = text.removeprefix("```").strip()

    if text.endswith("```"):
        text = text.removesuffix("```").strip()

    try:
        return True, json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return False, None


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def compare_json_fields(
    expected: Any,
    output: Any,
) -> tuple[int | None, int | None, float | None]:
    if not isinstance(expected, dict) or not isinstance(output, dict):
        return None, None, None

    expected_keys = list(expected.keys())

    if not expected_keys:
        return 0, 0, None

    matching = sum(
        1
        for key in expected_keys
        if key in output and output[key] == expected[key]
    )

    total = len(expected_keys)

    return total, matching, matching / total


def _empty_metric_values() -> dict[str, Any]:
    return {
        "exact_match": None,
        "normalized_exact_match": None,
        "expected_is_json": None,
        "output_is_json": None,
        "json_exact_match": None,
        "json_expected_field_count": None,
        "json_matching_field_count": None,
        "json_field_match_rate": None,
    }


def score_row(
    row: dict[str, Any],
    config: MetricsConfig,
) -> dict[str, Any]:
    evaluation_type = row["evaluation_type"]

    base_result: dict[str, Any] = {
        "case_id": row["case_id"],
        "model_name": row["model_name"],
        "evaluation_type": evaluation_type,
    }

    if evaluation_type not in DETERMINISTIC_EVALUATION_TYPES:
        return {
            **base_result,
            "metric_applicable": False,
            "metric_status": "skipped",
            "metric_passed": None,
            **_empty_metric_values(),
        }

    expected_text = (
        ""
        if pd.isna(row["expected_output"])
        else str(row["expected_output"])
    )

    output_text = (
        ""
        if pd.isna(row["model_output"])
        else str(row["model_output"])
    )

    exact_match = output_text == expected_text

    expected_normalized = normalize_text(
        expected_text,
        case_sensitive=config.metrics.case_sensitive,
        trim_whitespace=config.metrics.trim_whitespace,
        collapse_whitespace=config.metrics.collapse_whitespace,
    )

    output_normalized = normalize_text(
        output_text,
        case_sensitive=config.metrics.case_sensitive,
        trim_whitespace=config.metrics.trim_whitespace,
        collapse_whitespace=config.metrics.collapse_whitespace,
    )

    normalized_exact_match = (
        output_normalized == expected_normalized
    )

    if evaluation_type == EXACT:
        return {
            **base_result,
            "metric_applicable": True,
            "metric_status": "evaluated",
            "metric_passed": normalized_exact_match,
            "exact_match": exact_match,
            "normalized_exact_match": normalized_exact_match,
            "expected_is_json": None,
            "output_is_json": None,
            "json_exact_match": None,
            "json_expected_field_count": None,
            "json_matching_field_count": None,
            "json_field_match_rate": None,
        }

    expected_is_json, expected_json = try_parse_json(expected_text)
    output_is_json, output_json = try_parse_json(output_text)

    if not expected_is_json:
        raise ValueError(
            f"Case {row['case_id']} has evaluation_type=json, "
            "but expected_output is not valid JSON."
        )

    json_exact_match = False
    json_expected_field_count: int | None = None
    json_matching_field_count: int | None = None
    json_field_match_rate: float | None = None

    if output_is_json:
        json_exact_match = (
            canonical_json(output_json)
            == canonical_json(expected_json)
        )

        (
            json_expected_field_count,
            json_matching_field_count,
            json_field_match_rate,
        ) = compare_json_fields(expected_json, output_json)
    elif isinstance(expected_json, dict):
        json_expected_field_count = len(expected_json)
        json_matching_field_count = 0
        json_field_match_rate = 0.0

    return {
        **base_result,
        "metric_applicable": True,
        "metric_status": "evaluated",
        "metric_passed": bool(
            output_is_json and json_exact_match
        ),
        "exact_match": exact_match,
        "normalized_exact_match": normalized_exact_match,
        "expected_is_json": expected_is_json,
        "output_is_json": output_is_json,
        "json_exact_match": json_exact_match,
        "json_expected_field_count": json_expected_field_count,
        "json_matching_field_count": json_matching_field_count,
        "json_field_match_rate": json_field_match_rate,
    }


def _safe_rate(series: pd.Series) -> float | None:
    values = series.dropna()

    if values.empty:
        return None

    return float(values.astype(float).mean())


def build_metrics_summary(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []

    for model_name, group in results.groupby(
        "model_name",
        dropna=False,
    ):
        evaluated = group[group["metric_applicable"] == True]
        skipped = group[group["metric_applicable"] == False]

        exact_cases = evaluated[
            evaluated["evaluation_type"] == EXACT
        ]

        json_cases = evaluated[
            evaluated["evaluation_type"] == JSON
        ]

        records.append(
            {
                "model_name": model_name,
                "cases": int(len(group)),
                "evaluated_cases": int(len(evaluated)),
                "skipped_cases": int(len(skipped)),
                "exact_cases": int(len(exact_cases)),
                "json_cases": int(len(json_cases)),
                "metric_pass_rate": _safe_rate(
                    evaluated["metric_passed"]
                ),
                "exact_match_rate": _safe_rate(
                    exact_cases["exact_match"]
                ),
                "normalized_exact_match_rate": _safe_rate(
                    exact_cases["normalized_exact_match"]
                ),
                "valid_json_rate": _safe_rate(
                    json_cases["output_is_json"]
                ),
                "json_exact_match_rate": _safe_rate(
                    json_cases["json_exact_match"]
                ),
                "average_json_field_match_rate": _safe_rate(
                    json_cases["json_field_match_rate"]
                ),
            }
        )

    summary = pd.DataFrame(records)

    summary["rank"] = (
        summary["metric_pass_rate"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    ordered_columns = [
        "rank",
        "model_name",
        "cases",
        "evaluated_cases",
        "skipped_cases",
        "exact_cases",
        "json_cases",
        "metric_pass_rate",
        "exact_match_rate",
        "normalized_exact_match_rate",
        "valid_json_rate",
        "json_exact_match_rate",
        "average_json_field_match_rate",
    ]

    summary = summary[ordered_columns]

    return summary.sort_values(
        ["rank", "model_name"],
        ascending=[True, True],
    )


def save_metrics_summary(
    results: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)

    summary_path = output_path.with_name(
        output_path.stem + "_metrics_summary.csv"
    )

    summary = build_metrics_summary(results)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)

    return summary_path


def save_metrics_metadata(
    *,
    config_path: str | Path,
    config: MetricsConfig,
    inputs: pd.DataFrame,
    results: pd.DataFrame,
    output_path: str | Path,
    summary_path: str | Path,
) -> Path:
    output_path = Path(output_path)

    metadata_path = output_path.with_name(
        output_path.stem + "_metrics_metadata.json"
    )

    summary = build_metrics_summary(results)

    evaluated = results[
        results["metric_applicable"] == True
    ]

    skipped = results[
        results["metric_applicable"] == False
    ]

    metadata = {
        "schema_version": "0.2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evalanche_version": __version__,
        "run": {
            "name": config.run.name,
            "config_path": str(config_path),
            "input_path": str(config.run.input_path),
            "output_path": str(output_path),
            "summary_path": str(summary_path),
        },
        "hashes": {
            "config_sha256": sha256_file(config_path),
            "input_sha256": sha256_file(config.run.input_path),
            "output_sha256": sha256_file(output_path),
            "summary_sha256": sha256_file(summary_path),
        },
        "metrics": {
            "case_sensitive": config.metrics.case_sensitive,
            "trim_whitespace": config.metrics.trim_whitespace,
            "collapse_whitespace": (
                config.metrics.collapse_whitespace
            ),
        },
        "routing": {
            "supported_evaluation_types": [
                "exact",
                "json",
                "judge",
            ],
            "deterministic_evaluation_types": [
                "exact",
                "json",
            ],
        },
        "input_data": {
            "rows": int(len(inputs)),
            "columns": list(inputs.columns),
            "case_count": int(inputs["case_id"].nunique()),
            "model_count": int(inputs["model_name"].nunique()),
            "model_names": sorted(
                inputs["model_name"]
                .astype(str)
                .unique()
                .tolist()
            ),
        },
        "results": {
            "rows": int(len(results)),
            "evaluated_rows": int(len(evaluated)),
            "skipped_rows": int(len(skipped)),
            "metric_pass_rate": _safe_rate(
                evaluated["metric_passed"]
            ),
        },
        "model_summary": (
            summary.to_dict(orient="records")
            if not summary.empty
            else []
        ),
        "limitations": [
            "Deterministic metrics are only applied to cases explicitly routed to exact or json evaluation.",
            "Exact matching can reject semantically equivalent wording and should only be used for constrained outputs.",
            "Cases routed to judge evaluation are intentionally skipped by this command.",
        ],
    }

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    return metadata_path


def print_metrics_summary(summary: pd.DataFrame) -> None:
    if summary.empty:
        console.print(
            "[bold red]No metrics summary to display.[/bold red]"
        )
        return

    table = Table(title="Deterministic Metrics Summary")

    table.add_column("Rank")
    table.add_column("Model")
    table.add_column("Evaluated")
    table.add_column("Skipped")
    table.add_column("Pass rate")
    table.add_column("Exact")
    table.add_column("JSON exact")
    table.add_column("JSON fields")

    for _, row in summary.iterrows():
        table.add_row(
            str(row["rank"]),
            str(row["model_name"]),
            str(row["evaluated_cases"]),
            str(row["skipped_cases"]),
            (
                "N/A"
                if pd.isna(row["metric_pass_rate"])
                else f"{row['metric_pass_rate']:.1%}"
            ),
            (
                "N/A"
                if pd.isna(
                    row["normalized_exact_match_rate"]
                )
                else (
                    f"{row['normalized_exact_match_rate']:.1%}"
                )
            ),
            (
                "N/A"
                if pd.isna(row["json_exact_match_rate"])
                else f"{row['json_exact_match_rate']:.1%}"
            ),
            (
                "N/A"
                if pd.isna(
                    row["average_json_field_match_rate"]
                )
                else (
                    f"{row['average_json_field_match_rate']:.1%}"
                )
            ),
        )

    console.print(table)


def print_metric_failures(
    results: pd.DataFrame,
    max_rows: int = 10,
) -> None:
    failures = results[
        (results["metric_applicable"] == True)
        & (results["metric_passed"] == False)
    ].copy()

    if failures.empty:
        console.print(
            "[bold green]"
            "No deterministic metric failures."
            "[/bold green]"
        )
        return

    table = Table(
        title=f"Metric Failures, showing up to {max_rows}"
    )

    table.add_column("case_id")
    table.add_column("model_name")
    table.add_column("type")
    table.add_column("exact")
    table.add_column("json_exact")
    table.add_column("json_field_match")

    for _, row in failures.head(max_rows).iterrows():
        table.add_row(
            str(row["case_id"]),
            str(row["model_name"]),
            str(row["evaluation_type"]),
            (
                "N/A"
                if pd.isna(row["normalized_exact_match"])
                else str(row["normalized_exact_match"])
            ),
            (
                "N/A"
                if pd.isna(row["json_exact_match"])
                else str(row["json_exact_match"])
            ),
            (
                "N/A"
                if pd.isna(row["json_field_match_rate"])
                else f"{row['json_field_match_rate']:.1%}"
            ),
        )

    console.print(table)


def run_deterministic_metrics(
    *,
    config_path: str | Path,
    config: MetricsConfig,
) -> tuple[pd.DataFrame, Path, Path, Path]:
    inputs = load_metric_cases(config.run.input_path)

    metric_records = [
        score_row(row, config)
        for row in inputs.to_dict(orient="records")
    ]

    metric_results = pd.DataFrame(metric_records)

    results = inputs.merge(
        metric_results,
        on=[
            "case_id",
            "model_name",
            "evaluation_type",
        ],
        how="left",
    )

    output_path = Path(config.run.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    summary_path = save_metrics_summary(
        results,
        output_path,
    )

    metadata_path = save_metrics_metadata(
        config_path=config_path,
        config=config,
        inputs=inputs,
        results=results,
        output_path=output_path,
        summary_path=summary_path,
    )

    summary = build_metrics_summary(results)

    print_metrics_summary(summary)
    print_metric_failures(results)

    skipped_count = int(
        (results["metric_applicable"] == False).sum()
    )

    if skipped_count:
        console.print(
            f"\nSkipped {skipped_count} case(s) routed "
            "to LLM judge evaluation."
        )

    return results, output_path, summary_path, metadata_path
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


console = Console()


REQUIRED_METRIC_COLUMNS = {
    "case_id",
    "input",
    "expected_output",
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

    return df


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
    if pd.isna(value):
        return False, None

    if not isinstance(value, str):
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
    except Exception:
        return False, None


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def compare_json_fields(expected: Any, output: Any) -> tuple[int | None, int | None, float | None]:
    if not isinstance(expected, dict) or not isinstance(output, dict):
        return None, None, None

    expected_keys = list(expected.keys())

    if not expected_keys:
        return 0, 0, None

    matching = 0

    for key in expected_keys:
        if key not in output:
            continue

        if output[key] == expected[key]:
            matching += 1

    total = len(expected_keys)
    return total, matching, matching / total


def score_row(row: dict[str, Any], config: MetricsConfig) -> dict[str, Any]:
    expected_raw = row["expected_output"]
    output_raw = row["model_output"]

    expected_text = "" if pd.isna(expected_raw) else str(expected_raw)
    output_text = "" if pd.isna(output_raw) else str(output_raw)

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

    normalized_exact_match = output_normalized == expected_normalized

    expected_is_json, expected_json = try_parse_json(expected_text)
    output_is_json, output_json = try_parse_json(output_text)

    json_exact_match: bool | None = None
    json_expected_field_count: int | None = None
    json_matching_field_count: int | None = None
    json_field_match_rate: float | None = None

    if expected_is_json:
        if output_is_json:
            json_exact_match = canonical_json(output_json) == canonical_json(expected_json)
            (
                json_expected_field_count,
                json_matching_field_count,
                json_field_match_rate,
            ) = compare_json_fields(expected_json, output_json)
        else:
            json_exact_match = False
            if isinstance(expected_json, dict):
                json_expected_field_count = len(expected_json)
                json_matching_field_count = 0
                json_field_match_rate = 0.0

    if expected_is_json and config.metrics.use_json_when_expected_json:
        metric_passed = bool(output_is_json and json_exact_match)
    else:
        metric_passed = bool(normalized_exact_match)

    return {
        "case_id": row["case_id"],
        "model_name": row["model_name"],
        "exact_match": exact_match,
        "normalized_exact_match": normalized_exact_match,
        "metric_passed": metric_passed,
        "expected_is_json": expected_is_json,
        "output_is_json": output_is_json,
        "json_exact_match": json_exact_match,
        "json_expected_field_count": json_expected_field_count,
        "json_matching_field_count": json_matching_field_count,
        "json_field_match_rate": json_field_match_rate,
    }


def build_metrics_summary(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()

    aggregations = {
        "case_id": "count",
        "metric_passed": "mean",
        "exact_match": "mean",
        "normalized_exact_match": "mean",
        "expected_is_json": "sum",
        "output_is_json": "sum",
    }

    if "json_field_match_rate" in results.columns:
        aggregations["json_field_match_rate"] = "mean"

    summary = (
        results
        .groupby("model_name", dropna=False)
        .agg(aggregations)
        .reset_index()
    )

    summary = summary.rename(
        columns={
            "case_id": "cases",
            "metric_passed": "metric_pass_rate",
            "exact_match": "exact_match_rate",
            "normalized_exact_match": "normalized_exact_match_rate",
            "expected_is_json": "expected_json_cases",
            "output_is_json": "valid_json_outputs",
            "json_field_match_rate": "average_json_field_match_rate",
        }
    )

    summary["rank"] = (
        summary["metric_pass_rate"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    ordered_cols = [
        "rank",
        "model_name",
        "cases",
        "metric_pass_rate",
        "exact_match_rate",
        "normalized_exact_match_rate",
        "expected_json_cases",
        "valid_json_outputs",
        "average_json_field_match_rate",
    ]

    existing_ordered_cols = [col for col in ordered_cols if col in summary.columns]
    remaining_cols = [col for col in summary.columns if col not in existing_ordered_cols]

    summary = summary[existing_ordered_cols + remaining_cols]
    summary = summary.sort_values(["rank", "model_name"], ascending=[True, True])

    return summary


def save_metrics_summary(results: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    summary_path = output_path.with_name(output_path.stem + "_metrics_summary.csv")

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
    metadata_path = output_path.with_name(output_path.stem + "_metrics_metadata.json")

    summary = build_metrics_summary(results)

    metadata = {
        "schema_version": "0.1",
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
            "collapse_whitespace": config.metrics.collapse_whitespace,
            "use_json_when_expected_json": config.metrics.use_json_when_expected_json,
        },
        "input_data": {
            "rows": int(len(inputs)),
            "columns": list(inputs.columns),
            "case_count": int(inputs["case_id"].nunique()),
            "model_count": int(inputs["model_name"].nunique()),
            "model_names": sorted(inputs["model_name"].astype(str).unique().tolist()),
        },
        "results": {
            "rows": int(len(results)),
            "metric_pass_rate": (
                float(results["metric_passed"].mean())
                if len(results) > 0
                else None
            ),
            "exact_match_rate": (
                float(results["exact_match"].mean())
                if len(results) > 0
                else None
            ),
            "normalized_exact_match_rate": (
                float(results["normalized_exact_match"].mean())
                if len(results) > 0
                else None
            ),
        },
        "model_summary": (
            summary.to_dict(orient="records")
            if not summary.empty
            else []
        ),
        "limitations": [
            "Deterministic metrics are strict and may mark semantically equivalent outputs as incorrect.",
            "Exact-match metrics are most appropriate for classification, extraction, JSON, numeric, and constrained-output tasks.",
            "Open-ended generation tasks should also use rubric-based evaluation or human review.",
        ],
    }

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return metadata_path


def print_metrics_summary(summary: pd.DataFrame) -> None:
    if summary.empty:
        console.print("[bold red]No metrics summary to display.[/bold red]")
        return

    table = Table(title="Deterministic Metrics Summary")
    table.add_column("Rank")
    table.add_column("Model")
    table.add_column("Cases")
    table.add_column("Pass rate")
    table.add_column("Exact")
    table.add_column("Normalized exact")
    table.add_column("JSON field match")

    for _, row in summary.iterrows():
        json_field_match = row.get("average_json_field_match_rate")

        table.add_row(
            str(row["rank"]),
            str(row["model_name"]),
            str(row["cases"]),
            f"{row['metric_pass_rate']:.1%}",
            f"{row['exact_match_rate']:.1%}",
            f"{row['normalized_exact_match_rate']:.1%}",
            "N/A" if pd.isna(json_field_match) else f"{json_field_match:.1%}",
        )

    console.print(table)


def print_metric_failures(results: pd.DataFrame, max_rows: int = 10) -> None:
    failures = results[results["metric_passed"] == False].copy()

    if failures.empty:
        console.print("[bold green]No deterministic metric failures.[/bold green]")
        return

    table = Table(title=f"Metric Failures, showing up to {max_rows}")
    table.add_column("case_id")
    table.add_column("model_name")
    table.add_column("exact")
    table.add_column("normalized")
    table.add_column("json_exact")
    table.add_column("json_field_match")

    for _, row in failures.head(max_rows).iterrows():
        json_exact = row.get("json_exact_match")
        json_field_match = row.get("json_field_match_rate")

        table.add_row(
            str(row["case_id"]),
            str(row["model_name"]),
            str(row["exact_match"]),
            str(row["normalized_exact_match"]),
            "N/A" if pd.isna(json_exact) else str(json_exact),
            "N/A" if pd.isna(json_field_match) else f"{json_field_match:.1%}",
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
        on=["case_id", "model_name"],
        how="left",
    )

    output_path = Path(config.run.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    summary_path = save_metrics_summary(results, output_path)

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

    return results, output_path, summary_path, metadata_path
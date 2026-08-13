from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from integrations.inspect_ai.adapter import json_safe


def _correct(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().casefold() in {"c", "correct", "true", "1"}


def _find_dpd_score(scores: Any) -> Any | None:
    if not isinstance(scores, dict):
        return None
    for score in scores.values():
        value = getattr(score, "value", None)
        if isinstance(value, dict) and {
            "strict",
            "field_score",
            "valid_json",
        } <= set(value):
            return score
    return None


def _usage_totals(model_usage: Any) -> dict[str, int | float]:
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
        "generation_cost_usd": 0.0,
    }
    if not isinstance(model_usage, dict):
        return totals
    for usage in model_usage.values():
        for key in (
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "total_tokens",
        ):
            value = getattr(usage, key, None)
            if isinstance(value, int):
                totals[key] += value
        cost = getattr(usage, "total_cost", None)
        if isinstance(cost, (int, float)):
            totals["generation_cost_usd"] += float(cost)
    return totals


def _metadata_int(metadata: dict[str, Any], name: str) -> int:
    value = metadata.get(name)
    if value is None or str(value).strip() == "":
        return 0
    return int(float(str(value)))


def _metadata_float(
    metadata: dict[str, Any], name: str
) -> float | None:
    value = metadata.get(name)
    if value is None or str(value).strip() == "":
        return None
    return float(str(value))


def _historical_usage(metadata: dict[str, Any]) -> dict[str, int | float]:
    return {
        "input_tokens": _metadata_int(
            metadata, "generation_prompt_tokens"
        ),
        "output_tokens": _metadata_int(
            metadata, "generation_completion_tokens"
        ),
        "reasoning_tokens": _metadata_int(
            metadata, "generation_reasoning_tokens"
        ),
        "total_tokens": _metadata_int(
            metadata, "generation_total_tokens"
        ),
        "generation_cost_usd": _metadata_float(
            metadata, "generation_cost_usd"
        )
        or 0.0,
    }


def records_from_logs(log_dir: str | Path) -> tuple[pd.DataFrame, list[str]]:
    try:
        from inspect_ai.log import list_eval_logs, read_eval_log
    except ImportError as error:
        raise RuntimeError(
            "Inspect AI is not installed. Install requirements-inspect.txt."
        ) from error

    log_infos = list_eval_logs(str(Path(log_dir).resolve()), recursive=True)
    if not log_infos:
        raise FileNotFoundError(f"No Inspect evaluation logs found in {log_dir}.")

    records: list[dict[str, Any]] = []
    log_locations: list[str] = []
    for log_order, info in enumerate(
        sorted(log_infos, key=lambda item: str(item.name))
    ):
        log = read_eval_log(info)
        location = str(getattr(log, "location", None) or info.name)
        log_locations.append(location)
        eval_metadata = getattr(log.eval, "metadata", None) or {}
        eval_model = str(getattr(log.eval, "model", "unknown"))
        for sample in log.samples or []:
            metadata = sample.metadata or {}
            score = _find_dpd_score(sample.scores)
            value = getattr(score, "value", {}) if score else {}
            score_metadata = getattr(score, "metadata", {}) if score else {}
            usage = _usage_totals(sample.model_usage)
            if not int(usage["total_tokens"]):
                historical_usage = _historical_usage(metadata)
                if int(historical_usage["total_tokens"]):
                    usage = historical_usage
            error = getattr(sample, "error", None)
            parity = (score_metadata or {}).get("replay_parity")
            parity_checked = isinstance(parity, dict) and bool(parity)
            parity_values = (
                [bool(item) for item in parity.values()]
                if parity_checked
                else []
            )
            execution_mode = str(
                metadata.get("execution_mode")
                or eval_metadata.get("evalanche_execution_mode")
                or "live"
            )
            generation_time = _metadata_float(
                metadata, "generation_seconds"
            )
            if generation_time is None:
                generation_time = getattr(sample, "working_time", None)
            if generation_time is None:
                generation_time = getattr(sample, "total_time", None)
            records.append(
                {
                    "log_order": log_order,
                    "log_location": location,
                    "log_status": str(log.status),
                    "model_id": str(
                        metadata.get("historical_model_id")
                        or eval_metadata.get("evalanche_model_id")
                        or eval_model
                    ),
                    "inspect_model": eval_model,
                    "case_id": str(metadata.get("case_id", sample.id)),
                    "sample_id": str(sample.id),
                    "execution_mode": execution_mode,
                    "product_id": metadata.get("product_id"),
                    "language": metadata.get("language"),
                    "stratum": metadata.get("stratum"),
                    "strict_passed": (
                        _correct(value.get("strict")) if score else False
                    ),
                    "field_score": (
                        float(value.get("field_score", 0.0))
                        if score
                        else 0.0
                    ),
                    "valid_json": (
                        _correct(value.get("valid_json")) if score else False
                    ),
                    "mismatched_fields": json.dumps(
                        (score_metadata or {}).get(
                            "json_mismatched_fields", []
                        ),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    "parity_checked": parity_checked,
                    "parity_all": (
                        all(parity_values) if parity_checked else None
                    ),
                    "parity_strict": (
                        bool(parity.get("strict"))
                        if parity_checked and "strict" in parity
                        else None
                    ),
                    "parity_field_score": (
                        bool(parity.get("field_score"))
                        if parity_checked and "field_score" in parity
                        else None
                    ),
                    "parity_mismatched_fields": (
                        bool(parity.get("mismatched_fields"))
                        if parity_checked and "mismatched_fields" in parity
                        else None
                    ),
                    "parity_valid_json": (
                        bool(parity.get("valid_json"))
                        if parity_checked and "valid_json" in parity
                        else None
                    ),
                    "historical_run_id": (
                        metadata.get("historical_run_id")
                        or eval_metadata.get(
                            "evalanche_historical_run_id"
                        )
                    ),
                    "historical_source_sha256": eval_metadata.get(
                        "evalanche_historical_source_sha256"
                    ),
                    "sample_error": (
                        json.dumps(json_safe(error), ensure_ascii=False)
                        if error is not None
                        else ""
                    ),
                    "total_time_seconds": getattr(
                        sample, "total_time", None
                    ),
                    "working_time_seconds": getattr(
                        sample, "working_time", None
                    ),
                    "generation_time_seconds": generation_time,
                    **usage,
                }
            )
    frame = pd.DataFrame(records)
    if frame.empty:
        raise RuntimeError(
            f"Inspect logs in {log_dir} contain no sample records."
        )
    frame = frame.sort_values("log_order", kind="stable")
    frame = frame.drop_duplicates(
        subset=["model_id", "sample_id"], keep="last"
    ).drop(columns=["log_order"])
    return frame.reset_index(drop=True), log_locations


def _group_records(
    frame: pd.DataFrame,
    columns: Iterable[str],
) -> list[dict[str, Any]]:
    group_columns = list(columns)
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys, strict=True))
        rows.append(
            {
                **row,
                "cases": int(len(group)),
                "strict_pass_rate": float(group["strict_passed"].mean()),
                "mean_field_score": float(group["field_score"].mean()),
                "valid_json_rate": float(group["valid_json"].mean()),
                "sample_errors": int(
                    group["sample_error"].astype(bool).sum()
                ),
                "parity_checked": int(group["parity_checked"].sum()),
                "parity_disagreements": int(
                    (
                        group["parity_checked"]
                        & group["parity_all"].eq(False)
                    ).sum()
                ),
                "input_tokens": int(group["input_tokens"].sum()),
                "output_tokens": int(group["output_tokens"].sum()),
                "reasoning_tokens": int(
                    group["reasoning_tokens"].sum()
                ),
                "total_tokens": int(group["total_tokens"].sum()),
                "generation_cost_usd": float(
                    group["generation_cost_usd"].sum()
                ),
                "mean_generation_time_seconds": (
                    float(
                        group["generation_time_seconds"].dropna().mean()
                    )
                    if group["generation_time_seconds"].notna().any()
                    else None
                ),
                "mean_total_time_seconds": (
                    float(group["total_time_seconds"].dropna().mean())
                    if group["total_time_seconds"].notna().any()
                    else None
                ),
            }
        )
    return json_safe(rows)


def _summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Inspect AI DPD campaign summary",
        "",
        f"Status: **{summary['status']}**",
        "",
        "This is an optional framework-comparison artifact. It is not merged "
        "into the published Evalanche leaderboard automatically.",
        "",
        "## Model results",
        "",
        "| Model | Cases | Strict pass | Mean field score | Valid JSON | Errors | Parity |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["models"]:
        parity = (
            f"{row['parity_checked'] - row['parity_disagreements']}/"
            f"{row['parity_checked']}"
            if row["parity_checked"]
            else "N/A"
        )
        lines.append(
            "| {model_id} | {cases} | {strict_pass_rate:.4f} | "
            "{mean_field_score:.4f} | {valid_json_rate:.4f} | "
            "{sample_errors} | {parity} |".format(
                **row, parity=parity
            )
        )
    lines.extend(
        [
            "",
            "See `case_results.csv.gz` for case-level results and "
            "`summary.json` for language and stratum slices.",
            "",
        ]
    )
    return "\n".join(lines)


def write_campaign_report(
    log_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    frame, log_locations = records_from_logs(log_dir)
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    frame = frame.sort_values(
        ["model_id", "case_id", "sample_id"],
        kind="stable",
    ).reset_index(drop=True)
    frame.to_csv(
        output_path / "case_results.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    parity_checked = frame["parity_checked"].astype(bool)
    parity_disagreements = parity_checked & frame["parity_all"].eq(False)
    if frame["sample_error"].astype(bool).any():
        status = "complete_with_sample_errors"
    elif parity_disagreements.any():
        status = "complete_with_parity_disagreements"
    else:
        status = "complete"
    summary = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "log_count": len(log_locations),
        "sample_count": int(len(frame)),
        "log_locations": log_locations,
        "execution_modes": sorted(
            set(frame["execution_mode"].astype(str))
        ),
        "parity": {
            "status": (
                "passed"
                if parity_checked.any() and not parity_disagreements.any()
                else (
                    "failed"
                    if parity_disagreements.any()
                    else "not_checked"
                )
            ),
            "checked_samples": int(parity_checked.sum()),
            "disagreements": int(parity_disagreements.sum()),
            "strict_disagreements": int(
                (parity_checked & frame["parity_strict"].eq(False)).sum()
            ),
            "field_score_disagreements": int(
                (
                    parity_checked
                    & frame["parity_field_score"].eq(False)
                ).sum()
            ),
            "mismatched_field_disagreements": int(
                (
                    parity_checked
                    & frame["parity_mismatched_fields"].eq(False)
                ).sum()
            ),
            "valid_json_disagreements": int(
                (
                    parity_checked
                    & frame["parity_valid_json"].eq(False)
                ).sum()
            ),
        },
        "models": _group_records(frame, ["model_id"]),
        "by_language": _group_records(frame, ["model_id", "language"]),
        "by_stratum": _group_records(frame, ["model_id", "stratum"]),
    }
    (output_path / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_path / "summary.md").write_text(
        _summary_markdown(summary),
        encoding="utf-8",
    )
    return summary

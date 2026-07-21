from __future__ import annotations

from typing import Any

import pandas as pd


COST_CURRENCY = "USD"
COST_POLICY = "litellm_response_metadata_only"


def _numeric_values(
    frame: pd.DataFrame,
    column: str,
    request_mask: pd.Series,
) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(index=frame.index, dtype="float64")

    return pd.to_numeric(
        frame.loc[request_mask, column],
        errors="coerce",
    )


def _complete_sum(
    values: pd.Series,
    request_count: int,
    *,
    integer: bool,
) -> int | float | None:
    if request_count == 0 or int(values.notna().sum()) != request_count:
        return None

    total = float(values.sum())
    return int(total) if integer else total


def summarize_stage(
    frame: pd.DataFrame,
    prefix: str,
) -> dict[str, Any]:
    """Summarize one operational stage without treating missing data as zero."""
    status_column = f"{prefix}_status"

    if frame.empty or status_column not in frame.columns:
        return {
            "requests": 0,
            "successes": 0,
            "errors": 0,
            "failure_rate": None,
            "average_seconds": None,
            "p95_seconds": None,
            "latency_observations": 0,
            "latency_coverage": None,
            "api_attempts": None,
            "failed_api_attempts": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "token_observations": 0,
            "token_coverage": None,
            "cost_usd": None,
            "cost_observations": 0,
            "cost_coverage": None,
            "cost_complete": None,
            "cost_sources": "",
        }

    statuses = (
        frame[status_column]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    request_mask = statuses.isin({"success", "error"})
    request_count = int(request_mask.sum())
    success_count = int((statuses == "success").sum())
    error_count = int((statuses == "error").sum())

    seconds = _numeric_values(
        frame,
        f"{prefix}_seconds",
        request_mask,
    )
    seconds_observed = seconds.dropna()
    latency_observations = int(len(seconds_observed))

    attempts = _numeric_values(
        frame,
        f"{prefix}_attempts",
        request_mask,
    )
    failed_attempts = _numeric_values(
        frame,
        f"{prefix}_failed_attempts",
        request_mask,
    )
    prompt_tokens = _numeric_values(
        frame,
        f"{prefix}_prompt_tokens",
        request_mask,
    )
    completion_tokens = _numeric_values(
        frame,
        f"{prefix}_completion_tokens",
        request_mask,
    )
    total_tokens = _numeric_values(
        frame,
        f"{prefix}_total_tokens",
        request_mask,
    )
    token_observations = int(total_tokens.notna().sum())

    costs = _numeric_values(
        frame,
        f"{prefix}_cost_usd",
        request_mask,
    )
    cost_observations = int(costs.notna().sum())
    cost_complete = (
        cost_observations == request_count
        if request_count > 0
        else None
    )

    source_column = f"{prefix}_cost_source"
    if source_column in frame.columns:
        sources = sorted(
            {
                str(value).strip()
                for value in frame.loc[request_mask, source_column].dropna()
                if str(value).strip()
            }
        )
    else:
        sources = []

    return {
        "requests": request_count,
        "successes": success_count,
        "errors": error_count,
        "failure_rate": (
            error_count / request_count
            if request_count > 0
            else None
        ),
        "average_seconds": (
            float(seconds_observed.mean())
            if latency_observations
            else None
        ),
        "p95_seconds": (
            float(seconds_observed.quantile(0.95))
            if latency_observations
            else None
        ),
        "latency_observations": latency_observations,
        "latency_coverage": (
            latency_observations / request_count
            if request_count > 0
            else None
        ),
        "api_attempts": _complete_sum(
            attempts,
            request_count,
            integer=True,
        ),
        "failed_api_attempts": _complete_sum(
            failed_attempts,
            request_count,
            integer=True,
        ),
        "prompt_tokens": _complete_sum(
            prompt_tokens,
            request_count,
            integer=True,
        ),
        "completion_tokens": _complete_sum(
            completion_tokens,
            request_count,
            integer=True,
        ),
        "total_tokens": _complete_sum(
            total_tokens,
            request_count,
            integer=True,
        ),
        "token_observations": token_observations,
        "token_coverage": (
            token_observations / request_count
            if request_count > 0
            else None
        ),
        "cost_usd": _complete_sum(
            costs,
            request_count,
            integer=False,
        ),
        "cost_observations": cost_observations,
        "cost_coverage": (
            cost_observations / request_count
            if request_count > 0
            else None
        ),
        "cost_complete": cost_complete,
        "cost_sources": ", ".join(sources),
    }


def summarize_operations(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "cost_currency": COST_CURRENCY,
        "cost_policy": COST_POLICY,
        "generation": summarize_stage(frame, "generation"),
        "judge": summarize_stage(frame, "judge"),
    }
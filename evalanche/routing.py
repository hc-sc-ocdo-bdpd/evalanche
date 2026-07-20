from __future__ import annotations

from typing import Final

import pandas as pd


EXACT: Final[str] = "exact"
JSON: Final[str] = "json"
JUDGE: Final[str] = "judge"

SUPPORTED_EVALUATION_TYPES: Final[set[str]] = {
    EXACT,
    JSON,
    JUDGE,
}

DETERMINISTIC_EVALUATION_TYPES: Final[set[str]] = {
    EXACT,
    JSON,
}


def normalize_evaluation_type(value: object) -> str:
    if pd.isna(value):
        raise ValueError("evaluation_type cannot be empty")

    normalized = str(value).strip().lower()

    if normalized not in SUPPORTED_EVALUATION_TYPES:
        supported = ", ".join(sorted(SUPPORTED_EVALUATION_TYPES))
        raise ValueError(
            f"Unsupported evaluation_type '{value}'. "
            f"Supported values are: {supported}"
        )

    return normalized


def validate_evaluation_types(
    df: pd.DataFrame,
    *,
    source_name: str,
) -> pd.DataFrame:
    if "evaluation_type" not in df.columns:
        raise ValueError(
            f"{source_name} is missing required column: evaluation_type"
        )

    validated = df.copy()

    try:
        validated["evaluation_type"] = validated["evaluation_type"].apply(
            normalize_evaluation_type
        )
    except ValueError as exc:
        raise ValueError(f"Invalid evaluation routing in {source_name}: {exc}") from exc

    return validated
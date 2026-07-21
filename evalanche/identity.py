from __future__ import annotations

import pandas as pd


CASE_DEFINITION_COLUMNS = (
    "input",
    "expected_output",
    "evaluation_type",
)


def _normalize_identifier_column(
    frame: pd.DataFrame,
    *,
    column: str,
    source_name: str,
) -> None:
    missing_mask = frame[column].isna()
    if missing_mask.any():
        rows = frame.index[missing_mask].tolist()
        raise ValueError(
            f"{source_name} contains missing {column} values "
            f"at rows: {rows[:10]}"
        )

    normalized = frame[column].astype(str).str.strip()
    blank_mask = normalized.eq("")
    if blank_mask.any():
        rows = frame.index[blank_mask].tolist()
        raise ValueError(
            f"{source_name} contains blank {column} values "
            f"at rows: {rows[:10]}"
        )

    frame[column] = normalized


def validate_generation_case_ids(
    cases: pd.DataFrame,
    *,
    source_name: str,
) -> pd.DataFrame:
    validated = cases.copy()
    _normalize_identifier_column(
        validated,
        column="case_id",
        source_name=source_name,
    )

    duplicate_mask = validated.duplicated("case_id", keep=False)
    if duplicate_mask.any():
        duplicate_ids = sorted(
            validated.loc[duplicate_mask, "case_id"].unique().tolist()
        )
        raise ValueError(
            f"{source_name} contains duplicate case_id values: "
            f"{duplicate_ids[:10]}"
        )

    return validated


def validate_evaluation_keys(
    cases: pd.DataFrame,
    *,
    source_name: str,
) -> pd.DataFrame:
    validated = cases.copy()

    for column in ("case_id", "model_name"):
        _normalize_identifier_column(
            validated,
            column=column,
            source_name=source_name,
        )

    key_columns = ["case_id", "model_name"]
    duplicate_mask = validated.duplicated(key_columns, keep=False)
    if duplicate_mask.any():
        duplicate_pairs = (
            validated.loc[duplicate_mask, key_columns]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        formatted_pairs = [
            f"({case_id!r}, {model_name!r})"
            for case_id, model_name in duplicate_pairs
        ]
        raise ValueError(
            f"{source_name} contains duplicate "
            "(case_id, model_name) pairs: "
            f"{formatted_pairs[:10]}"
        )

    inconsistent_details: list[str] = []

    for column in CASE_DEFINITION_COLUMNS:
        if column not in validated.columns:
            continue

        distinct_counts = validated.groupby("case_id")[column].nunique(
            dropna=False
        )
        inconsistent_ids = sorted(
            distinct_counts[distinct_counts > 1]
            .index.astype(str)
            .tolist()
        )

        if inconsistent_ids:
            inconsistent_details.append(
                f"{column}: {inconsistent_ids[:10]}"
            )

    if inconsistent_details:
        raise ValueError(
            f"{source_name} reuses case_id values for inconsistent "
            "case definitions ("
            + "; ".join(inconsistent_details)
            + ")"
        )

    return validated



def validate_balanced_model_coverage(
    cases: pd.DataFrame,
    *,
    source_name: str,
) -> None:
    if cases.empty:
        return

    case_sets = {
        str(model_name): set(group["case_id"].astype(str))
        for model_name, group in cases.groupby("model_name", sort=True)
    }

    all_case_ids = set().union(*case_sets.values())
    incomplete_models: list[str] = []

    for model_name, model_case_ids in case_sets.items():
        missing_ids = sorted(all_case_ids - model_case_ids)
        if missing_ids:
            incomplete_models.append(
                f"{model_name!r} is missing {missing_ids[:10]}"
            )

    if incomplete_models:
        raise ValueError(
            f"{source_name} does not evaluate every model on the same "
            "case set: "
            + "; ".join(incomplete_models)
        )
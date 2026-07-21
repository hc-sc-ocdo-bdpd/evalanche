from __future__ import annotations

from itertools import combinations
from math import comb
from pathlib import Path
from statistics import NormalDist
from typing import Any, Sequence

import pandas as pd


CONFIDENCE_LEVEL = 0.95
SIGNIFICANCE_LEVEL = 0.05


PAIRWISE_COLUMNS = [
    "model_a",
    "model_b",
    "paired_cases",
    "excluded_cases",
    "model_a_pass_rate",
    "model_b_pass_rate",
    "pass_rate_difference",
    "model_a_only_passed",
    "model_b_only_passed",
    "both_passed",
    "both_failed",
    "average_score_difference",
    "exact_mcnemar_p_value",
    "holm_adjusted_p_value",
    "statistically_distinguishable",
    "clear_winner",
]


def wilson_score_interval(
    passed: int,
    total: int,
    *,
    confidence_level: float = CONFIDENCE_LEVEL,
) -> tuple[float, float]:
    """Return a Wilson score interval for a binomial pass rate."""
    if total <= 0:
        raise ValueError("total must be greater than zero")
    if passed < 0 or passed > total:
        raise ValueError("passed must be between zero and total")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between zero and one")

    proportion = passed / total
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2)
    z_squared = z**2
    denominator = 1 + z_squared / total
    centre = (proportion + z_squared / (2 * total)) / denominator
    margin = (
        z
        * (
            (
                proportion * (1 - proportion) / total
                + z_squared / (4 * total**2)
            )
            ** 0.5
        )
        / denominator
    )

    return max(0.0, centre - margin), min(1.0, centre + margin)


def exact_mcnemar_p_value(
    model_a_only_passed: int,
    model_b_only_passed: int,
) -> float:
    """Return the two-sided exact McNemar p-value."""
    if model_a_only_passed < 0 or model_b_only_passed < 0:
        raise ValueError("discordant counts cannot be negative")

    discordant = model_a_only_passed + model_b_only_passed
    if discordant == 0:
        return 1.0

    smaller_count = min(
        model_a_only_passed,
        model_b_only_passed,
    )
    lower_tail = sum(
        comb(discordant, value)
        for value in range(smaller_count + 1)
    ) / (2**discordant)

    return min(1.0, 2 * lower_tail)


def holm_adjust_p_values(p_values: Sequence[float]) -> list[float]:
    """Apply Holm's family-wise error correction to p-values."""
    if any(value < 0 or value > 1 for value in p_values):
        raise ValueError("p-values must be between zero and one")
    if not p_values:
        return []

    ordered_indexes = sorted(
        range(len(p_values)),
        key=lambda index: p_values[index],
    )
    adjusted = [0.0] * len(p_values)
    running_max = 0.0

    for order, original_index in enumerate(ordered_indexes):
        multiplier = len(p_values) - order
        candidate = min(1.0, multiplier * p_values[original_index])
        running_max = max(running_max, candidate)
        adjusted[original_index] = running_max

    return adjusted


def _empty_pairwise_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=PAIRWISE_COLUMNS)


def build_pairwise_comparisons(results: pd.DataFrame) -> pd.DataFrame:
    """Compare every model pair on their shared pass/fail outcomes."""
    if results.empty or results["model_name"].nunique() < 2:
        return _empty_pairwise_frame()

    required_columns = {
        "case_id",
        "model_name",
        "final_passed",
        "final_score",
    }
    missing_columns = sorted(required_columns - set(results.columns))
    if missing_columns:
        raise ValueError(
            "pairwise comparisons require columns: "
            + ", ".join(missing_columns)
        )

    duplicate_mask = results.duplicated(
        ["case_id", "model_name"],
        keep=False,
    )
    if duplicate_mask.any():
        raise ValueError(
            "pairwise comparisons require one result per case and model"
        )

    coverage_by_case = (
        results.assign(_present=True)
        .pivot(
            index="case_id",
            columns="model_name",
            values="_present",
        )
    )

    if coverage_by_case.isna().any().any():
        raise ValueError(
            "pairwise comparisons require every model to have a result "
            "for every case"
        )

    pass_by_case = results.pivot(
        index="case_id",
        columns="model_name",
        values="final_passed",
    )
    score_by_case = results.pivot(
        index="case_id",
        columns="model_name",
        values="final_score",
    )

    records: list[dict[str, Any]] = []
    model_names = sorted(pass_by_case.columns.astype(str).tolist())

    for model_a, model_b in combinations(model_names, 2):
        paired_mask = (
            pass_by_case[model_a].notna()
            & pass_by_case[model_b].notna()
            & score_by_case[model_a].notna()
            & score_by_case[model_b].notna()
        )
        model_a_passed = (
            pass_by_case.loc[paired_mask, model_a].astype(bool)
        )
        model_b_passed = (
            pass_by_case.loc[paired_mask, model_b].astype(bool)
        )
        model_a_only = int(
            (model_a_passed & ~model_b_passed).sum()
        )
        model_b_only = int(
            (~model_a_passed & model_b_passed).sum()
        )
        both_passed = int((model_a_passed & model_b_passed).sum())
        both_failed = int((~model_a_passed & ~model_b_passed).sum())
        paired_cases = int(len(model_a_passed))
        model_a_rate = (
            float(model_a_passed.mean())
            if paired_cases
            else None
        )
        model_b_rate = (
            float(model_b_passed.mean())
            if paired_cases
            else None
        )
        rate_difference = (
            model_a_rate - model_b_rate
            if model_a_rate is not None
            and model_b_rate is not None
            else None
        )
        average_score_difference = (
            float(
                (
                    score_by_case.loc[
                        paired_mask,
                        model_a,
                    ].astype(float)
                    - score_by_case.loc[
                        paired_mask,
                        model_b,
                    ].astype(float)
                ).mean()
            )
            if paired_cases
            else None
        )

        records.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "paired_cases": paired_cases,
                "excluded_cases": int(
                    len(pass_by_case) - paired_cases
                ),
                "model_a_pass_rate": model_a_rate,
                "model_b_pass_rate": model_b_rate,
                "pass_rate_difference": rate_difference,
                "model_a_only_passed": model_a_only,
                "model_b_only_passed": model_b_only,
                "both_passed": both_passed,
                "both_failed": both_failed,
                "average_score_difference": (
                    average_score_difference
                ),
                "exact_mcnemar_p_value": exact_mcnemar_p_value(
                    model_a_only,
                    model_b_only,
                ),
            }
        )

    comparisons = pd.DataFrame(records)
    comparisons["holm_adjusted_p_value"] = holm_adjust_p_values(
        comparisons["exact_mcnemar_p_value"].tolist()
    )
    comparisons["statistically_distinguishable"] = (
        comparisons["holm_adjusted_p_value"] < SIGNIFICANCE_LEVEL
    )

    def clear_winner(row: pd.Series) -> str | None:
        if not row["statistically_distinguishable"]:
            return None
        if pd.isna(row["pass_rate_difference"]):
            return None
        if row["pass_rate_difference"] > 0:
            return str(row["model_a"])
        if row["pass_rate_difference"] < 0:
            return str(row["model_b"])
        return None

    comparisons["clear_winner"] = comparisons.apply(
        clear_winner,
        axis=1,
    )

    return comparisons[PAIRWISE_COLUMNS]


def save_pairwise_comparisons(
    comparisons: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    comparison_path = output_path.with_name(
        output_path.stem + "_pairwise_comparisons.csv"
    )
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparisons.to_csv(comparison_path, index=False)
    return comparison_path
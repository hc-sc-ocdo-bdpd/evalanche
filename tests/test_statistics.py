import pandas as pd
import pytest

from evalanche.statistics import (
    build_pairwise_comparisons,
    exact_mcnemar_p_value,
    holm_adjust_p_values,
    wilson_score_interval,
)


def make_results(
    model_passes: dict[str, list[bool]],
) -> pd.DataFrame:
    records = []

    for model_name, pass_values in model_passes.items():
        for index, passed in enumerate(pass_values, start=1):
            records.append(
                {
                    "case_id": f"case_{index:03d}",
                    "model_name": model_name,
                    "final_passed": passed,
                    "final_score": float(passed),
                }
            )

    return pd.DataFrame(records)


def test_wilson_interval_for_five_of_six() -> None:
    low, high = wilson_score_interval(5, 6)

    assert low == pytest.approx(0.4365, abs=0.0001)
    assert high == pytest.approx(0.9699, abs=0.0001)


def test_wilson_interval_is_not_zero_width_for_all_passes() -> None:
    low, high = wilson_score_interval(6, 6)

    assert low == pytest.approx(0.6097, abs=0.0001)
    assert high == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("passed", "total", "confidence_level"),
    [
        (0, 0, 0.95),
        (2, 1, 0.95),
        (1, 2, 1.0),
    ],
)
def test_wilson_interval_rejects_invalid_inputs(
    passed: int,
    total: int,
    confidence_level: float,
) -> None:
    with pytest.raises(ValueError):
        wilson_score_interval(
            passed,
            total,
            confidence_level=confidence_level,
        )


def test_exact_mcnemar_returns_one_without_disagreements() -> None:
    assert exact_mcnemar_p_value(0, 0) == 1.0


def test_exact_mcnemar_detects_six_one_sided_disagreements() -> None:
    assert exact_mcnemar_p_value(6, 0) == pytest.approx(0.03125)


def test_exact_mcnemar_is_symmetric() -> None:
    assert exact_mcnemar_p_value(2, 7) == exact_mcnemar_p_value(7, 2)


def test_holm_adjustment_preserves_ordered_familywise_control() -> None:
    adjusted = holm_adjust_p_values([0.01, 0.04, 0.03])

    assert adjusted == pytest.approx([0.03, 0.06, 0.06])


def test_pairwise_comparison_counts_paired_outcomes() -> None:
    results = make_results(
        {
            "model_a": [True, True, False, False],
            "model_b": [True, False, True, False],
        }
    )

    comparison = build_pairwise_comparisons(results).iloc[0]

    assert comparison["paired_cases"] == 4
    assert comparison["both_passed"] == 1
    assert comparison["model_a_only_passed"] == 1
    assert comparison["model_b_only_passed"] == 1
    assert comparison["both_failed"] == 1
    assert comparison["pass_rate_difference"] == 0
    assert comparison["clear_winner"] is None


def test_pairwise_comparison_can_identify_clear_winner() -> None:
    results = make_results(
        {
            "model_a": [True] * 6,
            "model_b": [False] * 6,
        }
    )

    comparison = build_pairwise_comparisons(results).iloc[0]

    assert comparison["exact_mcnemar_p_value"] == pytest.approx(0.03125)
    assert comparison["holm_adjusted_p_value"] == pytest.approx(0.03125)
    assert bool(comparison["statistically_distinguishable"]) is True
    assert comparison["clear_winner"] == "model_a"


def test_pairwise_comparison_does_not_overclaim_one_difference() -> None:
    results = make_results(
        {
            "model_a": [True, True, True, True, True, True],
            "model_b": [False, True, True, True, True, True],
        }
    )

    comparison = build_pairwise_comparisons(results).iloc[0]

    assert comparison["pass_rate_difference"] == pytest.approx(1 / 6)
    assert bool(comparison["statistically_distinguishable"]) is False
    assert comparison["clear_winner"] is None


def test_pairwise_comparison_rejects_incomplete_case_coverage() -> None:
    results = make_results(
        {
            "model_a": [True, False],
            "model_b": [True],
        }
    )

    with pytest.raises(ValueError, match="every model"):
        build_pairwise_comparisons(results)
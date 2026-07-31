from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from evalanche.config import (
    DeterministicMetricsSettingsConfig,
    JsonComparisonSettingsConfig,
    MetricsConfig,
    RunConfig,
)
from evalanche.metrics.deterministic import (
    build_metrics_summary,
    canonical_json,
    normalize_text,
    score_row,
    try_parse_json,
)


def make_config(
    *,
    metrics: DeterministicMetricsSettingsConfig | None = None,
) -> MetricsConfig:
    return MetricsConfig(
        run=RunConfig(
            name="test",
            input_path=Path("input.csv"),
            output_path=Path("output.csv"),
        ),
        metrics=(
            metrics
            if metrics is not None
            else DeterministicMetricsSettingsConfig()
        ),
    )


def make_row(
    *,
    evaluation_type: str,
    expected_output: str,
    model_output: str,
    case_id: str = "case_001",
    model_name: str = "model_a",
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "input": "Test input",
        "expected_output": expected_output,
        "evaluation_type": evaluation_type,
        "model_name": model_name,
        "model_output": model_output,
    }


def test_normalize_text_applies_requested_normalization() -> None:
    normalized = normalize_text(
        "  Multi\n  LINE Text  ",
        case_sensitive=False,
        trim_whitespace=True,
        collapse_whitespace=True,
    )

    assert normalized == "multi line text"


def test_normalize_text_can_ignore_diacritics_and_punctuation() -> None:
    normalized = normalize_text(
        "  Shampoing Reddy-Clobétasol™  ",
        case_sensitive=False,
        trim_whitespace=True,
        collapse_whitespace=True,
        strip_diacritics=True,
        strip_punctuation=True,
    )

    assert normalized == "shampoing reddy clobetasol"


def test_try_parse_json_accepts_json_code_fence() -> None:
    parsed, value = try_parse_json('```json\n{"name": "Jordan"}\n```')

    assert parsed is True
    assert value == {"name": "Jordan"}


def test_try_parse_json_rejects_invalid_json() -> None:
    parsed, value = try_parse_json("Jordan paid 42.50")

    assert parsed is False
    assert value is None


def test_canonical_json_ignores_object_key_order() -> None:
    first = canonical_json({"name": "Jordan", "amount": 42.5})
    second = canonical_json({"amount": 42.5, "name": "Jordan"})

    assert first == second


def test_exact_metric_distinguishes_raw_and_normalized_match() -> None:
    result = score_row(
        make_row(
            evaluation_type="exact",
            expected_output="Negative",
            model_output="  negative  ",
        ),
        make_config(),
    )

    assert result["metric_applicable"] is True
    assert result["exact_match"] is False
    assert result["normalized_exact_match"] is True
    assert result["metric_passed"] is True


def test_judge_case_is_skipped_by_deterministic_metrics() -> None:
    result = score_row(
        make_row(
            evaluation_type="judge",
            expected_output="A clear rewrite.",
            model_output="Another valid clear rewrite.",
        ),
        make_config(),
    )

    assert result["metric_applicable"] is False
    assert result["metric_status"] == "skipped"
    assert result["metric_passed"] is None


def test_json_metric_accepts_equivalent_key_order() -> None:
    result = score_row(
        make_row(
            evaluation_type="json",
            expected_output='{"name": "Jordan", "amount": 42.5}',
            model_output='{"amount": 42.5, "name": "Jordan"}',
        ),
        make_config(),
    )

    assert result["output_is_json"] is True
    assert result["json_exact_match"] is True
    assert result["json_canonical_match"] is True
    assert result["json_field_match_rate"] == 1.0
    assert result["metric_passed"] is True


def test_json_metric_normalizes_text_values() -> None:
    result = score_row(
        make_row(
            evaluation_type="json",
            expected_output='{"name": "CAFÉ"}',
            model_output='{"name": "  café  "}',
        ),
        make_config(),
    )

    assert result["json_exact_match"] is False
    assert result["json_canonical_match"] is True
    assert result["json_mismatched_fields"] == "[]"
    assert result["metric_passed"] is True


def test_json_metric_can_compare_configured_lists_without_order() -> None:
    metrics = DeterministicMetricsSettingsConfig(
        json_comparison=JsonComparisonSettingsConfig(
            unordered_list_paths=["/items"],
        )
    )
    result = score_row(
        make_row(
            evaluation_type="json",
            expected_output='{"items": ["A", "B"]}',
            model_output='{"items": ["b", "a"]}',
        ),
        make_config(metrics=metrics),
    )

    assert result["json_exact_match"] is False
    assert result["json_canonical_match"] is True
    assert result["json_field_match_rate"] == 1.0
    assert result["metric_passed"] is True


def test_json_unordered_list_comparison_rejects_duplicates() -> None:
    metrics = DeterministicMetricsSettingsConfig(
        json_comparison=JsonComparisonSettingsConfig(
            unordered_list_paths=["/items"],
        )
    )
    result = score_row(
        make_row(
            evaluation_type="json",
            expected_output='{"items": ["A", "B"]}',
            model_output='{"items": ["A", "B", "B"]}',
        ),
        make_config(metrics=metrics),
    )

    assert result["json_canonical_match"] is False
    assert result["json_mismatched_fields"] == '["items"]'
    assert result["metric_passed"] is False


def test_json_metric_applies_wildcard_zero_padding() -> None:
    metrics = DeterministicMetricsSettingsConfig(
        json_comparison=JsonComparisonSettingsConfig(
            unordered_list_paths=["/din"],
            zero_pad_numeric_string_paths={"/din/*": 8},
        )
    )
    result = score_row(
        make_row(
            evaluation_type="json",
            expected_output='{"din": ["02297884", "00001234"]}',
            model_output='{"din": ["1234", "2297884"]}',
        ),
        make_config(metrics=metrics),
    )

    assert result["json_exact_match"] is False
    assert result["json_canonical_match"] is True
    assert result["metric_passed"] is True


@pytest.mark.parametrize(
    ("expected_strength", "model_strength"),
    [
        ("15", '"15"'),
        ("15.0", '"15"'),
        ("0.25", '"0.2500"'),
        ("15", '"1.5e1"'),
        ("-0.0", '"0"'),
    ],
)
def test_json_metric_normalizes_configured_numeric_values(
    expected_strength: str,
    model_strength: str,
) -> None:
    metrics = DeterministicMetricsSettingsConfig(
        json_comparison=JsonComparisonSettingsConfig(
            numeric_value_paths=["/active_ingredients/*/strength"],
        )
    )
    result = score_row(
        make_row(
            evaluation_type="json",
            expected_output=(
                '{"active_ingredients": [{"name": "A", '
                f'"strength": {expected_strength}' + "}]}"
            ),
            model_output=(
                '{"active_ingredients": [{"name": "A", '
                f'"strength": {model_strength}' + "}]}"
            ),
        ),
        make_config(metrics=metrics),
    )

    assert result["json_exact_match"] is False
    assert result["json_canonical_match"] is True
    assert result["metric_passed"] is True


@pytest.mark.parametrize(
    "model_strength",
    ['"15 mg"', '["15"]', "true", "null"],
)
def test_json_numeric_normalization_rejects_non_numeric_or_wrong_types(
    model_strength: str,
) -> None:
    metrics = DeterministicMetricsSettingsConfig(
        json_comparison=JsonComparisonSettingsConfig(
            numeric_value_paths=["/active_ingredients/*/strength"],
        )
    )
    result = score_row(
        make_row(
            evaluation_type="json",
            expected_output=(
                '{"active_ingredients": [{"name": "A", '
                '"strength": 15}]}'
            ),
            model_output=(
                '{"active_ingredients": [{"name": "A", '
                f'"strength": {model_strength}' + "}]}"
            ),
        ),
        make_config(metrics=metrics),
    )

    assert result["json_canonical_match"] is False
    assert result["metric_passed"] is False


def test_json_numeric_normalization_is_path_scoped() -> None:
    metrics = DeterministicMetricsSettingsConfig(
        json_comparison=JsonComparisonSettingsConfig(
            numeric_value_paths=["/strength"],
        )
    )
    result = score_row(
        make_row(
            evaluation_type="json",
            expected_output='{"strength": 15, "other": 15}',
            model_output='{"strength": "15.0", "other": "15"}',
        ),
        make_config(metrics=metrics),
    )

    assert result["json_canonical_match"] is False
    assert result["json_matching_field_count"] == 1
    assert result["json_mismatched_fields"] == '["other"]'


def test_json_numeric_normalization_keeps_duplicate_list_members() -> None:
    metrics = DeterministicMetricsSettingsConfig(
        json_comparison=JsonComparisonSettingsConfig(
            unordered_list_paths=["/active_ingredients"],
            numeric_value_paths=["/active_ingredients/*/strength"],
        )
    )
    ingredient = '{"name": "A", "strength": "15"}'
    result = score_row(
        make_row(
            evaluation_type="json",
            expected_output=(
                '{"active_ingredients": '
                '[{"name": "A", "strength": 15}]}'
            ),
            model_output=(
                '{"active_ingredients": ['
                f"{ingredient}, {ingredient}" + "]}"
            ),
        ),
        make_config(metrics=metrics),
    )

    assert result["json_canonical_match"] is False
    assert result["metric_passed"] is False


def test_json_metric_applies_configured_value_alias() -> None:
    metrics = DeterministicMetricsSettingsConfig(
        json_comparison=JsonComparisonSettingsConfig(
            value_aliases={
                "/unit": {
                    "milligram": "mg",
                }
            },
        )
    )
    result = score_row(
        make_row(
            evaluation_type="json",
            expected_output='{"unit": "mg"}',
            model_output='{"unit": "MILLIGRAM"}',
        ),
        make_config(metrics=metrics),
    )

    assert result["json_canonical_match"] is True
    assert result["metric_passed"] is True


def test_json_metric_reports_partial_field_match() -> None:
    result = score_row(
        make_row(
            evaluation_type="json",
            expected_output='{"name": "Jordan", "amount": 42.5}',
            model_output='{"name": "permit", "amount": 42.5}',
        ),
        make_config(),
    )

    assert result["json_exact_match"] is False
    assert result["json_expected_field_count"] == 2
    assert result["json_matching_field_count"] == 1
    assert result["json_field_match_rate"] == 0.5
    assert result["json_mismatched_fields"] == '["name"]'
    assert result["json_missing_fields"] == "[]"
    assert result["json_extra_fields"] == "[]"
    assert result["metric_passed"] is False


def test_json_metric_reports_missing_and_extra_fields() -> None:
    result = score_row(
        make_row(
            evaluation_type="json",
            expected_output='{"name": "Jordan", "amount": 42.5}',
            model_output='{"name": "Jordan", "currency": "CAD"}',
        ),
        make_config(),
    )

    assert result["json_expected_field_count"] == 2
    assert result["json_output_field_count"] == 2
    assert result["json_matching_field_count"] == 1
    assert result["json_missing_fields"] == '["amount"]'
    assert result["json_extra_fields"] == '["currency"]'
    assert result["json_mismatched_fields"] == "[]"
    assert result["metric_passed"] is False


def test_json_metric_reports_invalid_model_output() -> None:
    result = score_row(
        make_row(
            evaluation_type="json",
            expected_output='{"name": "Jordan", "amount": 42.5}',
            model_output="Jordan paid 42.50",
        ),
        make_config(),
    )

    assert result["output_is_json"] is False
    assert result["json_field_match_rate"] == 0.0
    assert result["metric_passed"] is False


def test_json_metric_rejects_invalid_expected_output() -> None:
    with pytest.raises(ValueError, match="expected_output is not valid JSON"):
        score_row(
            make_row(
                evaluation_type="json",
                expected_output="not json",
                model_output='{"name": "Jordan"}',
            ),
            make_config(),
        )


def test_metrics_summary_ranks_models_by_pass_rate() -> None:
    config = make_config()
    records = [
        score_row(
            make_row(
                case_id="case_001",
                model_name="model_a",
                evaluation_type="exact",
                expected_output="negative",
                model_output="negative",
            ),
            config,
        ),
        score_row(
            make_row(
                case_id="case_002",
                model_name="model_a",
                evaluation_type="exact",
                expected_output="48",
                model_output="48",
            ),
            config,
        ),
        score_row(
            make_row(
                case_id="case_001",
                model_name="model_b",
                evaluation_type="exact",
                expected_output="negative",
                model_output="neutral",
            ),
            config,
        ),
        score_row(
            make_row(
                case_id="case_002",
                model_name="model_b",
                evaluation_type="exact",
                expected_output="48",
                model_output="48",
            ),
            config,
        ),
    ]

    summary = build_metrics_summary(pd.DataFrame(records))

    assert summary["model_name"].tolist() == ["model_a", "model_b"]
    assert summary["rank"].tolist() == [1, 2]
    assert summary["metric_pass_rate"].tolist() == [1.0, 0.5]


def test_metrics_summary_assigns_same_rank_to_ties() -> None:
    config = make_config()
    records = [
        score_row(
            make_row(
                model_name=model_name,
                evaluation_type="exact",
                expected_output="negative",
                model_output="negative",
            ),
            config,
        )
        for model_name in ["model_a", "model_b"]
    ]

    summary = build_metrics_summary(pd.DataFrame(records))

    assert summary["rank"].tolist() == [1, 1]

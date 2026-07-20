from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from evalanche.config import MetricsConfig, RunConfig
from evalanche.metrics.deterministic import (
    build_metrics_summary,
    canonical_json,
    normalize_text,
    score_row,
    try_parse_json,
)


def make_config() -> MetricsConfig:
    return MetricsConfig(
        run=RunConfig(
            name="test",
            input_path=Path("input.csv"),
            output_path=Path("output.csv"),
        )
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
    assert result["json_field_match_rate"] == 1.0
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
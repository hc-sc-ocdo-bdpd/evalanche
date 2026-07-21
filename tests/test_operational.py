import pandas as pd
import pytest

from evalanche.operational import (
    summarize_operations,
    summarize_stage,
)


def test_stage_summary_reports_latency_tokens_cost_and_failures() -> None:
    rows = pd.DataFrame(
        [
            {
                "generation_status": "success",
                "generation_seconds": 1.0,
                "generation_attempts": 1,
                "generation_failed_attempts": 0,
                "generation_prompt_tokens": 10,
                "generation_completion_tokens": 5,
                "generation_total_tokens": 15,
                "generation_cost_usd": 0.001,
                "generation_cost_source": (
                    "litellm_response_metadata"
                ),
            },
            {
                "generation_status": "error",
                "generation_seconds": 3.0,
                "generation_attempts": 2,
                "generation_failed_attempts": 2,
                "generation_prompt_tokens": 20,
                "generation_completion_tokens": 10,
                "generation_total_tokens": 30,
                "generation_cost_usd": 0.002,
                "generation_cost_source": (
                    "litellm_response_metadata"
                ),
            },
        ]
    )

    summary = summarize_stage(rows, "generation")

    assert summary["requests"] == 2
    assert summary["successes"] == 1
    assert summary["errors"] == 1
    assert summary["failure_rate"] == 0.5
    assert summary["average_seconds"] == 2.0
    assert summary["p95_seconds"] == pytest.approx(2.9)
    assert summary["api_attempts"] == 3
    assert summary["failed_api_attempts"] == 2
    assert summary["prompt_tokens"] == 30
    assert summary["completion_tokens"] == 15
    assert summary["total_tokens"] == 45
    assert summary["cost_usd"] == pytest.approx(0.003)
    assert summary["cost_coverage"] == 1.0


def test_stage_summary_does_not_turn_missing_cost_into_zero() -> None:
    rows = pd.DataFrame(
        [
            {
                "judge_status": "success",
                "judge_seconds": 1.0,
                "judge_total_tokens": 10,
                "judge_cost_usd": 0.001,
            },
            {
                "judge_status": "success",
                "judge_seconds": 2.0,
                "judge_total_tokens": 20,
                "judge_cost_usd": None,
            },
        ]
    )

    summary = summarize_stage(rows, "judge")

    assert summary["total_tokens"] == 30
    assert summary["cost_usd"] is None
    assert summary["cost_coverage"] == 0.5
    assert summary["cost_complete"] is False


def test_operations_ignore_not_requested_judge_rows() -> None:
    rows = pd.DataFrame(
        [
            {
                "generation_status": "success",
                "judge_status": "not_requested",
            },
            {
                "generation_status": "success",
                "judge_status": "success",
            },
        ]
    )

    summary = summarize_operations(rows)

    assert summary["generation"]["requests"] == 2
    assert summary["judge"]["requests"] == 1
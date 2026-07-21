from pathlib import Path
from typing import Any

import pytest

from evalanche.config import (
    CriterionConfig,
    EvalConfig,
    JudgeConfig,
    RunConfig,
    ScoringConfig,
    TaskConfig,
)
from evalanche.judges.criteria import CriteriaJudge
from evalanche.judges.validation import (
    JudgeResponseValidationError,
    validate_judge_response,
)


def make_config() -> EvalConfig:
    return EvalConfig(
        run=RunConfig(
            name="judge-validation-test",
            input_path=Path("input.csv"),
            output_path=Path("output.csv"),
        ),
        judge=JudgeConfig(model="azure/test-judge"),
        task=TaskConfig(
            name="test-task",
            description="Evaluate the candidate response.",
        ),
        scoring=ScoringConfig(
            score_min=0,
            score_max=5,
            pass_threshold=0.6,
        ),
        criteria=[
            CriterionConfig(
                name="correctness",
                weight=2,
                description="The response is correct.",
            ),
            CriterionConfig(
                name="completeness",
                weight=1,
                description="The response is complete.",
            ),
        ],
    )


def make_response() -> dict[str, Any]:
    return {
        "criteria": {
            "correctness": {
                "score": 4,
                "reason": "The answer is correct.",
            },
            "completeness": {
                "score": 2,
                "reason": "Some detail is missing.",
            },
        },
        "overall_reason": "Correct but incomplete.",
    }


def test_valid_judge_response_is_accepted() -> None:
    validate_judge_response(make_response(), make_config())


def test_judge_response_requires_criteria_object() -> None:
    response = make_response()
    response["criteria"] = []

    with pytest.raises(
        JudgeResponseValidationError,
        match="'criteria' must be a JSON object",
    ):
        validate_judge_response(response, make_config())


def test_judge_response_rejects_missing_criterion() -> None:
    response = make_response()
    del response["criteria"]["completeness"]

    with pytest.raises(
        JudgeResponseValidationError,
        match="missing criteria.*completeness",
    ):
        validate_judge_response(response, make_config())


def test_judge_response_rejects_unexpected_criterion() -> None:
    response = make_response()
    response["criteria"]["style"] = {
        "score": 5,
        "reason": "Well written.",
    }

    with pytest.raises(
        JudgeResponseValidationError,
        match="unexpected criteria.*style",
    ):
        validate_judge_response(response, make_config())


@pytest.mark.parametrize("score", [-1, 6])
def test_judge_response_rejects_out_of_range_score(score: int) -> None:
    response = make_response()
    response["criteria"]["correctness"]["score"] = score

    with pytest.raises(
        JudgeResponseValidationError,
        match="score must be between 0 and 5",
    ):
        validate_judge_response(response, make_config())


@pytest.mark.parametrize("score", [True, "4", None])
def test_judge_response_rejects_non_numeric_score(score: Any) -> None:
    response = make_response()
    response["criteria"]["correctness"]["score"] = score

    with pytest.raises(
        JudgeResponseValidationError,
        match="score must be a number",
    ):
        validate_judge_response(response, make_config())


def test_judge_response_requires_criterion_reason() -> None:
    response = make_response()
    response["criteria"]["correctness"]["reason"] = "   "

    with pytest.raises(
        JudgeResponseValidationError,
        match="reason must be a non-empty string",
    ):
        validate_judge_response(response, make_config())


def test_judge_response_requires_overall_reason() -> None:
    response = make_response()
    response["overall_reason"] = ""

    with pytest.raises(
        JudgeResponseValidationError,
        match="'overall_reason'.*non-empty string",
    ):
        validate_judge_response(response, make_config())


def test_criteria_judge_scores_a_validated_response() -> None:
    config = make_config()
    judge = CriteriaJudge(config)

    class StubClient:
        def complete_json(
            self,
            messages: list[dict[str, str]],
            validator: Any = None,
        ) -> dict[str, Any]:
            response = make_response()
            validator(response)
            response["_usage"] = {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            }
            response["_operational"] = {
                "latency_seconds": 2.5,
                "api_seconds": 2.0,
                "attempts": 2,
                "failed_attempts": 1,
                "cost_usd": 0.004,
                "cost_source": "litellm_response_metadata",
            }
            return response

    judge.client = StubClient()
    result = judge.judge_case(
        {
            "case_id": "case_001",
            "model_name": "model_a",
            "input": "Question",
            "expected_output": "Expected",
            "model_output": "Candidate",
        }
    )

    assert result["weighted_score"] == 0.6667
    assert result["passed"] is True
    assert result["correctness_score"] == 4
    assert result["completeness_score"] == 2
    assert result["total_tokens"] == 30
    assert result["latency_seconds"] == 2.5
    assert result["attempts"] == 2
    assert result["failed_attempts"] == 1
    assert result["cost_usd"] == 0.004


def test_judge_prompt_names_every_required_criterion() -> None:
    judge = CriteriaJudge(make_config())

    messages = judge._build_messages(
        {
            "input": "Question",
            "expected_output": "Expected",
            "model_output": "Candidate",
        }
    )

    system_prompt = messages[0]["content"]
    assert '"correctness"' in system_prompt
    assert '"completeness"' in system_prompt
    assert "Include every configured criterion exactly once" in system_prompt
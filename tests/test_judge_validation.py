from pathlib import Path
from typing import Any

import pytest

import evalanche.judges.criteria as criteria_module
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
    assert result["evidence_level"] == "exploratory"
    assert result["protocol_id"] is None


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


def test_judge_prompt_blinds_candidate_identity_by_default() -> None:
    judge = CriteriaJudge(make_config())
    row = {
        "input": "Question",
        "expected_output": "Expected",
        "model_name": "secret-model-name",
        "model_output": "Candidate",
    }

    user_prompt = judge._build_messages(row)[1]["content"]
    assert "secret-model-name" not in user_prompt

    judge.config.judge.candidate_identity_blinded = False
    named_prompt = judge._build_messages(row)[1]["content"]
    assert "Candidate model identity" in named_prompt
    assert "secret-model-name" in named_prompt


def test_reference_free_prompt_omits_expected_output() -> None:
    config = make_config()
    config.judge.reference_mode = "none"
    judge = CriteriaJudge(config)

    messages = judge._build_messages(
        {
            "input": "Question",
            "expected_output": "Hidden reference",
            "model_output": "Candidate",
        }
    )

    assert "Hidden reference" not in messages[1]["content"]
    assert "No reference answer is provided" in messages[0]["content"]


def test_required_reference_rejects_blank_expected_output() -> None:
    judge = CriteriaJudge(make_config())

    with pytest.raises(ValueError, match="non-empty expected output"):
        judge._build_messages(
            {
                "input": "Question",
                "expected_output": "",
                "model_output": "Candidate",
            }
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("prompt_id", "custom.prompt", "only supports prompt_id"),
        ("prompt_version", "2.0", "only supports prompt_version"),
    ],
)
def test_builtin_judge_rejects_unimplemented_prompt_contracts(
    field: str,
    value: str,
    message: str,
) -> None:
    config = make_config()
    setattr(config.judge, field, value)

    with pytest.raises(ValueError, match=message):
        CriteriaJudge(config)


def test_criteria_judge_resolves_explicit_endpoint_price(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pricing_path = tmp_path / "pricing.yaml"
    pricing_path.write_text(
        "schema_version: '1.0'\n"
        "catalog_version: '2026-07-01'\n"
        "endpoints:\n"
        "  - pricing_id: judge_global\n"
        "    model: azure/test-judge\n"
        "    currency: USD\n"
        "    input_per_million_tokens: 1.0\n"
        "    output_per_million_tokens: 4.0\n"
        "    effective_date: 2026-07-01\n"
        "    source: test rate card\n",
        encoding="utf-8",
    )
    seen: dict[str, Any] = {}

    class StubClient:
        def __init__(self, **kwargs: Any) -> None:
            seen.update(kwargs)

    monkeypatch.setattr(criteria_module, "LLMClient", StubClient)
    config = make_config()
    config.endpoint_pricing_path = pricing_path
    config.judge.pricing_id = "judge_global"

    judge = CriteriaJudge(config)

    assert seen["endpoint_price"].pricing_id == "judge_global"
    assert seen["endpoint_price"].model == "azure/test-judge"
    assert judge.pricing_catalog_version == "2026-07-01"
    assert len(judge.pricing_catalog_sha256) == 64

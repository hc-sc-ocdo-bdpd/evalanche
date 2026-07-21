from pathlib import Path

import pytest
from pydantic import ValidationError

from evalanche.config import (
    CandidateModelConfig,
    CandidateModelsConfig,
    CriterionConfig,
    EvalConfig,
    JudgeConfig,
    ModelProfileConfig,
    RunConfig,
    ScoringConfig,
    SelectionConfig,
    SelectionConstraintsConfig,
    SelectionWeightsConfig,
    TaskConfig,
    resolve_env_vars,
)


def test_resolve_env_vars_uses_environment_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EVALANCHE_TEST_MODEL", "azure/deployed-model")

    resolved = resolve_env_vars(
        {"model": "${EVALANCHE_TEST_MODEL:-azure/fallback-model}"}
    )

    assert resolved == {"model": "azure/deployed-model"}


def test_resolve_env_vars_uses_default_when_variable_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EVALANCHE_TEST_MODEL", raising=False)

    resolved = resolve_env_vars(
        {"model": "${EVALANCHE_TEST_MODEL:-azure/fallback-model}"}
    )

    assert resolved == {"model": "azure/fallback-model"}


@pytest.mark.parametrize("threshold", [-0.01, 1.01])
def test_pass_threshold_must_be_between_zero_and_one(
    threshold: float,
) -> None:
    with pytest.raises(ValidationError, match="pass_threshold"):
        ScoringConfig(pass_threshold=threshold)


def test_eval_config_rejects_non_positive_total_criterion_weight() -> None:
    with pytest.raises(ValidationError, match="positive"):
        EvalConfig(
            run=RunConfig(
                name="test",
                input_path=Path("input.csv"),
                output_path=Path("output.csv"),
            ),
            judge=JudgeConfig(model="azure/judge"),
            task=TaskConfig(name="test", description="Test task"),
            scoring=ScoringConfig(),
            criteria=[
                CriterionConfig(
                    name="correctness",
                    weight=0,
                    description="Correctness",
                )
            ],
        )


@pytest.mark.parametrize(
    ("score_min", "score_max"),
    [(5, 5), (5, 4)],
)
def test_score_max_must_be_greater_than_score_min(
    score_min: int,
    score_max: int,
) -> None:
    with pytest.raises(ValidationError, match="score_max"):
        ScoringConfig(score_min=score_min, score_max=score_max)


def test_criterion_weight_cannot_be_negative() -> None:
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        CriterionConfig(
            name="correctness",
            weight=-1,
            description="Correctness",
        )


def test_eval_config_rejects_duplicate_criterion_names() -> None:
    with pytest.raises(ValidationError, match="criterion names must be unique"):
        EvalConfig(
            run=RunConfig(
                name="test",
                input_path=Path("input.csv"),
                output_path=Path("output.csv"),
            ),
            judge=JudgeConfig(model="azure/judge"),
            task=TaskConfig(name="test", description="Test task"),
            scoring=ScoringConfig(),
            criteria=[
                CriterionConfig(
                    name="correctness",
                    weight=1,
                    description="Correctness",
                ),
                CriterionConfig(
                    name="correctness",
                    weight=1,
                    description="Duplicate correctness",
                ),
            ],
        )

def test_candidate_model_name_is_trimmed() -> None:
    candidate = CandidateModelConfig(
        name=" model_a ",
        model="azure/deployment-a",
    )

    assert candidate.name == "model_a"


def test_candidate_model_name_cannot_be_blank() -> None:
    with pytest.raises(ValidationError, match="cannot be blank"):
        CandidateModelConfig(
            name="   ",
            model="azure/deployment-a",
        )


def test_candidate_model_names_must_be_unique() -> None:
    with pytest.raises(ValidationError, match="names must be unique"):
        CandidateModelsConfig(
            models=[
                CandidateModelConfig(
                    name="model_a",
                    model="azure/deployment-a",
                ),
                CandidateModelConfig(
                    name=" model_a ",
                    model="azure/deployment-b",
                ),
            ]
        )


def test_selection_weights_must_have_positive_total() -> None:
    with pytest.raises(ValidationError, match="positive"):
        SelectionWeightsConfig(
            quality=0,
            cost=0,
            latency=0,
            reliability=0,
        )


def test_cost_weight_requires_cost_scale() -> None:
    with pytest.raises(
        ValidationError,
        match="maximum_average_cost_usd",
    ):
        SelectionConfig(
            weights=SelectionWeightsConfig(
                quality=0.5,
                cost=0.5,
            )
        )


def test_latency_weight_requires_latency_scale() -> None:
    with pytest.raises(
        ValidationError,
        match="maximum_p95_latency_seconds",
    ):
        SelectionConfig(
            weights=SelectionWeightsConfig(
                quality=0.5,
                latency=0.5,
            )
        )


def test_selection_capabilities_are_normalized() -> None:
    constraints = SelectionConstraintsConfig(
        required_capabilities=[" Bilingual ", "Approved-Hosting"],
    )

    assert constraints.required_capabilities == [
        "bilingual",
        "approved-hosting",
    ]


def test_model_profile_names_must_be_unique() -> None:
    with pytest.raises(ValidationError, match="profile names"):
        SelectionConfig(
            model_profiles=[
                ModelProfileConfig(name="model_a"),
                ModelProfileConfig(name="model_a"),
            ]
        )


def test_selection_rejects_unknown_policy_fields() -> None:
    with pytest.raises(
        ValidationError,
        match="maximum_p95_latncy_seconds",
    ):
        SelectionConfig.model_validate(
            {
                "constraints": {
                    "maximum_p95_latncy_seconds": 5,
                }
            }
        )
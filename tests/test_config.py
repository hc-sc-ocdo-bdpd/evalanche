from pathlib import Path

import pytest
from pydantic import ValidationError

from evalanche.config import (
    CandidateModelConfig,
    CandidateModelsConfig,
    CriterionConfig,
    EvalConfig,
    JudgeConfig,
    RunConfig,
    ScoringConfig,
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
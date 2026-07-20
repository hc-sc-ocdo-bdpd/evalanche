from pathlib import Path

import pytest
from pydantic import ValidationError

from evalanche.config import (
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
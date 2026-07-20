from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

_ENV_PATTERN = re.compile(r"\$\{([^}:]+)(:-([^}]*))?\}")


def _resolve_env_string(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        default = match.group(3)
        env_value = os.getenv(name)

        if env_value is not None:
            return env_value

        if default is not None:
            return default

        return match.group(0)

    return _ENV_PATTERN.sub(replace, value)


def resolve_env_vars(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: resolve_env_vars(v) for k, v in value.items()}

    if isinstance(value, list):
        return [resolve_env_vars(v) for v in value]

    if isinstance(value, str):
        return _resolve_env_string(value)

    return value


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return resolve_env_vars(raw)


class RunConfig(BaseModel):
    name: str
    input_path: Path
    output_path: Path


class JudgeConfig(BaseModel):
    model: str
    temperature: float = 0
    max_retries: int = 3


class TaskConfig(BaseModel):
    name: str
    description: str


class ScoringConfig(BaseModel):
    score_min: int = 0
    score_max: int = 5
    pass_threshold: float = 0.75

    @field_validator("pass_threshold")
    @classmethod
    def validate_pass_threshold(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("pass_threshold must be between 0 and 1")
        return value
    @model_validator(mode="after")
    def validate_score_range(self) -> "ScoringConfig":
        if self.score_max <= self.score_min:
            raise ValueError("score_max must be greater than score_min")
        return self

    
class CriterionConfig(BaseModel):
    name: str
    weight: float = Field(ge=0, allow_inf_nan=False)
    description: str


class EvalConfig(BaseModel):
    run: RunConfig
    judge: JudgeConfig
    task: TaskConfig
    scoring: ScoringConfig
    criteria: list[CriterionConfig] = Field(min_length=1)

    @field_validator("criteria")
    @classmethod
    def validate_weights(cls, criteria: list[CriterionConfig]) -> list[CriterionConfig]:
        total = sum(c.weight for c in criteria)

        if total <= 0:
            raise ValueError("criteria weights must sum to a positive number")
        
        names = [criterion.name for criterion in criteria]
        if len(names) != len(set(names)):
            raise ValueError("criterion names must be unique")
        return criteria


class PromptConfig(BaseModel):
    system: str = "You are a helpful assistant."
    template: str = "{input}"


class GenerationSettingsConfig(BaseModel):
    continue_on_error: bool = True
    max_completion_tokens: int | None = None


class GenerationConfig(BaseModel):
    run: RunConfig
    candidate_models_path: Path
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    generation: GenerationSettingsConfig = Field(
        default_factory=GenerationSettingsConfig
    )


class CandidateModelConfig(BaseModel):
    name: str
    model: str
    temperature: float = 0
    max_retries: int = 3
    max_completion_tokens: int | None = None
    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("candidate model name cannot be blank")
        return normalized


class CandidateModelsConfig(BaseModel):
    models: list[CandidateModelConfig] = Field(min_length=1)
    @field_validator("models")
    @classmethod
    def validate_unique_names(
        cls,
        models: list[CandidateModelConfig],
        ) -> list[CandidateModelConfig]:
        names = [model.name for model in models]
        if len(names) != len(set(names)):
            raise ValueError("candidate model names must be unique")
        return models
    

class DeterministicMetricsSettingsConfig(BaseModel):
    case_sensitive: bool = False
    trim_whitespace: bool = True
    collapse_whitespace: bool = True


class MetricsConfig(BaseModel):
    run: RunConfig
    metrics: DeterministicMetricsSettingsConfig = Field(
        default_factory=DeterministicMetricsSettingsConfig
    )


def load_config(path: str | Path) -> EvalConfig:
    resolved = load_yaml(path)
    return EvalConfig.model_validate(resolved)


def load_generation_config(path: str | Path) -> GenerationConfig:
    resolved = load_yaml(path)
    return GenerationConfig.model_validate(resolved)


def load_candidate_models(path: str | Path) -> CandidateModelsConfig:
    resolved = load_yaml(path)
    return CandidateModelsConfig.model_validate(resolved)


def load_metrics_config(path: str | Path) -> MetricsConfig:
    resolved = load_yaml(path)
    return MetricsConfig.model_validate(resolved)
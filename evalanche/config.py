from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

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
    continue_on_error: bool = False
    pricing_id: str | None = None

    @field_validator("pricing_id")
    @classmethod
    def validate_pricing_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("pricing_id cannot be blank")
        return normalized


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
    endpoint_pricing_path: Path | None = None

    @field_validator("criteria")
    @classmethod
    def validate_weights(
        cls,
        criteria: list[CriterionConfig],
    ) -> list[CriterionConfig]:
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
    model_config = ConfigDict(extra="forbid")

    continue_on_error: bool = True
    max_completion_tokens: int | None = None
    max_workers: int = Field(default=1, ge=1, le=64)
    checkpoint_every: int = Field(default=100, ge=1)
    resume: bool = False
    max_requests_per_minute: float | None = Field(
        default=None,
        gt=0,
        allow_inf_nan=False,
    )
    cost_preflight_sample_path: Path | None = None
    maximum_estimated_cost_usd: float | None = Field(
        default=None,
        gt=0,
        allow_inf_nan=False,
    )
    cost_safety_multiplier: float = Field(
        default=1.0,
        ge=1.0,
        allow_inf_nan=False,
    )

    @model_validator(mode="after")
    def validate_cost_guard(self) -> "GenerationSettingsConfig":
        if (
            self.maximum_estimated_cost_usd is not None
            and self.cost_preflight_sample_path is None
        ):
            raise ValueError(
                "cost_preflight_sample_path is required when "
                "maximum_estimated_cost_usd is configured"
            )
        if (
            self.maximum_estimated_cost_usd is None
            and self.cost_safety_multiplier != 1.0
        ):
            raise ValueError(
                "maximum_estimated_cost_usd is required when "
                "cost_safety_multiplier is greater than 1"
            )
        return self


class GenerationConfig(BaseModel):
    run: RunConfig
    candidate_models_path: Path
    endpoint_pricing_path: Path | None = None
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    generation: GenerationSettingsConfig = Field(
        default_factory=GenerationSettingsConfig
    )


class CandidateModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    model: str
    temperature: float | None = 0
    reasoning_effort: Literal[
        "none",
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
    ] | None = None
    max_retries: int = Field(default=3, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    pricing_id: str | None = None
    provider_model_version: str | None = None
    deployment_type: str | None = None
    resource_region: str | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("candidate model name cannot be blank")
        return normalized

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("candidate model route cannot be blank")
        return normalized

    @field_validator("pricing_id")
    @classmethod
    def validate_pricing_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("pricing_id cannot be blank")
        return normalized

    @field_validator(
        "provider_model_version",
        "deployment_type",
        "resource_region",
    )
    @classmethod
    def normalize_optional_metadata(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError(
                "candidate deployment metadata cannot be blank"
            )
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


def _validate_json_pointer(value: str) -> str:
    if value == "":
        return value
    if not value.startswith("/"):
        raise ValueError(
            "JSON comparison paths must be JSON Pointers beginning with '/'"
        )
    if re.search(r"~(?![01])", value):
        raise ValueError(
            "JSON comparison paths may only use '~0' and '~1' escapes"
        )
    return value


class JsonComparisonSettingsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    unordered_list_paths: list[str] = Field(default_factory=list)
    numeric_value_paths: list[str] = Field(default_factory=list)
    zero_pad_numeric_string_paths: dict[str, int] = Field(
        default_factory=dict
    )
    value_aliases: dict[str, dict[str, str]] = Field(
        default_factory=dict
    )

    @field_validator("unordered_list_paths")
    @classmethod
    def validate_unordered_list_paths(
        cls,
        values: list[str],
    ) -> list[str]:
        validated = [_validate_json_pointer(value) for value in values]
        if len(validated) != len(set(validated)):
            raise ValueError("unordered_list_paths must be unique")
        return validated

    @field_validator("numeric_value_paths")
    @classmethod
    def validate_numeric_value_paths(
        cls,
        values: list[str],
    ) -> list[str]:
        validated = [_validate_json_pointer(value) for value in values]
        if len(validated) != len(set(validated)):
            raise ValueError("numeric_value_paths must be unique")
        return validated

    @field_validator(
        "zero_pad_numeric_string_paths",
        "value_aliases",
    )
    @classmethod
    def validate_rule_paths(
        cls,
        values: dict[str, Any],
    ) -> dict[str, Any]:
        for path in values:
            _validate_json_pointer(path)
        return values

    @field_validator("zero_pad_numeric_string_paths")
    @classmethod
    def validate_zero_pad_widths(
        cls,
        values: dict[str, int],
    ) -> dict[str, int]:
        if any(width <= 0 for width in values.values()):
            raise ValueError(
                "zero-pad widths must be positive integers"
            )
        return values


class DeterministicMetricsSettingsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_sensitive: bool = False
    trim_whitespace: bool = True
    collapse_whitespace: bool = True
    unicode_normalization: Literal[
        "NFC",
        "NFD",
        "NFKC",
        "NFKD",
    ] = "NFC"
    strip_diacritics: bool = False
    strip_punctuation: bool = False
    json_comparison: JsonComparisonSettingsConfig = Field(
        default_factory=JsonComparisonSettingsConfig
    )


class MetricsConfig(BaseModel):
    run: RunConfig
    metrics: DeterministicMetricsSettingsConfig = Field(
        default_factory=DeterministicMetricsSettingsConfig
    )


class SelectionWeightsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quality: float = Field(default=1.0, ge=0, allow_inf_nan=False)
    cost: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    latency: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    reliability: float = Field(default=0.0, ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_positive_total(self) -> "SelectionWeightsConfig":
        if (
            self.quality
            + self.cost
            + self.latency
            + self.reliability
            <= 0
        ):
            raise ValueError(
                "selection weights must sum to a positive number"
            )
        return self


def _normalize_capabilities(values: list[str]) -> list[str]:
    normalized = [value.strip().casefold() for value in values]
    if any(not value for value in normalized):
        raise ValueError("capabilities cannot contain blank values")
    if len(normalized) != len(set(normalized)):
        raise ValueError("capabilities must be unique")
    return normalized


class SelectionConstraintsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    require_model_profile: bool = False
    minimum_pass_rate: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )
    minimum_pass_rate_ci_low: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )
    maximum_average_cost_usd: float | None = Field(
        default=None,
        gt=0,
        allow_inf_nan=False,
    )
    maximum_p95_latency_seconds: float | None = Field(
        default=None,
        gt=0,
        allow_inf_nan=False,
    )
    maximum_generation_failure_rate: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )
    required_capabilities: list[str] = Field(default_factory=list)

    @field_validator("required_capabilities")
    @classmethod
    def validate_required_capabilities(
        cls,
        values: list[str],
    ) -> list[str]:
        return _normalize_capabilities(values)


class ModelProfileConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    available: bool = True
    capabilities: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("model profile name cannot be blank")
        return normalized

    @field_validator("capabilities")
    @classmethod
    def validate_capabilities(
        cls,
        values: list[str],
    ) -> list[str]:
        return _normalize_capabilities(values)


class SelectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    minimum_score_margin: float = Field(default=0.01, ge=0, le=1)
    weights: SelectionWeightsConfig = Field(
        default_factory=SelectionWeightsConfig
    )
    constraints: SelectionConstraintsConfig = Field(
        default_factory=SelectionConstraintsConfig
    )
    model_profiles: list[ModelProfileConfig] = Field(default_factory=list)

    @field_validator("model_profiles")
    @classmethod
    def validate_unique_model_profiles(
        cls,
        profiles: list[ModelProfileConfig],
    ) -> list[ModelProfileConfig]:
        names = [profile.name for profile in profiles]
        if len(names) != len(set(names)):
            raise ValueError("model profile names must be unique")
        return profiles

    @model_validator(mode="after")
    def validate_weight_scales(self) -> "SelectionConfig":
        if (
            self.weights.cost > 0
            and self.constraints.maximum_average_cost_usd is None
        ):
            raise ValueError(
                "maximum_average_cost_usd is required when the cost "
                "weight is positive"
            )
        if (
            self.weights.latency > 0
            and self.constraints.maximum_p95_latency_seconds is None
        ):
            raise ValueError(
                "maximum_p95_latency_seconds is required when the "
                "latency weight is positive"
            )
        return self


class EvaluationConfig(EvalConfig):
    model_config = ConfigDict(extra="forbid")

    metrics: DeterministicMetricsSettingsConfig = Field(
        default_factory=DeterministicMetricsSettingsConfig
    )
    selection: SelectionConfig = Field(default_factory=SelectionConfig)


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

def load_evaluation_config(path: str | Path) -> EvaluationConfig:
    resolved = load_yaml(path)
    return EvaluationConfig.model_validate(resolved)

from __future__ import annotations

import csv
import json
import re
import shutil
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from evalanche.access_sets import load_access_set
from evalanche.config import load_yaml
from evalanche.judges.protocol import load_judge_protocol
from evalanche.pricing import (
    load_endpoint_pricing_catalog,
    resolve_endpoint_price,
)
from evalanche.registry import (
    BenchmarkManifest,
    ModelManifest,
    load_registry,
    validate_registry,
)


TASK_BUNDLE_SCHEMA_VERSION = "1.0"
TASK_BUNDLE_VERSION = "0.1.0"
DEFAULT_TASKS_DIR = Path("local_tasks")
TASK_FAMILIES = (
    "classification",
    "structured_extraction",
    "factual_response",
    "open_ended_response",
)

TaskFamily = Literal[
    "classification",
    "structured_extraction",
    "factual_response",
    "open_ended_response",
]
EvaluationType = Literal["exact", "json", "judge"]

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_COLUMN = re.compile(r"^[a-z][a-z0-9_]*$")
_LANGUAGE = re.compile(r"^[a-z][a-z0-9-]*$")
_PLACEHOLDER = re.compile(
    r"(?:^|[^a-z0-9])(?:TODO|REPLACE_ME)(?:[^a-z0-9]|$)",
    re.IGNORECASE,
)
_RESERVED_COLUMNS = {
    "case_id",
    "group_id",
    "input",
    "expected_output",
    "evaluation_type",
}


def _identifier(value: str, field_name: str) -> str:
    normalized = value.strip().casefold().replace(" ", "_")
    if not _IDENTIFIER.fullmatch(normalized):
        raise ValueError(
            f"{field_name} must contain lowercase letters, numbers, dots, "
            "underscores, and hyphens"
        )
    return normalized


def _nonblank(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} cannot be blank")
    return normalized


def _unique(values: list[str], field_name: str) -> list[str]:
    normalized = [_nonblank(value, field_name) for value in values]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field_name} values must be unique")
    return normalized


def _evaluation_type(family: TaskFamily) -> EvaluationType:
    if family == "structured_extraction":
        return "json"
    if family == "open_ended_response":
        return "judge"
    return "exact"


class TaskBundleModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TaskModelRoute(TaskBundleModel):
    model_id: str
    provider_route: str
    display_name: str | None = None

    @field_validator("model_id")
    @classmethod
    def validate_model_id(cls, value: str) -> str:
        return _identifier(value, "model_id")

    @field_validator("provider_route")
    @classmethod
    def validate_provider_route(cls, value: str) -> str:
        return _nonblank(value, "provider_route")

    @field_validator("display_name")
    @classmethod
    def validate_display_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _nonblank(value, "display_name")


class TaskCriterion(TaskBundleModel):
    name: str
    weight: float = Field(gt=0, allow_inf_nan=False)
    description: str

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return _identifier(value, "criterion name")

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str) -> str:
        return _nonblank(value, "criterion description")


class TaskReview(TaskBundleModel):
    task_definition: bool = False
    cases_and_expected_outputs: bool = False
    prompt_and_scoring: bool = False
    privacy_and_artifacts: bool = False
    judge_route_and_cost: bool = False


class TaskProfile(TaskBundleModel):
    schema_version: Literal["1.0"] = TASK_BUNDLE_SCHEMA_VERSION
    task_id: str
    version: str = TASK_BUNDLE_VERSION
    title: str
    description: str
    family: TaskFamily
    evaluation_type: EvaluationType
    languages: list[str] = Field(min_length=1)
    slice_columns: list[str] = Field(min_length=1)
    unacceptable_errors: list[str] = Field(min_length=1)
    required_output_fields: list[str] = Field(default_factory=list)
    criteria: list[TaskCriterion] = Field(default_factory=list)
    measured_construct: str | None = None
    intended_use: str | None = None
    artifact_policy: Literal["local_only", "shareable"] = "local_only"
    contains_sensitive_data: bool = False
    privacy_notes: str
    starting_tier: Literal["smoke", "screen"] = "smoke"
    models: list[TaskModelRoute] = Field(default_factory=list)
    review: TaskReview = Field(default_factory=TaskReview)
    created_on: date
    status: Literal["starter", "ready", "retired"] = "starter"

    @field_validator("task_id")
    @classmethod
    def validate_task_id(cls, value: str) -> str:
        return _identifier(value, "task_id")

    @field_validator("title", "description", "privacy_notes")
    @classmethod
    def validate_required_text(cls, value: str, info: Any) -> str:
        return _nonblank(value, info.field_name)

    @field_validator("languages")
    @classmethod
    def validate_languages(cls, values: list[str]) -> list[str]:
        normalized = [value.strip().casefold() for value in values]
        if any(not _LANGUAGE.fullmatch(value) for value in normalized):
            raise ValueError("languages must use simple BCP 47-style tags")
        return _unique(normalized, "languages")

    @field_validator("slice_columns")
    @classmethod
    def validate_slice_columns(cls, values: list[str]) -> list[str]:
        normalized = [value.strip().casefold() for value in values]
        for value in normalized:
            if not _COLUMN.fullmatch(value):
                raise ValueError(
                    "slice columns must use lowercase snake_case"
                )
            if value in _RESERVED_COLUMNS:
                raise ValueError(f"{value!r} is a reserved case column")
        return _unique(normalized, "slice_columns")

    @field_validator("unacceptable_errors", "required_output_fields")
    @classmethod
    def validate_text_lists(
        cls,
        values: list[str],
        info: Any,
    ) -> list[str]:
        return _unique(values, info.field_name)

    @field_validator("measured_construct", "intended_use")
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _nonblank(value, "open-ended task description")

    @model_validator(mode="after")
    def validate_family_contract(self) -> "TaskProfile":
        if self.contains_sensitive_data and self.artifact_policy == "shareable":
            raise ValueError(
                "a task that may contain sensitive data must use local_only "
                "artifacts until separate review changes that decision"
            )
        expected_type = _evaluation_type(self.family)
        if self.evaluation_type != expected_type:
            raise ValueError(
                f"{self.family} requires evaluation_type {expected_type}"
            )
        if self.family == "structured_extraction":
            if not self.required_output_fields:
                raise ValueError(
                    "structured extraction requires output fields"
                )
        elif self.required_output_fields:
            raise ValueError(
                "required_output_fields are only used for structured extraction"
            )
        if self.family == "open_ended_response":
            if not self.criteria:
                raise ValueError("open-ended response requires criteria")
            if self.measured_construct is None or self.intended_use is None:
                raise ValueError(
                    "open-ended response requires measured_construct and "
                    "intended_use"
                )
        elif self.criteria:
            raise ValueError("criteria are only used for open-ended response")
        model_ids = [model.model_id for model in self.models]
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("task model IDs must be unique")
        return self


class TaskDatasetRecord(TaskBundleModel):
    schema_version: Literal["1.0"] = "1.0"
    dataset_id: str
    version: str
    title: str
    description: str
    created_on: date
    status: Literal["development"] = "development"
    limitations: list[str] = Field(min_length=1)

    @field_validator("dataset_id")
    @classmethod
    def validate_dataset_id(cls, value: str) -> str:
        return _identifier(value, "dataset_id")

    @field_validator("title", "description")
    @classmethod
    def validate_text(cls, value: str, info: Any) -> str:
        return _nonblank(value, info.field_name)


def parse_model_route(value: str) -> TaskModelRoute:
    if "=" not in value:
        raise ValueError("model routes must use MODEL_ID=PROVIDER_ROUTE")
    model_id, provider_route = value.split("=", 1)
    return TaskModelRoute(
        model_id=model_id,
        provider_route=provider_route,
    )


def parse_criterion(value: str) -> TaskCriterion:
    parts = value.split("=", 2)
    if len(parts) != 3:
        raise ValueError(
            "criteria must use NAME=WEIGHT=DESCRIPTION"
        )
    name, weight, description = parts
    try:
        numeric_weight = float(weight)
    except ValueError as error:
        raise ValueError("criterion weight must be numeric") from error
    return TaskCriterion(
        name=name,
        weight=numeric_weight,
        description=description,
    )


def normalize_family(value: str) -> TaskFamily:
    normalized = value.strip().casefold().replace("-", "_").replace(" ", "_")
    if normalized not in TASK_FAMILIES:
        raise ValueError(
            "task family must be classification, structured-extraction, "
            "factual-response, or open-ended-response"
        )
    return normalized  # type: ignore[return-value]


def _display_name(identifier: str) -> str:
    return identifier.replace("-", " ").replace("_", " ").title()


def _contains_placeholder(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_placeholder(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_placeholder(item) for item in value)
    return bool(_PLACEHOLDER.search(str(value)))


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            value,
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
        newline="\n",
    )


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8", newline="\n")


def _write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _default_prompt(family: TaskFamily) -> tuple[str, int]:
    if family == "classification":
        return (
            "Return only one allowed label with no explanation.",
            64,
        )
    if family == "structured_extraction":
        return (
            "Return one valid JSON object with no Markdown or explanation.",
            400,
        )
    if family == "factual_response":
        return (
            "Return only the short factual answer requested.",
            128,
        )
    return (
        "Follow the task instruction and return only the requested response.",
        900,
    )


def _placeholder_output(
    family: TaskFamily,
    required_output_fields: list[str],
) -> str:
    if family == "structured_extraction":
        return json.dumps(
            {
                field: f"TODO: expected value for {field}"
                for field in required_output_fields
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    if family == "classification":
        return "TODO_LABEL"
    if family == "factual_response":
        return "TODO: add the authoritative short answer"
    return "TODO: add a human-reviewed reference or evaluation anchor"


def _case_rows(
    profile: TaskProfile,
    *,
    sample_input: str | None,
    sample_expected_output: str | None,
) -> tuple[list[str], list[dict[str, Any]]]:
    columns = [
        "case_id",
        "group_id",
        *profile.slice_columns,
        "input",
        "expected_output",
        "evaluation_type",
    ]
    columns = list(dict.fromkeys(columns))
    placeholder_output = _placeholder_output(
        profile.family,
        profile.required_output_fields,
    )
    rows: list[dict[str, Any]] = []
    for index, split in enumerate(("development", "heldout"), start=1):
        language = profile.languages[min(index - 1, len(profile.languages) - 1)]
        row: dict[str, Any] = {
            "case_id": f"{profile.task_id}_{split}_{index:03d}",
            "group_id": f"{profile.task_id}_group_{index:03d}",
            "input": (
                sample_input
                if index == 1 and sample_input is not None
                else f"TODO: add a representative {split} input"
            ),
            "expected_output": (
                sample_expected_output
                if index == 1 and sample_expected_output is not None
                else placeholder_output
            ),
            "evaluation_type": profile.evaluation_type,
        }
        for column in profile.slice_columns:
            if column == "language":
                row[column] = language
            elif column == "split":
                row[column] = split
            elif column == "risk":
                row[column] = "normal" if index == 1 else "high"
            else:
                row[column] = f"TODO: set {column}"
        rows.append(row)
    return columns, rows


def _dataset_record(profile: TaskProfile) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "dataset_id": f"{profile.task_id}_cases",
        "version": profile.version,
        "title": f"{profile.title} local comparison cases",
        "description": (
            "Development record for locally prepared representative cases. "
            "Complete the case template and document provenance before use."
        ),
        "created_on": profile.created_on.isoformat(),
        "status": "development",
        "limitations": [
            "This is a local starter record, not a frozen dataset release.",
            "Case sources, sampling, and expected-answer review remain user-owned.",
        ],
    }


def _benchmark(profile: TaskProfile) -> dict[str, Any]:
    system_prompt, max_completion_tokens = _default_prompt(profile.family)
    scoring: dict[str, Any] = {
        "evaluation_type": profile.evaluation_type,
        "version": "1.0",
        "metrics": {
            "case_sensitive": False,
            "trim_whitespace": True,
            "collapse_whitespace": True,
            "unicode_normalization": "NFC",
        },
    }
    if profile.family == "structured_extraction":
        scoring["required_output_fields"] = profile.required_output_fields
    if profile.family == "open_ended_response":
        scoring.update(
            {
                "judge": {
                    "model": "${JUDGE_MODEL}",
                    "variant_id": "primary",
                    "temperature": 0,
                    "max_retries": 3,
                    "continue_on_error": True,
                    "prompt_id": "evalanche.criteria_pointwise",
                    "prompt_version": "1.0",
                    "rubric_id": f"{profile.task_id}_rubric",
                    "rubric_version": "1.0",
                    "candidate_identity_blinded": True,
                    "reference_mode": "required",
                    "protocol_path": "judge_validation/protocol.yaml",
                    "minimum_validation_level_for_selection": "calibrated",
                },
                "task": {
                    "name": profile.task_id,
                    "description": profile.description,
                    "measured_construct": profile.measured_construct,
                    "intended_use": profile.intended_use,
                    "languages": profile.languages,
                },
                "score": {
                    "score_min": 0,
                    "score_max": 5,
                    "pass_threshold": 0.75,
                },
                "criteria": [
                    criterion.model_dump(mode="json")
                    for criterion in profile.criteria
                ],
            }
        )

    screen_inherits = "smoke" if profile.starting_tier == "smoke" else None
    screen: dict[str, Any] = {
        "description": "Screen every current starter group.",
        "sampling": {"method": "all", "unit": "group"},
        "cost": {
            "initial_prompt_tokens": 2000,
            "initial_completion_tokens": max_completion_tokens,
            "safety_multiplier": 1.5,
            "request_ceiling_multiplier": 2.0,
        },
    }
    if screen_inherits is not None:
        screen["inherits"] = screen_inherits
        screen["cost"] = {
            "sample_from": "smoke",
            "safety_multiplier": 1.5,
            "request_ceiling_multiplier": 2.0,
        }

    return {
        "schema_version": "1.0",
        "benchmark_id": profile.task_id,
        "version": profile.version,
        "title": profile.title,
        "description": profile.description,
        "status": "draft",
        "dataset": {
            "dataset_id": f"{profile.task_id}_cases",
            "version": profile.version,
            "manifest_path": (
                f"configs/datasets/{profile.task_id}_cases_"
                f"{profile.version}.yaml"
            ),
            "cases_path": "data/cases.csv",
        },
        "prompt": {
            "version": "1.0",
            "system": system_prompt,
            "template": "{input}",
        },
        "scoring": scoring,
        "slice_columns": profile.slice_columns,
        "group_key": "group_id",
        "language_column": (
            "language" if "language" in profile.slice_columns else None
        ),
        "required_capabilities": [],
        "runtime": {
            "request_api": "chat_completions",
            "max_completion_tokens": max_completion_tokens,
            "max_workers": 4,
            "max_requests_per_minute": 60,
            "checkpoint_every": 10,
            "resume": True,
        },
        "tiers": {
            "smoke": {
                "description": "Verify one route and artifact path.",
                "sampling": {
                    "method": "balanced",
                    "unit": "group",
                    "count": 1,
                    "seed": int(profile.created_on.strftime("%Y%m%d")),
                    "stratify_by": [
                        column
                        for column in ("language", "risk")
                        if column in profile.slice_columns
                    ],
                },
                "cost": {
                    "initial_prompt_tokens": 2000,
                    "initial_completion_tokens": max_completion_tokens,
                    "safety_multiplier": 2.0,
                    "request_ceiling_multiplier": 2.0,
                },
            },
            "screen": screen,
        },
        "limitations": [
            "Starter cases and expected outputs require task-owner review.",
            "Results apply only to the recorded task and model configurations.",
        ],
    }


def _model_manifest(
    profile: TaskProfile,
    model: TaskModelRoute,
) -> dict[str, Any]:
    _, max_completion_tokens = _default_prompt(profile.family)
    return {
        "schema_version": "1.0",
        "model_id": model.model_id,
        "display_name": model.display_name or _display_name(model.model_id),
        "provider_route": model.provider_route,
        "request": {
            "temperature": 0,
            "max_retries": 3,
            "max_completion_tokens": max_completion_tokens,
        },
        "provider_model_version": None,
        "deployment_type": None,
        "resource_region": None,
        "capabilities": [],
        "notes": (
            "Route explicitly supplied to init-task. Verify version, region, "
            "capabilities, and pricing before paid execution."
        ),
    }


def _access_set(profile: TaskProfile) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "access_set_id": f"{profile.task_id}_available_models",
        "description": f"Routes confirmed for {profile.title}.",
        "confirmed_on": profile.created_on.isoformat(),
        "models": [
            {
                "model_id": model.model_id,
                "access_confirmed": True,
                "confirmation_basis": (
                    "Explicitly supplied to init-task as available in the "
                    "intended environment."
                ),
                "notes": (
                    "Reconfirm after a deployment, version, region, or access "
                    "change."
                ),
            }
            for model in profile.models
        ],
    }


def _judge_protocol(profile: TaskProfile) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "protocol_id": f"{profile.task_id}_judge",
        "version": profile.version,
        "title": f"{profile.title} judge validation",
        "target_level": "exploratory",
        "synthetic_fixture": False,
        "contract": {
            "mode": "pointwise",
            "judge_variant_id": "primary",
            "judge_model": "${JUDGE_MODEL}",
            "provider_model_version": None,
            "temperature": 0,
            "prompt_id": "evalanche.criteria_pointwise",
            "prompt_version": "1.0",
            "rubric_id": f"{profile.task_id}_rubric",
            "rubric_version": "1.0",
            "candidate_identity_blinded": True,
            "reference_mode": "required",
            "task": {
                "name": profile.task_id,
                "description": profile.description,
                "measured_construct": profile.measured_construct,
                "intended_use": profile.intended_use,
                "languages": profile.languages,
            },
            "scoring": {
                "score_min": 0,
                "score_max": 5,
                "pass_threshold": 0.75,
            },
            "criteria": [
                criterion.model_dump(mode="json")
                for criterion in profile.criteria
            ],
        },
        "data": {
            "cases_path": "judge_validation/cases.csv",
            "human_annotations_path": (
                "judge_validation/human_annotations.csv"
            ),
            "judge_observations_path": (
                "judge_validation/judge_observations.csv"
            ),
            "labels": ["pass", "fail"],
            "pass_label": "pass",
            "failure_label": "fail",
            "criteria": ["overall"],
            "primary_criterion": "overall",
            "split_column": "split",
            "calibration_split": "calibration",
            "validation_split": "validation",
            "slice_columns": ["language", "difficulty", "risk"],
        },
        "human_review": {
            "reviewer_population": (
                "TODO: identify qualified independent reviewers"
            ),
            "instructions_version": "TODO: version the review instructions",
            "minimum_independent_raters": 2,
            "independent_before_adjudication": True,
            "candidate_identity_blinded": True,
            "judge_identity_blinded": True,
            "adjudication_rule": (
                "TODO: define how reviewer disagreements are adjudicated"
            ),
        },
        "controls": {
            "primary_judge_variant_id": "primary",
            "baseline_trial_id": "trial_1",
            "baseline_prompt_variant_id": "baseline",
            "baseline_identity_condition": "blinded",
            "baseline_presentation_order": "pointwise",
            "minimum_independent_trials": 1,
            "minimum_prompt_variants": 1,
            "require_identity_bias_test": False,
            "require_position_swap": False,
            "response_cache_disabled": False,
            "required_slice_values": {},
            "bootstrap_samples": 1000,
            "bootstrap_seed": int(profile.created_on.strftime("%Y%m%d")),
        },
        "thresholds": {
            "minimum_validation_cases": 1,
        },
        "governance": {
            "protocol_status": "draft",
            "heldout_not_used_for_development": False,
            "independent_review": False,
            "conflicts_documented": False,
            "raw_artifacts_preserved": True,
            "judge_selection_rationale": (
                "TODO: explain why this judge route fits the task"
            ),
            "publication_notes": (
                "Starter protocol. No calibrated or decision-grade claim is "
                "made."
            ),
        },
        "research_basis": [],
        "limitations": [
            "No task-specific human or judge observations are supplied.",
            "The generated judge path is exploratory until the exact protocol "
            "is validated against qualified human review.",
        ],
    }


def _expected_answer_guidance(profile: TaskProfile) -> str:
    common = f"""# Expected answers for {profile.title}

The expected output defines what success means. Replace every `TODO` value in
`data/cases.csv`, preserve the source used to determine each answer, and keep a
development set separate from cases reserved for the final comparison.

For every case:

1. Record a representative input from an approved source.
2. Record an expected output supported by that source.
3. Check ambiguity, missing context, and unacceptable errors.
4. Keep related variants under the same `group_id` when they should not be
   treated as independent.
5. Set language, split, risk, and other slices deliberately.

Unacceptable errors declared for this task:
{chr(10).join(f'- {value}' for value in profile.unacceptable_errors)}
"""
    if profile.family == "classification":
        route = """
## Classification contract

Use one stable closed label for each case. Define the complete allowed label
set before the held-out run. Exact scoring is appropriate only when case,
whitespace, and Unicode normalization capture every defensible equivalence.
"""
    elif profile.family == "structured_extraction":
        route = f"""
## Structured extraction contract

Each expected output must be one valid JSON object containing these fields:

{chr(10).join(f'- `{field}`' for field in profile.required_output_fields)}

Define any unordered lists, numeric equivalences, aliases, or other
canonicalization explicitly in the benchmark before inspecting held-out
failures. Strict pass means every required field matches. Field scores remain
diagnostic only.
"""
    elif profile.family == "factual_response":
        route = """
## Factual response contract

Use an authoritative, traceable source for each answer. The starter uses exact
normalized text, so add a different deterministic contract if more than one
wording is legitimately correct. Do not use fuzzy similarity as an implicit
primary pass rule.
"""
    else:
        route = """
## Open-ended response contract

There may be no single gold string. Supply a human-reviewed reference or
anchor, observable criteria, source-grounded facts, and prohibited errors.
The generated judge protocol is exploratory. It does not contain human labels
or judge observations and cannot support a calibrated claim until those are
collected and validated for the exact task, rubric, prompt, judge route, and
settings.
"""
    return common + route


def _pricing_template(profile: TaskProfile) -> str:
    routes = "\n".join(
        f"# - {model.model_id}: {model.provider_route}"
        for model in profile.models
    ) or "# - TODO: add an explicitly available model route"
    return f"""# Pricing template for {profile.title}
#
# This file is outside configs/ so registry validation will not treat the
# placeholders as evidence. Copy it to configs/pricing.yaml only after every
# rate, route, date, and source is current and verified. Then add that path and
# the matching pricing_id to each model manifest.
#
# Named routes:
{routes}
schema_version: "1.0"
catalog_version: TODO_REPLACE_ME
endpoints:
  - pricing_id: TODO_REPLACE_ME
    model: TODO_REPLACE_ME
    currency: USD
    input_per_million_tokens: TODO_REPLACE_ME
    output_per_million_tokens: TODO_REPLACE_ME
    effective_date: TODO_REPLACE_ME
    source: TODO_REPLACE_ME
    notes: Do not use until every placeholder has been replaced and reviewed.
"""


def _model_template(profile: TaskProfile) -> str:
    _, max_completion_tokens = _default_prompt(profile.family)
    return f"""# Move a completed copy to configs/models/<model_id>.yaml.
# Never add a model merely because it exists in the main Evalanche registry.
schema_version: "1.0"
model_id: TODO_REPLACE_ME
display_name: TODO_REPLACE_ME
provider_route: TODO_REPLACE_ME
request:
  temperature: 0
  max_retries: 3
  max_completion_tokens: {max_completion_tokens}
provider_model_version:
deployment_type:
resource_region:
capabilities: []
notes: Confirm access, route metadata, capabilities, and pricing before use.
"""


def _access_template(profile: TaskProfile) -> str:
    return f"""# Move a completed copy to configs/access_sets/
# {profile.task_id}_available_models.yaml.
schema_version: "1.0"
access_set_id: {profile.task_id}_available_models
description: Routes explicitly confirmed for {profile.title}.
confirmed_on: TODO_REPLACE_ME
models:
  - model_id: TODO_REPLACE_ME
    access_confirmed: true
    confirmation_basis: TODO_REPLACE_ME
    notes: Reconfirm when the route or intended environment changes.
"""


def _readme(profile: TaskProfile, bundle_path: str) -> str:
    benchmark = f"{profile.task_id}@{profile.version}"
    model_flags = " ".join(
        f"--model {model.model_id}" for model in profile.models
    )
    if not model_flags:
        model_flags = "--model MODEL_ID"
    access_path = (
        f"configs/access_sets/{profile.task_id}_available_models.yaml"
    )
    chosen_tier = profile.starting_tier
    validate_command = (
        "docker compose run --rm evalanche python -m evalanche.cli "
        f"validate-task --root {bundle_path}"
    )
    registry_command = (
        "docker compose run --rm evalanche python -m evalanche.cli "
        f"registry-validate --root {bundle_path}"
    )
    plan_command = (
        "docker compose run --rm evalanche python -m evalanche.cli "
        f"run-benchmark --root {bundle_path} --benchmark {benchmark} "
        f"{model_flags} --plan-only"
    )
    cost_validation_command = f"{validate_command} --require-cost-ready"
    preflight_command = (
        "docker compose run --rm evalanche python -m evalanche.cli "
        f"run-benchmark --root {bundle_path} --benchmark {benchmark} "
        f"--tier {chosen_tier} --all-compatible --access-set {access_path} "
        "--preflight-only"
    )
    run_command = (
        "docker compose run --rm evalanche python -m evalanche.cli "
        f"run-benchmark --root {bundle_path} --benchmark {benchmark} "
        f"--tier {chosen_tier} --all-compatible --access-set {access_path} "
        "--experiment --max-cost-usd REVIEWED_BUDGET"
    )
    summary_command = (
        "docker compose run --rm evalanche python -m evalanche.cli "
        f"summarize-benchmark --root {bundle_path} --benchmark {benchmark} "
        f"--tier {chosen_tier} --access-set {access_path}"
    )
    judge_validation_command = (
        "docker compose run --rm evalanche python -m evalanche.cli "
        f"validate-judge --root {bundle_path} --protocol "
        f"{bundle_path}/judge_validation/protocol.yaml"
    )
    open_ended_note = ""
    execution_command = f"""```bash
{run_command}
```"""
    if profile.family == "open_ended_response":
        open_ended_note = f"""
The open-ended route also makes judge-model calls. The current tier preflight
prices candidate generation, not judge evaluation. Keep this bundle
planning-only until the judge route, pricing, evaluation evidence, and total
cost exposure have been reviewed explicitly. `validate-task` will keep
execution blocked while `review.judge_route_and_cost` is false.

After replacing the judge-validation templates with real independent human
annotations and saved judge observations, analyze them offline:

```bash
{judge_validation_command}
```
"""
        execution_command = """```text
No paid open-ended execution command is generated by default. Review
docs/llm_judges.md and the note above before approving that step.
```"""

    models_note = (
        "The model manifests and dated access set were created only for the "
        "routes supplied to `init-task`. Reconfirm their metadata and add "
        "current pricing before preflight."
        if profile.models
        else "No model was supplied. Complete the model and access-set "
        "templates before planning a comparison."
    )

    return f"""# {profile.title}

This local starter bundle was created offline by Evalanche. It is isolated
from the repository's main model registry and is ignored by Git by default.
Run every command below from the Evalanche repository root.

## 1. Complete the task

1. Replace every `TODO` in `data/cases.csv` and `task.yaml`.
2. Read `guidance/expected_answers.md`.
3. Review the generated prompt, scoring contract, slices, and tiers in
   `configs/benchmarks/{profile.task_id}_{profile.version}.yaml`.
4. Set the applicable `task.yaml` review fields to `true` only after review.
5. Keep real local cases and provider outputs out of Git unless they are
   deliberately approved for sharing.

{models_note}

## 2. Validate offline

This command makes no provider calls and exits unsuccessfully while required
task-owned values remain unresolved:

```bash
{validate_command}
```

Validate the generated registry separately:

```bash
{registry_command}
```

## 3. Build a no-call plan

```bash
{plan_command}
```

## 4. Add verified pricing, then preflight

Evalanche does not invent prices. Use `templates/pricing_catalog.yaml` as a
worksheet, then add a valid pricing catalog path and pricing ID to each model
manifest. Confirm full readiness:

```bash
{cost_validation_command}
```

Run the aggregate no-call cost preflight:

```bash
{preflight_command}
```

{open_ended_note}

## 5. Run the reviewed tier

Choose `REVIEWED_BUDGET` only after reading the preflight.

{execution_command}

## 6. Build the comparison

```bash
{summary_command}
```

The report preserves tradeoffs and does not select a model automatically.
For method details, return to `docs/local_comparison.md` in the main repository.
"""


def _judge_templates(
    profile: TaskProfile,
    case_rows: list[dict[str, Any]],
) -> dict[str, tuple[list[str], list[dict[str, Any]]]]:
    cases = [
        {
            "case_id": row["case_id"],
            "split": (
                "calibration"
                if row.get("split") == "development"
                else "validation"
            ),
            "language": row.get("language", profile.languages[0]),
            "difficulty": "TODO",
            "risk": row.get("risk", "TODO"),
            "input": row["input"],
            "reference_answer": row["expected_output"],
            "candidate_output": "TODO: add one saved candidate output",
        }
        for row in case_rows
    ]
    return {
        "cases.csv": (
            [
                "case_id",
                "split",
                "language",
                "difficulty",
                "risk",
                "input",
                "reference_answer",
                "candidate_output",
            ],
            cases,
        ),
        "human_annotations.csv": (
            [
                "case_id",
                "criterion",
                "reviewer_id",
                "annotation_stage",
                "label",
                "notes",
            ],
            [],
        ),
        "judge_observations.csv": (
            [
                "case_id",
                "criterion",
                "judge_variant_id",
                "trial_id",
                "prompt_variant_id",
                "identity_condition",
                "presentation_order",
                "label",
                "status",
                "uncertainty",
                "reason",
            ],
            [],
        ),
    }


def initialize_task_bundle(
    *,
    root_path: str | Path,
    task_id: str,
    description: str,
    family: str,
    languages: list[str] | None = None,
    slice_columns: list[str] | None = None,
    unacceptable_errors: list[str] | None = None,
    models: list[TaskModelRoute] | None = None,
    required_output_fields: list[str] | None = None,
    criteria: list[TaskCriterion] | None = None,
    measured_construct: str | None = None,
    intended_use: str | None = None,
    sample_input: str | None = None,
    sample_expected_output: str | None = None,
    artifact_policy: Literal["local_only", "shareable"] = "local_only",
    contains_sensitive_data: bool = False,
    privacy_notes: str | None = None,
    starting_tier: Literal["smoke", "screen"] = "smoke",
    output_dir: str | Path = DEFAULT_TASKS_DIR,
    replace: bool = False,
    created_on: date | None = None,
) -> dict[str, Any]:
    """Create one isolated task bundle without making provider calls."""

    root = Path(root_path).resolve()
    if not root.is_dir():
        raise ValueError(f"Repository root does not exist: {root_path}")

    normalized_task_id = _identifier(task_id, "task_id")
    normalized_family = normalize_family(family)
    normalized_languages = languages or ["en"]
    normalized_slices = list(
        dict.fromkeys(["language", "split", "risk", *(slice_columns or [])])
    )
    normalized_models = models or []
    normalized_fields = required_output_fields or []
    normalized_criteria = criteria or []

    if contains_sensitive_data and (
        sample_input is not None or sample_expected_output is not None
    ):
        raise ValueError(
            "Do not pass sample values on the command line when the task may "
            "contain sensitive data. Add approved cases inside the intended "
            "environment after initialization."
        )

    if normalized_family == "structured_extraction":
        if sample_expected_output is not None:
            try:
                parsed_output = json.loads(sample_expected_output)
            except json.JSONDecodeError as error:
                raise ValueError(
                    "structured sample expected output must be valid JSON"
                ) from error
            if not isinstance(parsed_output, dict):
                raise ValueError(
                    "structured sample expected output must be a JSON object"
                )
            if not normalized_fields:
                normalized_fields = list(parsed_output)
            missing_fields = sorted(
                set(normalized_fields) - set(parsed_output)
            )
            if missing_fields:
                raise ValueError(
                    "structured sample output is missing required fields: "
                    + ", ".join(missing_fields)
                )
            sample_expected_output = json.dumps(
                parsed_output,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        if not normalized_fields:
            normalized_fields = ["todo_required_field"]
    elif normalized_fields:
        raise ValueError(
            "output fields can only be supplied for structured extraction"
        )

    if normalized_family == "open_ended_response":
        if not normalized_criteria:
            normalized_criteria = [
                TaskCriterion(
                    name="todo_criterion",
                    weight=1.0,
                    description=(
                        "TODO: define one observable evaluation criterion"
                    ),
                )
            ]
        measured_construct = measured_construct or (
            "TODO: define the construct this judge is intended to measure"
        )
        intended_use = intended_use or (
            "TODO: define how the resulting evidence may be used"
        )
    elif normalized_criteria:
        raise ValueError(
            "criteria can only be supplied for open-ended response"
        )

    created = created_on or date.today()
    profile = TaskProfile(
        task_id=normalized_task_id,
        title=_display_name(normalized_task_id),
        description=description,
        family=normalized_family,
        evaluation_type=_evaluation_type(normalized_family),
        languages=normalized_languages,
        slice_columns=normalized_slices,
        unacceptable_errors=(
            unacceptable_errors
            or ["TODO: describe one error that would make an output unusable"]
        ),
        required_output_fields=normalized_fields,
        criteria=normalized_criteria,
        measured_construct=measured_construct,
        intended_use=intended_use,
        artifact_policy=artifact_policy,
        contains_sensitive_data=contains_sensitive_data,
        privacy_notes=(
            privacy_notes
            or (
                "Keep cases and generated outputs local until their sharing "
                "status is reviewed."
            )
        ),
        starting_tier=starting_tier,
        models=normalized_models,
        created_on=created,
    )

    output_root = Path(output_dir)
    if not output_root.is_absolute():
        output_root = root / output_root
    output_root = output_root.resolve()
    try:
        output_root.relative_to(root)
    except ValueError as error:
        raise ValueError(
            "output_dir must stay inside the repository so Docker commands "
            "can reach the generated bundle"
        ) from error
    output_root.mkdir(parents=True, exist_ok=True)

    target = output_root / normalized_task_id
    if target.is_symlink():
        raise ValueError("refusing to replace a symlinked task bundle")
    backup: Path | None = None
    if target.exists():
        if not replace:
            raise FileExistsError(
                f"Task bundle already exists: {target}. Choose a new task ID "
                "or use --replace for a recoverable backup."
            )
        timestamp = datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%S%fZ"
        )
        backup = output_root / f"{normalized_task_id}.backup-{timestamp}"
        if backup.exists():
            raise FileExistsError(f"Backup path already exists: {backup}")
        target.rename(backup)

    staging = output_root / (
        f".{normalized_task_id}.creating-{uuid.uuid4().hex[:10]}"
    )
    try:
        staging.mkdir()
        columns, rows = _case_rows(
            profile,
            sample_input=sample_input,
            sample_expected_output=sample_expected_output,
        )
        _write_yaml(staging / "task.yaml", profile.model_dump(mode="json"))
        _write_csv(staging / "data/cases.csv", columns, rows)
        _write_yaml(
            staging
            / "configs/datasets"
            / f"{profile.task_id}_cases_{profile.version}.yaml",
            _dataset_record(profile),
        )
        _write_yaml(
            staging
            / "configs/benchmarks"
            / f"{profile.task_id}_{profile.version}.yaml",
            _benchmark(profile),
        )
        for model in profile.models:
            _write_yaml(
                staging / "configs/models" / f"{model.model_id}.yaml",
                _model_manifest(profile, model),
            )
        if profile.models:
            _write_yaml(
                staging
                / "configs/access_sets"
                / f"{profile.task_id}_available_models.yaml",
                _access_set(profile),
            )
        else:
            _write_text(
                staging / "templates/access_set.yaml",
                _access_template(profile),
            )
        _write_text(
            staging / "templates/model.yaml",
            _model_template(profile),
        )
        _write_text(
            staging / "templates/pricing_catalog.yaml",
            _pricing_template(profile),
        )
        _write_text(
            staging / "guidance/expected_answers.md",
            _expected_answer_guidance(profile),
        )
        if profile.family == "open_ended_response":
            _write_yaml(
                staging / "judge_validation/protocol.yaml",
                _judge_protocol(profile),
            )
            for filename, (fieldnames, template_rows) in _judge_templates(
                profile,
                rows,
            ).items():
                _write_csv(
                    staging / "judge_validation" / filename,
                    fieldnames,
                    template_rows,
                )

        bundle_relative = target.relative_to(root).as_posix()
        _write_text(
            staging / "README.md",
            _readme(profile, bundle_relative),
        )
        staging.rename(target)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        if backup is not None and backup.exists() and not target.exists():
            backup.rename(target)
        raise

    validation = validate_task_bundle(target)
    return {
        "task_root": target,
        "task_id": profile.task_id,
        "benchmark": f"{profile.task_id}@{profile.version}",
        "backup": backup,
        "validation": validation,
        "provider_calls": 0,
    }


def _validate_cases(
    *,
    cases: pd.DataFrame,
    profile: TaskProfile,
    benchmark: BenchmarkManifest,
) -> tuple[list[str], list[str]]:
    structural: list[str] = []
    unresolved: list[str] = []
    required = {
        "case_id",
        "group_id",
        "input",
        "expected_output",
        "evaluation_type",
        *profile.slice_columns,
    }
    missing = sorted(required - set(cases.columns))
    if missing:
        structural.append(f"data/cases.csv is missing columns: {missing}")
        return structural, unresolved
    if cases.empty:
        structural.append("data/cases.csv contains no cases")
        return structural, unresolved
    if cases["case_id"].astype(str).duplicated().any():
        structural.append("data/cases.csv contains duplicate case IDs")
    observed_types = set(cases["evaluation_type"].astype(str))
    if observed_types != {profile.evaluation_type}:
        structural.append(
            "case evaluation types do not match task.yaml: "
            f"{sorted(observed_types)}"
        )
    if benchmark.scoring.evaluation_type != profile.evaluation_type:
        structural.append("benchmark scoring does not match task.yaml")
    for column in ("input", "expected_output", *profile.slice_columns):
        values = cases[column].fillna("").astype(str)
        if values.str.strip().eq("").any():
            unresolved.append(f"complete every {column!r} case value")
        if values.map(_contains_placeholder).any():
            unresolved.append(f"replace every placeholder in {column!r}")
    if "split" in cases.columns:
        splits = set(cases["split"].astype(str).str.casefold())
        for required_split in ("development", "heldout"):
            if required_split not in splits:
                unresolved.append(
                    f"add at least one {required_split} case"
                )
    if "language" in cases.columns:
        unknown = sorted(
            set(cases["language"].astype(str).str.casefold())
            - set(profile.languages)
        )
        if unknown:
            structural.append(
                "case languages are not declared in task.yaml: "
                + ", ".join(unknown)
            )
    if profile.family == "structured_extraction":
        for row_index, value in enumerate(cases["expected_output"]):
            try:
                parsed = json.loads(str(value))
            except json.JSONDecodeError:
                structural.append(
                    f"expected_output row {row_index} is not valid JSON"
                )
                break
            if not isinstance(parsed, dict):
                structural.append(
                    f"expected_output row {row_index} is not a JSON object"
                )
                break
            missing_fields = sorted(
                set(profile.required_output_fields) - set(parsed)
            )
            if missing_fields:
                structural.append(
                    f"expected_output row {row_index} is missing fields: "
                    + ", ".join(missing_fields)
                )
                break
    return structural, list(dict.fromkeys(unresolved))


def _pricing_readiness(
    *,
    root: Path,
    models: list[ModelManifest],
) -> tuple[list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []
    for model in models:
        if model.provider_model_version is None:
            warnings.append(
                f"{model.model_id}: provider model version is not recorded"
            )
        if model.pricing_catalog_path is None or model.pricing_id is None:
            blockers.append(
                f"{model.model_id}: add a verified pricing catalog and pricing ID"
            )
            continue
        try:
            catalog = load_endpoint_pricing_catalog(
                root / model.pricing_catalog_path
            )
            resolve_endpoint_price(
                catalog=catalog,
                pricing_id=model.pricing_id,
                model=model.provider_route,
            )
        except (OSError, ValueError) as error:
            blockers.append(f"{model.model_id}: pricing is invalid: {error}")
    return blockers, warnings


def validate_task_bundle(root_path: str | Path) -> dict[str, Any]:
    """Validate an initialized task without making provider calls."""

    root = Path(root_path).resolve()
    structural: list[str] = []
    unresolved: list[str] = []
    warnings: list[str] = []
    profile: TaskProfile | None = None
    benchmark: BenchmarkManifest | None = None
    model_manifests: list[ModelManifest] = []

    profile_path = root / "task.yaml"
    if not profile_path.is_file():
        structural.append("task.yaml was not found")
    else:
        try:
            profile = TaskProfile.model_validate(load_yaml(profile_path))
        except (OSError, ValueError) as error:
            structural.append(f"task.yaml is invalid: {error}")

    required_paths = (
        "README.md",
        "data/cases.csv",
        "guidance/expected_answers.md",
    )
    for relative_path in required_paths:
        if not (root / relative_path).is_file():
            structural.append(f"required file is missing: {relative_path}")

    if profile is not None:
        dataset_path = (
            root
            / "configs/datasets"
            / f"{profile.task_id}_cases_{profile.version}.yaml"
        )
        benchmark_path = (
            root
            / "configs/benchmarks"
            / f"{profile.task_id}_{profile.version}.yaml"
        )
        try:
            dataset = TaskDatasetRecord.model_validate(load_yaml(dataset_path))
            if dataset.dataset_id != f"{profile.task_id}_cases":
                structural.append("dataset record ID does not match task.yaml")
            if dataset.version != profile.version:
                structural.append(
                    "dataset record version does not match task.yaml"
                )
        except (OSError, ValueError) as error:
            structural.append(f"dataset starter record is invalid: {error}")
        try:
            benchmark = BenchmarkManifest.model_validate(
                load_yaml(benchmark_path)
            )
            if benchmark.benchmark_id != profile.task_id:
                structural.append("benchmark ID does not match task.yaml")
            if benchmark.version != profile.version:
                structural.append("benchmark version does not match task.yaml")
        except (OSError, ValueError) as error:
            structural.append(f"benchmark manifest is invalid: {error}")

        if benchmark is not None:
            try:
                cases = pd.read_csv(root / "data/cases.csv")
                case_structural, case_unresolved = _validate_cases(
                    cases=cases,
                    profile=profile,
                    benchmark=benchmark,
                )
                structural.extend(case_structural)
                unresolved.extend(case_unresolved)
            except (OSError, ValueError) as error:
                structural.append(f"data/cases.csv is invalid: {error}")

        if _contains_placeholder(profile.unacceptable_errors):
            unresolved.append("replace unacceptable-error placeholders")
        if _contains_placeholder(profile.required_output_fields):
            unresolved.append("replace required-output-field placeholders")
        if _contains_placeholder(profile.criteria):
            unresolved.append("replace judge-criterion placeholders")
        if _contains_placeholder(profile.measured_construct):
            unresolved.append("define the measured construct")
        if _contains_placeholder(profile.intended_use):
            unresolved.append("define the judge evidence's intended use")
        for name, completed in profile.review.model_dump().items():
            if name == "judge_route_and_cost" and (
                profile.family != "open_ended_response"
            ):
                continue
            if not completed:
                unresolved.append(f"complete review.{name}")

        expected_model_ids = {model.model_id for model in profile.models}
        model_paths = sorted((root / "configs/models").glob("*.yaml"))
        observed_model_ids: set[str] = set()
        for model_path in model_paths:
            try:
                model = ModelManifest.model_validate(load_yaml(model_path))
                observed_model_ids.add(model.model_id)
                model_manifests.append(model)
            except (OSError, ValueError) as error:
                structural.append(
                    f"model manifest {model_path.name} is invalid: {error}"
                )
        if observed_model_ids != expected_model_ids:
            structural.append(
                "task.yaml models and configs/models do not match: "
                f"expected {sorted(expected_model_ids)}, found "
                f"{sorted(observed_model_ids)}"
            )
        if not expected_model_ids:
            unresolved.append(
                "add at least one explicitly available model route"
            )
        else:
            access_path = (
                root
                / "configs/access_sets"
                / f"{profile.task_id}_available_models.yaml"
            )
            try:
                access_set = load_access_set(access_path)
                if set(access_set.model_ids) != expected_model_ids:
                    structural.append(
                        "access set does not match task.yaml model IDs"
                    )
            except (OSError, ValueError) as error:
                structural.append(f"access set is invalid: {error}")

        if profile.family == "open_ended_response":
            protocol_path = root / "judge_validation/protocol.yaml"
            try:
                protocol = load_judge_protocol(protocol_path)
                if protocol.target_level != "exploratory":
                    structural.append(
                        "generated judge protocol must start as exploratory"
                    )
                if protocol.contract.task.name != profile.task_id:
                    structural.append(
                        "judge protocol task does not match task.yaml"
                    )
                if benchmark is not None and benchmark.scoring.judge is not None:
                    judge = benchmark.scoring.judge
                    if protocol.contract.judge_model != judge.model:
                        structural.append(
                            "judge protocol model does not match benchmark"
                        )
                    protocol_criteria = [
                        criterion.model_dump(mode="json")
                        for criterion in protocol.contract.criteria
                    ]
                    benchmark_criteria = [
                        criterion.model_dump(mode="json")
                        for criterion in benchmark.scoring.criteria
                    ]
                    if protocol_criteria != benchmark_criteria:
                        structural.append(
                            "judge protocol criteria do not match benchmark"
                        )
                warnings.append(
                    "judge evidence remains exploratory until real human and "
                    "judge observations validate the exact protocol"
                )
            except (OSError, ValueError) as error:
                structural.append(f"judge protocol is invalid: {error}")

    try:
        registry = load_registry(root)
        registry_result = validate_registry(registry)
        for issue in registry_result["issues"]:
            if issue == "No model manifests were found.":
                if profile is None or profile.models:
                    structural.append(issue)
            else:
                structural.append(f"registry: {issue}")
    except (OSError, ValueError) as error:
        structural.append(f"registry could not be loaded: {error}")

    pricing_blockers, pricing_warnings = _pricing_readiness(
        root=root,
        models=model_manifests,
    )
    warnings.extend(pricing_warnings)
    structural = list(dict.fromkeys(structural))
    unresolved = list(dict.fromkeys(unresolved))
    warnings = list(dict.fromkeys(warnings))
    schema_valid = not structural
    ready_for_plan = schema_valid and not unresolved
    ready_for_cost_preflight = (
        ready_for_plan
        and bool(model_manifests)
        and not pricing_blockers
    )
    execution_supported = bool(
        profile is not None and profile.family != "open_ended_response"
    )
    return {
        "schema_version": TASK_BUNDLE_SCHEMA_VERSION,
        "task_root": str(root),
        "task_id": profile.task_id if profile is not None else None,
        "family": profile.family if profile is not None else None,
        "schema_valid": schema_valid,
        "ready_for_plan": ready_for_plan,
        "ready_for_cost_preflight": ready_for_cost_preflight,
        "paid_execution_supported": execution_supported,
        "structural_issues": structural,
        "unresolved_values": unresolved,
        "cost_blockers": pricing_blockers,
        "warnings": warnings,
        "provider_calls": 0,
    }


def task_execution_blocker(
    root_path: str | Path,
    *,
    require_paid_execution: bool = True,
) -> str | None:
    root = Path(root_path).resolve()
    if not (root / "task.yaml").is_file():
        return None
    result = validate_task_bundle(root)
    if not result["ready_for_cost_preflight"]:
        details = (
            result["structural_issues"]
            + result["unresolved_values"]
            + result["cost_blockers"]
        )
        return "Generated task bundle is not ready: " + "; ".join(details)
    if require_paid_execution and not result["paid_execution_supported"]:
        return (
            "Generated open-ended task bundles remain planning-only because "
            "the tier budget does not include judge-model calls."
        )
    return None

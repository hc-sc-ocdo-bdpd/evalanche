from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from evalanche.config import EvalConfig, load_yaml


ValidationLevel = Literal[
    "exploratory",
    "calibrated",
    "decision_grade",
    "publishable",
]
VALIDATION_LEVELS: tuple[str, ...] = (
    "exploratory",
    "calibrated",
    "decision_grade",
    "publishable",
)


def validation_level_rank(level: str) -> int:
    try:
        return VALIDATION_LEVELS.index(level)
    except ValueError as error:
        raise ValueError(f"Unknown judge validation level: {level!r}") from error


def level_at_least(observed: str, required: str) -> bool:
    return validation_level_rank(observed) >= validation_level_rank(required)


def achieved_level_from_gates(
    gates: list[dict[str, Any]],
    *,
    synthetic_fixture: bool,
) -> str:
    """Derive an evidence level from cumulative protocol gate results."""
    if synthetic_fixture:
        return "exploratory"
    if not gates:
        raise ValueError("Judge validation report contains no protocol gates")

    normalized: list[tuple[str, bool]] = []
    for gate in gates:
        level = str(gate.get("required_for_level", ""))
        if level == "exploratory":
            raise ValueError(
                "Exploratory evidence cannot depend on a validation gate"
            )
        validation_level_rank(level)
        passed = gate.get("passed")
        if not isinstance(passed, bool):
            raise ValueError("Judge validation gate status must be boolean")
        normalized.append((level, passed))

    achieved = "exploratory"
    for level in ("calibrated", "decision_grade", "publishable"):
        rank = validation_level_rank(level)
        required = [
            passed
            for required_level, passed in normalized
            if 0 < validation_level_rank(required_level) <= rank
        ]
        if required and all(required):
            achieved = level
        else:
            break
    return achieved


def _normalized_nonempty(value: str, *, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} cannot be blank")
    return normalized


class ProtocolCriterion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    weight: float = Field(ge=0, allow_inf_nan=False)
    description: str

    @field_validator("name", "description")
    @classmethod
    def validate_text(cls, value: str, info: Any) -> str:
        return _normalized_nonempty(value, field_name=info.field_name)


class ProtocolTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    measured_construct: str
    intended_use: str
    languages: list[str] = Field(min_length=1)

    @field_validator(
        "name",
        "description",
        "measured_construct",
        "intended_use",
    )
    @classmethod
    def validate_text(cls, value: str, info: Any) -> str:
        return _normalized_nonempty(value, field_name=info.field_name)

    @field_validator("languages")
    @classmethod
    def validate_languages(cls, values: list[str]) -> list[str]:
        normalized = [
            _normalized_nonempty(value, field_name="languages")
            for value in values
        ]
        if len(normalized) != len(set(normalized)):
            raise ValueError("languages must be unique")
        return normalized


class ProtocolScoring(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score_min: int = 0
    score_max: int = 5
    pass_threshold: float = Field(default=0.75, ge=0, le=1)

    @model_validator(mode="after")
    def validate_range(self) -> "ProtocolScoring":
        if self.score_max <= self.score_min:
            raise ValueError("score_max must be greater than score_min")
        return self


class JudgeEvaluationContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["pointwise", "pairwise"]
    judge_variant_id: str
    judge_model: str
    provider_model_version: str | None = None
    temperature: float | None = 0
    prompt_id: str
    prompt_version: str
    rubric_id: str
    rubric_version: str
    candidate_identity_blinded: bool
    reference_mode: Literal["required", "optional", "none"]
    task: ProtocolTask
    scoring: ProtocolScoring
    criteria: list[ProtocolCriterion] = Field(min_length=1)

    @field_validator(
        "judge_variant_id",
        "judge_model",
        "prompt_id",
        "prompt_version",
        "rubric_id",
        "rubric_version",
    )
    @classmethod
    def validate_identity(cls, value: str, info: Any) -> str:
        return _normalized_nonempty(value, field_name=info.field_name)

    @field_validator("provider_model_version")
    @classmethod
    def validate_optional_version(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _normalized_nonempty(value, field_name="provider_model_version")

    @field_validator("criteria")
    @classmethod
    def validate_criteria(
        cls,
        criteria: list[ProtocolCriterion],
    ) -> list[ProtocolCriterion]:
        names = [criterion.name for criterion in criteria]
        if len(names) != len(set(names)):
            raise ValueError("protocol criterion names must be unique")
        if sum(criterion.weight for criterion in criteria) <= 0:
            raise ValueError("protocol criterion weights must sum positively")
        return criteria


class PairwiseLabels(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_a_label: str
    candidate_b_label: str
    tie_label: str

    @field_validator("candidate_a_label", "candidate_b_label", "tie_label")
    @classmethod
    def validate_label(cls, value: str, info: Any) -> str:
        return _normalized_nonempty(value, field_name=info.field_name)

    @model_validator(mode="after")
    def validate_unique(self) -> "PairwiseLabels":
        labels = {
            self.candidate_a_label,
            self.candidate_b_label,
            self.tie_label,
        }
        if len(labels) != 3:
            raise ValueError("pairwise labels must be distinct")
        return self


class JudgeValidationData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cases_path: Path
    human_annotations_path: Path
    judge_observations_path: Path
    labels: list[str] = Field(min_length=2)
    pass_label: str | None = None
    failure_label: str | None = None
    criteria: list[str] = Field(default_factory=lambda: ["overall"])
    primary_criterion: str = "overall"
    split_column: str = "split"
    calibration_split: str = "calibration"
    validation_split: str = "validation"
    slice_columns: list[str] = Field(default_factory=list)
    pairwise: PairwiseLabels | None = None

    @field_validator(
        "labels",
        "criteria",
        "slice_columns",
    )
    @classmethod
    def validate_unique_strings(
        cls,
        values: list[str],
        info: Any,
    ) -> list[str]:
        normalized = [
            _normalized_nonempty(value, field_name=info.field_name)
            for value in values
        ]
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"{info.field_name} must be unique")
        return normalized

    @field_validator(
        "primary_criterion",
        "split_column",
        "calibration_split",
        "validation_split",
    )
    @classmethod
    def validate_required_text(cls, value: str, info: Any) -> str:
        return _normalized_nonempty(value, field_name=info.field_name)

    @model_validator(mode="after")
    def validate_labels_and_criteria(self) -> "JudgeValidationData":
        if self.primary_criterion not in self.criteria:
            raise ValueError("primary_criterion must be listed in criteria")
        if (self.pass_label is None) != (self.failure_label is None):
            raise ValueError(
                "pass_label and failure_label must be configured together"
            )
        if self.pass_label is not None:
            if self.pass_label == self.failure_label:
                raise ValueError("pass_label and failure_label must differ")
            if self.pass_label not in self.labels:
                raise ValueError("pass_label must be listed in labels")
            if self.failure_label not in self.labels:
                raise ValueError("failure_label must be listed in labels")
            if set(self.labels) != {self.pass_label, self.failure_label}:
                raise ValueError(
                    "pass/failure diagnostics require exactly two labels"
                )
        if self.calibration_split == self.validation_split:
            raise ValueError("calibration and validation splits must differ")
        return self


class HumanReviewProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reviewer_population: str
    instructions_version: str
    minimum_independent_raters: int = Field(default=1, ge=1)
    independent_before_adjudication: bool = False
    candidate_identity_blinded: bool = False
    judge_identity_blinded: bool = False
    adjudication_rule: str

    @field_validator(
        "reviewer_population",
        "instructions_version",
        "adjudication_rule",
    )
    @classmethod
    def validate_text(cls, value: str, info: Any) -> str:
        return _normalized_nonempty(value, field_name=info.field_name)


class JudgeValidationControls(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primary_judge_variant_id: str
    baseline_trial_id: str = "trial_1"
    baseline_prompt_variant_id: str = "baseline"
    baseline_identity_condition: Literal["blinded", "named"] = "blinded"
    baseline_presentation_order: Literal["pointwise", "AB", "BA"] = (
        "pointwise"
    )
    minimum_independent_trials: int = Field(default=1, ge=1)
    minimum_prompt_variants: int = Field(default=1, ge=1)
    require_identity_bias_test: bool = False
    require_position_swap: bool = False
    response_cache_disabled: bool = False
    required_slice_values: dict[str, list[str]] = Field(default_factory=dict)
    bootstrap_samples: int = Field(default=1000, ge=100, le=10000)
    bootstrap_seed: int = 20260806

    @field_validator(
        "primary_judge_variant_id",
        "baseline_trial_id",
        "baseline_prompt_variant_id",
    )
    @classmethod
    def validate_identity(cls, value: str, info: Any) -> str:
        return _normalized_nonempty(value, field_name=info.field_name)

    @field_validator("required_slice_values")
    @classmethod
    def validate_slice_values(
        cls,
        values: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        normalized: dict[str, list[str]] = {}
        for raw_column, raw_values in values.items():
            column = _normalized_nonempty(
                raw_column,
                field_name="required_slice_values",
            )
            clean_values = [
                _normalized_nonempty(
                    value,
                    field_name=f"required_slice_values.{column}",
                )
                for value in raw_values
            ]
            if not clean_values:
                raise ValueError(
                    f"required_slice_values.{column} cannot be empty"
                )
            if len(clean_values) != len(set(clean_values)):
                raise ValueError(
                    f"required_slice_values.{column} must be unique"
                )
            normalized[column] = clean_values
        return normalized


class JudgeValidationThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    minimum_validation_cases: int | None = Field(default=None, ge=1)
    minimum_cases_per_required_slice: int | None = Field(default=None, ge=1)
    minimum_human_alpha: float | None = Field(default=None, ge=-1, le=1)
    minimum_judge_human_kappa: float | None = Field(
        default=None,
        ge=-1,
        le=1,
    )
    minimum_failure_precision: float | None = Field(default=None, ge=0, le=1)
    minimum_failure_recall: float | None = Field(default=None, ge=0, le=1)
    maximum_false_approval_rate: float | None = Field(default=None, ge=0, le=1)
    maximum_observation_failure_rate: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )
    minimum_stability_alpha: float | None = Field(default=None, ge=-1, le=1)
    maximum_prompt_flip_rate: float | None = Field(default=None, ge=0, le=1)
    maximum_identity_flip_rate: float | None = Field(default=None, ge=0, le=1)
    maximum_self_preference_shift: float | None = Field(default=None, ge=0, le=1)
    maximum_position_flip_rate: float | None = Field(default=None, ge=0, le=1)
    maximum_position_bias: float | None = Field(default=None, ge=0, le=0.5)


class JudgeProtocolGovernance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protocol_status: Literal["draft", "frozen"] = "draft"
    frozen_on: date | None = None
    heldout_not_used_for_development: bool = False
    independent_review: bool = False
    conflicts_documented: bool = False
    raw_artifacts_preserved: bool = True
    judge_selection_rationale: str | None = None
    publication_notes: str | None = None

    @field_validator("judge_selection_rationale", "publication_notes")
    @classmethod
    def validate_optional_notes(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _normalized_nonempty(value, field_name="governance notes")

    @model_validator(mode="after")
    def validate_frozen_date(self) -> "JudgeProtocolGovernance":
        if self.protocol_status == "frozen" and self.frozen_on is None:
            raise ValueError("frozen_on is required for a frozen protocol")
        return self


class JudgeProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    protocol_id: str
    version: str
    title: str
    target_level: ValidationLevel = "exploratory"
    synthetic_fixture: bool = False
    contract: JudgeEvaluationContract
    data: JudgeValidationData
    human_review: HumanReviewProtocol
    controls: JudgeValidationControls
    thresholds: JudgeValidationThresholds = Field(
        default_factory=JudgeValidationThresholds
    )
    governance: JudgeProtocolGovernance = Field(
        default_factory=JudgeProtocolGovernance
    )
    research_basis: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(min_length=1)

    @field_validator("protocol_id", "version", "title")
    @classmethod
    def validate_identity(cls, value: str, info: Any) -> str:
        return _normalized_nonempty(value, field_name=info.field_name)

    @field_validator("research_basis", "limitations")
    @classmethod
    def validate_text_lists(
        cls,
        values: list[str],
        info: Any,
    ) -> list[str]:
        return [
            _normalized_nonempty(value, field_name=info.field_name)
            for value in values
        ]

    @model_validator(mode="after")
    def validate_mode(self) -> "JudgeProtocol":
        if self.contract.mode == "pairwise" and self.data.pairwise is None:
            raise ValueError("pairwise mode requires data.pairwise labels")
        if self.contract.mode == "pointwise" and self.data.pairwise is not None:
            raise ValueError("pointwise mode cannot configure data.pairwise")
        if (
            self.controls.primary_judge_variant_id
            != self.contract.judge_variant_id
        ):
            raise ValueError(
                "controls.primary_judge_variant_id must match "
                "contract.judge_variant_id"
            )
        expected_order = (
            "pointwise" if self.contract.mode == "pointwise" else "AB"
        )
        if self.controls.baseline_presentation_order != expected_order:
            raise ValueError(
                "baseline_presentation_order must be "
                f"{expected_order!r} for {self.contract.mode} mode"
            )
        unknown_slices = set(self.controls.required_slice_values) - set(
            self.data.slice_columns
        )
        if unknown_slices:
            raise ValueError(
                "required_slice_values references undeclared slice columns: "
                f"{sorted(unknown_slices)}"
            )
        return self


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_hashed_files(
    *,
    paths: Any,
    hashes: Any,
    required_keys: set[str],
    description: str,
) -> dict[str, Path]:
    if not isinstance(paths, dict) or not isinstance(hashes, dict):
        raise ValueError(
            f"Judge validation report has invalid {description} metadata"
        )
    missing = sorted(
        key for key in required_keys if key not in paths or key not in hashes
    )
    if missing:
        raise ValueError(
            f"Judge validation report is missing {description}: {missing}"
        )

    resolved: dict[str, Path] = {}
    for key in required_keys:
        path = Path(str(paths[key]))
        if not path.is_file():
            raise ValueError(
                f"Judge validation {description} file is missing: {path}"
            )
        if _sha256_file(path) != str(hashes[key]):
            raise ValueError(
                f"Judge validation {description} hash does not match: {path}"
            )
        resolved[key] = path
    return resolved


def _read_gate_statuses(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Judge validation gate artifact is empty")

    gates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        gate_id = str(row.get("gate_id", "")).strip()
        level = str(row.get("required_for_level", "")).strip()
        passed_text = str(row.get("passed", "")).strip().lower()
        if not gate_id or gate_id in seen:
            raise ValueError(
                "Judge validation gate artifact has blank or duplicate gate_id"
            )
        if passed_text not in {"true", "false"}:
            raise ValueError(
                "Judge validation gate artifact has invalid passed status"
            )
        seen.add(gate_id)
        gates.append(
            {
                "gate_id": gate_id,
                "required_for_level": level,
                "passed": passed_text == "true",
            }
        )
    return gates


def load_judge_protocol(path: str | Path) -> JudgeProtocol:
    return JudgeProtocol.model_validate(load_yaml(path))


def protocol_contract_sha256(protocol: JudgeProtocol) -> str:
    return _canonical_sha256(protocol.contract.model_dump(mode="json"))


def build_eval_contract(config: EvalConfig) -> dict[str, Any]:
    judge = config.judge
    if judge.rubric_id is None or judge.rubric_version is None:
        raise ValueError(
            "rubric_id and rubric_version are required when matching a "
            "judge validation protocol"
        )
    return {
        "mode": "pointwise",
        "judge_variant_id": judge.variant_id,
        "judge_model": judge.model,
        "provider_model_version": judge.provider_model_version,
        "temperature": (
            float(judge.temperature)
            if judge.temperature is not None
            else None
        ),
        "prompt_id": judge.prompt_id,
        "prompt_version": judge.prompt_version,
        "rubric_id": judge.rubric_id,
        "rubric_version": judge.rubric_version,
        "candidate_identity_blinded": judge.candidate_identity_blinded,
        "reference_mode": judge.reference_mode,
        "task": {
            "name": config.task.name,
            "description": config.task.description,
            "measured_construct": (
                config.task.measured_construct or config.task.description
            ),
            "intended_use": (
                config.task.intended_use or config.task.description
            ),
            "languages": config.task.languages,
        },
        "scoring": config.scoring.model_dump(mode="json"),
        "criteria": [
            criterion.model_dump(mode="json") for criterion in config.criteria
        ],
    }


def eval_contract_sha256(config: EvalConfig) -> str:
    return _canonical_sha256(build_eval_contract(config))


@dataclass(frozen=True)
class ResolvedJudgeEvidence:
    level: str
    protocol_id: str | None
    protocol_version: str | None
    protocol_path: str | None
    protocol_sha256: str | None
    contract_sha256: str | None
    validation_report_path: str | None
    validation_report_sha256: str | None
    target_met: bool | None
    synthetic_fixture: bool
    note: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_judge_evidence(config: EvalConfig) -> ResolvedJudgeEvidence:
    protocol_path = config.judge.protocol_path
    report_path = config.judge.validation_report_path

    if protocol_path is None:
        return ResolvedJudgeEvidence(
            level="exploratory",
            protocol_id=None,
            protocol_version=None,
            protocol_path=None,
            protocol_sha256=None,
            contract_sha256=None,
            validation_report_path=None,
            validation_report_sha256=None,
            target_met=None,
            synthetic_fixture=False,
            note=(
                "No task-specific judge validation protocol and report were "
                "attached."
            ),
        )

    protocol_path = Path(protocol_path)
    protocol = load_judge_protocol(protocol_path)
    protocol_hash = _sha256_file(protocol_path)
    protocol_contract = protocol_contract_sha256(protocol)
    runtime_contract = eval_contract_sha256(config)
    if runtime_contract != protocol_contract:
        raise ValueError(
            "Judge validation protocol does not match the active task, "
            "rubric, prompt, judge model, or settings"
        )

    if report_path is None:
        return ResolvedJudgeEvidence(
            level="exploratory",
            protocol_id=protocol.protocol_id,
            protocol_version=protocol.version,
            protocol_path=str(protocol_path),
            protocol_sha256=protocol_hash,
            contract_sha256=protocol_contract,
            validation_report_path=None,
            validation_report_sha256=None,
            target_met=None,
            synthetic_fixture=protocol.synthetic_fixture,
            note="The protocol matches, but no validation report was attached.",
        )

    report_path = Path(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != "1.0":
        raise ValueError("Unsupported judge validation report schema")
    report_protocol = report.get("protocol", {})
    if report_protocol.get("protocol_id") != protocol.protocol_id:
        raise ValueError("Judge validation report protocol_id does not match")
    if report_protocol.get("version") != protocol.version:
        raise ValueError("Judge validation report version does not match")
    if report_protocol.get("protocol_sha256") != protocol_hash:
        raise ValueError("Judge validation report protocol hash does not match")
    if report_protocol.get("contract_sha256") != protocol_contract:
        raise ValueError("Judge validation report contract hash does not match")
    if report.get("target_level") != protocol.target_level:
        raise ValueError("Judge validation report target level does not match")

    if report_protocol.get("synthetic_fixture") is not protocol.synthetic_fixture:
        raise ValueError(
            "Judge validation report synthetic-fixture status does not match"
        )

    data = report.get("data", {})
    _verify_hashed_files(
        paths=data.get("paths"),
        hashes=data.get("hashes"),
        required_keys={
            "cases",
            "human_annotations",
            "judge_observations",
        },
        description="source data",
    )

    artifacts = report.get("artifacts", {})
    artifact_paths = {
        "gates": artifacts.get("gates_path"),
        "disagreements": artifacts.get("disagreements_path"),
        "subgroups": artifacts.get("subgroups_path"),
    }
    artifact_hashes = {
        "gates": artifacts.get("gates_sha256"),
        "disagreements": artifacts.get("disagreements_sha256"),
        "subgroups": artifacts.get("subgroups_sha256"),
    }
    verified_artifacts = _verify_hashed_files(
        paths=artifact_paths,
        hashes=artifact_hashes,
        required_keys={"gates", "disagreements", "subgroups"},
        description="artifact",
    )

    gate_statuses = _read_gate_statuses(verified_artifacts["gates"])
    report_gates = report.get("gates")
    if not isinstance(report_gates, list):
        raise ValueError("Judge validation report gates are invalid")
    report_gate_statuses = [
        {
            "gate_id": gate.get("gate_id"),
            "required_for_level": gate.get("required_for_level"),
            "passed": gate.get("passed"),
        }
        for gate in report_gates
        if isinstance(gate, dict)
    ]
    if report_gate_statuses != gate_statuses:
        raise ValueError(
            "Judge validation report gates do not match the gate artifact"
        )

    level = str(report.get("achieved_level", ""))
    validation_level_rank(level)
    if protocol.synthetic_fixture and level != "exploratory":
        raise ValueError(
            "Synthetic judge fixtures cannot supply calibrated evidence"
        )
    derived_level = achieved_level_from_gates(
        gate_statuses,
        synthetic_fixture=protocol.synthetic_fixture,
    )
    if level != derived_level:
        raise ValueError(
            "Judge validation report level does not match its protocol gates"
        )
    target_met = report.get("target_met")
    if not isinstance(target_met, bool):
        raise ValueError("Judge validation report target_met must be boolean")
    expected_target_met = level_at_least(level, protocol.target_level)
    if target_met != expected_target_met:
        raise ValueError(
            "Judge validation report target status is inconsistent"
        )

    return ResolvedJudgeEvidence(
        level=level,
        protocol_id=protocol.protocol_id,
        protocol_version=protocol.version,
        protocol_path=str(protocol_path),
        protocol_sha256=protocol_hash,
        contract_sha256=protocol_contract,
        validation_report_path=str(report_path),
        validation_report_sha256=_sha256_file(report_path),
        target_met=target_met,
        synthetic_fixture=protocol.synthetic_fixture,
        note=(
            "Task-specific validation evidence matched the active judge "
            "contract."
        ),
    )


def judge_claim_block_reason(
    results: Any,
    config: EvalConfig,
) -> str | None:
    if results is None or len(results) == 0:
        return None
    if "evaluation_source" not in results.columns:
        return None
    judged = results[results["evaluation_source"] == "llm_judge"]
    if judged.empty:
        return None

    level_column = "judge_evidence_level"
    levels = (
        judged[level_column].dropna().astype(str).tolist()
        if level_column in judged.columns
        else ["exploratory"]
    )
    observed = min(
        levels or ["exploratory"],
        key=validation_level_rank,
    )
    required = config.judge.minimum_validation_level_for_selection
    if level_at_least(observed, required):
        return None
    return (
        f"Judge evidence level {observed!r} is below the configured "
        f"minimum {required!r} for comparative selection."
    )

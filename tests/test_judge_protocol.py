import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from evalanche.config import (
    CriterionConfig,
    EvalConfig,
    JudgeConfig,
    RunConfig,
    ScoringConfig,
    TaskConfig,
)
from evalanche.judges.calibration import validate_judge_protocol
from evalanche.judges.protocol import (
    JudgeProtocol,
    eval_contract_sha256,
    judge_claim_block_reason,
    level_at_least,
    load_judge_protocol,
    protocol_contract_sha256,
    resolve_judge_evidence,
    validation_level_rank,
)


PROTOCOL_PATH = Path("examples/judge_validation/protocol.yaml")


def copied_protocol(
    tmp_path: Path,
    *,
    synthetic_fixture: bool = True,
) -> Path:
    raw = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    raw["synthetic_fixture"] = synthetic_fixture
    for key in (
        "cases_path",
        "human_annotations_path",
        "judge_observations_path",
    ):
        source = Path(raw["data"][key])
        target = tmp_path / source.name
        target.write_bytes(source.read_bytes())
        raw["data"][key] = str(target)
    protocol_path = tmp_path / "protocol.yaml"
    protocol_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return protocol_path


def fixture_eval_config(
    *,
    protocol_path: Path | None = None,
    report_path: Path | None = None,
) -> EvalConfig:
    return EvalConfig(
        run=RunConfig(
            name="fixture-judge",
            input_path=Path("input.csv"),
            output_path=Path("output.csv"),
        ),
        judge=JudgeConfig(
            model="fixture/local-judge",
            variant_id="fixture_primary",
            provider_model_version="fixture-1",
            prompt_id="evalanche.criteria_pointwise",
            prompt_version="1.0",
            rubric_id="generic_open_ended_quality",
            rubric_version="1.0",
            candidate_identity_blinded=True,
            reference_mode="required",
            protocol_path=protocol_path,
            validation_report_path=report_path,
        ),
        task=TaskConfig(
            name="generic_open_ended_response",
            description="Assess whether a response is correct and complete.",
            measured_construct="Task-specific correctness and completeness.",
            intended_use="Demonstrate the offline judge-validation workflow.",
            languages=["en", "fr"],
        ),
        scoring=ScoringConfig(
            score_min=0,
            score_max=5,
            pass_threshold=0.75,
        ),
        criteria=[
            CriterionConfig(
                name="correctness",
                weight=0.75,
                description=(
                    "The response is supported by the reference and "
                    "contains no material error."
                ),
            ),
            CriterionConfig(
                name="completeness",
                weight=0.25,
                description=(
                    "The response includes every required point without "
                    "irrelevant padding."
                ),
            ),
        ],
    )


def test_validation_levels_have_a_strict_order() -> None:
    assert validation_level_rank("exploratory") == 0
    assert level_at_least("decision_grade", "calibrated") is True
    assert level_at_least("exploratory", "calibrated") is False
    with pytest.raises(ValueError, match="Unknown judge validation level"):
        validation_level_rank("trusted")


def test_protocol_contract_matches_the_exact_runtime_config() -> None:
    protocol = load_judge_protocol(PROTOCOL_PATH)
    config = fixture_eval_config()

    assert eval_contract_sha256(config) == protocol_contract_sha256(protocol)


def test_resolve_judge_evidence_requires_matching_protocol_and_report(
    tmp_path: Path,
) -> None:
    artifacts = validate_judge_protocol(PROTOCOL_PATH, output_dir=tmp_path)
    config = fixture_eval_config(
        protocol_path=PROTOCOL_PATH,
        report_path=artifacts.report_json_path,
    )

    evidence = resolve_judge_evidence(config)

    assert evidence.level == "exploratory"
    assert evidence.protocol_id == "generic_open_ended_quality_fixture"
    assert evidence.target_met is True
    assert len(str(evidence.validation_report_sha256)) == 64


def test_resolve_judge_evidence_rejects_runtime_contract_drift() -> None:
    config = fixture_eval_config(protocol_path=PROTOCOL_PATH)
    config.task.description = "Changed after validation."

    with pytest.raises(ValueError, match="does not match the active"):
        resolve_judge_evidence(config)


def test_synthetic_report_cannot_claim_calibrated_evidence(
    tmp_path: Path,
) -> None:
    artifacts = validate_judge_protocol(PROTOCOL_PATH, output_dir=tmp_path)
    report = json.loads(
        artifacts.report_json_path.read_text(encoding="utf-8")
    )
    report["achieved_level"] = "calibrated"
    modified = tmp_path / "modified.json"
    modified.write_text(json.dumps(report), encoding="utf-8")
    config = fixture_eval_config(
        protocol_path=PROTOCOL_PATH,
        report_path=modified,
    )

    with pytest.raises(ValueError, match="Synthetic judge fixtures"):
        resolve_judge_evidence(config)


def test_report_level_is_derived_from_preserved_gate_results(
    tmp_path: Path,
) -> None:
    protocol_path = copied_protocol(tmp_path, synthetic_fixture=False)
    artifacts = validate_judge_protocol(
        protocol_path,
        output_dir=tmp_path / "output",
    )
    assert artifacts.achieved_level == "decision_grade"
    report = json.loads(
        artifacts.report_json_path.read_text(encoding="utf-8")
    )
    assert report["achieved_level"] != "publishable"
    report["achieved_level"] = "publishable"
    modified = tmp_path / "raised-level.json"
    modified.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match its protocol gates"):
        resolve_judge_evidence(
            fixture_eval_config(
                protocol_path=protocol_path,
                report_path=modified,
            )
        )


def test_report_is_rejected_after_validation_source_data_changes(
    tmp_path: Path,
) -> None:
    protocol_path = copied_protocol(tmp_path)
    artifacts = validate_judge_protocol(
        protocol_path,
        output_dir=tmp_path / "output",
    )
    cases_path = tmp_path / "cases.csv"
    cases_path.write_text(
        cases_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="source data hash does not match"):
        resolve_judge_evidence(
            fixture_eval_config(
                protocol_path=protocol_path,
                report_path=artifacts.report_json_path,
            )
        )


def test_judge_claims_are_blocked_below_configured_level() -> None:
    config = fixture_eval_config()
    results = pd.DataFrame(
        [
            {
                "evaluation_source": "llm_judge",
                "judge_evidence_level": "exploratory",
            }
        ]
    )

    assert "below the configured minimum" in str(
        judge_claim_block_reason(results, config)
    )
    config.judge.minimum_validation_level_for_selection = "exploratory"
    assert judge_claim_block_reason(results, config) is None


def test_validation_report_requires_protocol_path() -> None:
    with pytest.raises(ValidationError, match="protocol_path is required"):
        JudgeConfig(
            model="fixture/judge",
            validation_report_path=Path("report.json"),
        )


def test_pairwise_protocol_requires_pairwise_labels() -> None:
    raw = load_judge_protocol(PROTOCOL_PATH).model_dump(mode="json")
    raw["contract"]["mode"] = "pairwise"
    raw["controls"]["baseline_presentation_order"] = "AB"

    with pytest.raises(ValidationError, match="pairwise mode requires"):
        JudgeProtocol.model_validate(raw)


def test_binary_failure_diagnostics_require_exactly_two_labels() -> None:
    raw = load_judge_protocol(PROTOCOL_PATH).model_dump(mode="json")
    raw["data"]["labels"].append("uncertain")

    with pytest.raises(ValidationError, match="exactly two labels"):
        JudgeProtocol.model_validate(raw)

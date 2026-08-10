import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

import evalanche.cli as cli
from evalanche.judges.calibration import (
    JudgeValidationArtifacts,
    validate_judge_protocol,
)


PROTOCOL_PATH = Path("examples/judge_validation/protocol.yaml")


def test_synthetic_worked_example_builds_complete_offline_artifacts(
    tmp_path: Path,
) -> None:
    artifacts = validate_judge_protocol(PROTOCOL_PATH, output_dir=tmp_path)
    report = json.loads(
        artifacts.report_json_path.read_text(encoding="utf-8")
    )

    assert artifacts.achieved_level == "exploratory"
    assert artifacts.target_met is True
    assert report["judge_human_agreement"]["validation"][
        "cohen_kappa"
    ] == pytest.approx(0.5)
    assert report["controls"]["stability"]["observed_value_count"] == 3
    assert report["controls"]["prompt"]["flip_rate"] == 0.0
    assert report["controls"]["prompt"][
        "complete_condition_coverage"
    ] == 1.0
    assert len(report["judge_variants"]) == 2
    assert len(report["subgroups"]) == 6
    assert report["disagreement_cases"] == 1
    assert artifacts.gates_path.is_file()
    assert artifacts.disagreements_path.is_file()
    assert artifacts.subgroups_path.is_file()
    markdown = artifacts.report_markdown_path.read_text(encoding="utf-8")
    assert "Synthetic fixtures can demonstrate the workflow only" in markdown
    assert "Failure is the positive class" in markdown


def test_declared_target_failure_is_saved_and_reported(
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    raw["target_level"] = "calibrated"
    protocol_path = tmp_path / "protocol.yaml"
    protocol_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    artifacts = validate_judge_protocol(
        protocol_path,
        output_dir=tmp_path / "output",
    )

    assert artifacts.achieved_level == "exploratory"
    assert artifacts.target_met is False
    assert artifacts.report_json_path.is_file()


def test_invalid_successful_observation_is_rejected(tmp_path: Path) -> None:
    observations = pd.read_csv(
        "examples/judge_validation/judge_observations.csv",
        dtype=str,
        keep_default_na=False,
    )
    observations.loc[0, "label"] = ""
    observations_path = tmp_path / "observations.csv"
    observations.to_csv(observations_path, index=False)
    raw = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    raw["data"]["judge_observations_path"] = str(observations_path)
    protocol_path = tmp_path / "protocol.yaml"
    protocol_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="require a label"):
        validate_judge_protocol(
            protocol_path,
            output_dir=tmp_path / "output",
        )


def test_decision_grade_requires_control_coverage_on_every_validation_case(
    tmp_path: Path,
) -> None:
    observations = pd.read_csv(
        "examples/judge_validation/judge_observations.csv",
        dtype=str,
        keep_default_na=False,
    )
    incomplete = observations[
        ~(
            (observations["case_id"] == "fixture_008")
            & (observations["prompt_variant_id"] == "paraphrase")
        )
    ]
    observations_path = tmp_path / "observations.csv"
    incomplete.to_csv(observations_path, index=False)
    raw = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    raw["synthetic_fixture"] = False
    raw["target_level"] = "decision_grade"
    raw["data"]["judge_observations_path"] = str(observations_path)
    protocol_path = tmp_path / "protocol.yaml"
    protocol_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    artifacts = validate_judge_protocol(
        protocol_path,
        output_dir=tmp_path / "output",
    )
    gates = pd.read_csv(artifacts.gates_path).set_index("gate_id")

    assert artifacts.achieved_level == "calibrated"
    assert artifacts.target_met is False
    assert float(gates.loc["prompt_variant_coverage", "observed"]) == 0.75
    assert bool(gates.loc["prompt_variant_coverage", "passed"]) is False


def _write_pairwise_fixture(tmp_path: Path) -> Path:
    cases = tmp_path / "cases.csv"
    cases.write_text(
        "case_id,split,language,input,response_a,response_b\n"
        "pair_1,calibration,en,Choose the better response.,A1,B1\n"
        "pair_2,validation,fr,Choisissez la meilleure reponse.,A2,B2\n",
        encoding="utf-8",
    )
    humans = tmp_path / "humans.csv"
    humans.write_text(
        "case_id,criterion,reviewer_id,annotation_stage,label\n"
        "pair_1,overall,h1,independent,a\n"
        "pair_1,overall,h2,independent,a\n"
        "pair_1,overall,adjudicator,adjudicated,a\n"
        "pair_2,overall,h1,independent,b\n"
        "pair_2,overall,h2,independent,b\n"
        "pair_2,overall,adjudicator,adjudicated,b\n",
        encoding="utf-8",
    )
    observations = tmp_path / "observations.csv"
    observations.write_text(
        "case_id,criterion,judge_variant_id,trial_id,prompt_variant_id,"
        "identity_condition,presentation_order,label,status,uncertainty\n"
        "pair_1,overall,pair_judge,trial_1,baseline,blinded,AB,a,success,0.1\n"
        "pair_1,overall,pair_judge,trial_1,baseline,blinded,BA,a,success,0.1\n"
        "pair_2,overall,pair_judge,trial_1,baseline,blinded,AB,b,success,0.1\n"
        "pair_2,overall,pair_judge,trial_1,baseline,blinded,BA,b,success,0.1\n",
        encoding="utf-8",
    )
    protocol = {
        "schema_version": "1.0",
        "protocol_id": "pairwise_fixture",
        "version": "0.1.0",
        "title": "Pairwise fixture",
        "target_level": "exploratory",
        "synthetic_fixture": True,
        "contract": {
            "mode": "pairwise",
            "judge_variant_id": "pair_judge",
            "judge_model": "fixture/pair-judge",
            "provider_model_version": "fixture-1",
            "temperature": 0,
            "prompt_id": "fixture.pairwise",
            "prompt_version": "1.0",
            "rubric_id": "pairwise_quality",
            "rubric_version": "1.0",
            "candidate_identity_blinded": True,
            "reference_mode": "required",
            "task": {
                "name": "pairwise_quality",
                "description": "Choose the better response.",
                "measured_construct": "Relative response quality.",
                "intended_use": "Exercise order controls.",
                "languages": ["en", "fr"],
            },
            "scoring": {
                "score_min": 0,
                "score_max": 5,
                "pass_threshold": 0.75,
            },
            "criteria": [
                {
                    "name": "quality",
                    "weight": 1.0,
                    "description": "Task-specific response quality.",
                }
            ],
        },
        "data": {
            "cases_path": str(cases),
            "human_annotations_path": str(humans),
            "judge_observations_path": str(observations),
            "labels": ["a", "b", "tie"],
            "criteria": ["overall"],
            "primary_criterion": "overall",
            "split_column": "split",
            "calibration_split": "calibration",
            "validation_split": "validation",
            "slice_columns": ["language"],
            "pairwise": {
                "candidate_a_label": "a",
                "candidate_b_label": "b",
                "tie_label": "tie",
            },
        },
        "human_review": {
            "reviewer_population": "Synthetic fixtures.",
            "instructions_version": "1.0",
            "minimum_independent_raters": 2,
            "independent_before_adjudication": True,
            "candidate_identity_blinded": True,
            "judge_identity_blinded": True,
            "adjudication_rule": "Synthetic agreement.",
        },
        "controls": {
            "primary_judge_variant_id": "pair_judge",
            "baseline_trial_id": "trial_1",
            "baseline_prompt_variant_id": "baseline",
            "baseline_identity_condition": "blinded",
            "baseline_presentation_order": "AB",
            "minimum_independent_trials": 1,
            "minimum_prompt_variants": 1,
            "require_identity_bias_test": False,
            "require_position_swap": True,
            "response_cache_disabled": True,
            "required_slice_values": {"language": ["fr"]},
            "bootstrap_samples": 100,
            "bootstrap_seed": 1,
        },
        "thresholds": {},
        "governance": {
            "protocol_status": "draft",
            "raw_artifacts_preserved": True,
        },
        "research_basis": ["https://arxiv.org/abs/2406.07791"],
        "limitations": ["Synthetic pairwise fixture."],
    }
    protocol_path = tmp_path / "protocol.yaml"
    protocol_path.write_text(yaml.safe_dump(protocol), encoding="utf-8")
    return protocol_path


def test_pairwise_validation_normalizes_ab_and_ba_results(
    tmp_path: Path,
) -> None:
    protocol_path = _write_pairwise_fixture(tmp_path)
    artifacts = validate_judge_protocol(
        protocol_path,
        output_dir=tmp_path / "output",
    )
    report = json.loads(
        artifacts.report_json_path.read_text(encoding="utf-8")
    )
    position = report["controls"]["position"]

    assert position["applicable"] is True
    assert position["observed_value_count"] == 2
    assert position["flip_rate"] == 0.0
    assert position["first_position_selection_rate"] == 0.5
    assert position["position_bias"] == 0.0


def test_validate_judge_cli_exits_two_when_target_is_not_met(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifacts = JudgeValidationArtifacts(
        report_json_path=tmp_path / "report.json",
        report_markdown_path=tmp_path / "report.md",
        gates_path=tmp_path / "gates.csv",
        disagreements_path=tmp_path / "disagreements.csv",
        subgroups_path=tmp_path / "subgroups.csv",
        achieved_level="exploratory",
        target_level="calibrated",
        target_met=False,
    )
    monkeypatch.setattr(
        cli,
        "validate_judge_protocol",
        lambda *args, **kwargs: artifacts,
    )

    with pytest.raises(SystemExit) as error:
        cli.run_validate_judge("protocol.yaml")

    assert error.value.code == 2


def test_validate_judge_command_is_exposed_by_parser() -> None:
    args = cli.build_parser().parse_args(
        [
            "validate-judge",
            "--protocol",
            "examples/judge_validation/protocol.yaml",
            "--output-dir",
            "results/check",
        ]
    )

    assert args.command == "validate-judge"
    assert args.protocol.endswith("protocol.yaml")
    assert args.output_dir == "results/check"

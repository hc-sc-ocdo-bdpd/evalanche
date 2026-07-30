from pathlib import Path

import pandas as pd
import pytest

import evalanche.dpd_census_comparison as census_comparison
from evalanche.cli import build_parser
from evalanche.config import (
    load_evaluation_config,
    load_generation_config,
)


def _cases() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "case_id": "case_001",
                "input": "Extract one.",
                "expected_output": '{"value": 1}',
                "evaluation_type": "json",
            },
            {
                "case_id": "case_002",
                "input": "Extract two.",
                "expected_output": '{"value": 2}',
                "evaluation_type": "json",
            },
        ]
    )


def _output(cases: pd.DataFrame, model_name: str) -> pd.DataFrame:
    frame = cases.copy()
    frame["model_name"] = model_name
    frame["candidate_model"] = (
        "azure/" + model_name.replace("_", "-")
    )
    frame["model_output"] = frame["expected_output"]
    frame["generation_status"] = "success"
    frame["generation_error"] = ""
    return frame


def _configure_tiny_census(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        census_comparison,
        "DPD_CENSUS_CASES_PATH",
        Path("data/cases.csv"),
    )
    monkeypatch.setattr(
        census_comparison,
        "DPD_CENSUS_CASE_COUNT",
        2,
    )
    monkeypatch.setattr(
        census_comparison,
        "DPD_CENSUS_COMBINED_OUTPUT_PATH",
        Path("data/combined.csv"),
    )
    monkeypatch.setattr(
        census_comparison,
        "DPD_CENSUS_ASSEMBLY_METADATA_PATH",
        Path("data/combined_metadata.json"),
    )
    monkeypatch.setattr(
        census_comparison,
        "MODEL_OUTPUT_SOURCES",
        (
            ("gpt_5_4_mini", Path("data/mini.csv")),
            ("gpt_5_6_luna", Path("data/luna.csv")),
            ("gpt_5_6_terra", Path("data/terra.csv")),
            ("gpt_5_6_sol", Path("data/sol.csv")),
        ),
    )


def test_cli_exposes_full_census_assembly_command() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "assemble-dpd-census-comparison",
            "--require-model",
            "gpt_5_6_luna",
        ]
    )

    assert args.command == "assemble-dpd-census-comparison"
    assert args.required_models == ["gpt_5_6_luna"]


def test_full_census_configs_are_separate_resumable_and_guarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "DPD_FULL_CENSUS_RPM",
        "DPD_FULL_CENSUS_WORKERS",
        "DPD_FULL_CENSUS_LUNA_MAX_USD",
        "DPD_FULL_CENSUS_TERRA_MAX_USD",
        "DPD_FULL_CENSUS_SOL_MAX_USD",
    ):
        monkeypatch.delenv(name, raising=False)

    expected_limits = {
        "luna": 40.0,
        "terra": 90.0,
        "sol": 170.0,
    }
    for tier, limit in expected_limits.items():
        generation = load_generation_config(
            "configs/"
            f"generate_hc_dpd_census_gpt_5_6_{tier}.yaml"
        )
        evaluation = load_evaluation_config(
            "configs/"
            f"evaluate_hc_dpd_census_gpt_5_6_{tier}.yaml"
        )

        assert generation.run.input_path == Path(
            "data/hc/benchmarks/"
            "dpd_structured_extraction_census/0.2.0/cases.csv.gz"
        )
        assert generation.run.output_path == Path(
            "data/generated/"
            f"hc_dpd_census_gpt_5_6_{tier}_outputs.csv"
        )
        assert generation.generation.resume is True
        assert generation.generation.checkpoint_every == 100
        assert generation.generation.max_workers == 4
        assert (
            generation.generation.max_requests_per_minute
            == 60
        )
        assert (
            generation.generation.cost_preflight_sample_path
            == Path(
                "data/generated/"
                f"hc_dpd_comparison_500_gpt_5_6_{tier}_outputs.csv"
            )
        )
        assert (
            generation.generation.maximum_estimated_cost_usd
            == limit
        )
        assert (
            generation.generation.cost_safety_multiplier
            == 1.5
        )
        assert evaluation.run.input_path == generation.run.output_path


def test_full_census_controls_accept_environment_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DPD_FULL_CENSUS_RPM", "75")
    monkeypatch.setenv("DPD_FULL_CENSUS_WORKERS", "6")
    monkeypatch.setenv(
        "DPD_FULL_CENSUS_LUNA_MAX_USD",
        "55.50",
    )

    generation = load_generation_config(
        "configs/generate_hc_dpd_census_gpt_5_6_luna.yaml"
    )

    assert generation.generation.max_requests_per_minute == 75
    assert generation.generation.max_workers == 6
    assert (
        generation.generation.maximum_estimated_cost_usd
        == 55.5
    )


def test_full_census_assembly_combines_complete_models(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_tiny_census(monkeypatch)
    data = tmp_path / "data"
    data.mkdir()
    cases = _cases()
    cases.to_csv(data / "cases.csv", index=False)
    _output(cases, "gpt_5_4_mini").to_csv(
        data / "mini.csv",
        index=False,
    )
    _output(cases, "gpt_5_6_luna").to_csv(
        data / "luna.csv",
        index=False,
    )

    result = (
        census_comparison
        .assemble_dpd_census_comparison_outputs(
            root_path=tmp_path,
            required_models=[
                "gpt_5_4_mini",
                "gpt_5_6_luna",
            ],
        )
    )
    combined = pd.read_csv(tmp_path / result["output_path"])

    assert result["available_models"] == [
        "gpt_5_4_mini",
        "gpt_5_6_luna",
    ]
    assert result["case_count_per_model"] == 2
    assert result["combined_row_count"] == 4
    assert combined["model_name"].tolist() == [
        "gpt_5_4_mini",
        "gpt_5_4_mini",
        "gpt_5_6_luna",
        "gpt_5_6_luna",
    ]


def test_full_census_assembly_skips_incomplete_optional_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_tiny_census(monkeypatch)
    data = tmp_path / "data"
    data.mkdir()
    cases = _cases()
    cases.to_csv(data / "cases.csv", index=False)
    _output(cases, "gpt_5_4_mini").to_csv(
        data / "mini.csv",
        index=False,
    )
    _output(
        cases.iloc[[0]],
        "gpt_5_6_luna",
    ).to_csv(data / "luna.csv", index=False)

    result = (
        census_comparison
        .assemble_dpd_census_comparison_outputs(
            root_path=tmp_path,
            required_models=["gpt_5_4_mini"],
        )
    )

    assert result["available_models"] == ["gpt_5_4_mini"]
    assert result["skipped_incomplete_models"][0][
        "model_name"
    ] == "gpt_5_6_luna"


def test_full_census_assembly_rejects_incomplete_required_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_tiny_census(monkeypatch)
    data = tmp_path / "data"
    data.mkdir()
    cases = _cases()
    cases.to_csv(data / "cases.csv", index=False)
    _output(
        cases.iloc[[0]],
        "gpt_5_6_luna",
    ).to_csv(data / "luna.csv", index=False)

    with pytest.raises(
        ValueError,
        match="does not contain the complete frozen census",
    ):
        (
            census_comparison
            .assemble_dpd_census_comparison_outputs(
                root_path=tmp_path,
                required_models=["gpt_5_6_luna"],
            )
        )


def test_combined_full_census_evaluation_has_no_composite_winner() -> None:
    evaluation = load_evaluation_config(
        "configs/evaluate_hc_dpd_census_all_models.yaml"
    )

    assert evaluation.run.input_path == Path(
        "data/generated/hc_dpd_census_all_models_outputs.csv"
    )
    assert evaluation.run.output_path == Path(
        "results/evaluate_hc_dpd_census_all_models_results.csv"
    )
    assert evaluation.selection.enabled is False

import json
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd

from evalanche.cli import build_parser
from evalanche.config import (
    load_candidate_models,
    load_evaluation_config,
    load_generation_config,
)
from evalanche.dpd_comparison import (
    DPD_COMPARISON_CASES_PATH,
    DPD_COMPARISON_CASE_COUNT,
    DPD_COMPARISON_PRODUCT_COUNT,
    DPD_COMPARISON_REPORT_PATH,
    DPD_CENSUS_CASES_PATH,
    assemble_dpd_comparison_outputs,
    build_dpd_comparison_sample,
)
from evalanche.pricing import load_endpoint_pricing_catalog


EXPECTED_PRODUCTS_BY_STRATUM = {
    "single_ingredient": 111,
    "multi_ingredient": 55,
    "multi_variant": 78,
    "multi_ingredient_multi_variant": 6,
}


def test_cli_exposes_dpd_comparison_commands() -> None:
    parser = build_parser()

    build_args = parser.parse_args(["build-dpd-comparison"])
    assemble_args = parser.parse_args(
        [
            "assemble-dpd-comparison",
            "--require-model",
            "gpt_5_4_mini",
        ]
    )

    assert build_args.products == DPD_COMPARISON_PRODUCT_COUNT
    assert build_args.seed == 20260728
    assert assemble_args.required_models == ["gpt_5_4_mini"]


def test_materialized_comparison_sample_is_bilingual_and_proportional() -> None:
    cases = pd.read_csv(DPD_COMPARISON_CASES_PATH)
    report = json.loads(
        DPD_COMPARISON_REPORT_PATH.read_text(encoding="utf-8")
    )

    assert len(cases) == DPD_COMPARISON_CASE_COUNT
    assert cases["case_id"].is_unique
    assert cases["product_id"].nunique() == DPD_COMPARISON_PRODUCT_COUNT
    assert Counter(cases["language"]) == {"en": 250, "fr": 250}
    assert Counter(
        cases.drop_duplicates("product_id")["stratum"]
    ) == Counter(EXPECTED_PRODUCTS_BY_STRATUM)
    assert all(
        set(group["language"]) == {"en", "fr"} and len(group) == 2
        for _, group in cases.groupby("product_id")
    )
    assert report["coverage"]["all_required_route_groups_present"] is True
    assert all(report["quality_checks"].values())


def test_comparison_sample_rebuild_is_byte_reproducible(
    tmp_path: Path,
) -> None:
    census_target = tmp_path / DPD_CENSUS_CASES_PATH
    census_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DPD_CENSUS_CASES_PATH, census_target)

    result = build_dpd_comparison_sample(root_path=tmp_path)

    rebuilt = tmp_path / DPD_COMPARISON_CASES_PATH
    assert result["case_count"] == DPD_COMPARISON_CASE_COUNT
    assert rebuilt.read_bytes() == DPD_COMPARISON_CASES_PATH.read_bytes()


def _mock_output(
    cases: pd.DataFrame,
    *,
    model_name: str,
    candidate_model: str,
) -> pd.DataFrame:
    output = cases.copy()
    output["model_name"] = model_name
    output["candidate_model"] = candidate_model
    output["model_output"] = output["expected_output"]
    output["generation_status"] = "success"
    output["generation_error"] = ""
    output["generation_seconds"] = 1.0
    output["generation_prompt_tokens"] = 100
    output["generation_completion_tokens"] = 50
    output["generation_total_tokens"] = 150
    output["generation_cost_usd"] = 0.001
    return output


def test_assembly_reuses_mini_and_combines_available_models(
    tmp_path: Path,
) -> None:
    sample_target = tmp_path / DPD_COMPARISON_CASES_PATH
    sample_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DPD_COMPARISON_CASES_PATH, sample_target)
    sample = pd.read_csv(sample_target)

    full_mini = _mock_output(
        sample,
        model_name="gpt_5_4_mini",
        candidate_model="azure/gpt-5.4-mini",
    )
    mini_path = (
        tmp_path
        / "data/generated/hc_dpd_census_gpt_5_4_mini_outputs.csv"
    )
    mini_path.parent.mkdir(parents=True, exist_ok=True)
    full_mini.to_csv(mini_path, index=False)

    terra = _mock_output(
        sample,
        model_name="gpt_5_6_terra",
        candidate_model="azure/gpt-5.6-terra",
    )
    terra_path = (
        tmp_path
        / "data/generated/"
        "hc_dpd_comparison_500_gpt_5_6_terra_outputs.csv"
    )
    terra.to_csv(terra_path, index=False)

    result = assemble_dpd_comparison_outputs(
        root_path=tmp_path,
        required_models=["gpt_5_4_mini", "gpt_5_6_terra"],
    )
    combined = pd.read_csv(tmp_path / result["output_path"])

    assert result["available_models"] == [
        "gpt_5_4_mini",
        "gpt_5_6_terra",
    ]
    assert len(combined) == 1000
    assert Counter(combined["model_name"]) == {
        "gpt_5_4_mini": 500,
        "gpt_5_6_terra": 500,
    }


def test_all_gpt_5_6_comparison_configs_are_pinned_and_guarded() -> None:
    expected = {
        "luna": {
            "input": 1.0,
            "output": 6.0,
            "limit": 5.0,
        },
        "terra": {
            "input": 2.5,
            "output": 15.0,
            "limit": 10.0,
        },
        "sol": {
            "input": 5.0,
            "output": 30.0,
            "limit": 20.0,
        },
    }

    for tier, values in expected.items():
        pilot = load_generation_config(
            f"configs/generate_hc_dpd_census_demo_gpt_5_6_{tier}.yaml"
        )
        comparison = load_generation_config(
            "configs/"
            f"generate_hc_dpd_comparison_500_gpt_5_6_{tier}.yaml"
        )
        candidate = load_candidate_models(
            comparison.candidate_models_path
        ).models[0]
        price = load_endpoint_pricing_catalog(
            comparison.endpoint_pricing_path
        ).endpoints[0]

        assert candidate.name == f"gpt_5_6_{tier}"
        assert candidate.model == f"azure/gpt-5.6-{tier}"
        assert candidate.provider_model_version == "2026-07-09"
        assert candidate.deployment_type == "Global Standard"
        assert candidate.resource_region == "Canada East"
        assert candidate.temperature is None
        assert candidate.reasoning_effort == "none"
        assert pilot.generation.resume is False
        assert comparison.run.input_path == DPD_COMPARISON_CASES_PATH
        assert comparison.generation.max_workers == 4
        assert comparison.generation.max_requests_per_minute == 60
        assert comparison.generation.resume is True
        assert comparison.generation.checkpoint_every == 50
        assert (
            comparison.generation.cost_preflight_sample_path
            == pilot.run.output_path
        )
        assert (
            comparison.generation.maximum_estimated_cost_usd
            == values["limit"]
        )
        assert comparison.generation.cost_safety_multiplier == 1.5
        assert price.input_per_million_tokens == values["input"]
        assert price.output_per_million_tokens == values["output"]
        assert price.cached_input_per_million_tokens is None


def test_comparison_evaluation_keeps_pairing_and_no_composite_winner() -> None:
    evaluation = load_evaluation_config(
        "configs/evaluate_hc_dpd_comparison_500.yaml"
    )

    assert evaluation.run.input_path == Path(
        "data/generated/hc_dpd_comparison_500_outputs.csv"
    )
    assert evaluation.run.output_path == Path(
        "results/evaluate_hc_dpd_comparison_500_results.csv"
    )
    assert evaluation.selection.enabled is False
    assert evaluation.metrics.json_comparison.numeric_value_paths == [
        "/active_ingredients/*/strength"
    ]

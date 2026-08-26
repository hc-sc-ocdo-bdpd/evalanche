import json
from pathlib import Path

import pandas as pd
import pytest

from evalanche.cli import build_parser
from evalanche.config import load_evaluation_config
from evalanche.dpd_analysis import (
    DPD_FIELDS,
    _validate_results,
    build_dpd_census_analysis,
    classify_dpd_json_error,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT / "configs/evaluate_hc_dpd_census_all_models.yaml"
)
MODELS = (
    "gpt_5_4_mini",
    "gpt_5_6_luna",
    "gpt_5_6_terra",
    "gpt_5_6_sol",
)
STRATA = (
    "single_ingredient",
    "multi_ingredient",
    "multi_variant",
    "multi_ingredient_multi_variant",
)


def _expected_output(
    *,
    brand_name: str = "TEST DRUG",
    dosage_forms: list[str] | None = None,
) -> dict:
    return {
        "din": ["01234567"],
        "brand_name": brand_name,
        "active_ingredients": [
            {
                "name": "INGREDIENT",
                "strength": 5,
                "unit": "MG",
            }
        ],
        "dosage_forms": dosage_forms or ["TABLET"],
        "routes": ["ORAL"],
        "schedule": ["PRESCRIPTION"],
        "product_status": "MARKETED",
        "company": "TEST COMPANY",
    }


def _fixture_results() -> pd.DataFrame:
    records = []
    for product_index, stratum in enumerate(STRATA):
        product_id = f"product_{product_index}"
        for language in ("en", "fr"):
            case_id = f"{product_id}_{language}"
            dosage_forms = (
                ["SPRAY, BAG-ON-VALVE"]
                if product_index == 1
                else ["TABLET"]
            )
            expected = _expected_output(
                brand_name=f"TEST DRUG {product_index}",
                dosage_forms=dosage_forms,
            )
            for model_index, model_name in enumerate(MODELS):
                output = json.loads(json.dumps(expected))
                passed = True
                if model_name == "gpt_5_4_mini" and case_id == "product_0_en":
                    output["brand_name"] = [output["brand_name"]]
                    passed = False
                if model_name == "gpt_5_6_luna" and case_id == "product_0_en":
                    output["product_status"] = [
                        output["product_status"]
                    ]
                    passed = False
                if model_name == "gpt_5_6_terra" and case_id == "product_1_en":
                    output["dosage_forms"] = [
                        "SPRAY",
                        "BAG-ON-VALVE",
                    ]
                    passed = False
                if model_name == "gpt_5_6_sol" and case_id == "product_2_fr":
                    output["company"] = "TEST COMPANI"
                    passed = False

                records.append(
                    {
                        "case_id": case_id,
                        "product_id": product_id,
                        "brand_name": expected["brand_name"],
                        "language": language,
                        "stratum": stratum,
                        "input": f"Rendered source for {case_id}",
                        "expected_output": json.dumps(
                            expected,
                            ensure_ascii=False,
                        ),
                        "source_metadata": json.dumps(
                            {"drug_codes": [product_index]}
                        ),
                        "model_name": model_name,
                        "model_output": json.dumps(
                            output,
                            ensure_ascii=False,
                        ),
                        "generation_status": "success",
                        "generation_seconds": 1.0 + model_index,
                        "generation_cost_usd": (
                            0.01 * (model_index + 1)
                        ),
                        "final_score": 1.0 if passed else 0.875,
                        "final_passed": passed,
                    }
                )
    return pd.DataFrame(records)


def test_classify_dpd_json_error_diagnostic_categories() -> None:
    config = load_evaluation_config(CONFIG_PATH)
    expected = _expected_output(
        dosage_forms=["SPRAY, BAG-ON-VALVE"]
    )

    numeric_string = json.loads(json.dumps(expected))
    numeric_string["active_ingredients"][0]["strength"] = "5.0"
    assert classify_dpd_json_error(
        expected_output=expected,
        model_output=numeric_string,
        config=config,
    ) == "pass"

    schema_error = json.loads(json.dumps(expected))
    schema_error["company"] = ["TEST COMPANY"]
    assert classify_dpd_json_error(
        expected_output=expected,
        model_output=schema_error,
        config=config,
    ) == "schema_type_only"

    duplicate_error = json.loads(json.dumps(expected))
    duplicate_error["active_ingredients"].append(
        duplicate_error["active_ingredients"][0].copy()
    )
    assert classify_dpd_json_error(
        expected_output=expected,
        model_output=duplicate_error,
        config=config,
    ) == "duplicate_items_only"

    compound_error = json.loads(json.dumps(expected))
    compound_error["dosage_forms"] = [
        "SPRAY",
        "BAG-ON-VALVE",
    ]
    assert classify_dpd_json_error(
        expected_output=expected,
        model_output=compound_error,
        config=config,
    ) == "compound_label_split_only"

    text_error = json.loads(json.dumps(expected))
    text_error["company"] = "TEST COMPANI"
    assert classify_dpd_json_error(
        expected_output=expected,
        model_output=text_error,
        config=config,
    ) == "text_or_value_error"

    missing_field = json.loads(json.dumps(expected))
    del missing_field["company"]
    assert classify_dpd_json_error(
        expected_output=expected,
        model_output=missing_field,
        config=config,
    ) == "missing_or_extra_fields"
    assert classify_dpd_json_error(
        expected_output=expected,
        model_output="not json",
        config=config,
    ) == "invalid_json"
    assert classify_dpd_json_error(
        expected_output=expected,
        model_output=expected,
        config=config,
        generation_status="error",
    ) == "generation_error"


def test_build_dpd_census_analysis_outputs_selected_audit_cases(
    tmp_path: Path,
) -> None:
    results = _fixture_results()
    results_path = tmp_path / "results.csv"
    output_dir = tmp_path / "analysis"
    results.to_csv(results_path, index=False)

    result = build_dpd_census_analysis(
        root_path=tmp_path,
        results_path=results_path,
        config_path=CONFIG_PATH,
        output_dir=output_dir,
        shared_failure_sample=1,
        all_pass_sample=1,
        audit_seed=17,
    )

    assert result["benchmark"]["case_count"] == 8
    assert result["benchmark"]["product_family_count"] == 4
    assert result["audit_set"] == {
        "seed": 17,
        "primary_model_failures": 1,
        "comparison_only_failures": 1,
        "shared_lower_tier_failure_sample": 1,
        "all_model_pass_sample": 1,
        "total_unique_cases": 4,
        "status": "selected_for_automated_audit",
    }

    overview = pd.read_csv(output_dir / "model_overview.csv")
    rows = overview.set_index("model_name")
    assert rows.loc["gpt_5_6_sol", "passed_cases"] == 7
    assert rows.loc["gpt_5_6_terra", "passed_cases"] == 7
    assert (
        rows.loc[
            "gpt_5_4_mini",
            "schema_type_only_failures",
        ]
        == 1
    )

    taxonomy = pd.read_csv(output_dir / "error_taxonomy.csv")
    terra_overall = taxonomy[
        (taxonomy["model_name"] == "gpt_5_6_terra")
        & (taxonomy["language"] == "all")
    ]
    assert terra_overall.iloc[0]["error_type"] == (
        "compound_label_split_only"
    )

    frontier = pd.read_csv(output_dir / "frontier_cases.csv")
    assert len(frontier) == 2
    assert set(frontier["case_id"]) == {
        "product_1_en",
        "product_2_fr",
    }

    selected = pd.read_csv(output_dir / "selected_cases.csv")
    assert len(selected) == 4
    assert selected["case_id"].nunique() == 4
    assert set(selected["selection_group"]) == {
        "primary_model_failure",
        "comparison_only_failure",
        "lower_tier_shared_failure_sample",
        "all_model_pass_sample",
    }

    first_manifest = (
        output_dir / "analysis_manifest.json"
    ).read_bytes()
    first_selected = (output_dir / "selected_cases.csv").read_bytes()
    build_dpd_census_analysis(
        root_path=tmp_path,
        results_path=results_path,
        config_path=CONFIG_PATH,
        output_dir=output_dir,
        shared_failure_sample=1,
        all_pass_sample=1,
        audit_seed=17,
    )
    assert (
        output_dir / "analysis_manifest.json"
    ).read_bytes() == first_manifest
    assert (
        output_dir / "selected_cases.csv"
    ).read_bytes() == first_selected


def test_validate_results_rejects_invalid_contracts() -> None:
    results = _fixture_results()
    validated, model_order = _validate_results(
        results,
        required_models=MODELS,
    )
    assert tuple(model_order) == MODELS
    assert set(validated["expected_output"].map(json.loads).iloc[0]) == set(
        DPD_FIELDS
    )

    duplicate = pd.concat(
        [results, results.iloc[[0]]],
        ignore_index=True,
    )
    with pytest.raises(
        ValueError,
        match="one row per case and model",
    ):
        _validate_results(
            duplicate,
            required_models=MODELS,
        )

    incomplete = results[
        ~(
            (results["model_name"] == "gpt_5_6_sol")
            & (results["case_id"] == "product_3_fr")
        )
    ]
    with pytest.raises(
        ValueError,
        match="identical case coverage",
    ):
        _validate_results(
            incomplete,
            required_models=MODELS,
        )

    with pytest.raises(
        ValueError,
        match="missing required models",
    ):
        _validate_results(
            results[results["model_name"] != "gpt_5_6_sol"],
            required_models=MODELS,
        )


def test_analysis_cli_defaults() -> None:
    args = build_parser().parse_args(["analyze-dpd-census"])
    assert args.command == "analyze-dpd-census"
    assert args.primary_model == "gpt_5_6_sol"
    assert args.comparison_model == "gpt_5_6_terra"
    assert args.shared_failure_sample == 25
    assert args.all_pass_sample == 25

    audit_args = build_parser().parse_args(["audit-dpd-evidence"])
    assert audit_args.command == "audit-dpd-evidence"
    assert audit_args.output.endswith("evidence_audit.csv")
    assert audit_args.summary.endswith("evidence_audit_summary.json")

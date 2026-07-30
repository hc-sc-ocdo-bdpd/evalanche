import csv
import gzip
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pytest

from evalanche.cli import build_parser
from evalanche.config import (
    MetricsConfig,
    load_candidate_models,
    load_evaluation_config,
    load_generation_config,
)
from evalanche.dataset_manifest import (
    load_dataset_manifest,
    sha256_file,
    verify_dataset_manifest_file,
)
from evalanche.dpd_benchmark import (
    CASE_FIELDS,
    DPD_SOURCE_MANIFEST,
    OUTPUT_FIELDS,
    build_candidate_products,
    load_marketed_tables,
)
from evalanche.dpd_census import (
    DPD_CENSUS_DATASET_ID,
    DPD_CENSUS_DEMO_TARGETS,
    DPD_CENSUS_VERSION,
    create_dpd_benchmark_census,
    select_demo_products,
)
from evalanche.generation import load_generation_cases
from evalanche.metrics.deterministic import score_row
from evalanche.pricing import load_endpoint_pricing_catalog


MATERIALIZED_MANIFEST = Path(
    "configs/datasets/"
    "hc_dpd_structured_extraction_census_0.2.0_manifest.yaml"
)
MATERIALIZED_DIRECTORY = Path(
    "data/hc/benchmarks/dpd_structured_extraction_census/0.2.0"
)


def _read_gzip_csv(path: Path) -> list[dict[str, str]]:
    with gzip.open(
        path,
        mode="rt",
        encoding="utf-8",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _copy_source_release(destination: Path) -> None:
    source_manifest = load_dataset_manifest(DPD_SOURCE_MANIFEST)
    manifest_destination = destination / DPD_SOURCE_MANIFEST
    manifest_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DPD_SOURCE_MANIFEST, manifest_destination)
    for file_entry in source_manifest.files:
        source = Path(file_entry.relative_path)
        target = destination / file_entry.relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


@pytest.fixture(scope="module")
def census_products() -> list[dict[str, str]]:
    return _read_gzip_csv(MATERIALIZED_DIRECTORY / "products.csv.gz")


@pytest.fixture(scope="module")
def census_cases() -> list[dict[str, str]]:
    return _read_gzip_csv(MATERIALIZED_DIRECTORY / "cases.csv.gz")


@pytest.fixture(scope="module")
def census_candidates() -> list[Any]:
    source_manifest = load_dataset_manifest(DPD_SOURCE_MANIFEST)
    marketed_entry = next(
        file
        for file in source_manifest.files
        if file.role == "raw_dpd_marketed_archive"
    )
    tables = load_marketed_tables(marketed_entry.relative_path)
    candidates, _ = build_candidate_products(
        tables,
        include_complex_families=True,
    )
    return candidates


def test_cli_exposes_reproducible_dpd_census_command() -> None:
    parser = build_parser()

    args = parser.parse_args(["build-dpd-census"])

    assert args.command == "build-dpd-census"
    assert args.root == "."
    assert args.source_manifest == DPD_SOURCE_MANIFEST.as_posix()


def test_materialized_dpd_census_manifest_verifies() -> None:
    verification = verify_dataset_manifest_file(
        MATERIALIZED_MANIFEST,
        root_path=".",
    )

    assert verification["valid"] is True, verification
    assert verification["files_checked"] == 4
    assert verification["files_passed"] == 4
    products_result = next(
        result
        for result in verification["files"]
        if result["file_id"] == "dpd_census_products"
    )
    assert products_result["record_count_verified"] is True
    assert products_result["sampling_membership_verified"] is True


def test_census_contains_every_representable_product_family(
    census_products: list[dict[str, str]],
    census_cases: list[dict[str, str]],
) -> None:
    assert len(census_products) == 7017
    assert len(census_cases) == 14034
    assert Counter(
        row["stratum"] for row in census_products
    ) == Counter(
        {
            "single_ingredient": 3123,
            "multi_ingredient": 1528,
            "multi_variant": 2197,
            "multi_ingredient_multi_variant": 169,
        }
    )
    assert {row["split"] for row in census_products} == {"census"}
    assert {row["split"] for row in census_cases} == {"census"}


def test_census_case_contract_is_bilingual_and_traceable(
    census_products: list[dict[str, str]],
    census_cases: list[dict[str, str]],
) -> None:
    cases_by_product: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in census_cases:
        cases_by_product[row["product_id"]].append(row)

    assert tuple(census_cases[0]) == CASE_FIELDS
    assert len({row["case_id"] for row in census_cases}) == len(
        census_cases
    )
    for product in census_products:
        product_cases = cases_by_product[product["product_id"]]
        assert {row["language"] for row in product_cases} == {
            "en",
            "fr",
        }
        assert {
            row["case_id"] for row in product_cases
        } == set(json.loads(product["language_case_ids"]))
        for case in product_cases:
            expected = json.loads(case["expected_output"])
            metadata = json.loads(case["source_metadata"])
            assert tuple(expected) == OUTPUT_FIELDS
            assert metadata["source_dataset_id"] == (
                "hc_dpd_source_snapshot"
            )
            assert metadata["archive_members"]


def test_complex_product_families_are_represented(
    census_products: list[dict[str, str]],
    census_cases: list[dict[str, str]],
) -> None:
    complex_products = [
        row
        for row in census_products
        if row["stratum"] == "multi_ingredient_multi_variant"
    ]
    complex_ids = {row["product_id"] for row in complex_products}
    complex_cases = [
        row for row in census_cases if row["product_id"] in complex_ids
    ]

    assert len(complex_products) == 169
    assert len(complex_cases) == 338
    assert all(
        int(product["variant_count"]) >= 2
        and len(json.loads(product["active_ingredient_codes"])) >= 2
        for product in complex_products
    )


def test_build_report_records_population_and_honest_boundaries() -> None:
    report = json.loads(
        (MATERIALIZED_DIRECTORY / "build_report.json").read_text(
            encoding="utf-8"
        )
    )

    assert report["dataset_id"] == DPD_CENSUS_DATASET_ID
    assert report["dataset_version"] == DPD_CENSUS_VERSION
    assert report["population"]["source_drug_rows"] == 13538
    assert report["population"]["eligible_drug_rows"] == 11613
    assert report["population"]["eligible_product_families"] == 7030
    assert report["population"]["included_product_families"] == 7017
    assert report["population"][
        "product_family_exclusions_by_reason"
    ] == {"inconsistent_french_brand_name": 13}
    assert report["quality_checks"][
        "all_candidate_product_families_included"
    ] is True
    assert report["quality_checks"]["complex_families_included"] is True
    assert report["quality_checks"][
        "manual_product_monograph_audit_status"
    ] == "not_started"
    assert any(
        "not Product Monograph" in limitation
        for limitation in report["limitations"]
    )


def test_demo_view_is_bounded_balanced_and_part_of_census(
    census_cases: list[dict[str, str]],
) -> None:
    demo_cases = _read_csv(MATERIALIZED_DIRECTORY / "demo_cases.csv")
    census_ids = {row["case_id"] for row in census_cases}

    assert len(demo_cases) == 24
    assert {row["case_id"] for row in demo_cases} <= census_ids
    assert Counter(row["language"] for row in demo_cases) == {
        "en": 12,
        "fr": 12,
    }
    assert Counter(row["stratum"] for row in demo_cases) == {
        stratum: target * 2
        for stratum, target in DPD_CENSUS_DEMO_TARGETS.items()
    }


def test_demo_selection_is_independent_of_candidate_input_order(
    census_candidates: list[Any],
) -> None:
    first = select_demo_products(census_candidates)
    second = select_demo_products(reversed(census_candidates))

    assert [candidate.product_id for candidate in first] == [
        candidate.product_id for candidate in second
    ]
    assert Counter(candidate.stratum for candidate in first) == Counter(
        DPD_CENSUS_DEMO_TARGETS
    )


def test_compressed_census_cases_are_generation_ready() -> None:
    cases = load_generation_cases(
        MATERIALIZED_DIRECTORY / "cases.csv.gz"
    )
    demo_cases = load_generation_cases(
        MATERIALIZED_DIRECTORY / "demo_cases.csv"
    )

    assert len(cases) == 14034
    assert len(demo_cases) == 24
    assert cases["case_id"].is_unique
    assert demo_cases["case_id"].is_unique
    assert set(cases["evaluation_type"]) == {"json"}


def test_demo_configs_load_and_target_bounded_view() -> None:
    generation = load_generation_config(
        "configs/generate_hc_dpd_census_demo.yaml"
    )
    evaluation = load_evaluation_config(
        "configs/evaluate_hc_dpd_census_demo.yaml"
    )

    assert generation.run.input_path == (
        MATERIALIZED_DIRECTORY / "demo_cases.csv"
    )
    assert generation.generation.max_completion_tokens == 900
    assert evaluation.run.input_path == Path(
        "data/generated/hc_dpd_census_demo_outputs.csv"
    )
    assert set(
        evaluation.metrics.json_comparison.unordered_list_paths
    ) == {
        "/din",
        "/active_ingredients",
        "/dosage_forms",
        "/routes",
        "/schedule",
    }
    assert (
        evaluation.metrics.json_comparison
        .zero_pad_numeric_string_paths["/din/*"]
        == 8
    )
    assert (
        evaluation.metrics.json_comparison.numeric_value_paths
        == ["/active_ingredients/*/strength"]
    )


def test_full_census_gpt_5_4_mini_configs_are_guarded_and_separate() -> None:
    generation = load_generation_config(
        "configs/generate_hc_dpd_census_gpt_5_4_mini.yaml"
    )
    evaluation = load_evaluation_config(
        "configs/evaluate_hc_dpd_census_gpt_5_4_mini.yaml"
    )

    assert generation.run.input_path == (
        MATERIALIZED_DIRECTORY / "cases.csv.gz"
    )
    assert generation.run.output_path == Path(
        "data/generated/hc_dpd_census_gpt_5_4_mini_outputs.csv"
    )
    assert generation.candidate_models_path == Path(
        "configs/candidate_gpt_5_4_mini.yaml"
    )
    candidates = load_candidate_models(
        generation.candidate_models_path
    )
    assert candidates.models[0].pricing_id == (
        "azure_global_gpt_5_4_mini_2026_03_17"
    )
    assert candidates.models[0].provider_model_version == "2026-03-17"
    assert candidates.models[0].deployment_type == "Global Standard"
    assert candidates.models[0].resource_region == "Canada East"
    assert generation.generation.resume is True
    assert generation.generation.max_workers == 2
    assert generation.generation.max_requests_per_minute == 30
    assert generation.generation.checkpoint_every == 100
    assert (
        generation.generation.maximum_estimated_cost_usd
        == 30
    )
    assert generation.generation.cost_safety_multiplier == 1.5
    assert evaluation.run.input_path == generation.run.output_path
    assert evaluation.run.output_path == Path(
        "results/evaluate_hc_dpd_census_gpt_5_4_mini_results.csv"
    )


def test_gpt_5_6_sol_configs_are_pinned_guarded_and_separate() -> None:
    pilot = load_generation_config(
        "configs/generate_hc_dpd_census_demo_gpt_5_6_sol.yaml"
    )
    full = load_generation_config(
        "configs/generate_hc_dpd_census_gpt_5_6_sol.yaml"
    )
    pilot_evaluation = load_evaluation_config(
        "configs/evaluate_hc_dpd_census_demo_gpt_5_6_sol.yaml"
    )
    full_evaluation = load_evaluation_config(
        "configs/evaluate_hc_dpd_census_gpt_5_6_sol.yaml"
    )
    candidates = load_candidate_models(full.candidate_models_path)
    candidate = candidates.models[0]
    pricing = load_endpoint_pricing_catalog(
        full.endpoint_pricing_path
    )
    endpoint = pricing.endpoints[0]

    assert pilot.run.input_path == (
        MATERIALIZED_DIRECTORY / "demo_cases.csv"
    )
    assert pilot.run.output_path == Path(
        "data/generated/hc_dpd_census_demo_gpt_5_6_sol_outputs.csv"
    )
    assert full.run.input_path == (
        MATERIALIZED_DIRECTORY / "cases.csv.gz"
    )
    assert full.run.output_path == Path(
        "data/generated/hc_dpd_census_gpt_5_6_sol_outputs.csv"
    )
    assert full.candidate_models_path == Path(
        "configs/candidate_gpt_5_6_sol.yaml"
    )
    assert candidate.name == "gpt_5_6_sol"
    assert candidate.model == "azure/gpt-5.6-sol"
    assert candidate.temperature is None
    assert candidate.reasoning_effort == "none"
    assert candidate.provider_model_version == "2026-07-09"
    assert candidate.deployment_type == "Global Standard"
    assert candidate.resource_region == "Canada East"
    assert candidate.pricing_id == (
        "azure_global_gpt_5_6_sol_2026_07_09"
    )
    assert endpoint.model == candidate.model
    assert endpoint.input_per_million_tokens == 5
    assert endpoint.cached_input_per_million_tokens is None
    assert endpoint.output_per_million_tokens == 30
    assert pilot.prompt == full.prompt
    assert full.generation.resume is True
    assert full.generation.max_workers == 4
    assert full.generation.max_requests_per_minute == 60
    assert full.generation.checkpoint_every == 100
    assert full.generation.cost_preflight_sample_path == (
        Path(
            "data/generated/"
            "hc_dpd_comparison_500_gpt_5_6_sol_outputs.csv"
        )
    )
    assert full.generation.maximum_estimated_cost_usd == 170
    assert full.generation.cost_safety_multiplier == 1.5
    assert pilot_evaluation.run.input_path == pilot.run.output_path
    assert full_evaluation.run.input_path == full.run.output_path
    assert full_evaluation.run.output_path == Path(
        "results/evaluate_hc_dpd_census_gpt_5_6_sol_results.csv"
    )
    assert (
        pilot_evaluation.metrics
        == full_evaluation.metrics
    )


def _reorder_and_recase_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _reorder_and_recase_json(nested)
            for key, nested in value.items()
        }
    if isinstance(value, list):
        return [
            _reorder_and_recase_json(nested)
            for nested in reversed(value)
        ]
    if isinstance(value, str):
        return value.lower()
    return value


def test_demo_json_rules_accept_order_only_variation() -> None:
    evaluation = load_evaluation_config(
        "configs/evaluate_hc_dpd_census_demo.yaml"
    )
    metrics_config = MetricsConfig(
        run=evaluation.run,
        metrics=evaluation.metrics,
    )
    demo_cases = _read_csv(
        MATERIALIZED_DIRECTORY / "demo_cases.csv"
    )

    for case in demo_cases:
        output = _reorder_and_recase_json(
            json.loads(case["expected_output"])
        )
        result = score_row(
            {
                **case,
                "model_name": "order_variant",
                "model_output": json.dumps(
                    output,
                    ensure_ascii=False,
                ),
            },
            metrics_config,
        )
        assert result["metric_passed"] is True, case["case_id"]
        assert result["json_canonical_match"] is True


def test_demo_json_rules_accept_numeric_strength_strings() -> None:
    evaluation = load_evaluation_config(
        "configs/evaluate_hc_dpd_census_demo.yaml"
    )
    metrics_config = MetricsConfig(
        run=evaluation.run,
        metrics=evaluation.metrics,
    )
    demo_cases = _read_csv(
        MATERIALIZED_DIRECTORY / "demo_cases.csv"
    )

    for case in demo_cases:
        output = json.loads(case["expected_output"])
        for ingredient in output["active_ingredients"]:
            ingredient["strength"] = str(ingredient["strength"])
        result = score_row(
            {
                **case,
                "model_name": "numeric_string_variant",
                "model_output": json.dumps(
                    output,
                    ensure_ascii=False,
                ),
            },
            metrics_config,
        )
        assert result["metric_passed"] is True, case["case_id"]
        assert result["json_canonical_match"] is True


def test_demo_json_rules_still_reject_factual_error() -> None:
    evaluation = load_evaluation_config(
        "configs/evaluate_hc_dpd_census_demo.yaml"
    )
    metrics_config = MetricsConfig(
        run=evaluation.run,
        metrics=evaluation.metrics,
    )
    case = _read_csv(
        MATERIALIZED_DIRECTORY / "demo_cases.csv"
    )[0]
    output = json.loads(case["expected_output"])
    output["company"] = "NOT THE SOURCE COMPANY"

    result = score_row(
        {
            **case,
            "model_name": "factual_error",
            "model_output": json.dumps(
                output,
                ensure_ascii=False,
            ),
        },
        metrics_config,
    )

    assert result["metric_passed"] is False
    assert result["json_mismatched_fields"] == '["company"]'


def test_builder_refuses_to_overwrite_materialized_release() -> None:
    with pytest.raises(
        ValueError,
        match="already exists and will not be overwritten",
    ):
        create_dpd_benchmark_census(root_path=".")


def test_builder_is_byte_reproducible(tmp_path: Path) -> None:
    roots = [tmp_path / "first", tmp_path / "second"]
    for root in roots:
        _copy_source_release(root)
        result = create_dpd_benchmark_census(root_path=root)
        assert result["product_count"] == 7017
        assert result["case_count"] == 14034
        assert result["demo_case_count"] == 24

    relative_paths = [
        Path(
            "data/hc/benchmarks/dpd_structured_extraction_census/"
            "0.2.0/cases.csv.gz"
        ),
        Path(
            "data/hc/benchmarks/dpd_structured_extraction_census/"
            "0.2.0/products.csv.gz"
        ),
        Path(
            "data/hc/benchmarks/dpd_structured_extraction_census/"
            "0.2.0/demo_cases.csv"
        ),
        Path(
            "data/hc/benchmarks/dpd_structured_extraction_census/"
            "0.2.0/build_report.json"
        ),
        MATERIALIZED_MANIFEST,
    ]
    for relative_path in relative_paths:
        assert sha256_file(roots[0] / relative_path) == sha256_file(
            roots[1] / relative_path
        )


def test_builder_rejects_wrong_parent_dataset_before_writing(
    tmp_path: Path,
) -> None:
    _copy_source_release(tmp_path)
    raw = DPD_SOURCE_MANIFEST.read_text(encoding="utf-8")
    source_path = tmp_path / DPD_SOURCE_MANIFEST
    source_path.write_text(
        raw.replace(
            "dataset_id: hc_dpd_source_snapshot",
            "dataset_id: wrong_source",
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="source manifest dataset_id",
    ):
        create_dpd_benchmark_census(root_path=tmp_path)
    assert not (
        tmp_path
        / "data/hc/benchmarks/dpd_structured_extraction_census/0.2.0"
    ).exists()

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import re
import shutil
from typing import Any

import pytest
import yaml

from evalanche.cli import build_parser
from evalanche.dataset_manifest import (
    load_dataset_manifest,
    sha256_file,
    verify_dataset_manifest_file,
)
from evalanche.dpd_benchmark import (
    CASE_FIELDS,
    DEFAULT_STRATUM_TARGETS,
    DPD_BENCHMARK_SEED,
    DPD_BENCHMARK_VERSION,
    DPD_SOURCE_MANIFEST,
    OUTPUT_FIELDS,
    REQUIRED_ROUTE_COVERAGE,
    build_candidate_products,
    create_dpd_benchmark_slice,
    load_marketed_tables,
    select_products,
)
from evalanche.generation import load_generation_cases


MATERIALIZED_MANIFEST = Path(
    "configs/datasets/"
    "hc_dpd_structured_extraction_0.1.0_manifest.yaml"
)
MATERIALIZED_DIRECTORY = Path(
    "data/hc/benchmarks/dpd_structured_extraction/0.1.0"
)
_DIN = re.compile(r"^\d{8}$")


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
def candidates() -> list[Any]:
    source_manifest = load_dataset_manifest(DPD_SOURCE_MANIFEST)
    marketed_entry = next(
        file
        for file in source_manifest.files
        if file.role == "raw_dpd_marketed_archive"
    )
    tables = load_marketed_tables(marketed_entry.relative_path)
    result, _ = build_candidate_products(tables)
    return result


def test_cli_exposes_reproducible_dpd_benchmark_command() -> None:
    parser = build_parser()

    args = parser.parse_args(["build-dpd-benchmark"])

    assert args.command == "build-dpd-benchmark"
    assert args.root == "."
    assert args.source_manifest == DPD_SOURCE_MANIFEST.as_posix()
    assert args.version == DPD_BENCHMARK_VERSION
    assert args.seed == DPD_BENCHMARK_SEED


def test_materialized_dpd_benchmark_manifest_verifies() -> None:
    verification = verify_dataset_manifest_file(
        MATERIALIZED_MANIFEST,
        root_path=".",
    )

    assert verification["valid"] is True, verification
    assert verification["files_passed"] == 3
    assert all(
        not file_result["issues"]
        for file_result in verification["files"]
    )
    products_result = next(
        file_result
        for file_result in verification["files"]
        if file_result["file_id"] == "dpd_benchmark_products"
    )
    assert products_result["sampling_membership_verified"] is True


def test_materialized_slice_has_expected_sampling_and_splits() -> None:
    products = _read_csv(MATERIALIZED_DIRECTORY / "products.csv")
    cases = _read_csv(MATERIALIZED_DIRECTORY / "cases.csv")

    assert len(products) == 40
    assert len(cases) == 80
    assert Counter(row["stratum"] for row in products) == Counter(
        DEFAULT_STRATUM_TARGETS
    )
    assert Counter(row["split"] for row in products) == {
        "development": 30,
        "heldout": 10,
    }
    assert Counter(row["language"] for row in cases) == {
        "en": 40,
        "fr": 40,
    }
    assert Counter(row["split"] for row in cases) == {
        "development": 60,
        "heldout": 20,
    }


def test_every_product_has_paired_unique_language_cases() -> None:
    products = _read_csv(MATERIALIZED_DIRECTORY / "products.csv")
    cases = _read_csv(MATERIALIZED_DIRECTORY / "cases.csv")
    cases_by_product: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in cases:
        cases_by_product[row["product_id"]].append(row)

    assert len({row["case_id"] for row in cases}) == len(cases)
    for product in products:
        product_cases = cases_by_product[product["product_id"]]
        assert {row["language"] for row in product_cases} == {
            "en",
            "fr",
        }
        assert {
            row["case_id"] for row in product_cases
        } == set(json.loads(product["language_case_ids"]))


def test_cases_have_valid_contract_and_exact_source_traceability() -> None:
    cases = _read_csv(MATERIALIZED_DIRECTORY / "cases.csv")
    expected_members = {
        "comp.txt",
        "drug.txt",
        "form.txt",
        "ingred.txt",
        "route.txt",
        "schedule.txt",
        "status.txt",
    }

    assert tuple(cases[0]) == CASE_FIELDS
    for row in cases:
        expected = json.loads(row["expected_output"])
        metadata = json.loads(row["source_metadata"])
        assert tuple(expected) == OUTPUT_FIELDS
        assert all(_DIN.fullmatch(value) for value in expected["din"])
        assert expected["active_ingredients"]
        assert all(
            set(ingredient) == {"name", "strength", "unit"}
            for ingredient in expected["active_ingredients"]
        )
        assert metadata["source_dataset_id"] == (
            "hc_dpd_source_snapshot"
        )
        assert metadata["source_dataset_version"] == "2026.7.2"
        assert len(metadata["source_manifest_sha256"]) == 64
        assert len(metadata["archive_sha256"]) == 64
        assert set(metadata["archive_members"]) == expected_members
        assert all(
            row_numbers
            and all(
                isinstance(row_number, int) and row_number > 0
                for row_number in row_numbers
            )
            for row_numbers in metadata["archive_members"].values()
        )

def test_french_cases_use_french_reference_values() -> None:
    cases = _read_csv(MATERIALIZED_DIRECTORY / "cases.csv")
    french_cases = [row for row in cases if row["language"] == "fr"]

    assert french_cases
    assert all(
        json.loads(row["expected_output"])["product_status"]
        == "COMMERCIALISÉ"
        for row in french_cases
    )
    assert all(
        "DOSSIER SOURCE DPD (FRANÇAIS)" in row["input"]
        for row in french_cases
    )


def test_build_report_records_quality_gates_and_route_coverage() -> None:
    report = json.loads(
        (
            MATERIALIZED_DIRECTORY / "build_report.json"
        ).read_text(encoding="utf-8")
    )

    assert report["sampling"]["selected_product_count"] == 40
    assert report["sampling"]["generated_case_count"] == 80
    assert report["quality_checks"]["unique_product_ids"] is True
    assert report["quality_checks"]["unique_case_ids"] is True
    assert (
        report["quality_checks"]["two_language_cases_per_product"]
        is True
    )
    assert (
        report["quality_checks"][
            "development_heldout_ingredient_overlap_count"
        ]
        == 0
    )
    assert (
        report["quality_checks"]["all_required_route_groups_present"]
        is True
    )
    assert report["coverage"]["required_route_groups"] == {
        route_group: True
        for route_group in REQUIRED_ROUTE_COVERAGE
    }
    assert (
        report["quality_checks"][
            "manual_product_monograph_audit_status"
        ]
        == "not_started"
    )


def test_selected_ingredient_codes_do_not_cross_splits() -> None:
    products = _read_csv(MATERIALIZED_DIRECTORY / "products.csv")
    by_split: dict[str, set[str]] = defaultdict(set)
    for product in products:
        by_split[product["split"]].update(
            json.loads(product["active_ingredient_codes"])
        )

    assert not (
        by_split["development"] & by_split["heldout"]
    )


def test_selection_is_independent_of_candidate_input_order(
    candidates: list[Any],
) -> None:
    first = select_products(
        candidates,
        seed=DPD_BENCHMARK_SEED,
    )
    second = select_products(
        reversed(candidates),
        seed=DPD_BENCHMARK_SEED,
    )

    first_membership = [
        (selected.candidate.product_id, selected.split)
        for selected in first
    ]
    second_membership = [
        (selected.candidate.product_id, selected.split)
        for selected in second
    ]
    assert first_membership == second_membership
    assert Counter(
        selected.candidate.stratum for selected in first
    ) == Counter(DEFAULT_STRATUM_TARGETS)
    assert Counter(selected.split for selected in first) == {
        "development": 30,
        "heldout": 10,
    }


def test_selection_fails_closed_when_route_coverage_is_impossible(
    candidates: list[Any],
) -> None:
    with pytest.raises(
        ValueError,
        match="required route coverage: underwater",
    ):
        select_products(
            candidates,
            seed=DPD_BENCHMARK_SEED,
            required_route_coverage={
                "underwater": {"SUBMARINE"},
            },
        )


def test_builder_refuses_to_overwrite_materialized_release() -> None:
    with pytest.raises(
        ValueError,
        match="already exists and will not be overwritten",
    ):
        create_dpd_benchmark_slice(root_path=".")


def test_builder_rejects_wrong_parent_dataset_before_writing(
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(
        DPD_SOURCE_MANIFEST.read_text(encoding="utf-8")
    )
    raw["release"]["dataset_id"] = "wrong_source"
    source_path = tmp_path / "source.yaml"
    source_path.write_text(
        yaml.safe_dump(raw, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="source manifest dataset_id",
    ):
        create_dpd_benchmark_slice(
            root_path=tmp_path,
            source_manifest_path="source.yaml",
        )
    assert not (
        tmp_path
        / "data/hc/benchmarks/dpd_structured_extraction/0.1.0"
    ).exists()


def test_builder_is_byte_reproducible(tmp_path: Path) -> None:
    roots = [tmp_path / "first", tmp_path / "second"]
    results = []
    for root in roots:
        _copy_source_release(root)
        results.append(create_dpd_benchmark_slice(root_path=root))

    relative_paths = [
        Path(
            "data/hc/benchmarks/dpd_structured_extraction/0.1.0/"
            "cases.csv"
        ),
        Path(
            "data/hc/benchmarks/dpd_structured_extraction/0.1.0/"
            "products.csv"
        ),
        Path(
            "data/hc/benchmarks/dpd_structured_extraction/0.1.0/"
            "build_report.json"
        ),
        MATERIALIZED_MANIFEST,
    ]
    for relative_path in relative_paths:
        assert sha256_file(roots[0] / relative_path) == sha256_file(
            roots[1] / relative_path
        )
    assert results[0]["product_count"] == 40
    assert results[1]["case_count"] == 80


def test_materialized_cases_are_generation_ready() -> None:
    cases = load_generation_cases(
        MATERIALIZED_DIRECTORY / "cases.csv"
    )

    assert len(cases) == 80
    assert cases["case_id"].is_unique
    assert set(cases["evaluation_type"]) == {"json"}
    assert set(cases["language"]) == {"en", "fr"}

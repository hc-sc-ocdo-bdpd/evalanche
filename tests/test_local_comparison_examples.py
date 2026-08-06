from pathlib import Path

import pandas as pd
import pytest

from evalanche.access_sets import AccessSet
from evalanche.config import load_yaml
from evalanche.registry import (
    BenchmarkManifest,
    load_registry,
    validate_registry,
)


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples/local_comparison"


@pytest.mark.parametrize(
    ("benchmark_file", "expected_type"),
    [
        ("exact_benchmark.example.yaml", "exact"),
        ("json_benchmark.example.yaml", "json"),
    ],
)
def test_benchmark_starter_is_a_valid_registry_record(
    benchmark_file: str,
    expected_type: str,
) -> None:
    manifest_path = EXAMPLES / benchmark_file
    benchmark = BenchmarkManifest.model_validate(load_yaml(manifest_path))
    registry = load_registry(ROOT)
    key = f"{benchmark.benchmark_id}@{benchmark.version}"
    registry.benchmarks[key] = benchmark
    registry.benchmark_paths[key] = manifest_path

    validation = validate_registry(registry)

    assert validation["valid"] is True
    assert benchmark.scoring.evaluation_type == expected_type
    assert key in validation["fingerprints"]


@pytest.mark.parametrize(
    ("filename", "expected_type"),
    [
        ("classification_cases.csv", "exact"),
        ("factual_cases.csv", "exact"),
        ("structured_extraction_cases.csv", "json"),
        ("open_ended_cases.csv", "judge"),
    ],
)
def test_case_starters_expose_the_documented_contract(
    filename: str,
    expected_type: str,
) -> None:
    cases = pd.read_csv(EXAMPLES / filename)

    assert {
        "case_id",
        "group_id",
        "language",
        "split",
        "risk",
        "input",
        "expected_output",
        "evaluation_type",
    }.issubset(cases.columns)
    assert cases["case_id"].is_unique
    assert set(cases["evaluation_type"]) == {expected_type}


def test_access_set_starter_requires_explicit_confirmation() -> None:
    access_set = AccessSet.model_validate(
        load_yaml(EXAMPLES / "access_set.example.yaml")
    )

    assert access_set.model_ids == ["replace_with_registered_model_id"]
    assert all(model.access_confirmed is True for model in access_set.models)

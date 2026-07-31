from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from evalanche.benchmark_runner import build_run_plan
from evalanche.leaderboard import build_benchmark_index, build_leaderboard
from evalanche.registry import (
    benchmark_fingerprint,
    load_registry,
    sha256_file,
    validate_registry,
)
from evalanche.result_bundle import (
    load_active_bundles,
    register_result_bundle,
)

ROOT = Path(__file__).resolve().parents[1]


def _write_yaml(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(value, sort_keys=False),
        encoding="utf-8",
    )


def _synthetic_registry(tmp_path: Path) -> None:
    cases = pd.DataFrame(
        [
            {
                "case_id": "case_en",
                "group_id": "group_1",
                "language": "en",
                "input": "one",
                "expected_output": '{"answer":"one"}',
                "evaluation_type": "json",
            },
            {
                "case_id": "case_fr",
                "group_id": "group_1",
                "language": "fr",
                "input": "un",
                "expected_output": '{"answer":"un"}',
                "evaluation_type": "json",
            },
        ]
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    cases.to_csv(data_dir / "cases.csv", index=False)
    (data_dir / "manifest.yaml").write_text(
        "schema_version: 'test'\n",
        encoding="utf-8",
    )
    for model_id in ("model_a", "model_b"):
        _write_yaml(
            tmp_path / "configs/models" / f"{model_id}.yaml",
            {
                "schema_version": "1.0",
                "model_id": model_id,
                "display_name": model_id.replace("_", " ").title(),
                "provider_route": f"test/{model_id}",
                "request": {"temperature": 0},
            },
        )
    _write_yaml(
        tmp_path / "configs/benchmarks/synthetic_1.0.0.yaml",
        {
            "schema_version": "1.0",
            "benchmark_id": "synthetic",
            "version": "1.0.0",
            "title": "Synthetic benchmark",
            "description": "Registry integration test.",
            "status": "frozen",
            "dataset": {
                "dataset_id": "synthetic_cases",
                "version": "1.0.0",
                "manifest_path": "data/manifest.yaml",
                "cases_path": "data/cases.csv",
            },
            "prompt": {
                "system": "Return JSON.",
                "template": "{input}",
                "version": "1.0",
            },
            "scoring": {
                "evaluation_type": "json",
                "version": "1.0",
                "required_output_fields": ["answer"],
            },
            "slice_columns": ["language"],
            "group_key": "group_id",
        },
    )


def test_repository_registry_is_valid() -> None:
    registry = load_registry(ROOT)
    result = validate_registry(registry)

    assert result["valid"] is True
    assert set(registry.models) == {
        "gpt_5_4_mini",
        "gpt_5_6_luna",
        "gpt_5_6_terra",
        "gpt_5_6_sol",
    }
    assert set(registry.benchmarks) == {
        "hc_dpd_structured_extraction@0.2.0",
    }


def test_manifest_only_model_and_dataset_extension(
    tmp_path: Path,
) -> None:
    _synthetic_registry(tmp_path)
    registry = load_registry(tmp_path)
    validation = validate_registry(registry)
    assert validation["valid"] is True

    benchmark = registry.resolve_benchmark("synthetic@1.0.0")
    for model_id, passed in (
        ("model_a", [True, False]),
        ("model_b", [True, True]),
    ):
        results = pd.DataFrame(
            {
                "case_id": ["case_en", "case_fr"],
                "model_name": [model_id, model_id],
                "final_passed": passed,
                "final_score": [float(value) for value in passed],
                "evaluation_source": ["deterministic", "deterministic"],
                "generation_status": ["success", "success"],
            }
        )
        result_path = tmp_path / f"{model_id}.csv"
        results.to_csv(result_path, index=False)
        register_result_bundle(
            registry=registry,
            benchmark=benchmark,
            model_id=model_id,
            results_path=result_path,
            reports_root="reports",
        )

    built = build_leaderboard(
        registry=registry,
        benchmark=benchmark,
        reports_root="reports",
    )
    leaderboard = pd.read_csv(built["leaderboard_path"])
    assert leaderboard["model_id"].tolist() == ["model_b", "model_a"]
    pairwise = pd.read_csv(built["pairwise_path"])
    assert pairwise.loc[0, "b_only_passed"] == 1
    assert build_benchmark_index(
        registry=registry,
        reports_root="reports",
    ).is_file()

    plan = build_run_plan(
        registry=registry,
        benchmark=benchmark,
        model=registry.resolve_model("model_b"),
    )
    assert plan["planned_model_calls"] == 2
    assert plan["execution"] == {
        "generation_complete": False,
        "evaluation_complete": False,
    }


def test_result_registration_is_byte_reproducible(
    tmp_path: Path,
) -> None:
    _synthetic_registry(tmp_path)
    registry = load_registry(tmp_path)
    benchmark = registry.resolve_benchmark("synthetic@1.0.0")
    results = pd.DataFrame(
        {
            "case_id": ["case_en", "case_fr"],
            "model_name": ["model_a", "model_a"],
            "final_passed": [True, False],
            "final_score": [1.0, 0.5],
            "generated_at_utc": [
                "2026-07-30T01:00:00Z",
                "2026-07-30T01:01:00Z",
            ],
        }
    )
    result_path = tmp_path / "model_a.csv"
    results.to_csv(result_path, index=False)

    first = register_result_bundle(
        registry=registry,
        benchmark=benchmark,
        model_id="model_a",
        results_path=result_path,
        reports_root="reports",
    )
    first_hashes = (
        sha256_file(first["metadata_path"]),
        sha256_file(first["case_scores_path"]),
        sha256_file(first["active_path"]),
    )
    second = register_result_bundle(
        registry=registry,
        benchmark=benchmark,
        model_id="model_a",
        results_path=result_path,
        reports_root="reports",
    )
    second_hashes = (
        sha256_file(second["metadata_path"]),
        sha256_file(second["case_scores_path"]),
        sha256_file(second["active_path"]),
    )
    assert first["run_id"] == second["run_id"]
    assert first_hashes == second_hashes


def test_incompatible_result_bundle_is_rejected(
    tmp_path: Path,
) -> None:
    _synthetic_registry(tmp_path)
    registry = load_registry(tmp_path)
    benchmark = registry.resolve_benchmark("synthetic@1.0.0")
    result_path = tmp_path / "model_a.csv"
    pd.DataFrame(
        {
            "case_id": ["case_en", "case_fr"],
            "model_name": ["model_a", "model_a"],
            "final_passed": [True, True],
            "final_score": [1.0, 1.0],
        }
    ).to_csv(result_path, index=False)
    register_result_bundle(
        registry=registry,
        benchmark=benchmark,
        model_id="model_a",
        results_path=result_path,
        reports_root="reports",
    )

    cases_path = tmp_path / "data/cases.csv"
    cases = pd.read_csv(cases_path)
    cases.loc[0, "input"] = "changed"
    cases.to_csv(cases_path, index=False)
    with pytest.raises(ValueError, match="Incompatible active run"):
        load_active_bundles(
            registry=registry,
            benchmark=benchmark,
            reports_root="reports",
        )


def test_manifest_line_endings_do_not_change_compatibility(
    tmp_path: Path,
) -> None:
    _synthetic_registry(tmp_path)
    registry = load_registry(tmp_path)
    benchmark = registry.resolve_benchmark("synthetic@1.0.0")
    before = benchmark_fingerprint(registry, benchmark)

    manifest_path = tmp_path / "data/manifest.yaml"
    manifest_path.write_bytes(
        manifest_path.read_bytes().replace(b"\n", b"\r\n")
    )
    after = benchmark_fingerprint(registry, benchmark)

    assert before == after


def test_published_dpd_registry_scores_match_frozen_release() -> None:
    path = (
        ROOT
        / "reports/benchmarks/hc_dpd_structured_extraction/"
        "0.2.0/leaderboard.json"
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    rows = {row["model_id"]: row for row in document["models"]}
    assert {
        model: row["passed_cases"] for model, row in rows.items()
    } == {
        "gpt_5_6_sol": 14_021,
        "gpt_5_6_terra": 13_986,
        "gpt_5_4_mini": 10_827,
        "gpt_5_6_luna": 10_662,
    }

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from evalanche import __version__
from evalanche.benchmark_runner import (
    build_run_plan,
    rescore_registered_benchmark,
    run_registered_benchmark,
)
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
    assert registry.models
    assert {
        "hc_dpd_structured_extraction@0.2.0",
        "hc_product_monograph_native_pdf_extraction@0.1.0",
        "hc_product_monograph_structured_extraction@0.1.0",
    }.issubset(registry.benchmarks)
    assert len(registry.benchmarks) == 3
    fingerprint = benchmark_fingerprint(
        registry,
        registry.resolve_benchmark("hc_dpd_structured_extraction@0.2.0"),
    )
    assert fingerprint["evaluator_version"] == "0.3.0"
    assert fingerprint["software_version"] == __version__


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
    index_path = build_benchmark_index(
        registry=registry,
        reports_root="reports",
    )
    assert index_path.is_file()
    index_html = index_path.read_text(encoding="utf-8")
    assert "Status: frozen" in index_html
    assert 'href="synthetic/1.0.0/leaderboard.html"' in index_html

    draft_config = yaml.safe_load(
        (tmp_path / "configs/benchmarks/synthetic_1.0.0.yaml").read_text(
            encoding="utf-8"
        )
    )
    draft_config.update(
        {
            "benchmark_id": "synthetic_draft",
            "title": "Synthetic draft",
            "status": "draft",
        }
    )
    _write_yaml(
        tmp_path / "configs/benchmarks/synthetic_draft_1.0.0.yaml",
        draft_config,
    )
    draft_index = build_benchmark_index(
        registry=load_registry(tmp_path),
        reports_root="reports",
    ).read_text(encoding="utf-8")
    assert "Status: draft" in draft_index
    assert 'href="synthetic_draft/1.0.0/leaderboard.html"' not in draft_index
    assert '<div class="card unavailable">' in draft_index

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


def test_registered_benchmark_can_rescore_saved_outputs_offline(
    tmp_path: Path,
) -> None:
    _synthetic_registry(tmp_path)
    registry = load_registry(tmp_path)
    benchmark = registry.resolve_benchmark("synthetic@1.0.0")
    model = registry.resolve_model("model_a")
    plan = build_run_plan(
        registry=registry,
        benchmark=benchmark,
        model=model,
    )
    generation_path = tmp_path / plan["paths"]["generation_output"]
    generation_path.parent.mkdir(parents=True, exist_ok=True)
    cases = pd.read_csv(tmp_path / "data/cases.csv")
    cases["model_name"] = "model_a"
    cases["model_output"] = [
        '{"answer":"ONE"}',
        '{"answer":"incorrect"}',
    ]
    cases["generation_status"] = "success"
    cases.to_csv(generation_path, index=False)

    result = rescore_registered_benchmark(
        registry=registry,
        benchmark=benchmark,
        model=model,
    )

    assert result["generation_calls"] == 0
    evaluation_path = tmp_path / plan["paths"]["evaluation_output"]
    rescored = pd.read_csv(evaluation_path)
    assert rescored["final_passed"].tolist() == [True, False]
    assert result["bundle"]["active_path"].is_file()
    assert result["leaderboard"]["html_path"].is_file()


def test_draft_and_capability_gates_prevent_accidental_execution(
    tmp_path: Path,
) -> None:
    _synthetic_registry(tmp_path)
    benchmark_path = tmp_path / "configs/benchmarks/synthetic_1.0.0.yaml"
    benchmark_document = yaml.safe_load(benchmark_path.read_text(encoding="utf-8"))
    benchmark_document["status"] = "draft"
    benchmark_document["required_capabilities"] = ["pdf_input"]
    _write_yaml(benchmark_path, benchmark_document)

    registry = load_registry(tmp_path)
    benchmark = registry.resolve_benchmark("synthetic@1.0.0")
    with pytest.raises(ValueError, match="pdf_input"):
        build_run_plan(
            registry=registry,
            benchmark=benchmark,
            model=registry.resolve_model("model_a"),
        )

    model_path = tmp_path / "configs/models/model_a.yaml"
    model_document = yaml.safe_load(model_path.read_text(encoding="utf-8"))
    model_document["capabilities"] = ["pdf_input"]
    _write_yaml(model_path, model_document)
    registry = load_registry(tmp_path)
    benchmark = registry.resolve_benchmark("synthetic@1.0.0")
    model = registry.resolve_model("model_a")

    planned = run_registered_benchmark(
        registry=registry,
        benchmark=benchmark,
        model=model,
        plan_only=True,
    )
    assert planned["executed"] is False
    with pytest.raises(ValueError, match="Only ready or frozen"):
        run_registered_benchmark(
            registry=registry,
            benchmark=benchmark,
            model=model,
        )


def test_generation_failed_bundle_is_visible_but_unranked(
    tmp_path: Path,
) -> None:
    _synthetic_registry(tmp_path)
    registry = load_registry(tmp_path)
    benchmark = registry.resolve_benchmark("synthetic@1.0.0")
    result_path = tmp_path / "failed.csv"
    pd.DataFrame(
        {
            "case_id": ["case_en", "case_fr"],
            "model_name": ["model_a", "model_a"],
            "final_passed": [False, False],
            "final_score": [0.0, 0.0],
            "evaluation_source": ["deterministic", "deterministic"],
            "generation_status": ["error", "error"],
        }
    ).to_csv(result_path, index=False)
    register_result_bundle(
        registry=registry,
        benchmark=benchmark,
        model_id="model_a",
        results_path=result_path,
        reports_root="reports",
    )

    built = build_leaderboard(
        registry=registry,
        benchmark=benchmark,
        reports_root="reports",
    )
    leaderboard = pd.read_csv(built["leaderboard_path"])
    assert leaderboard.loc[0, "result_status"] == "ineligible"
    assert pd.isna(leaderboard.loc[0, "rank"])
    document = json.loads(built["json_path"].read_text(encoding="utf-8"))
    assert document["models"][0]["rank"] is None
    assert document["pairwise"] == []


def test_descriptive_reporting_keeps_complete_results_unranked(
    tmp_path: Path,
) -> None:
    _synthetic_registry(tmp_path)
    benchmark_path = tmp_path / "configs/benchmarks/synthetic_1.0.0.yaml"
    document = yaml.safe_load(benchmark_path.read_text(encoding="utf-8"))
    document["reporting"] = {
        "mode": "descriptive",
        "reason": "Reference labels do not have independent human sign-off.",
    }
    _write_yaml(benchmark_path, document)
    registry = load_registry(tmp_path)
    benchmark = registry.resolve_benchmark("synthetic@1.0.0")

    for model_id, passed in (
        ("model_a", [True, False]),
        ("model_b", [True, True]),
    ):
        result_path = tmp_path / f"{model_id}.csv"
        pd.DataFrame(
            {
                "case_id": ["case_en", "case_fr"],
                "model_name": [model_id, model_id],
                "final_passed": passed,
                "final_score": [float(value) for value in passed],
                "generation_status": ["success", "success"],
            }
        ).to_csv(result_path, index=False)
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
    assert set(leaderboard["result_status"]) == {"descriptive"}
    assert leaderboard["rank"].isna().all()
    assert len(pd.read_csv(built["pairwise_path"])) == 1
    leaderboard_json = json.loads(
        built["json_path"].read_text(encoding="utf-8")
    )
    assert leaderboard_json["reporting_policy"]["mode"] == "descriptive"
    assert "without an official rank" in (
        built["markdown_path"].read_text(encoding="utf-8")
    )


def test_descriptive_reporting_requires_a_reason(tmp_path: Path) -> None:
    _synthetic_registry(tmp_path)
    benchmark_path = tmp_path / "configs/benchmarks/synthetic_1.0.0.yaml"
    document = yaml.safe_load(benchmark_path.read_text(encoding="utf-8"))
    document["reporting"] = {"mode": "descriptive"}
    _write_yaml(benchmark_path, document)

    with pytest.raises(ValueError, match="requires a reason"):
        load_registry(tmp_path)


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
    manifest_path.write_bytes(manifest_path.read_bytes().replace(b"\n", b"\r\n"))
    after = benchmark_fingerprint(registry, benchmark)

    assert before == after


def test_published_dpd_registry_scores_match_frozen_release() -> None:
    path = (
        ROOT / "reports/benchmarks/hc_dpd_structured_extraction/0.2.0/leaderboard.json"
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    rows = {row["model_id"]: row for row in document["models"]}
    assert {model: row["passed_cases"] for model, row in rows.items()} == {
        "gpt_5_6_sol": 14_021,
        "gpt_5_6_terra": 13_986,
        "gpt_5_4_mini": 10_827,
        "gpt_5_6_luna": 10_662,
    }

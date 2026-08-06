from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from evalanche.benchmark_experiment import (
    build_benchmark_experiment_summary,
)
from evalanche.benchmark_runner import build_run_plan
from evalanche.registry import load_registry


def _write_yaml(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(value, sort_keys=False),
        encoding="utf-8",
    )


def _registry(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    pd.DataFrame(
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
    ).to_csv(data / "cases.csv", index=False)
    (data / "manifest.yaml").write_text(
        "schema_version: test\n",
        encoding="utf-8",
    )
    for model_id in ("model_a", "model_b", "model_c"):
        _write_yaml(
            tmp_path / "configs/models" / f"{model_id}.yaml",
            {
                "schema_version": "1.0",
                "model_id": model_id,
                "display_name": model_id,
                "provider_route": f"test/{model_id}",
                "request": {"temperature": 0},
                "capabilities": ["json"],
            },
        )
    _write_yaml(
        tmp_path / "configs/benchmarks/synthetic.yaml",
        {
            "schema_version": "1.0",
            "benchmark_id": "synthetic",
            "version": "1.0.0",
            "title": "Synthetic benchmark",
            "description": "Synthetic experiment test.",
            "status": "draft",
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
            "required_capabilities": ["json"],
        },
    )


def _write_result(
    tmp_path: Path,
    *,
    model_id: str,
    passed: list[bool],
    costs: list[float],
) -> None:
    registry = load_registry(tmp_path)
    benchmark = registry.resolve_benchmark("synthetic@1.0.0")
    plan = build_run_plan(
        registry=registry,
        benchmark=benchmark,
        model=registry.resolve_model(model_id),
    )
    output = tmp_path / plan["paths"]["evaluation_output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "case_id": ["case_en", "case_fr"],
            "model_name": [model_id, model_id],
            "final_passed": passed,
            "final_score": [1.0 if value else 0.0 for value in passed],
            "evaluation_source": ["deterministic", "deterministic"],
            "generation_status": ["success", "success"],
            "generation_seconds": [1.0, 2.0],
            "generation_total_tokens": [100, 120],
            "generation_cost_usd": costs,
            "generation_cost_source": ["configured", "configured"],
            "output_is_json": [True, True],
            "json_missing_fields": ["[]", "[]"],
            "json_mismatched_fields": [
                "[]" if passed[0] else '["answer"]',
                "[]" if passed[1] else '["answer"]',
            ],
        }
    ).to_csv(output, index=False)


def test_summary_discovers_new_completed_models_without_code_changes(
    tmp_path: Path,
) -> None:
    _registry(tmp_path)
    _write_result(
        tmp_path,
        model_id="model_a",
        passed=[True, False],
        costs=[0.01, 0.02],
    )
    _write_result(
        tmp_path,
        model_id="model_c",
        passed=[True, True],
        costs=[0.03, 0.04],
    )
    registry = load_registry(tmp_path)
    benchmark = registry.resolve_benchmark("synthetic@1.0.0")

    first = build_benchmark_experiment_summary(
        registry=registry,
        benchmark=benchmark,
    )
    assert first["summary"]["model_name"].tolist() == [
        "model_c",
        "model_a",
    ]
    assert first["comparison_status"] == "draft_experimental"
    assert len(first["comparisons"]) == 1
    model_a = first["summary"].query("model_name == 'model_a'").iloc[0]
    assert model_a["generation_cost_usd"] == pytest.approx(0.03)
    assert model_a["generation_average_cost_usd"] == pytest.approx(0.015)
    assert model_a["generation_cost_coverage"] == 1.0
    assert set(first["fields"]["field"]) == {"answer"}
    assert set(first["outcomes"]["outcome"]) == {
        "all_passed",
        "disagreement",
    }

    _write_result(
        tmp_path,
        model_id="model_b",
        passed=[False, False],
        costs=[0.005, 0.005],
    )
    second = build_benchmark_experiment_summary(
        registry=registry,
        benchmark=benchmark,
    )
    assert set(second["summary"]["model_name"]) == {
        "model_a",
        "model_b",
        "model_c",
    }
    assert len(second["comparisons"]) == 3
    metadata = json.loads(
        second["paths"]["metadata"].read_text(encoding="utf-8")
    )
    assert metadata["model_count"] == 3
    assert metadata["comparison_status"] == "draft_experimental"
    report = second["paths"]["report"].read_text(encoding="utf-8")
    assert "Current candidate access was not confirmed" in report
    assert "Rerunning the summary rebuilds every table" in report
    assert "experimental and unranked" in report


def test_requested_model_requires_a_completed_compatible_evaluation(
    tmp_path: Path,
) -> None:
    _registry(tmp_path)
    _write_result(
        tmp_path,
        model_id="model_a",
        passed=[True, True],
        costs=[0.01, 0.01],
    )
    registry = load_registry(tmp_path)
    benchmark = registry.resolve_benchmark("synthetic@1.0.0")
    with pytest.raises(ValueError, match="model_b"):
        build_benchmark_experiment_summary(
            registry=registry,
            benchmark=benchmark,
            model_ids=["model_b"],
        )

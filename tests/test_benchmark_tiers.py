from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from evalanche.benchmark_campaign import (
    build_tier_campaign_preflight,
    compatible_model_ids,
    execute_tier_campaign,
    select_models_for_promotion,
)
import evalanche.benchmark_campaign as campaign_module
from evalanche.benchmark_experiment import build_benchmark_experiment_summary
from evalanche.benchmark_runner import build_run_plan
from evalanche.benchmark_tiers import materialize_tier_cohort
from evalanche.registry import load_registry, sha256_file, validate_registry


def _write_yaml(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(value, sort_keys=False),
        encoding="utf-8",
    )


def _build_registry(tmp_path: Path) -> None:
    records = []
    for group_index in range(6):
        for language in ("en", "fr"):
            records.append(
                {
                    "case_id": f"case_{group_index}_{language}",
                    "group_id": f"group_{group_index}",
                    "language": language,
                    "split": "heldout" if group_index >= 4 else "development",
                    "stratum": "complex" if group_index % 2 else "simple",
                    "input": f"input {group_index} {language}",
                    "expected_output": '{"answer":"ok"}',
                    "evaluation_type": "json",
                }
            )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(records).to_csv(data_dir / "cases.csv", index=False)
    (data_dir / "manifest.yaml").write_text(
        "schema_version: test\n",
        encoding="utf-8",
    )

    endpoints = []
    for index, model_id in enumerate(("model_a", "model_b"), start=1):
        route = f"test/{model_id}"
        pricing_id = f"price_{model_id}"
        endpoints.append(
            {
                "pricing_id": pricing_id,
                "model": route,
                "input_per_million_tokens": float(index),
                "output_per_million_tokens": float(index * 2),
                "effective_date": "2026-08-05",
                "source": "synthetic test price",
            }
        )
        _write_yaml(
            tmp_path / "configs/models" / f"{model_id}.yaml",
            {
                "schema_version": "1.0",
                "model_id": model_id,
                "display_name": model_id,
                "provider_route": route,
                "pricing_catalog_path": "configs/pricing.yaml",
                "pricing_id": pricing_id,
                "capabilities": ["json"],
            },
        )
    _write_yaml(
        tmp_path / "configs/pricing.yaml",
        {
            "schema_version": "1.0",
            "catalog_version": "test",
            "endpoints": endpoints,
        },
    )
    _write_yaml(
        tmp_path / "configs/benchmarks/synthetic.yaml",
        {
            "schema_version": "1.0",
            "benchmark_id": "synthetic",
            "version": "1.0.0",
            "title": "Synthetic tier benchmark",
            "description": "Tier integration test.",
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
            "slice_columns": ["language", "split", "stratum"],
            "group_key": "group_id",
            "required_capabilities": ["json"],
            "tiers": {
                "smoke": {
                    "description": "One group.",
                    "sampling": {
                        "method": "balanced",
                        "unit": "group",
                        "count": 1,
                        "seed": 7,
                        "stratify_by": ["language", "split", "stratum"],
                    },
                    "cost": {
                        "initial_prompt_tokens": 1000,
                        "initial_completion_tokens": 100,
                        "safety_multiplier": 2.0,
                        "request_ceiling_multiplier": 2.0,
                    },
                },
                "screen": {
                    "description": "Three cumulative groups.",
                    "inherits": "smoke",
                    "sampling": {
                        "method": "balanced",
                        "unit": "group",
                        "count": 3,
                        "seed": 7,
                        "stratify_by": ["language", "split", "stratum"],
                    },
                    "cost": {
                        "sample_from": "smoke",
                        "initial_prompt_tokens": 1000,
                        "initial_completion_tokens": 100,
                        "safety_multiplier": 1.5,
                        "request_ceiling_multiplier": 2.0,
                    },
                    "promotion": {
                        "minimum_pass_rate": 0.75,
                        "maximum_generation_failure_rate": 0.0,
                    },
                },
                "standard": {
                    "description": "All groups.",
                    "inherits": "screen",
                    "sampling": {"method": "all", "unit": "group"},
                    "cost": {
                        "sample_from": "screen",
                        "initial_prompt_tokens": 1000,
                        "initial_completion_tokens": 100,
                        "safety_multiplier": 1.25,
                        "request_ceiling_multiplier": 1.5,
                    },
                },
            },
        },
    )


def _write_tier_result(
    tmp_path: Path,
    *,
    model_id: str,
    tier_name: str | None,
    passed: bool,
) -> None:
    registry = load_registry(tmp_path)
    benchmark = registry.resolve_benchmark("synthetic@1.0.0")
    model = registry.resolve_model(model_id)
    plan = build_run_plan(
        registry=registry,
        benchmark=benchmark,
        model=model,
        tier_name=tier_name,
    )
    input_path = tmp_path / plan["paths"]["generation_config"]
    generation_config = yaml.safe_load(input_path.read_text(encoding="utf-8"))
    cases = pd.read_csv(tmp_path / generation_config["run"]["input_path"])
    generation = cases.copy()
    generation["model_name"] = model_id
    generation["candidate_model"] = model.provider_route
    generation["model_output"] = '{"answer":"ok"}'
    generation["generation_status"] = "success"
    generation["generation_prompt_tokens"] = 1000
    generation["generation_completion_tokens"] = 100
    generation["generation_total_tokens"] = 1100
    generation["generation_cost_usd"] = 0.001
    generation["generation_configured_cost_usd"] = 0.001
    generation["generation_budget_cost_usd"] = 0.001
    generation["generation_seconds"] = 1.0
    output_path = tmp_path / plan["paths"]["generation_output"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generation.to_csv(output_path, index=False)

    evaluation = generation.copy()
    evaluation["final_passed"] = passed
    evaluation["final_score"] = 1.0 if passed else 0.0
    evaluation["evaluation_source"] = "deterministic"
    evaluation["output_is_json"] = True
    evaluation["json_missing_fields"] = "[]"
    evaluation["json_mismatched_fields"] = "[]" if passed else '["answer"]'
    evaluation_path = tmp_path / plan["paths"]["evaluation_output"]
    evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation.to_csv(evaluation_path, index=False)


def test_tiers_are_nested_deterministic_and_incremental(tmp_path: Path) -> None:
    _build_registry(tmp_path)
    registry = load_registry(tmp_path)
    benchmark = registry.resolve_benchmark("synthetic@1.0.0")

    smoke = materialize_tier_cohort(
        registry=registry, benchmark=benchmark, tier_name="smoke"
    )
    screen = materialize_tier_cohort(
        registry=registry, benchmark=benchmark, tier_name="screen"
    )
    standard = materialize_tier_cohort(
        registry=registry, benchmark=benchmark, tier_name="standard"
    )

    assert [
        len(smoke["cohort"].cumulative_cases),
        len(screen["cohort"].cumulative_cases),
        len(standard["cohort"].cumulative_cases),
    ] == [2, 6, 12]
    assert [
        len(smoke["cohort"].execution_cases),
        len(screen["cohort"].execution_cases),
        len(standard["cohort"].execution_cases),
    ] == [2, 4, 6]
    smoke_ids = set(smoke["cohort"].cumulative_cases["case_id"])
    screen_ids = set(screen["cohort"].cumulative_cases["case_id"])
    standard_ids = set(standard["cohort"].cumulative_cases["case_id"])
    assert smoke_ids < screen_ids < standard_ids

    first_hash = sha256_file(screen["cohort_path"])
    second = materialize_tier_cohort(
        registry=registry, benchmark=benchmark, tier_name="screen"
    )
    assert sha256_file(second["cohort_path"]) == first_hash
    assert validate_registry(registry)["valid"] is True


def test_tier_summary_combines_fragments_and_accepts_legacy_full_run(
    tmp_path: Path,
) -> None:
    _build_registry(tmp_path)
    for tier_name in ("smoke", "screen", "standard"):
        _write_tier_result(
            tmp_path,
            model_id="model_a",
            tier_name=tier_name,
            passed=True,
        )
    _write_tier_result(
        tmp_path,
        model_id="model_b",
        tier_name=None,
        passed=False,
    )
    registry = load_registry(tmp_path)
    result = build_benchmark_experiment_summary(
        registry=registry,
        benchmark=registry.resolve_benchmark("synthetic@1.0.0"),
        tier_name="standard",
    )

    assert set(result["summary"]["model_name"]) == {"model_a", "model_b"}
    assert set(result["summary"]["cases"]) == {12}
    assert len(result["results"]) == 24
    assert not result["results"].duplicated(["model_name", "case_id"]).any()
    assert result["paths"]["report"].parent.name == "standard"


def test_campaign_preflight_is_aggregate_budgeted_and_registry_driven(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _build_registry(tmp_path)
    monkeypatch.chdir(tmp_path)
    registry = load_registry(tmp_path)
    benchmark = registry.resolve_benchmark("synthetic@1.0.0")

    result = build_tier_campaign_preflight(
        registry=registry,
        benchmark=benchmark,
        tier_name="smoke",
        model_ids=compatible_model_ids(
            registry=registry,
            benchmark=benchmark,
        ),
        maximum_cost_usd=0.01,
    )

    assert result["pending_requests"] == 4
    assert result["projected_budgeted_total_usd"] == pytest.approx(0.0144)
    assert result["within_budget"] is False
    assert {row["model_id"] for row in result["rows"]} == {
        "model_a",
        "model_b",
    }
    assert all(
        row["cost_sample_source"] == "initial_token_estimate"
        for row in result["rows"]
    )

    with pytest.raises(ValueError, match="completed 'smoke' results"):
        build_tier_campaign_preflight(
            registry=registry,
            benchmark=benchmark,
            tier_name="screen",
            model_ids=["model_a"],
        )

    _write_tier_result(
        tmp_path,
        model_id="model_a",
        tier_name="smoke",
        passed=True,
    )
    screen = build_tier_campaign_preflight(
        registry=load_registry(tmp_path),
        benchmark=load_registry(tmp_path).resolve_benchmark(
            "synthetic@1.0.0"
        ),
        tier_name="screen",
        model_ids=["model_a"],
    )
    assert screen["pending_requests"] == 4
    assert screen["rows"][0]["cost_sample_source"] == (
        "observed_smoke_tier"
    )


def test_campaign_identity_includes_access_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _build_registry(tmp_path)
    monkeypatch.chdir(tmp_path)
    registry = load_registry(tmp_path)
    benchmark = registry.resolve_benchmark("synthetic@1.0.0")

    first = build_tier_campaign_preflight(
        registry=registry,
        benchmark=benchmark,
        tier_name="smoke",
        model_ids=["model_a"],
        access_scope={"mode": "explicit_models", "confirmed_on": "first"},
    )
    second = build_tier_campaign_preflight(
        registry=registry,
        benchmark=benchmark,
        tier_name="smoke",
        model_ids=["model_a"],
        access_scope={"mode": "explicit_models", "confirmed_on": "second"},
    )

    assert first["campaign_id"] != second["campaign_id"]


def test_promotion_rejects_an_empty_access_confirmed_scope(
    tmp_path: Path,
) -> None:
    _build_registry(tmp_path)
    registry = load_registry(tmp_path)

    with pytest.raises(ValueError, match="No access-confirmed models"):
        select_models_for_promotion(
            registry=registry,
            benchmark=registry.resolve_benchmark("synthetic@1.0.0"),
            source_tier="screen",
            model_ids=[],
        )


def test_promotion_uses_manifest_gates_for_every_discovered_model(
    tmp_path: Path,
) -> None:
    _build_registry(tmp_path)
    for model_id, passed in (("model_a", True), ("model_b", False)):
        _write_tier_result(
            tmp_path,
            model_id=model_id,
            tier_name="smoke",
            passed=passed,
        )
        _write_tier_result(
            tmp_path,
            model_id=model_id,
            tier_name="screen",
            passed=passed,
        )
    registry = load_registry(tmp_path)
    promotion = select_models_for_promotion(
        registry=registry,
        benchmark=registry.resolve_benchmark("synthetic@1.0.0"),
        source_tier="screen",
    )

    assert promotion["selected_model_ids"] == ["model_a"]
    decisions = promotion["decisions"].set_index("model_id")
    assert bool(decisions.loc["model_a", "promoted"]) is True
    assert bool(decisions.loc["model_b", "promoted"]) is False


def test_tier_campaign_executes_pending_model_and_rebuilds_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _build_registry(tmp_path)
    monkeypatch.chdir(tmp_path)
    registry = load_registry(tmp_path)
    benchmark = registry.resolve_benchmark("synthetic@1.0.0")
    preflight = build_tier_campaign_preflight(
        registry=registry,
        benchmark=benchmark,
        tier_name="smoke",
        model_ids=["model_a"],
        maximum_cost_usd=1.0,
    )

    def complete_without_provider_calls(**kwargs: Any) -> dict[str, Any]:
        model = kwargs["model"]
        assert model.model_id == "model_a"
        _write_tier_result(
            tmp_path,
            model_id="model_a",
            tier_name="smoke",
            passed=True,
        )
        plan = build_run_plan(
            registry=registry,
            benchmark=benchmark,
            model=registry.resolve_model("model_a"),
            tier_name="smoke",
        )
        return {"plan": plan}

    monkeypatch.setattr(
        campaign_module,
        "run_experimental_benchmark",
        complete_without_provider_calls,
    )
    result = execute_tier_campaign(
        registry=registry,
        preflight=preflight,
    )

    assert result["status"] == "complete"
    assert result["completed_models"] == ["model_a"]
    assert result["observed_budgeted_cost_usd"] == pytest.approx(0.004)
    assert result["summary"]["summary"]["cases"].tolist() == [2]
    campaign = yaml.safe_load(
        result["paths"]["campaign"].read_text(encoding="utf-8")
    )
    assert campaign["status"] == "complete"


def test_tier_campaign_refuses_failed_aggregate_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _build_registry(tmp_path)
    monkeypatch.chdir(tmp_path)
    registry = load_registry(tmp_path)
    preflight = build_tier_campaign_preflight(
        registry=registry,
        benchmark=registry.resolve_benchmark("synthetic@1.0.0"),
        tier_name="smoke",
        model_ids=["model_a", "model_b"],
        maximum_cost_usd=0.001,
    )

    with pytest.raises(ValueError, match="Aggregate cost preflight failed"):
        execute_tier_campaign(registry=registry, preflight=preflight)

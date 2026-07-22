from pathlib import Path

import pandas as pd

from evalanche.config import (
    CriterionConfig,
    EvalConfig,
    JudgeConfig,
    RunConfig,
    ScoringConfig,
    TaskConfig,
)
from evalanche.metadata import build_run_metadata


def make_config(tmp_path: Path) -> EvalConfig:
    return EvalConfig(
        run=RunConfig(
            name="metadata-test",
            input_path=tmp_path / "input.csv",
            output_path=tmp_path / "results.csv",
        ),
        judge=JudgeConfig(model="azure/test-judge"),
        task=TaskConfig(name="test", description="Test task"),
        scoring=ScoringConfig(),
        criteria=[
            CriterionConfig(
                name="correctness",
                weight=1,
                description="Correctness",
            )
        ],
    )


def build_metadata(
    tmp_path: Path,
    config: EvalConfig,
    results: pd.DataFrame,
) -> dict[str, object]:
    config_path = tmp_path / "config.yaml"
    summary_path = tmp_path / "summary.csv"
    config_path.write_text("test config", encoding="utf-8")
    config.run.input_path.write_text("input", encoding="utf-8")
    config.run.output_path.write_text("results", encoding="utf-8")
    summary_path.write_text("summary", encoding="utf-8")
    return build_run_metadata(
        config_path=config_path,
        config=config,
        cases=results,
        results=results,
        case_results_path=config.run.output_path,
        model_summary_path=summary_path,
    )


def test_standalone_judge_metadata_keeps_cost_sources_separate(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    results = pd.DataFrame(
        [
            {
                "case_id": "case_001",
                "model_name": "model_a",
                "weighted_score": 0.8,
                "passed": True,
                "total_tokens": 30,
                "cost_usd": 0.001,
                "cost_source": "configured_endpoint_pricing",
                "configured_cost_usd": 0.001,
                "provider_reported_cost_usd": 0.0012,
            },
            {
                "case_id": "case_002",
                "model_name": "model_a",
                "weighted_score": 0.9,
                "passed": True,
                "total_tokens": 40,
                "cost_usd": 0.002,
                "cost_source": "configured_endpoint_pricing",
                "configured_cost_usd": 0.002,
                "provider_reported_cost_usd": None,
            },
        ]
    )

    metadata = build_metadata(tmp_path, config, results)
    summary = metadata["results"]

    assert summary["total_tokens"] == 70
    assert summary["cost_usd"] == 0.003
    assert summary["cost_coverage"] == 1.0
    assert summary["configured_cost_usd"] == 0.003
    assert summary["provider_reported_cost_usd"] is None
    assert summary["provider_reported_cost_coverage"] == 0.5


def test_standalone_judge_metadata_snapshots_resolved_price(
    tmp_path: Path,
) -> None:
    pricing_path = tmp_path / "pricing.yaml"
    pricing_path.write_text(
        "schema_version: '1.0'\n"
        "catalog_version: '2026-07-01'\n"
        "endpoints:\n"
        "  - pricing_id: judge_global\n"
        "    model: azure/test-judge\n"
        "    input_per_million_tokens: 1.0\n"
        "    output_per_million_tokens: 4.0\n"
        "    effective_date: 2026-07-01\n"
        "    source: test rate card\n",
        encoding="utf-8",
    )
    config = make_config(tmp_path)
    config.endpoint_pricing_path = pricing_path
    config.judge.pricing_id = "judge_global"
    results = pd.DataFrame(
        [
            {
                "case_id": "case_001",
                "model_name": "model_a",
                "weighted_score": 0.8,
                "passed": True,
            }
        ]
    )

    metadata = build_metadata(tmp_path, config, results)
    pricing = metadata["endpoint_pricing"]

    assert metadata["schema_version"] == "0.2"
    assert len(pricing["catalog_sha256"]) == 64
    assert pricing["resolved_endpoints"][0]["pricing_id"] == (
        "judge_global"
    )
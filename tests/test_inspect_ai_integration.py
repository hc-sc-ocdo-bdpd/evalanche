from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from integrations.inspect_ai import DEFAULT_BENCHMARK_REFERENCE
from integrations.inspect_ai.adapter import (
    _load_published_case_scores,
    _validate_historical_source_frame,
    build_parity_report,
    configure_inspect_azure_environment,
    inspect_model_route,
    load_dpd_cases,
    load_dpd_context,
    load_dpd_replay_records,
)

ROOT = Path(__file__).resolve().parents[1]


def test_dpd_scopes_preserve_frozen_release_contract() -> None:
    context = load_dpd_context(ROOT, DEFAULT_BENCHMARK_REFERENCE)

    assert context.benchmark.status == "frozen"
    assert len(context.fingerprint["compatibility_sha256"]) == 64
    assert len(load_dpd_cases(context, "demo")) == 24
    assert len(load_dpd_cases(context, "audit")) == 105
    assert len(load_dpd_cases(context, "full")) == 14_034
    assert len(load_dpd_replay_records(context)) == 420


def test_saved_output_parity_is_exact() -> None:
    report = build_parity_report(load_dpd_context(ROOT))

    assert report["status"] == "passed"
    assert report["review_case_count"] == 105
    assert report["saved_output_count"] == 420
    assert report["strict_disagreements"] == 0
    assert report["field_score_disagreements"] == 0
    assert report["mismatched_field_disagreements"] == 0
    assert report["issues"] == []


def test_azure_route_and_environment_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AZURE_API_KEY", "secret")
    monkeypatch.setenv("AZURE_API_BASE", "https://example.invalid")
    monkeypatch.setenv("AZURE_API_VERSION", "2026-01-01")
    for name in (
        "AZUREAI_OPENAI_API_KEY",
        "AZUREAI_OPENAI_BASE_URL",
        "AZUREAI_OPENAI_API_VERSION",
    ):
        monkeypatch.delenv(name, raising=False)

    present = configure_inspect_azure_environment()

    assert all(present.values())
    assert inspect_model_route("azure/deployment") == (
        "openai/azure/deployment"
    )
    with pytest.raises(ValueError, match="azure/<deployment>"):
        inspect_model_route("unsupported/model")


def test_inspect_native_replay_and_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("inspect_ai")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    from inspect_ai import eval as inspect_eval

    from integrations.inspect_ai.dpd_task import dpd_replay
    from integrations.inspect_ai.reporting import write_campaign_report

    logs = inspect_eval(
        dpd_replay(root=str(ROOT)),
        model="mockllm/model",
        log_dir=str(tmp_path / "logs"),
        limit=2,
        display="none",
    )
    assert [str(log.status) for log in logs] == ["success"]

    summary = write_campaign_report(
        tmp_path / "logs", tmp_path / "report"
    )
    assert summary["status"] == "complete"
    assert summary["sample_count"] == 2
    assert (tmp_path / "report/case_results.csv.gz").is_file()
    saved = json.loads((tmp_path / "report/summary.json").read_text())
    assert saved["sample_count"] == 2


def test_historical_source_membership_and_frozen_inputs_are_strict() -> None:
    cases = pd.DataFrame(
        [
            {
                "case_id": "case-1",
                "input": "input 1",
                "expected_output": '{"value":1}',
                "evaluation_type": "json",
                "product_id": "product-1",
                "language": "en",
                "stratum": "single_ingredient",
            },
            {
                "case_id": "case-2",
                "input": "input 2",
                "expected_output": '{"value":2}',
                "evaluation_type": "json",
                "product_id": "product-2",
                "language": "fr",
                "stratum": "multi_variant",
            },
        ]
    )
    rows = []
    for model_id in ("model-a", "model-b"):
        for row in cases.to_dict(orient="records"):
            rows.append(
                {
                    **row,
                    "model_name": model_id,
                    "model_output": row["expected_output"],
                    "generation_status": "success",
                }
            )
    frame = pd.DataFrame(rows)

    validated = _validate_historical_source_frame(
        frame,
        cases,
        ("model-a", "model-b"),
        source=Path("historical.csv"),
    )
    assert len(validated) == 4

    tampered = frame.copy()
    tampered.loc[0, "input"] = "changed"
    with pytest.raises(ValueError, match="frozen census"):
        _validate_historical_source_frame(
            tampered,
            cases,
            ("model-a", "model-b"),
            source=Path("historical.csv"),
        )


def test_published_score_bundles_are_hash_checked() -> None:
    context = load_dpd_context(ROOT)
    expected_runs = {
        "gpt_5_4_mini": "21a2133d90f17f63",
        "gpt_5_6_luna": "28add3264b58f7e7",
        "gpt_5_6_terra": "4262a9bcd13c8ecd",
        "gpt_5_6_sol": "b127e027b3f34940",
    }

    for model_id, run_id in expected_runs.items():
        scores, run_info = _load_published_case_scores(context, model_id)
        assert len(scores) == 14_034
        assert not scores["case_id"].duplicated().any()
        assert run_info["run_id"] == run_id
        assert len(run_info["case_scores_sha256"]) == 64


def test_inspect_native_historical_import_and_parity_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("inspect_ai")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    from inspect_ai import eval as inspect_eval

    from integrations.inspect_ai.dpd_task import build_dpd_historical_task
    from integrations.inspect_ai.reporting import write_campaign_report

    context = load_dpd_context(ROOT)
    model_id = "gpt_5_6_sol"
    records = []
    for row in load_dpd_cases(context, "full").head(2).to_dict(
        orient="records"
    ):
        records.append(
            {
                **row,
                "model_output": row["expected_output"],
                "historical_model_id": model_id,
                "historical_run_id": "test-run",
                "expected_metric_passed": True,
                "expected_output_is_json": True,
                "expected_field_score": 1.0,
                "expected_mismatched_fields": [],
                "generation_seconds": "1.5",
                "generation_prompt_tokens": "10",
                "generation_completion_tokens": "5",
                "generation_total_tokens": "15",
                "generation_cost_usd": "0.01",
            }
        )
    task = build_dpd_historical_task(
        context=context,
        records=records,
        model_id=model_id,
        source_path="historical.csv",
        source_sha256="a" * 64,
        release_id="test-release",
        run_info={"run_id": "test-run", "published_compatibility": {}},
    )
    logs = inspect_eval(
        task,
        model="mockllm/model",
        log_dir=str(tmp_path / "historical-logs"),
        display="none",
    )
    assert [str(log.status) for log in logs] == ["success"]

    summary = write_campaign_report(
        tmp_path / "historical-logs", tmp_path / "historical-report"
    )
    assert summary["status"] == "complete"
    assert summary["sample_count"] == 2
    assert summary["execution_modes"] == ["historical_import"]
    assert summary["parity"] == {
        "status": "passed",
        "checked_samples": 2,
        "disagreements": 0,
        "strict_disagreements": 0,
        "field_score_disagreements": 0,
        "mismatched_field_disagreements": 0,
        "valid_json_disagreements": 0,
    }
    assert summary["models"][0]["model_id"] == model_id
    assert summary["models"][0]["total_tokens"] == 30
    assert summary["models"][0]["generation_cost_usd"] == pytest.approx(
        0.02
    )

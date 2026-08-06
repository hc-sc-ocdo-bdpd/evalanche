from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import evalanche.generation as generation_module
from evalanche.config import (
    GenerationConfig,
    GenerationSettingsConfig,
    RunConfig,
)
from evalanche.generation import (
    _candidate_fingerprint_payload,
    _request_cost_ceiling,
    generate_outputs,
    preflight_generation,
)


def _write_run_files(
    tmp_path: Path,
    *,
    case_count: int,
    input_rate: float = 1.0,
    output_rate: float = 4.0,
    sample_prompt_tokens: int = 10,
    sample_completion_tokens: int = 2,
    maximum_cost: float = 10.0,
    safety_multiplier: float = 1.5,
    max_retries: int = 1,
) -> tuple[GenerationConfig, Path, Path]:
    input_path = tmp_path / "cases.csv"
    output_path = tmp_path / "outputs.csv"
    candidates_path = tmp_path / "candidates.yaml"
    pricing_path = tmp_path / "pricing.yaml"
    sample_path = tmp_path / "sample.csv"
    config_path = tmp_path / "generate.yaml"

    pd.DataFrame(
        [
            {
                "case_id": f"case_{index:03d}",
                "input": f"input {index}",
                "expected_output": f"expected {index}",
                "evaluation_type": "exact",
            }
            for index in range(1, case_count + 1)
        ]
    ).to_csv(input_path, index=False)
    candidates_path.write_text(
        "models:\n"
        "  - name: model_a\n"
        "    model: azure/model-a\n"
        f"    max_retries: {max_retries}\n"
        "    max_completion_tokens: 10\n",
        encoding="utf-8",
    )
    pricing_path.write_text(
        "schema_version: '1.0'\n"
        "catalog_version: test-1\n"
        "endpoints:\n"
        "  - pricing_id: model_a_global\n"
        "    model: azure/model-a\n"
        "    currency: USD\n"
        f"    input_per_million_tokens: {input_rate}\n"
        f"    output_per_million_tokens: {output_rate}\n"
        "    effective_date: 2026-07-27\n"
        "    source: test rate\n",
        encoding="utf-8",
    )
    candidates_path.write_text(
        candidates_path.read_text(encoding="utf-8")
        + "    pricing_id: model_a_global\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "model_name": "model_a",
                "candidate_model": "azure/model-a",
                "generation_status": "success",
                "generation_prompt_tokens": sample_prompt_tokens,
                "generation_completion_tokens": (
                    sample_completion_tokens
                ),
            }
        ]
    ).to_csv(sample_path, index=False)
    config_path.write_text("test generation config\n", encoding="utf-8")

    config = GenerationConfig(
        run=RunConfig(
            name="resume-test",
            input_path=input_path,
            output_path=output_path,
        ),
        candidate_models_path=candidates_path,
        endpoint_pricing_path=pricing_path,
        generation=GenerationSettingsConfig(
            continue_on_error=True,
            max_completion_tokens=10,
            max_workers=1,
            checkpoint_every=1,
            resume=True,
            cost_preflight_sample_path=sample_path,
            maximum_estimated_cost_usd=maximum_cost,
            cost_safety_multiplier=safety_multiplier,
        ),
    )
    return config, config_path, output_path


def _successful_record(
    *,
    candidate: Any,
    row: dict[str, Any],
    configured_cost: float = 0.0001,
) -> dict[str, Any]:
    record = dict(row)
    record.update(
        {
            "model_name": candidate.name,
            "candidate_model": candidate.model,
            "model_output": str(row["expected_output"]),
            "generation_status": "success",
            "generation_error": "",
            "generated_at_utc": "2026-07-27T00:00:00+00:00",
            "generation_seconds": 0.01,
            "generation_api_seconds": 0.01,
            "generation_attempts": 1,
            "generation_failed_attempts": 0,
            "generation_prompt_tokens": 10,
            "generation_cached_prompt_tokens": 0,
            "generation_completion_tokens": 2,
            "generation_total_tokens": 12,
            "generation_cost_usd": configured_cost,
            "generation_cost_source": (
                "configured_endpoint_pricing"
            ),
            "generation_configured_cost_usd": configured_cost,
            "generation_provider_reported_cost_usd": None,
            "generation_pricing_id": "model_a_global",
            "generation_pricing_model": candidate.model,
            "generation_pricing_currency": "USD",
            "generation_pricing_input_per_million_tokens": 1.0,
            "generation_pricing_cached_input_per_million_tokens": None,
            "generation_pricing_output_per_million_tokens": 4.0,
            "generation_pricing_effective_date": "2026-07-27",
            "generation_pricing_source": "test rate",
        }
    )
    return record


def test_legacy_candidate_fingerprint_omits_new_null_reasoning_field() -> None:
    candidate = generation_module.CandidateModelConfig(
        name="legacy",
        model="azure/legacy",
    )

    payload = _candidate_fingerprint_payload(candidate)

    assert "reasoning_effort" not in payload
    assert payload["temperature"] == 0


def test_explicit_request_ceiling_supports_native_file_budget_reserve() -> None:
    config = GenerationConfig(
        run=RunConfig(
            name="native-file",
            input_path=Path("input.csv"),
            output_path=Path("output.csv"),
        ),
        candidate_models_path=Path("models.yaml"),
        generation=GenerationSettingsConfig(
            request_api="responses",
            request_cost_ceiling_usd=0.25,
        ),
    )
    candidate = generation_module.CandidateModelConfig(
        name="model_a",
        model="azure/model-a",
    )

    assert _request_cost_ceiling(
        config=config,
        candidate=candidate,
        row={"input_files": '[{"path":"document.pdf"}]'},
        endpoint_price=None,
    ) == pytest.approx(0.25)


def test_interrupted_run_resumes_without_repeating_checkpointed_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, config_path, output_path = _write_run_files(
        tmp_path,
        case_count=3,
    )
    first_calls: list[str] = []

    def interrupt_on_second(**kwargs: Any) -> dict[str, Any]:
        row = kwargs["row"]
        first_calls.append(str(row["case_id"]))
        if row["case_id"] == "case_002":
            raise KeyboardInterrupt
        return _successful_record(
            candidate=kwargs["candidate"],
            row=row,
        )

    monkeypatch.setattr(
        generation_module,
        "generate_one",
        interrupt_on_second,
    )
    with pytest.raises(KeyboardInterrupt):
        generate_outputs(config=config, config_path=config_path)

    checkpoint_path = output_path.with_name(
        output_path.name + ".checkpoint.jsonl"
    )
    assert first_calls == ["case_001", "case_002"]
    assert len(checkpoint_path.read_text(encoding="utf-8").splitlines()) == 1
    assert list(pd.read_csv(output_path)["case_id"]) == ["case_001"]

    resumed_calls: list[str] = []

    def finish_remaining(**kwargs: Any) -> dict[str, Any]:
        row = kwargs["row"]
        resumed_calls.append(str(row["case_id"]))
        return _successful_record(
            candidate=kwargs["candidate"],
            row=row,
        )

    monkeypatch.setattr(
        generation_module,
        "generate_one",
        finish_remaining,
    )
    outputs, _, metadata_path = generate_outputs(
        config=config,
        config_path=config_path,
    )

    assert resumed_calls == ["case_002", "case_003"]
    assert list(outputs["case_id"]) == [
        "case_001",
        "case_002",
        "case_003",
    ]
    assert outputs.attrs["run_status"] == "completed"
    assert outputs.attrs["resumed_output_count"] == 1
    assert len(checkpoint_path.read_text(encoding="utf-8").splitlines()) == 3

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["execution"]["status"] == "completed"
    assert metadata["execution"]["resumed_output_count"] == 1
    assert metadata["execution"]["pending_output_count"] == 0

    resumed_calls.clear()
    generate_outputs(config=config, config_path=config_path)
    assert resumed_calls == []


def test_cost_preflight_aborts_before_first_model_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, config_path, output_path = _write_run_files(
        tmp_path,
        case_count=2,
        sample_prompt_tokens=1_000_000,
        sample_completion_tokens=1_000_000,
        maximum_cost=1.0,
        safety_multiplier=1.0,
    )
    calls = 0

    def should_not_run(**kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        raise AssertionError("model call should not occur")

    monkeypatch.setattr(
        generation_module,
        "generate_one",
        should_not_run,
    )

    with pytest.raises(ValueError, match="cost preflight failed"):
        generate_outputs(config=config, config_path=config_path)

    assert calls == 0
    assert not output_path.exists()


def test_preflight_only_is_read_only_and_makes_no_model_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _, output_path = _write_run_files(
        tmp_path,
        case_count=3,
    )
    calls = 0

    def should_not_run(**kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        raise AssertionError("model call should not occur")

    monkeypatch.setattr(
        generation_module,
        "generate_one",
        should_not_run,
    )
    preflight = preflight_generation(config=config)

    assert calls == 0
    assert preflight["expected_output_count"] == 3
    assert preflight["checkpointed_output_count"] == 0
    assert preflight["pending_output_count"] == 3
    assert preflight["within_limit"] is True
    assert not output_path.exists()
    assert not output_path.with_name(
        output_path.name + ".checkpoint.jsonl"
    ).exists()


def test_runtime_cost_guard_stops_and_preserves_partial_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, config_path, output_path = _write_run_files(
        tmp_path,
        case_count=5,
        input_rate=1.0,
        output_rate=100.0,
        sample_prompt_tokens=10,
        sample_completion_tokens=1,
        maximum_cost=0.003,
        safety_multiplier=1.0,
        max_retries=1,
    )
    calls: list[str] = []

    def expensive_generation(**kwargs: Any) -> dict[str, Any]:
        row = kwargs["row"]
        calls.append(str(row["case_id"]))
        return _successful_record(
            candidate=kwargs["candidate"],
            row=row,
            configured_cost=0.00101,
        )

    monkeypatch.setattr(
        generation_module,
        "generate_one",
        expensive_generation,
    )

    outputs, _, metadata_path = generate_outputs(
        config=config,
        config_path=config_path,
    )

    assert calls == ["case_001", "case_002"]
    assert len(outputs) == 2
    assert outputs.attrs["run_status"] == "budget_exhausted"
    assert list(pd.read_csv(output_path)["case_id"]) == [
        "case_001",
        "case_002",
    ]

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["execution"]["status"] == "budget_exhausted"
    assert metadata["execution"]["completed_output_count"] == 2
    assert metadata["execution"]["pending_output_count"] == 3
    assert metadata["outputs"]["budgeted_cost_usd"] == pytest.approx(
        0.00202
    )


def test_resume_rejects_changed_azure_route_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, config_path, _ = _write_run_files(
        tmp_path,
        case_count=1,
    )
    monkeypatch.setenv(
        "AZURE_API_BASE",
        "https://first-resource.example/",
    )
    monkeypatch.setenv("AZURE_API_VERSION", "2024-12-01-preview")
    monkeypatch.setattr(
        generation_module,
        "generate_one",
        lambda **kwargs: _successful_record(
            candidate=kwargs["candidate"],
            row=kwargs["row"],
        ),
    )
    generate_outputs(config=config, config_path=config_path)

    monkeypatch.setenv(
        "AZURE_API_BASE",
        "https://second-resource.example/",
    )
    with pytest.raises(ValueError, match="fingerprint"):
        preflight_generation(config=config)


def test_resume_rejects_duplicate_checkpoint_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, config_path, output_path = _write_run_files(
        tmp_path,
        case_count=1,
    )
    monkeypatch.setattr(
        generation_module,
        "generate_one",
        lambda **kwargs: _successful_record(
            candidate=kwargs["candidate"],
            row=kwargs["row"],
        ),
    )
    generate_outputs(config=config, config_path=config_path)
    checkpoint_path = output_path.with_name(
        output_path.name + ".checkpoint.jsonl"
    )
    line = checkpoint_path.read_text(encoding="utf-8")
    checkpoint_path.write_text(line + line, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate case/model"):
        preflight_generation(config=config)

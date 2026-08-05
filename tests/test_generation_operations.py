from datetime import date
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import evalanche.generation as generation_module
from evalanche.config import (
    CandidateModelConfig,
    GenerationConfig,
    RunConfig,
)
from evalanche.generation import generate_one, generate_outputs
from evalanche.pricing import EndpointPriceConfig


def test_generate_one_preserves_operational_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    class StubClient:
        def __init__(self, **kwargs: Any) -> None:
            seen.update(kwargs)

        def complete_text(
            self,
            messages: list[dict[str, str]],
            max_completion_tokens: int | None = None,
        ) -> dict[str, Any]:
            return {
                "content": "negative",
                "usage": {
                    "prompt_tokens": 10,
                    "cached_prompt_tokens": 1,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                },
                "operational": {
                    "latency_seconds": 1.25,
                    "api_seconds": 1.0,
                    "attempts": 2,
                    "failed_attempts": 1,
                    "cost_usd": 0.003,
                    "cost_source": "litellm_response_metadata",
                    "configured_cost_usd": 0.0025,
                    "provider_reported_cost_usd": 0.003,
                    "pricing_id": "model_a_global",
                    "pricing_model": "azure/model-a",
                    "pricing_currency": "USD",
                    "pricing_input_per_million_tokens": 1.0,
                    "pricing_cached_input_per_million_tokens": 0.25,
                    "pricing_output_per_million_tokens": 4.0,
                    "pricing_effective_date": "2026-07-01",
                    "pricing_source": "test rate card",
                },
            }

    monkeypatch.setattr(
        generation_module,
        "LLMClient",
        StubClient,
    )
    config = GenerationConfig(
        run=RunConfig(
            name="test",
            input_path=Path("input.csv"),
            output_path=Path("output.csv"),
        ),
        candidate_models_path=Path("models.yaml"),
    )
    candidate = CandidateModelConfig(
        name="model_a",
        model="azure/model-a",
        temperature=None,
        reasoning_effort="none",
    )
    endpoint_price = EndpointPriceConfig(
        pricing_id="model_a_global",
        model="azure/model-a",
        input_per_million_tokens=1.0,
        cached_input_per_million_tokens=0.25,
        output_per_million_tokens=4.0,
        effective_date=date(2026, 7, 1),
        source="test rate card",
    )

    result = generate_one(
        config=config,
        candidate=candidate,
        endpoint_price=endpoint_price,
        row={
            "case_id": "case_001",
            "input": "Return the sentiment.",
            "expected_output": "negative",
            "evaluation_type": "exact",
        },
    )

    assert result["generation_status"] == "success"
    assert result["generation_seconds"] == 1.25
    assert result["generation_attempts"] == 2
    assert result["generation_failed_attempts"] == 1
    assert result["generation_prompt_tokens"] == 10
    assert result["generation_cached_prompt_tokens"] == 1
    assert result["generation_completion_tokens"] == 2
    assert result["generation_total_tokens"] == 12
    assert result["generation_cost_usd"] == 0.003
    assert result["generation_cost_source"] == (
        "litellm_response_metadata"
    )
    assert result["generation_configured_cost_usd"] == 0.0025
    assert result["generation_provider_reported_cost_usd"] == 0.003
    assert result["generation_pricing_id"] == "model_a_global"
    assert result["generation_pricing_effective_date"] == "2026-07-01"
    assert seen["endpoint_price"] is endpoint_price
    assert seen["temperature"] is None
    assert seen["reasoning_effort"] == "none"


def test_generation_resolves_and_snapshots_explicit_endpoint_price(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "cases.csv"
    output_path = tmp_path / "outputs.csv"
    candidates_path = tmp_path / "candidates.yaml"
    pricing_path = tmp_path / "pricing.yaml"
    config_path = tmp_path / "generate.yaml"
    pd.DataFrame(
        [
            {
                "case_id": "case_001",
                "input": "Return negative.",
                "expected_output": "negative",
                "evaluation_type": "exact",
            }
        ]
    ).to_csv(input_path, index=False)
    candidates_path.write_text(
        "models:\n"
        "  - name: model_a\n"
        "    model: azure/model-a\n"
        "    provider_model_version: '2026-03-17'\n"
        "    deployment_type: Global Standard\n"
        "    resource_region: Canada East\n"
        "    pricing_id: model_a_global\n",
        encoding="utf-8",
    )
    pricing_path.write_text(
        "schema_version: '1.0'\n"
        "catalog_version: '2026-07-01'\n"
        "endpoints:\n"
        "  - pricing_id: model_a_global\n"
        "    model: azure/model-a\n"
        "    currency: USD\n"
        "    input_per_million_tokens: 1.0\n"
        "    output_per_million_tokens: 4.0\n"
        "    effective_date: 2026-07-01\n"
        "    source: test rate card\n",
        encoding="utf-8",
    )
    config_path.write_text("generation config", encoding="utf-8")

    class StubClient:
        def __init__(self, **kwargs: Any) -> None:
            self.endpoint_price = kwargs["endpoint_price"]

        def complete_text(
            self,
            messages: list[dict[str, str]],
            max_completion_tokens: int | None = None,
        ) -> dict[str, Any]:
            assert self.endpoint_price.pricing_id == "model_a_global"
            return {
                "content": "negative",
                "usage": {
                    "prompt_tokens": 10,
                    "cached_prompt_tokens": 0,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                },
                "operational": {
                    "latency_seconds": 0.1,
                    "api_seconds": 0.1,
                    "attempts": 1,
                    "failed_attempts": 0,
                    "cost_usd": 0.000018,
                    "cost_source": "configured_endpoint_pricing",
                    "configured_cost_usd": 0.000018,
                    "provider_reported_cost_usd": None,
                    "pricing_id": "model_a_global",
                    "pricing_model": "azure/model-a",
                    "pricing_currency": "USD",
                    "pricing_input_per_million_tokens": 1.0,
                    "pricing_cached_input_per_million_tokens": None,
                    "pricing_output_per_million_tokens": 4.0,
                    "pricing_effective_date": "2026-07-01",
                    "pricing_source": "test rate card",
                },
            }

    monkeypatch.setattr(generation_module, "LLMClient", StubClient)
    config = GenerationConfig(
        run=RunConfig(
            name="test",
            input_path=input_path,
            output_path=output_path,
        ),
        candidate_models_path=candidates_path,
        endpoint_pricing_path=pricing_path,
    )

    outputs, _, metadata_path = generate_outputs(
        config=config,
        config_path=config_path,
    )

    assert outputs.iloc[0]["generation_cost_source"] == (
        "configured_endpoint_pricing"
    )
    assert outputs.iloc[0]["generation_pricing_catalog_version"] == (
        "2026-07-01"
    )
    assert len(
        outputs.iloc[0]["generation_pricing_catalog_sha256"]
    ) == 64
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["schema_version"] == "0.6"
    assert metadata["execution"]["status"] == "completed"
    assert metadata["execution"]["expected_output_count"] == 1
    assert metadata["execution"]["completed_output_count"] == 1
    assert metadata["generation"]["request_api"] == "chat_completions"
    assert len(metadata["hashes"]["endpoint_pricing_sha256"]) == 64
    assert metadata["endpoint_pricing"]["catalog_version"] == (
        "2026-07-01"
    )
    assert metadata["endpoint_pricing"]["resolved_endpoints"][0][
        "pricing_id"
    ] == "model_a_global"
    assert metadata["candidate_models"][0][
        "provider_model_version"
    ] == "2026-03-17"
    assert metadata["candidate_models"][0]["deployment_type"] == (
        "Global Standard"
    )
    assert metadata["candidate_models"][0]["resource_region"] == (
        "Canada East"
    )

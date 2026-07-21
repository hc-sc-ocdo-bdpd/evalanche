from pathlib import Path
from typing import Any

import pytest

import evalanche.generation as generation_module
from evalanche.config import (
    CandidateModelConfig,
    GenerationConfig,
    RunConfig,
)
from evalanche.generation import generate_one


def test_generate_one_preserves_operational_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StubClient:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def complete_text(
            self,
            messages: list[dict[str, str]],
            max_completion_tokens: int | None = None,
        ) -> dict[str, Any]:
            return {
                "content": "negative",
                "usage": {
                    "prompt_tokens": 10,
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
    )

    result = generate_one(
        config=config,
        candidate=candidate,
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
    assert result["generation_completion_tokens"] == 2
    assert result["generation_total_tokens"] == 12
    assert result["generation_cost_usd"] == 0.003
    assert result["generation_cost_source"] == (
        "litellm_response_metadata"
    )
from datetime import date
from types import SimpleNamespace
from typing import Any

import pytest
from tenacity import wait_none

import evalanche.llm as llm_module
from evalanche.llm import (
    LLMClient,
    _parse_json_content,
    operational_from_error,
)
from evalanche.pricing import EndpointPriceConfig


def make_response(
    content: str,
    *,
    cost_usd: float | None = 0.001,
    cached_tokens: int | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            prompt_tokens_details=(
                SimpleNamespace(cached_tokens=cached_tokens)
                if cached_tokens is not None
                else None
            ),
        ),
        _hidden_params=(
            {"response_cost": cost_usd}
            if cost_usd is not None
            else {}
        ),
    )


def test_parse_json_content_accepts_code_fence() -> None:
    parsed = _parse_json_content('```json\n{"answer": 42}\n```')

    assert parsed == {"answer": 42}


def test_complete_json_retries_when_validator_rejects_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(
        [
            make_response('{"status": "invalid"}'),
            make_response('{"status": "valid"}'),
        ]
    )
    call_count = 0
    attempt_hook_count = 0

    def fake_completion(**kwargs: Any) -> SimpleNamespace:
        nonlocal call_count
        call_count += 1
        return next(responses)

    def before_attempt() -> None:
        nonlocal attempt_hook_count
        attempt_hook_count += 1

    def validator(response: dict[str, Any]) -> None:
        if response.get("status") != "valid":
            raise ValueError("invalid schema")

    monkeypatch.setattr(llm_module, "_completion", fake_completion)
    monkeypatch.setattr(
        llm_module,
        "wait_exponential",
        lambda **kwargs: wait_none(),
    )

    result = LLMClient(
        model="azure/test-model",
        max_retries=2,
        before_attempt=before_attempt,
    ).complete_json(
        [{"role": "user", "content": "Test"}],
        validator=validator,
    )

    assert call_count == 2
    assert attempt_hook_count == 2
    assert result["status"] == "valid"
    assert result["_usage"] == {
        "prompt_tokens": 20,
        "cached_prompt_tokens": None,
        "completion_tokens": 10,
        "total_tokens": 30,
    }
    assert result["_operational"]["attempts"] == 2
    assert result["_operational"]["failed_attempts"] == 0
    assert result["_operational"]["cost_usd"] == pytest.approx(
        0.002
    )
    assert result["_operational"]["cost_source"] == (
        "litellm_response_metadata"
    )


def test_complete_text_leaves_unreported_cost_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        llm_module,
        "_completion",
        lambda **kwargs: make_response("answer", cost_usd=None),
    )

    result = LLMClient(
        model="azure/test-model",
        max_retries=1,
    ).complete_text([{"role": "user", "content": "Test"}])

    assert result["content"] == "answer"
    assert result["usage"]["total_tokens"] == 15
    assert result["operational"]["cost_usd"] is None
    assert result["operational"]["cost_source"] is None


def test_complete_text_sends_reasoning_and_omits_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def fake_completion(**kwargs: Any) -> SimpleNamespace:
        seen.update(kwargs)
        return make_response("answer")

    monkeypatch.setattr(llm_module, "_completion", fake_completion)

    result = LLMClient(
        model="azure/gpt-5.6-sol",
        temperature=None,
        reasoning_effort="none",
        max_retries=1,
    ).complete_text(
        [{"role": "user", "content": "Test"}],
        max_completion_tokens=900,
    )

    assert result["content"] == "answer"
    assert seen["model"] == "azure/gpt-5.6-sol"
    assert seen["reasoning_effort"] == "none"
    assert seen["max_completion_tokens"] == 900
    assert "temperature" not in seen


def test_complete_response_maps_responses_usage_and_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def fake_responses(**kwargs: Any) -> SimpleNamespace:
        seen.update(kwargs)
        return SimpleNamespace(
            output_text='{"answer":"pdf"}',
            usage=SimpleNamespace(
                input_tokens=120,
                output_tokens=8,
                total_tokens=128,
                input_tokens_details=SimpleNamespace(cached_tokens=20),
            ),
            _hidden_params={},
        )

    monkeypatch.setattr(llm_module, "_responses", fake_responses)

    result = LLMClient(
        model="azure/gpt-5.6-sol",
        temperature=None,
        reasoning_effort="none",
        max_retries=1,
    ).complete_response(
        [{"role": "user", "content": "Read the PDF"}],
        instructions="Use only the attached source.",
        max_output_tokens=900,
    )

    assert result["content"] == '{"answer":"pdf"}'
    assert result["usage"] == {
        "prompt_tokens": 120,
        "cached_prompt_tokens": 20,
        "completion_tokens": 8,
        "total_tokens": 128,
    }
    assert seen["reasoning"] == {"effort": "none"}
    assert seen["instructions"] == "Use only the attached source."
    assert seen["max_output_tokens"] == 900
    assert "temperature" not in seen


def test_installed_litellm_supports_gpt_5_6_azure_reasoning_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "true")
    from litellm import get_supported_openai_params

    for tier in ("luna", "terra", "sol"):
        supported = get_supported_openai_params(
            model=f"azure/gpt-5.6-{tier}"
        )

        assert supported is not None
        assert "reasoning_effort" in supported


def test_configured_endpoint_price_precedes_provider_cost_and_counts_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(
        [
            make_response(
                '{"status": "invalid"}',
                cached_tokens=2,
            ),
            make_response(
                '{"status": "valid"}',
                cached_tokens=2,
            ),
        ]
    )
    monkeypatch.setattr(
        llm_module,
        "_completion",
        lambda **kwargs: next(responses),
    )
    monkeypatch.setattr(
        llm_module,
        "wait_exponential",
        lambda **kwargs: wait_none(),
    )
    endpoint_price = EndpointPriceConfig(
        pricing_id="test_global",
        model="azure/test-model",
        input_per_million_tokens=1.0,
        cached_input_per_million_tokens=0.25,
        output_per_million_tokens=4.0,
        effective_date=date(2026, 7, 1),
        source="test rate card",
    )

    result = LLMClient(
        model="azure/test-model",
        max_retries=2,
        endpoint_price=endpoint_price,
    ).complete_json(
        [{"role": "user", "content": "Test"}],
        validator=lambda response: (
            None
            if response.get("status") == "valid"
            else (_ for _ in ()).throw(ValueError("invalid"))
        ),
    )

    operational = result["_operational"]
    assert result["_usage"]["cached_prompt_tokens"] == 4
    assert operational["configured_cost_usd"] == pytest.approx(
        0.000057
    )
    assert operational["provider_reported_cost_usd"] == pytest.approx(
        0.002
    )
    assert operational["cost_usd"] == pytest.approx(0.000057)
    assert operational["cost_source"] == "configured_endpoint_pricing"
    assert operational["pricing_id"] == "test_global"
    assert operational["pricing_effective_date"] == "2026-07-01"


def test_provider_cost_is_fallback_when_configured_estimate_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = make_response("answer", cost_usd=0.003)
    response.usage.prompt_tokens = None
    monkeypatch.setattr(
        llm_module,
        "_completion",
        lambda **kwargs: response,
    )
    endpoint_price = EndpointPriceConfig(
        pricing_id="test_global",
        model="azure/test-model",
        input_per_million_tokens=1.0,
        output_per_million_tokens=4.0,
        effective_date=date(2026, 7, 1),
        source="test rate card",
    )

    result = LLMClient(
        model="azure/test-model",
        max_retries=1,
        endpoint_price=endpoint_price,
    ).complete_text([{"role": "user", "content": "Test"}])

    operational = result["operational"]
    assert operational["configured_cost_usd"] is None
    assert operational["provider_reported_cost_usd"] == 0.003
    assert operational["cost_usd"] == 0.003
    assert operational["cost_source"] == "litellm_response_metadata"


def test_failed_call_exposes_operational_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_completion(**kwargs: Any) -> SimpleNamespace:
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(llm_module, "_completion", fail_completion)

    with pytest.raises(RuntimeError) as caught:
        LLMClient(
            model="azure/test-model",
            max_retries=1,
        ).complete_text(
            [{"role": "user", "content": "Test"}]
        )

    operational = operational_from_error(caught.value)
    assert operational["attempts"] == 1
    assert operational["failed_attempts"] == 1
    assert operational["successful_responses"] == 0
    assert operational["latency_seconds"] >= 0

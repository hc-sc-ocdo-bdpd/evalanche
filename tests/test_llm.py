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


def make_response(
    content: str,
    *,
    cost_usd: float | None = 0.001,
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

    def fake_completion(**kwargs: Any) -> SimpleNamespace:
        nonlocal call_count
        call_count += 1
        return next(responses)

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
    ).complete_json(
        [{"role": "user", "content": "Test"}],
        validator=validator,
    )

    assert call_count == 2
    assert result["status"] == "valid"
    assert result["_usage"] == {
        "prompt_tokens": 20,
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
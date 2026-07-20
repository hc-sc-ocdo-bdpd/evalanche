from types import SimpleNamespace
from typing import Any

import pytest
from tenacity import wait_none

import evalanche.llm as llm_module
from evalanche.llm import LLMClient, _parse_json_content


def make_response(content: str) -> SimpleNamespace:
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
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }
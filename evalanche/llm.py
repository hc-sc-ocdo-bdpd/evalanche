from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential


JsonValidator = Callable[[dict[str, Any]], None]


def _completion(**kwargs: Any) -> Any:
    # Keep provider initialization at the actual network boundary. This makes
    # offline commands and unit tests independent of LiteLLM client startup.
    from litellm import completion

    return completion(**kwargs)

class LLMClient:
    def __init__(
        self,
        model: str,
        temperature: float = 0,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries

    def complete_json(
        self,
        messages: list[dict[str, str]],
        validator: JsonValidator | None = None,
    ) -> dict[str, Any]:
        retrying_call = retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            reraise=True,
        )(self._single_complete_json_call)

        return retrying_call(messages, validator)

    def complete_text(
        self,
        messages: list[dict[str, str]],
        max_completion_tokens: int | None = None,
    ) -> dict[str, Any]:
        retrying_call = retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            reraise=True,
        )(self._single_complete_text_call)

        return retrying_call(messages, max_completion_tokens)

    def _single_complete_json_call(
        self,
        messages: list[dict[str, str]],
        validator: JsonValidator | None = None,
    ) -> dict[str, Any]:
        try:
            response = _completion(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )
        except Exception:
            response = _completion(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
            )

        content = response.choices[0].message.content
        parsed = _parse_json_content(content)

        if validator is not None:
            validator(parsed)

        parsed["_usage"] = _usage_to_dict(response)

        return parsed

    def _single_complete_text_call(
        self,
        messages: list[dict[str, str]],
        max_completion_tokens: int | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }

        if max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = max_completion_tokens

        response = _completion(**kwargs)
        content = response.choices[0].message.content or ""

        return {
            "content": content,
            "usage": _usage_to_dict(response),
        }


def _usage_to_dict(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)

    if usage is None:
        return {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        }

    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def _parse_json_content(content: str) -> dict[str, Any]:
    text = content.strip()

    if text.startswith("```json"):
        text = text.removeprefix("```json").strip()

    if text.startswith("```"):
        text = text.removeprefix("```").strip()

    if text.endswith("```"):
        text = text.removesuffix("```").strip()

    return json.loads(text)
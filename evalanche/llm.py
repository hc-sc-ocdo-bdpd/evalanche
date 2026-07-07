from __future__ import annotations

import json
from typing import Any

from litellm import completion
from tenacity import retry, stop_after_attempt, wait_exponential


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

    def complete_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        return self._complete_json(messages)

    def _complete_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        retrying_call = retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            reraise=True,
        )(self._single_complete_json_call)

        return retrying_call(messages)

    def _single_complete_json_call(
        self,
        messages: list[dict[str, str]],
    ) -> dict[str, Any]:
        try:
            response = completion(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )
        except Exception:
            response = completion(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
            )

        content = response.choices[0].message.content
        parsed = _parse_json_content(content)

        usage = getattr(response, "usage", None)
        if usage is not None:
            parsed["_usage"] = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }

        return parsed


def _parse_json_content(content: str) -> dict[str, Any]:
    text = content.strip()

    if text.startswith("```json"):
        text = text.removeprefix("```json").strip()

    if text.startswith("```"):
        text = text.removeprefix("```").strip()

    if text.endswith("```"):
        text = text.removesuffix("```").strip()

    return json.loads(text)
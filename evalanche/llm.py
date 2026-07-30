from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from math import isfinite
from time import perf_counter
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from evalanche.pricing import (
    CONFIGURED_COST_SOURCE,
    PROVIDER_COST_SOURCE,
    EndpointPriceConfig,
    endpoint_price_fields,
    estimate_endpoint_cost_usd,
)


JsonValidator = Callable[[dict[str, Any]], None]
AttemptHook = Callable[[], None]
_ERROR_OPERATIONAL_ATTRIBUTE = "_evalanche_operational"


def _completion(**kwargs: Any) -> Any:
    # Keep provider initialization at the actual network boundary. This makes
    # offline commands and unit tests independent of LiteLLM client startup.
    from litellm import completion

    return completion(**kwargs)


def _value(container: Any, name: str) -> Any:
    if isinstance(container, dict):
        return container.get(name)
    return getattr(container, name, None)


def _nonnegative_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None

    if not isfinite(numeric) or numeric < 0:
        return None

    return numeric


def _response_cost_usd(response: Any) -> float | None:
    """Read LiteLLM's response cost when it supplied one."""
    hidden_params = _value(response, "_hidden_params")

    if hidden_params is None:
        return None

    for key in ("response_cost", "response_cost_usd"):
        cost = _nonnegative_number(_value(hidden_params, key))
        if cost is not None:
            return cost

    return None


def _usage_to_dict(response: Any) -> dict[str, Any]:
    usage = _value(response, "usage")

    if usage is None:
        return {
            "prompt_tokens": None,
            "cached_prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        }

    prompt_details = _value(usage, "prompt_tokens_details")

    return {
        "prompt_tokens": _value(usage, "prompt_tokens"),
        "cached_prompt_tokens": _value(
            prompt_details,
            "cached_tokens",
        ),
        "completion_tokens": _value(usage, "completion_tokens"),
        "total_tokens": _value(usage, "total_tokens"),
    }


@dataclass
class _CallTracker:
    endpoint_price: EndpointPriceConfig | None = None
    before_attempt: AttemptHook | None = None
    started_at: float = field(default_factory=perf_counter)
    attempts: int = 0
    failed_attempts: int = 0
    successful_responses: int = 0
    api_seconds: float = 0.0
    prompt_tokens: int = 0
    cached_prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_token_observations: int = 0
    cached_prompt_token_observations: int = 0
    completion_token_observations: int = 0
    total_token_observations: int = 0
    known_provider_cost_responses: int = 0
    provider_reported_cost_usd: float = 0.0

    def completion(self, **kwargs: Any) -> Any:
        if self.before_attempt is not None:
            self.before_attempt()
        self.attempts += 1
        attempt_started_at = perf_counter()

        try:
            response = _completion(**kwargs)
        except Exception:
            self.failed_attempts += 1
            raise
        finally:
            self.api_seconds += perf_counter() - attempt_started_at

        self.successful_responses += 1
        usage = _usage_to_dict(response)
        self._record_tokens("prompt", usage.get("prompt_tokens"))
        self._record_tokens(
            "cached_prompt",
            usage.get("cached_prompt_tokens"),
        )
        self._record_tokens(
            "completion",
            usage.get("completion_tokens"),
        )
        self._record_tokens("total", usage.get("total_tokens"))

        cost = _response_cost_usd(response)
        if cost is not None:
            self.known_provider_cost_responses += 1
            self.provider_reported_cost_usd += cost

        return response

    def _record_tokens(self, kind: str, value: Any) -> None:
        numeric = _nonnegative_number(value)
        if numeric is None:
            return

        setattr(
            self,
            f"{kind}_tokens",
            getattr(self, f"{kind}_tokens") + int(numeric),
        )
        observation_name = f"{kind}_token_observations"
        setattr(
            self,
            observation_name,
            getattr(self, observation_name) + 1,
        )

    def usage(self) -> dict[str, int | None]:
        def complete_value(kind: str) -> int | None:
            observations = getattr(
                self,
                f"{kind}_token_observations",
            )
            if (
                self.successful_responses == 0
                or observations != self.successful_responses
            ):
                return None
            return int(getattr(self, f"{kind}_tokens"))

        return {
            "prompt_tokens": complete_value("prompt"),
            "cached_prompt_tokens": complete_value("cached_prompt"),
            "completion_tokens": complete_value("completion"),
            "total_tokens": complete_value("total"),
        }

    def snapshot(self) -> dict[str, Any]:
        usage = self.usage()
        provider_cost_complete = (
            self.successful_responses > 0
            and self.known_provider_cost_responses
            == self.successful_responses
        )
        provider_cost = (
            self.provider_reported_cost_usd
            if provider_cost_complete
            else None
        )
        configured_cost = (
            estimate_endpoint_cost_usd(
                self.endpoint_price,
                prompt_tokens=usage["prompt_tokens"],
                cached_prompt_tokens=usage[
                    "cached_prompt_tokens"
                ],
                completion_tokens=usage["completion_tokens"],
            )
            if self.endpoint_price is not None
            and self.successful_responses > 0
            else None
        )

        if configured_cost is not None:
            selected_cost = configured_cost
            selected_source = CONFIGURED_COST_SOURCE
        elif provider_cost is not None:
            selected_cost = provider_cost
            selected_source = PROVIDER_COST_SOURCE
        else:
            selected_cost = None
            selected_source = None

        return {
            "latency_seconds": perf_counter() - self.started_at,
            "api_seconds": self.api_seconds,
            "attempts": self.attempts,
            "failed_attempts": self.failed_attempts,
            "successful_responses": self.successful_responses,
            **usage,
            "cost_usd": selected_cost,
            "cost_source": selected_source,
            "configured_cost_usd": configured_cost,
            "provider_reported_cost_usd": provider_cost,
            **endpoint_price_fields(self.endpoint_price),
        }


def operational_from_error(error: BaseException) -> dict[str, Any]:
    telemetry = getattr(
        error,
        _ERROR_OPERATIONAL_ATTRIBUTE,
        None,
    )
    return telemetry if isinstance(telemetry, dict) else {}


def _attach_operational_metrics(
    error: BaseException,
    telemetry: dict[str, Any],
) -> None:
    try:
        setattr(error, _ERROR_OPERATIONAL_ATTRIBUTE, telemetry)
    except Exception:
        # Some third-party exception types may reject custom attributes. The
        # original error must remain the one presented to the caller.
        pass


class LLMClient:
    def __init__(
        self,
        model: str,
        temperature: float | None = 0,
        reasoning_effort: str | None = None,
        max_retries: int = 3,
        endpoint_price: EndpointPriceConfig | None = None,
        before_attempt: AttemptHook | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.max_retries = max_retries
        self.endpoint_price = endpoint_price
        self.before_attempt = before_attempt

    def _request_kwargs(
        self,
        messages: list[dict[str, str]],
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
        return kwargs

    def complete_json(
        self,
        messages: list[dict[str, str]],
        validator: JsonValidator | None = None,
    ) -> dict[str, Any]:
        tracker = _CallTracker(
            endpoint_price=self.endpoint_price,
            before_attempt=self.before_attempt,
        )
        retrying_call = retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            reraise=True,
        )(self._single_complete_json_call)

        try:
            result = retrying_call(messages, validator, tracker)
        except Exception as error:
            _attach_operational_metrics(error, tracker.snapshot())
            raise

        telemetry = tracker.snapshot()
        result["_usage"] = {
            key: telemetry.get(key)
            for key in (
                "prompt_tokens",
                "cached_prompt_tokens",
                "completion_tokens",
                "total_tokens",
            )
        }
        result["_operational"] = telemetry
        return result

    def complete_text(
        self,
        messages: list[dict[str, str]],
        max_completion_tokens: int | None = None,
    ) -> dict[str, Any]:
        tracker = _CallTracker(
            endpoint_price=self.endpoint_price,
            before_attempt=self.before_attempt,
        )
        retrying_call = retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            reraise=True,
        )(self._single_complete_text_call)

        try:
            result = retrying_call(
                messages,
                max_completion_tokens,
                tracker,
            )
        except Exception as error:
            _attach_operational_metrics(error, tracker.snapshot())
            raise

        telemetry = tracker.snapshot()
        result["usage"] = {
            key: telemetry.get(key)
            for key in (
                "prompt_tokens",
                "cached_prompt_tokens",
                "completion_tokens",
                "total_tokens",
            )
        }
        result["operational"] = telemetry
        return result

    def _single_complete_json_call(
        self,
        messages: list[dict[str, str]],
        validator: JsonValidator | None,
        tracker: _CallTracker,
    ) -> dict[str, Any]:
        kwargs = self._request_kwargs(messages)
        try:
            response = tracker.completion(
                **kwargs,
                response_format={"type": "json_object"},
            )
        except Exception:
            response = tracker.completion(**kwargs)

        content = response.choices[0].message.content
        parsed = _parse_json_content(content)

        if validator is not None:
            validator(parsed)

        return parsed

    def _single_complete_text_call(
        self,
        messages: list[dict[str, str]],
        max_completion_tokens: int | None,
        tracker: _CallTracker,
    ) -> dict[str, Any]:
        kwargs = self._request_kwargs(messages)

        if max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = max_completion_tokens

        response = tracker.completion(**kwargs)
        content = response.choices[0].message.content or ""
        return {"content": content}


def _parse_json_content(content: str) -> dict[str, Any]:
    text = content.strip()

    if text.startswith("```json"):
        text = text.removeprefix("```json").strip()

    if text.startswith("```"):
        text = text.removeprefix("```").strip()

    if text.endswith("```"):
        text = text.removesuffix("```").strip()

    return json.loads(text)

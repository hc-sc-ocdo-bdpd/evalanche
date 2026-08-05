from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    wait,
)
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from time import monotonic, perf_counter, sleep
from typing import Any, Callable

import pandas as pd
from tqdm.auto import tqdm

from evalanche import __version__
from evalanche.config import (
    CandidateModelConfig,
    GenerationConfig,
    load_candidate_models,
)
from evalanche.identity import validate_generation_case_ids
from evalanche.llm import LLMClient, operational_from_error
from evalanche.operational import (
    COST_CURRENCY,
    COST_POLICY,
    summarize_stage,
)
from evalanche.pricing import (
    EndpointPriceConfig,
    EndpointPricingCatalogConfig,
    build_pricing_snapshot,
    estimate_endpoint_cost_usd,
    load_optional_endpoint_pricing_catalog_with_hash,
    resolve_endpoint_price,
)
from evalanche.routing import validate_evaluation_types


REQUIRED_GENERATION_COLUMNS = {
    "case_id",
    "input",
    "expected_output",
    "evaluation_type",
}


class SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def sha256_file(path: str | Path | None) -> str | None:
    if path is None:
        return None

    file_path = Path(path)

    if not file_path.exists() or not file_path.is_file():
        return None

    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_generation_cases(path: str | Path) -> pd.DataFrame:
    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Generation input file not found: {input_path}"
        )

    cases = pd.read_csv(input_path)

    missing_columns = REQUIRED_GENERATION_COLUMNS - set(cases.columns)

    if missing_columns:
        raise ValueError(
            "Generation input is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    cases = validate_evaluation_types(
        cases,
        source_name=str(input_path),
    )

    return validate_generation_case_ids(
        cases,
        source_name=str(input_path),
    )


def render_prompt(
    template: str,
    row: dict[str, Any],
) -> str:
    values = SafeFormatDict(
        {
            key: "" if pd.isna(value) else str(value)
            for key, value in row.items()
        }
    )

    return template.format_map(values)


def build_messages(
    config: GenerationConfig,
    row: dict[str, Any],
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": config.prompt.system,
        },
        {
            "role": "user",
            "content": render_prompt(
                config.prompt.template,
                row,
            ),
        },
    ]


def generate_one(
    *,
    config: GenerationConfig,
    candidate: CandidateModelConfig,
    row: dict[str, Any],
    endpoint_price: EndpointPriceConfig | None = None,
    before_attempt: Callable[[], None] | None = None,
) -> dict[str, Any]:
    client = LLMClient(
        model=candidate.model,
        temperature=candidate.temperature,
        reasoning_effort=candidate.reasoning_effort,
        max_retries=candidate.max_retries,
        endpoint_price=endpoint_price,
        before_attempt=before_attempt,
    )

    max_completion_tokens = (
        candidate.max_completion_tokens
        if candidate.max_completion_tokens is not None
        else config.generation.max_completion_tokens
    )

    started_at = perf_counter()

    if config.generation.request_api == "responses":
        response = client.complete_response(
            input_items=build_response_input(config, row),
            instructions=config.prompt.system,
            max_output_tokens=max_completion_tokens,
        )
    else:
        response = client.complete_text(
            messages=build_messages(config, row),
            max_completion_tokens=max_completion_tokens,
        )

    finished_at = datetime.now(timezone.utc)
    usage = response.get("usage", {})
    operational = response.get("operational", {})

    record = dict(row)

    record.update(
        {
            "model_name": candidate.name,
            "candidate_model": candidate.model,
            "model_output": response.get("content", ""),
            "generation_status": "success",
            "generation_error": "",
            "generated_at_utc": finished_at.isoformat(),
            "generation_seconds": operational.get(
                "latency_seconds",
                perf_counter() - started_at,
            ),
            "generation_api_seconds": operational.get(
                "api_seconds"
            ),
            "generation_attempts": operational.get("attempts"),
            "generation_failed_attempts": operational.get(
                "failed_attempts"
            ),
            "generation_prompt_tokens": usage.get(
                "prompt_tokens"
            ),
            "generation_cached_prompt_tokens": usage.get(
                "cached_prompt_tokens"
            ),
            "generation_completion_tokens": usage.get(
                "completion_tokens"
            ),
            "generation_total_tokens": usage.get(
                "total_tokens"
            ),
            "generation_cost_usd": operational.get("cost_usd"),
            "generation_cost_source": operational.get(
                "cost_source"
            ),
            "generation_configured_cost_usd": operational.get(
                "configured_cost_usd"
            ),
            "generation_provider_reported_cost_usd": operational.get(
                "provider_reported_cost_usd"
            ),
            "generation_pricing_id": operational.get(
                "pricing_id"
            ),
            "generation_pricing_model": operational.get(
                "pricing_model"
            ),
            "generation_pricing_currency": operational.get(
                "pricing_currency"
            ),
            "generation_pricing_input_per_million_tokens": operational.get(
                "pricing_input_per_million_tokens"
            ),
            "generation_pricing_cached_input_per_million_tokens": operational.get(
                "pricing_cached_input_per_million_tokens"
            ),
            "generation_pricing_output_per_million_tokens": operational.get(
                "pricing_output_per_million_tokens"
            ),
            "generation_pricing_effective_date": operational.get(
                "pricing_effective_date"
            ),
            "generation_pricing_source": operational.get(
                "pricing_source"
            ),
        }
    )

    return record


GENERATION_CHECKPOINT_SCHEMA_VERSION = "1.0"
GENERATION_FINGERPRINT_SCHEMA_VERSION = "1.1"
_CHECKPOINT_SUFFIX = ".checkpoint.jsonl"
_PROMPT_TOKEN_OVERHEAD_CEILING = 256
_MAX_FILE_INPUT_BYTES = 50 * 1024 * 1024
_MAX_FILE_INPUT_COUNT = 50
_FILE_INPUT_DETAILS = {"auto", "low", "high"}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(missing, bool) and missing


def load_input_file_descriptors(
    row: dict[str, Any],
) -> list[dict[str, Any]]:
    """Load and verify local file descriptors stored in a case row.

    The CSV stores descriptors, never file bytes. Every descriptor must carry
    the expected SHA-256 so a changed download cannot silently enter a run.
    Paths follow the same repository-root-relative convention as other
    Evalanche config paths.
    """
    raw_value = row.get("input_files")
    if _is_missing(raw_value) or str(raw_value).strip() == "":
        return []

    try:
        raw_descriptors = json.loads(str(raw_value))
    except json.JSONDecodeError as error:
        raise ValueError("input_files must contain a JSON list") from error
    if not isinstance(raw_descriptors, list) or not raw_descriptors:
        raise ValueError("input_files must contain a non-empty JSON list")
    if len(raw_descriptors) > _MAX_FILE_INPUT_COUNT:
        raise ValueError("input_files cannot contain more than 50 files")

    descriptors: list[dict[str, Any]] = []
    total_bytes = 0
    root = Path.cwd().resolve()
    for index, raw in enumerate(raw_descriptors):
        if not isinstance(raw, dict):
            raise ValueError(
                f"input_files[{index}] must be a JSON object"
            )
        path_value = raw.get("path")
        expected_sha256 = str(raw.get("sha256", "")).casefold()
        media_type = str(
            raw.get("media_type", "application/pdf")
        ).casefold()
        raw_detail = raw.get("detail")
        detail = (
            None
            if _is_missing(raw_detail) or str(raw_detail).strip() == ""
            else str(raw_detail).casefold()
        )
        if not isinstance(path_value, str) or not path_value.strip():
            raise ValueError(f"input_files[{index}].path is required")
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise ValueError(
                f"input_files[{index}].sha256 must be 64 hex characters"
            )
        if media_type != "application/pdf":
            raise ValueError(
                "Native file generation currently supports only "
                "application/pdf"
            )
        if detail is not None and detail not in _FILE_INPUT_DETAILS:
            raise ValueError(
                f"input_files[{index}].detail must be one of "
                f"{sorted(_FILE_INPUT_DETAILS)}"
            )

        path = Path(path_value)
        if path.is_absolute():
            raise ValueError(
                f"input_files[{index}].path must be repository-relative"
            )
        path = (root / path).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(
                f"input_files[{index}].path resolves outside the "
                "repository root"
            ) from error
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found: {path}")
        byte_size = path.stat().st_size
        total_bytes += byte_size
        if byte_size > _MAX_FILE_INPUT_BYTES:
            raise ValueError(f"Input file exceeds 50 MB: {path}")
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Input file SHA-256 mismatch for {path}: expected "
                f"{expected_sha256}, found {actual_sha256}"
            )
        filename = str(raw.get("filename") or path.name).strip()
        if (
            not filename
            or "/" in filename
            or "\\" in filename
            or Path(filename).name != filename
        ):
            raise ValueError(
                f"input_files[{index}].filename must be a basename"
            )
        descriptor = {
            "path": path,
            "filename": filename,
            "media_type": media_type,
            "byte_size": byte_size,
            "sha256": expected_sha256,
        }
        if detail is not None:
            descriptor["detail"] = detail
        descriptors.append(descriptor)

    if total_bytes > _MAX_FILE_INPUT_BYTES:
        raise ValueError("Combined input files exceed 50 MB")
    return descriptors


def build_response_input(
    config: GenerationConfig,
    row: dict[str, Any],
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": render_prompt(config.prompt.template, row),
        }
    ]
    for descriptor in load_input_file_descriptors(row):
        encoded = base64.b64encode(
            descriptor["path"].read_bytes()
        ).decode("ascii")
        file_item = {
            "type": "input_file",
            "filename": descriptor["filename"],
            "file_data": (
                f"data:{descriptor['media_type']};base64,{encoded}"
            ),
        }
        if descriptor.get("detail") is not None:
            file_item["detail"] = descriptor["detail"]
        content.append(file_item)
    return [{"role": "user", "content": content}]


class _RequestStartLimiter:
    def __init__(self, requests_per_minute: float | None) -> None:
        self.interval_seconds = (
            60.0 / requests_per_minute
            if requests_per_minute is not None
            else 0.0
        )
        self._lock = Lock()
        self._next_start = 0.0

    def wait(self) -> None:
        if self.interval_seconds <= 0:
            return

        with self._lock:
            now = monotonic()
            wait_seconds = max(0.0, self._next_start - now)
            self._next_start = (
                max(now, self._next_start) + self.interval_seconds
            )

        if wait_seconds > 0:
            sleep(wait_seconds)


def _checkpoint_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    return path.with_name(path.name + _CHECKPOINT_SUFFIX)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)

    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass

    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False

    if isinstance(missing, bool) and missing:
        return None
    return value


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name("." + path.name + ".tmp")
    with temporary_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        file.write(content)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary_path, path)


def _atomic_write_dataframe(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name("." + path.name + ".tmp")
    frame.to_csv(temporary_path, index=False)
    os.replace(temporary_path, path)


def _append_checkpoint_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": GENERATION_CHECKPOINT_SCHEMA_VERSION,
        "record": _json_safe(record),
    }
    with path.open("a", encoding="utf-8", newline="\n") as file:
        file.write(
            json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
            )
        )
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())


def _rewrite_checkpoint_records(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    lines = [
        json.dumps(
            {
                "schema_version": GENERATION_CHECKPOINT_SCHEMA_VERSION,
                "record": _json_safe(record),
            },
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        )
        for record in records
    ]
    content = "\n".join(lines)
    if lines:
        content += "\n"
    _atomic_write_text(path, content)


def _load_checkpoint_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    lines = path.read_text(encoding="utf-8").splitlines()
    records: list[dict[str, Any]] = []
    repaired_trailing_line = False

    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            if index == len(lines) - 1:
                repaired_trailing_line = True
                break
            raise ValueError(
                f"generation checkpoint contains invalid JSON on line "
                f"{index + 1}: {path}"
            ) from error

        if payload.get("schema_version") != (
            GENERATION_CHECKPOINT_SCHEMA_VERSION
        ):
            raise ValueError(
                "generation checkpoint schema version is not supported: "
                f"{payload.get('schema_version')!r}"
            )

        record = payload.get("record")
        if not isinstance(record, dict):
            raise ValueError(
                "generation checkpoint entries must contain an object record"
            )
        records.append(record)

    if repaired_trailing_line:
        _rewrite_checkpoint_records(path, records)

    return records


def _record_key(record: dict[str, Any]) -> tuple[str, str]:
    return (
        str(record.get("case_id", "")),
        str(record.get("model_name", "")),
    )


def _candidate_fingerprint_payload(
    candidate: CandidateModelConfig,
) -> dict[str, Any]:
    payload = candidate.model_dump(mode="json")
    if payload.get("reasoning_effort") is None:
        payload.pop("reasoning_effort")
    return payload


def _generation_fingerprint(
    *,
    config: GenerationConfig,
    candidates: list[CandidateModelConfig],
    input_sha256: str | None,
    pricing_catalog_sha256: str | None,
) -> str:
    payload = {
        "schema_version": GENERATION_FINGERPRINT_SCHEMA_VERSION,
        "input_sha256": input_sha256,
        "prompt": config.prompt.model_dump(mode="json"),
        "request_api": config.generation.request_api,
        "max_completion_tokens": (
            config.generation.max_completion_tokens
        ),
        "candidates": [
            _candidate_fingerprint_payload(candidate)
            for candidate in candidates
        ],
        "endpoint_pricing_sha256": pricing_catalog_sha256,
        "provider_route_environment": {
            "AZURE_API_BASE": os.getenv("AZURE_API_BASE"),
            "AZURE_API_VERSION": os.getenv("AZURE_API_VERSION"),
        },
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_resumed_records(
    *,
    records: list[dict[str, Any]],
    work_items: list[tuple[dict[str, Any], CandidateModelConfig]],
    fingerprint: str,
) -> None:
    expected = {
        (str(row["case_id"]), candidate.name): candidate
        for row, candidate in work_items
    }
    seen: set[tuple[str, str]] = set()

    for record in records:
        key = _record_key(record)
        if key in seen:
            raise ValueError(
                "generation checkpoint contains a duplicate case/model "
                f"record: {key}"
            )
        seen.add(key)

        candidate = expected.get(key)
        if candidate is None:
            raise ValueError(
                "generation checkpoint contains a case/model pair that is "
                f"not part of the configured run: {key}"
            )
        if record.get("candidate_model") != candidate.model:
            raise ValueError(
                "generation checkpoint model route does not match the "
                f"configured candidate for {key}"
            )
        if record.get("generation_run_fingerprint") != fingerprint:
            raise ValueError(
                "generation checkpoint fingerprint does not match the "
                "current input, prompt, candidate, or price catalog"
            )
        if record.get("generation_status") not in {"success", "error"}:
            raise ValueError(
                f"generation checkpoint has an invalid status for {key}"
            )


def _request_cost_ceiling(
    *,
    config: GenerationConfig,
    candidate: CandidateModelConfig,
    row: dict[str, Any],
    endpoint_price: EndpointPriceConfig | None,
) -> float | None:
    if endpoint_price is None:
        return None

    # PDF tokenization includes rendered page images. A byte-count estimate
    # would materially understate retry reserves, so only observed provider
    # usage is treated as valid for native-file requests.
    if not _is_missing(row.get("input_files")):
        if str(row.get("input_files")).strip():
            return None

    completion_limit = (
        candidate.max_completion_tokens
        if candidate.max_completion_tokens is not None
        else config.generation.max_completion_tokens
    )
    if completion_limit is None:
        return None

    messages = build_messages(config, row)
    prompt_bytes = sum(
        len(message["content"].encode("utf-8"))
        for message in messages
    )
    prompt_token_ceiling = (
        prompt_bytes + _PROMPT_TOKEN_OVERHEAD_CEILING
    )
    return estimate_endpoint_cost_usd(
        endpoint_price,
        prompt_tokens=prompt_token_ceiling,
        cached_prompt_tokens=0,
        completion_tokens=completion_limit,
    )


def _error_generation_record(
    *,
    row: dict[str, Any],
    candidate: CandidateModelConfig,
    error: BaseException,
) -> dict[str, Any]:
    operational = operational_from_error(error)
    record = dict(row)
    record.update(
        {
            "model_name": candidate.name,
            "candidate_model": candidate.model,
            "model_output": "",
            "generation_status": "error",
            "generation_error": repr(error),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "generation_seconds": operational.get("latency_seconds"),
            "generation_api_seconds": operational.get("api_seconds"),
            "generation_attempts": operational.get("attempts"),
            "generation_failed_attempts": operational.get(
                "failed_attempts"
            ),
            "generation_prompt_tokens": operational.get(
                "prompt_tokens"
            ),
            "generation_cached_prompt_tokens": operational.get(
                "cached_prompt_tokens"
            ),
            "generation_completion_tokens": operational.get(
                "completion_tokens"
            ),
            "generation_total_tokens": operational.get("total_tokens"),
            "generation_cost_usd": operational.get("cost_usd"),
            "generation_cost_source": operational.get("cost_source"),
            "generation_configured_cost_usd": operational.get(
                "configured_cost_usd"
            ),
            "generation_provider_reported_cost_usd": operational.get(
                "provider_reported_cost_usd"
            ),
            "generation_pricing_id": operational.get("pricing_id"),
            "generation_pricing_model": operational.get(
                "pricing_model"
            ),
            "generation_pricing_currency": operational.get(
                "pricing_currency"
            ),
            "generation_pricing_input_per_million_tokens": (
                operational.get("pricing_input_per_million_tokens")
            ),
            "generation_pricing_cached_input_per_million_tokens": (
                operational.get(
                    "pricing_cached_input_per_million_tokens"
                )
            ),
            "generation_pricing_output_per_million_tokens": (
                operational.get("pricing_output_per_million_tokens")
            ),
            "generation_pricing_effective_date": operational.get(
                "pricing_effective_date"
            ),
            "generation_pricing_source": operational.get(
                "pricing_source"
            ),
        }
    )
    return record


def _complete_generation_item(
    *,
    config: GenerationConfig,
    candidate: CandidateModelConfig,
    row: dict[str, Any],
    endpoint_price: EndpointPriceConfig | None,
    request_cost_ceiling: float | None,
    limiter: _RequestStartLimiter,
    fingerprint: str,
    pricing_catalog_version: str | None,
    pricing_catalog_sha256: str | None,
) -> dict[str, Any]:
    try:
        record = generate_one(
            config=config,
            candidate=candidate,
            row=row,
            endpoint_price=endpoint_price,
            before_attempt=limiter.wait,
        )
    except Exception as error:
        if not config.generation.continue_on_error:
            raise
        record = _error_generation_record(
            row=row,
            candidate=candidate,
            error=error,
        )

    configured_cost = record.get("generation_configured_cost_usd")
    raw_budget_cost = (
        float(configured_cost)
        if configured_cost is not None
        else 0.0
    )
    failed_attempts = record.get("generation_failed_attempts")
    if (
        request_cost_ceiling is not None
        and failed_attempts is not None
    ):
        raw_budget_cost += (
            max(0, int(failed_attempts)) * request_cost_ceiling
        )

    record.update(
        {
            "generation_unreported_retry_cost_reserve_usd": (
                raw_budget_cost
                - (
                    float(configured_cost)
                    if configured_cost is not None
                    else 0.0
                )
            ),
            "generation_budget_cost_usd": (
                raw_budget_cost
                if request_cost_ceiling is not None
                else configured_cost
            ),
            "generation_run_fingerprint": fingerprint,
            "generation_pricing_catalog_version": (
                pricing_catalog_version
            ),
            "generation_pricing_catalog_sha256": (
                pricing_catalog_sha256
            ),
        }
    )
    return record


def _raw_budget_cost(
    records: list[dict[str, Any]],
    *,
    require_complete: bool,
) -> float:
    total = 0.0
    for record in records:
        value = record.get("generation_budget_cost_usd")
        if value is None:
            if require_complete:
                raise ValueError(
                    "generation budget accounting is incomplete for "
                    f"{_record_key(record)}"
                )
            continue
        total += float(value)
    return total


def _build_cost_preflight(
    *,
    config: GenerationConfig,
    candidates: list[CandidateModelConfig],
    endpoint_prices: dict[str, EndpointPriceConfig | None],
    pending_items: list[
        tuple[dict[str, Any], CandidateModelConfig]
    ],
    resumed_records: list[dict[str, Any]],
) -> dict[str, Any] | None:
    maximum = config.generation.maximum_estimated_cost_usd
    sample_path = config.generation.cost_preflight_sample_path
    if maximum is None or sample_path is None:
        return None

    if not sample_path.exists():
        raise FileNotFoundError(
            f"generation cost-preflight sample not found: {sample_path}"
        )

    sample = pd.read_csv(sample_path)
    required_columns = {
        "model_name",
        "candidate_model",
        "generation_status",
        "generation_prompt_tokens",
        "generation_completion_tokens",
    }
    missing = required_columns - set(sample.columns)
    if missing:
        raise ValueError(
            "generation cost-preflight sample is missing columns: "
            f"{sorted(missing)}"
        )

    pending_by_model = {
        candidate.name: sum(
            1
            for _, pending_candidate in pending_items
            if pending_candidate.name == candidate.name
        )
        for candidate in candidates
    }
    by_model: dict[str, Any] = {}
    projected_remaining_cost = 0.0

    for candidate in candidates:
        endpoint_price = endpoint_prices[candidate.name]
        if endpoint_price is None:
            raise ValueError(
                "maximum_estimated_cost_usd requires an explicit "
                f"endpoint price for candidate {candidate.name!r}"
            )

        candidate_sample = sample[
            (sample["model_name"].astype(str) == candidate.name)
            & (
                sample["candidate_model"].astype(str)
                == candidate.model
            )
            & (
                sample["generation_status"]
                .astype(str)
                .str.lower()
                == "success"
            )
        ]
        if candidate_sample.empty:
            raise ValueError(
                "generation cost-preflight sample has no successful rows "
                f"for {candidate.name!r} using {candidate.model!r}"
            )

        costs: list[float] = []
        for _, sample_row in candidate_sample.iterrows():
            prompt_tokens = pd.to_numeric(
                sample_row["generation_prompt_tokens"],
                errors="coerce",
            )
            completion_tokens = pd.to_numeric(
                sample_row["generation_completion_tokens"],
                errors="coerce",
            )
            if pd.isna(prompt_tokens) or pd.isna(completion_tokens):
                raise ValueError(
                    "generation cost-preflight sample contains missing "
                    f"token usage for {candidate.name!r}"
                )

            cost = estimate_endpoint_cost_usd(
                endpoint_price,
                prompt_tokens=int(prompt_tokens),
                cached_prompt_tokens=0,
                completion_tokens=int(completion_tokens),
            )
            if cost is None:
                raise ValueError(
                    "generation cost preflight could not estimate a "
                    f"sample row for {candidate.name!r}"
                )
            costs.append(cost)

        average_cost = sum(costs) / len(costs)
        remaining_cost = (
            average_cost * pending_by_model[candidate.name]
        )
        projected_remaining_cost += remaining_cost
        by_model[candidate.name] = {
            "model": candidate.model,
            "sample_rows": len(costs),
            "average_sample_cost_usd": average_cost,
            "pending_requests": pending_by_model[candidate.name],
            "projected_remaining_cost_usd": remaining_cost,
        }

    existing_budget_cost = _raw_budget_cost(
        resumed_records,
        require_complete=True,
    )
    projected_total_cost = (
        existing_budget_cost + projected_remaining_cost
    )
    multiplier = config.generation.cost_safety_multiplier
    budgeted_projection = projected_total_cost * multiplier
    preflight = {
        "sample_path": str(sample_path),
        "sample_sha256": sha256_file(sample_path),
        "assumes_cached_input_tokens": 0,
        "existing_budget_cost_usd": existing_budget_cost,
        "projected_remaining_cost_usd": projected_remaining_cost,
        "projected_total_cost_usd": projected_total_cost,
        "cost_safety_multiplier": multiplier,
        "projected_budgeted_total_usd": budgeted_projection,
        "maximum_estimated_cost_usd": maximum,
        "within_limit": budgeted_projection <= maximum,
        "by_model": by_model,
    }
    if not preflight["within_limit"]:
        raise ValueError(
            "generation cost preflight failed: projected budgeted cost "
            f"${budgeted_projection:.2f} USD exceeds the configured "
            f"${maximum:.2f} USD limit"
        )
    return preflight


def _ordered_outputs(
    *,
    records: list[dict[str, Any]],
    order: dict[tuple[str, str], int],
) -> pd.DataFrame:
    ordered = sorted(records, key=lambda record: order[_record_key(record)])
    return pd.DataFrame(ordered)


def preflight_generation(
    *,
    config: GenerationConfig,
) -> dict[str, Any]:
    cases = load_generation_cases(config.run.input_path)
    candidate_config = load_candidate_models(
        config.candidate_models_path
    )
    pricing_catalog, pricing_catalog_sha256 = (
        load_optional_endpoint_pricing_catalog_with_hash(
            config.endpoint_pricing_path
        )
    )
    endpoint_prices = {
        candidate.name: resolve_endpoint_price(
            catalog=pricing_catalog,
            pricing_id=candidate.pricing_id,
            model=candidate.model,
        )
        for candidate in candidate_config.models
    }
    work_items = [
        (row, candidate)
        for row in cases.to_dict(orient="records")
        for candidate in candidate_config.models
    ]
    output_path = Path(config.run.output_path)
    checkpoint_path = _checkpoint_path(output_path)
    fingerprint = _generation_fingerprint(
        config=config,
        candidates=candidate_config.models,
        input_sha256=sha256_file(config.run.input_path),
        pricing_catalog_sha256=pricing_catalog_sha256,
    )

    if config.generation.resume and checkpoint_path.exists():
        records = _load_checkpoint_records(checkpoint_path)
    elif config.generation.resume and output_path.exists():
        records = [
            _json_safe(record)
            for record in pd.read_csv(output_path).to_dict(
                orient="records"
            )
        ]
    else:
        records = []

    _validate_resumed_records(
        records=records,
        work_items=work_items,
        fingerprint=fingerprint,
    )
    completed_keys = {_record_key(record) for record in records}
    pending_items = [
        item
        for item in work_items
        if (str(item[0]["case_id"]), item[1].name)
        not in completed_keys
    ]
    preflight = _build_cost_preflight(
        config=config,
        candidates=candidate_config.models,
        endpoint_prices=endpoint_prices,
        pending_items=pending_items,
        resumed_records=records,
    )
    if preflight is None:
        raise ValueError(
            "generation config does not declare a cost preflight and limit"
        )

    return {
        **preflight,
        "expected_output_count": len(work_items),
        "checkpointed_output_count": len(records),
        "pending_output_count": len(pending_items),
        "generation_run_fingerprint": fingerprint,
    }


def generate_outputs(
    *,
    config: GenerationConfig,
    config_path: str | Path,
) -> tuple[pd.DataFrame, Path, Path]:
    cases = load_generation_cases(config.run.input_path)

    candidate_config = load_candidate_models(
        config.candidate_models_path
    )
    pricing_catalog, pricing_catalog_sha256 = (
        load_optional_endpoint_pricing_catalog_with_hash(
            config.endpoint_pricing_path
        )
    )
    endpoint_prices = {
        candidate.name: resolve_endpoint_price(
            catalog=pricing_catalog,
            pricing_id=candidate.pricing_id,
            model=candidate.model,
        )
        for candidate in candidate_config.models
    }

    work_items = [
        (row, candidate)
        for row in cases.to_dict(orient="records")
        for candidate in candidate_config.models
    ]
    output_path = Path(config.run.output_path)
    checkpoint_path = _checkpoint_path(output_path)
    input_sha256 = sha256_file(config.run.input_path)
    fingerprint = _generation_fingerprint(
        config=config,
        candidates=candidate_config.models,
        input_sha256=input_sha256,
        pricing_catalog_sha256=pricing_catalog_sha256,
    )
    order = {
        (str(row["case_id"]), candidate.name): index
        for index, (row, candidate) in enumerate(work_items)
    }

    if config.generation.resume:
        if checkpoint_path.exists():
            records = _load_checkpoint_records(checkpoint_path)
        elif output_path.exists():
            records = [
                _json_safe(record)
                for record in pd.read_csv(output_path).to_dict(
                    orient="records"
                )
            ]
            _rewrite_checkpoint_records(checkpoint_path, records)
        else:
            records = []
            _rewrite_checkpoint_records(checkpoint_path, records)
    else:
        records = []
        _rewrite_checkpoint_records(checkpoint_path, records)

    _validate_resumed_records(
        records=records,
        work_items=work_items,
        fingerprint=fingerprint,
    )
    resumed_output_count = len(records)
    completed_keys = {_record_key(record) for record in records}
    pending_items = [
        item
        for item in work_items
        if (str(item[0]["case_id"]), item[1].name)
        not in completed_keys
    ]
    preflight = _build_cost_preflight(
        config=config,
        candidates=candidate_config.models,
        endpoint_prices=endpoint_prices,
        pending_items=pending_items,
        resumed_records=records,
    )

    if preflight is not None:
        print(
            "\nCost preflight: "
            f"${preflight['projected_total_cost_usd']:.2f} USD "
            "projected, "
            f"${preflight['projected_budgeted_total_usd']:.2f} USD "
            "after safety multiplier, "
            f"${preflight['maximum_estimated_cost_usd']:.2f} USD limit"
        )
    if resumed_output_count:
        print(
            f"Resuming from {resumed_output_count}/"
            f"{len(work_items)} checkpointed outputs."
        )

    maximum_cost = config.generation.maximum_estimated_cost_usd
    safety_multiplier = config.generation.cost_safety_multiplier
    budget_guard_enabled = maximum_cost is not None
    current_raw_budget_cost = _raw_budget_cost(
        records,
        require_complete=budget_guard_enabled,
    )
    limiter = _RequestStartLimiter(
        config.generation.max_requests_per_minute
    )
    pricing_catalog_version = (
        pricing_catalog.catalog_version
        if pricing_catalog is not None
        else None
    )
    run_status = "completed" if not pending_items else "in_progress"
    outputs = _ordered_outputs(records=records, order=order)
    metadata_path = output_path.with_name(
        output_path.stem + "_generation_metadata.json"
    )

    def persist_state(status: str) -> pd.DataFrame:
        persisted_outputs = _ordered_outputs(
            records=records,
            order=order,
        )
        _atomic_write_dataframe(persisted_outputs, output_path)
        save_generation_metadata(
            config=config,
            config_path=config_path,
            candidates=candidate_config.models,
            cases=cases,
            outputs=persisted_outputs,
            output_path=output_path,
            pricing_catalog=pricing_catalog,
            pricing_catalog_sha256=pricing_catalog_sha256,
            run_status=status,
            checkpoint_path=checkpoint_path,
            expected_output_count=len(work_items),
            resumed_output_count=resumed_output_count,
            generation_run_fingerprint=fingerprint,
            cost_preflight=preflight,
        )
        return persisted_outputs

    if pending_items:
        futures: dict[
            Future[dict[str, Any]],
            tuple[
                dict[str, Any],
                CandidateModelConfig,
                float,
            ],
        ] = {}
        pending_index = 0
        budget_blocked = False
        executor = ThreadPoolExecutor(
            max_workers=config.generation.max_workers,
            thread_name_prefix="evalanche-generation",
        )
        progress = tqdm(
            total=len(work_items),
            initial=resumed_output_count,
            desc="Generating outputs",
        )

        def submit_available() -> None:
            nonlocal pending_index, budget_blocked
            while (
                pending_index < len(pending_items)
                and len(futures) < config.generation.max_workers
            ):
                row, candidate = pending_items[pending_index]
                endpoint_price = endpoint_prices[candidate.name]
                ceiling = _request_cost_ceiling(
                    config=config,
                    candidate=candidate,
                    row=row,
                    endpoint_price=endpoint_price,
                )
                if budget_guard_enabled and ceiling is None:
                    raise ValueError(
                        "maximum_estimated_cost_usd requires endpoint "
                        "pricing and max_completion_tokens for every "
                        "candidate"
                    )

                retry_ceiling = (
                    float(ceiling)
                    * max(1, int(candidate.max_retries))
                    if ceiling is not None
                    else 0.0
                )
                in_flight_reserve = sum(
                    item[2] for item in futures.values()
                )
                projected_budgeted_cost = (
                    current_raw_budget_cost
                    + in_flight_reserve
                    + retry_ceiling
                ) * safety_multiplier
                if (
                    maximum_cost is not None
                    and projected_budgeted_cost > maximum_cost
                ):
                    budget_blocked = True
                    return

                future = executor.submit(
                    _complete_generation_item,
                    config=config,
                    candidate=candidate,
                    row=row,
                    endpoint_price=endpoint_price,
                    request_cost_ceiling=ceiling,
                    limiter=limiter,
                    fingerprint=fingerprint,
                    pricing_catalog_version=pricing_catalog_version,
                    pricing_catalog_sha256=pricing_catalog_sha256,
                )
                futures[future] = (row, candidate, retry_ceiling)
                pending_index += 1

        try:
            submit_available()
            since_materialized_checkpoint = 0

            while futures:
                completed, _ = wait(
                    futures,
                    return_when=FIRST_COMPLETED,
                )
                for future in completed:
                    futures.pop(future)
                    record = future.result()
                    key = _record_key(record)
                    if key in completed_keys:
                        raise ValueError(
                            "generation produced a duplicate case/model "
                            f"record: {key}"
                        )

                    _append_checkpoint_record(
                        checkpoint_path,
                        record,
                    )
                    records.append(record)
                    completed_keys.add(key)
                    current_raw_budget_cost += float(
                        record.get(
                            "generation_budget_cost_usd",
                            0.0,
                        )
                        or 0.0
                    )
                    since_materialized_checkpoint += 1
                    progress.update(1)

                    if (
                        since_materialized_checkpoint
                        >= config.generation.checkpoint_every
                    ):
                        persist_state("in_progress")
                        since_materialized_checkpoint = 0

                submit_available()

            if pending_index < len(pending_items) and budget_blocked:
                run_status = "budget_exhausted"
            else:
                run_status = "completed"
        except BaseException:
            run_status = "interrupted"
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)

            for future in list(futures):
                if future.cancelled() or not future.done():
                    continue
                try:
                    record = future.result()
                except BaseException:
                    continue
                key = _record_key(record)
                if key in completed_keys:
                    continue
                _append_checkpoint_record(checkpoint_path, record)
                records.append(record)
                completed_keys.add(key)

            persist_state(run_status)
            progress.close()
            raise
        else:
            executor.shutdown(wait=True)
            progress.close()

        outputs = persist_state(run_status)
    else:
        outputs = persist_state(run_status)

    outputs.attrs["run_status"] = run_status
    outputs.attrs["expected_output_count"] = len(work_items)
    outputs.attrs["resumed_output_count"] = resumed_output_count
    outputs.attrs["checkpoint_path"] = str(checkpoint_path)

    return outputs, output_path, metadata_path


def save_generation_metadata(
    *,
    config: GenerationConfig,
    config_path: str | Path,
    candidates: list[CandidateModelConfig],
    cases: pd.DataFrame,
    outputs: pd.DataFrame,
    output_path: str | Path,
    pricing_catalog: EndpointPricingCatalogConfig | None,
    pricing_catalog_sha256: str | None,
    run_status: str,
    checkpoint_path: str | Path,
    expected_output_count: int,
    resumed_output_count: int,
    generation_run_fingerprint: str,
    cost_preflight: dict[str, Any] | None,
) -> Path:
    output_path = Path(output_path)
    checkpoint_path = Path(checkpoint_path)

    metadata_path = output_path.with_name(
        output_path.stem
        + "_generation_metadata.json"
    )

    if "generation_status" in outputs.columns:
        successful_outputs = outputs[
            outputs["generation_status"] == "success"
        ]
        failed_outputs = outputs[
            outputs["generation_status"] == "error"
        ]
    else:
        successful_outputs = outputs.iloc[0:0]
        failed_outputs = outputs.iloc[0:0]
    operation_summary = summarize_stage(
        outputs,
        "generation",
    )
    operations_by_model = (
        {
            str(model_name): summarize_stage(
                group,
                "generation",
            )
            for model_name, group in outputs.groupby(
                "model_name",
                dropna=False,
            )
        }
        if "model_name" in outputs.columns
        else {}
    )
    pricing_snapshot = build_pricing_snapshot(
        path=config.endpoint_pricing_path,
        catalog=pricing_catalog,
        catalog_sha256=pricing_catalog_sha256,
        references=[
            {
                "name": candidate.name,
                "model": candidate.model,
                "pricing_id": candidate.pricing_id,
            }
            for candidate in candidates
        ],
    )
    if "generation_budget_cost_usd" in outputs.columns:
        budget_costs = pd.to_numeric(
            outputs["generation_budget_cost_usd"],
            errors="coerce",
        )
        budget_cost_observations = int(budget_costs.notna().sum())
        budget_cost_usd = (
            float(budget_costs.sum())
            if budget_cost_observations == len(outputs)
            and len(outputs) > 0
            else None
        )
    else:
        budget_cost_observations = 0
        budget_cost_usd = None
    budgeted_cost_usd = (
        budget_cost_usd
        * config.generation.cost_safety_multiplier
        if budget_cost_usd is not None
        else None
    )

    metadata = {
        "schema_version": "0.6",
        "created_at_utc": (
            datetime.now(timezone.utc).isoformat()
        ),
        "evalanche_version": __version__,
        "run": {
            "name": config.run.name,
            "config_path": str(config_path),
            "input_path": str(config.run.input_path),
            "output_path": str(output_path),
            "candidate_models_path": str(
                config.candidate_models_path
            ),
            "endpoint_pricing_path": (
                str(config.endpoint_pricing_path)
                if config.endpoint_pricing_path is not None
                else None
            ),
        },
        "hashes": {
            "config_sha256": sha256_file(config_path),
            "input_sha256": sha256_file(
                config.run.input_path
            ),
            "candidate_models_sha256": sha256_file(
                config.candidate_models_path
            ),
            "endpoint_pricing_sha256": pricing_catalog_sha256,
            "output_sha256": sha256_file(output_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "cost_preflight_sample_sha256": (
                cost_preflight.get("sample_sha256")
                if cost_preflight is not None
                else None
            ),
        },
        "prompt": {
            "system": config.prompt.system,
            "template": config.prompt.template,
        },
        "generation": {
            "request_api": config.generation.request_api,
            "continue_on_error": (
                config.generation.continue_on_error
            ),
            "max_completion_tokens": (
                config.generation.max_completion_tokens
            ),
            "max_workers": config.generation.max_workers,
            "checkpoint_every": (
                config.generation.checkpoint_every
            ),
            "resume": config.generation.resume,
            "max_requests_per_minute": (
                config.generation.max_requests_per_minute
            ),
            "cost_preflight_sample_path": (
                str(
                    config.generation.cost_preflight_sample_path
                )
                if config.generation.cost_preflight_sample_path
                is not None
                else None
            ),
            "maximum_estimated_cost_usd": (
                config.generation.maximum_estimated_cost_usd
            ),
            "cost_safety_multiplier": (
                config.generation.cost_safety_multiplier
            ),
        },
        "candidate_models": [
            {
                "name": candidate.name,
                "model": candidate.model,
                "temperature": candidate.temperature,
                "reasoning_effort": candidate.reasoning_effort,
                "max_retries": candidate.max_retries,
                "max_completion_tokens": (
                    candidate.max_completion_tokens
                ),
                "pricing_id": candidate.pricing_id,
                "provider_model_version": (
                    candidate.provider_model_version
                ),
                "deployment_type": candidate.deployment_type,
                "resource_region": candidate.resource_region,
            }
            for candidate in candidates
        ],
        "input_data": {
            "rows": int(len(cases)),
            "columns": list(cases.columns),
            "case_count": int(
                cases["case_id"].nunique()
            ),
            "evaluation_types": sorted(
                cases["evaluation_type"]
                .astype(str)
                .unique()
                .tolist()
            ),
        },
        "outputs": {
            "rows": int(len(outputs)),
            "success_count": int(
                len(successful_outputs)
            ),
            "error_count": int(
                len(failed_outputs)
            ),
            "models": sorted(
                outputs["model_name"].astype(str).unique().tolist()
                if "model_name" in outputs.columns
                else []
            ),
            "failure_rate": operation_summary[
                "failure_rate"
            ],
            "prompt_tokens": operation_summary[
                "prompt_tokens"
            ],
            "completion_tokens": operation_summary[
                "completion_tokens"
            ],
            "total_tokens": operation_summary["total_tokens"],
            "average_generation_seconds": operation_summary[
                "average_seconds"
            ],
            "p95_generation_seconds": operation_summary[
                "p95_seconds"
            ],
            "cost_usd": operation_summary["cost_usd"],
            "cost_coverage": operation_summary[
                "cost_coverage"
            ],
            "configured_cost_usd": operation_summary[
                "configured_cost_usd"
            ],
            "configured_cost_coverage": operation_summary[
                "configured_cost_coverage"
            ],
            "provider_reported_cost_usd": operation_summary[
                "provider_reported_cost_usd"
            ],
            "provider_reported_cost_coverage": operation_summary[
                "provider_reported_cost_coverage"
            ],
            "budget_cost_usd": budget_cost_usd,
            "budget_cost_observations": (
                budget_cost_observations
            ),
            "budgeted_cost_usd": budgeted_cost_usd,
        },
        "execution": {
            "status": run_status,
            "generation_run_fingerprint": (
                generation_run_fingerprint
            ),
            "expected_output_count": expected_output_count,
            "completed_output_count": int(len(outputs)),
            "pending_output_count": max(
                0,
                expected_output_count - len(outputs),
            ),
            "resumed_output_count": resumed_output_count,
            "checkpoint_path": str(checkpoint_path),
        },
        "cost_preflight": cost_preflight,
        "endpoint_pricing": pricing_snapshot,
        "operations": {
            "cost_currency": COST_CURRENCY,
            "cost_policy": COST_POLICY,
            "generation": operation_summary,
            "by_model": operations_by_model,
        },
        "limitations": [
            (
                "Generated outputs are specific to the "
                "candidate models, prompts, inputs, and "
                "API configuration used in this run."
            ),
            (
                "Generated outputs should be inspected "
                "before important evaluations."
            ),
            (
                "Using the same model as both candidate "
                "and judge may introduce evaluation bias."
            ),
            (
                "Configured token costs are estimates from recorded usage "
                "and declared rates, not provider invoices."
            ),
            (
                "The local cost guard applies a configured safety "
                "multiplier and reserves worst-case retry cost, but it is "
                "not an Azure billing hard stop and cannot price failed "
                "provider calls that return no usage."
            ),
        ],
    }

    _atomic_write_text(
        metadata_path,
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
    )

    return metadata_path

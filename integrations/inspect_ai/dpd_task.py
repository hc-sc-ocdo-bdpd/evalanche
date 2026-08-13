from __future__ import annotations

import asyncio
import json
from time import monotonic
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig
from inspect_ai.model import ModelOutput, ModelUsage
from inspect_ai.scorer import (
    Score,
    Scorer,
    Target,
    accuracy,
    mean,
    scorer,
    stderr,
)
from inspect_ai.solver import Generate, Solver, TaskState, solver

from integrations.inspect_ai import (
    DEFAULT_BENCHMARK_REFERENCE,
    DEFAULT_HISTORICAL_OUTPUTS_PATH,
)
from integrations.inspect_ai.adapter import (
    DpdContext,
    json_safe,
    load_dpd_cases,
    load_dpd_context,
    load_dpd_historical_import,
    load_dpd_replay_records,
    score_dpd_completion,
)


class _RequestRateLimiter:
    def __init__(self, requests_per_minute: float | None) -> None:
        self.interval = (
            60.0 / requests_per_minute if requests_per_minute else 0.0
        )
        self.next_request_at = 0.0
        self.lock = asyncio.Lock()

    async def wait(self) -> None:
        if not self.interval:
            return
        async with self.lock:
            now = monotonic()
            delay = max(0.0, self.next_request_at - now)
            if delay:
                await asyncio.sleep(delay)
            self.next_request_at = max(monotonic(), self.next_request_at) + (
                self.interval
            )


@solver
def evalanche_rate_limited_generate(
    requests_per_minute: float | None = None,
) -> Solver:
    limiter = _RequestRateLimiter(requests_per_minute)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        await limiter.wait()
        return await generate(state)

    return solve


@solver
def evalanche_saved_output() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        del generate
        saved_model_output = str(
            state.metadata.pop("saved_model_output")
        )
        output = ModelOutput.from_content(
            model=f"replay/{state.metadata['historical_model_id']}",
            content=saved_model_output,
        )
        input_tokens = _metadata_int(
            state.metadata, "generation_prompt_tokens"
        )
        output_tokens = _metadata_int(
            state.metadata, "generation_completion_tokens"
        )
        total_tokens = _metadata_int(
            state.metadata, "generation_total_tokens"
        )
        cached_tokens = _metadata_int(
            state.metadata, "generation_cached_prompt_tokens"
        )
        cost = _metadata_float(state.metadata, "generation_cost_usd")
        if input_tokens or output_tokens or total_tokens or cost is not None:
            output.usage = ModelUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens or input_tokens + output_tokens,
                input_tokens_cache_read=cached_tokens or None,
                total_cost=cost,
            )
        output.time = _metadata_float(
            state.metadata, "generation_seconds"
        )
        output.metadata = {
            "evalanche_execution_mode": state.metadata.get(
                "execution_mode", "saved_output_replay"
            ),
            "evalanche_historical_run_id": state.metadata.get(
                "historical_run_id"
            ),
            "evalanche_historical_source_sha256": state.metadata.get(
                "historical_source_sha256"
            ),
        }
        state.output = output
        return state

    return solve


def _metadata_int(metadata: dict[str, Any], name: str) -> int:
    value = metadata.get(name)
    if value is None or str(value).strip() == "":
        return 0
    return int(float(str(value)))


def _metadata_float(
    metadata: dict[str, Any], name: str
) -> float | None:
    value = metadata.get(name)
    if value is None or str(value).strip() == "":
        return None
    return float(str(value))


@scorer(
    name="evalanche_dpd",
    metrics={
        "strict": [accuracy(), stderr()],
        "field_score": [mean(), stderr()],
        "valid_json": [accuracy(), stderr()],
    },
)
def evalanche_dpd_scorer(
    root: str = ".",
    benchmark_reference: str = DEFAULT_BENCHMARK_REFERENCE,
) -> Scorer:
    context = load_dpd_context(root, benchmark_reference)

    async def score(state: TaskState, target: Target) -> Score:
        model_name = str(
            state.metadata.get("historical_model_id") or state.model
        )
        result = score_dpd_completion(
            context,
            case_id=str(state.metadata["case_id"]),
            expected_output=target.text,
            model_output=state.output.completion,
            model_name=model_name,
            evaluation_type=str(state.metadata["evaluation_type"]),
        )
        field_score = float(result["json_field_match_rate"] or 0.0)
        mismatched_fields = json.loads(
            str(result["json_mismatched_fields"] or "[]")
        )
        metadata: dict[str, Any] = {
            "case_id": result["case_id"],
            "model_name": result["model_name"],
            "metric_passed": bool(result["metric_passed"]),
            "output_is_json": bool(result["output_is_json"]),
            "json_field_match_rate": field_score,
            "json_missing_fields": json.loads(
                str(result["json_missing_fields"] or "[]")
            ),
            "json_extra_fields": json.loads(
                str(result["json_extra_fields"] or "[]")
            ),
            "json_mismatched_fields": mismatched_fields,
        }
        if "expected_metric_passed" in state.metadata:
            replay_parity = {
                "strict": bool(result["metric_passed"])
                == bool(state.metadata["expected_metric_passed"]),
                "field_score": abs(
                    field_score
                    - float(state.metadata["expected_field_score"])
                )
                <= 1e-12,
                "mismatched_fields": mismatched_fields
                == state.metadata["expected_mismatched_fields"],
            }
            if "expected_output_is_json" in state.metadata:
                replay_parity["valid_json"] = bool(
                    result["output_is_json"]
                ) == bool(state.metadata["expected_output_is_json"])
            metadata["replay_parity"] = replay_parity
        return Score(
            value={
                "strict": "C" if result["metric_passed"] else "I",
                "field_score": field_score,
                "valid_json": "C" if result["output_is_json"] else "I",
            },
            answer=state.output.completion,
            explanation=(
                "Canonical Evalanche deterministic JSON comparison."
            ),
            metadata=json_safe(metadata),
        )

    return score


def _case_sample(
    row: dict[str, Any],
    *,
    system_message: str,
    prompt_template: str,
    fingerprint: str,
    sample_id: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> Sample:
    metadata = {
        "case_id": row["case_id"],
        "product_id": row["product_id"],
        "language": row["language"],
        "stratum": row["stratum"],
        "split": row.get("split"),
        "evaluation_type": row["evaluation_type"],
        "benchmark_fingerprint": fingerprint,
    }
    metadata.update(extra_metadata or {})
    return Sample(
        id=sample_id or str(row["case_id"]),
        input=[
            ChatMessageSystem(content=system_message),
            ChatMessageUser(
                content=prompt_template.format(input=str(row["input"]))
            ),
        ],
        target=str(row["expected_output"]),
        metadata=json_safe(metadata),
    )


@task
def dpd(
    scope: str = "demo",
    benchmark_reference: str = DEFAULT_BENCHMARK_REFERENCE,
    root: str = ".",
    requests_per_minute: float | None = None,
) -> Task:
    context = load_dpd_context(root, benchmark_reference)
    cases = load_dpd_cases(context, scope)
    rpm = (
        requests_per_minute
        if requests_per_minute is not None
        else context.benchmark.runtime.max_requests_per_minute
    )
    fingerprint = context.fingerprint["compatibility_sha256"]
    samples = [
        _case_sample(
            row,
            system_message=context.benchmark.prompt.system,
            prompt_template=context.benchmark.prompt.template,
            fingerprint=fingerprint,
        )
        for row in cases.to_dict(orient="records")
    ]
    return Task(
        dataset=MemoryDataset(
            samples=samples,
            name=(
                f"{context.benchmark.benchmark_id}_"
                f"{context.benchmark.version}_{scope}"
            ),
            location=str(context.cases_path),
        ),
        solver=evalanche_rate_limited_generate(rpm),
        scorer=evalanche_dpd_scorer(root, benchmark_reference),
        config=GenerateConfig(
            max_tokens=context.benchmark.runtime.max_completion_tokens
        ),
        name="evalanche_dpd",
        version=context.benchmark.version,
        metadata={
            "evalanche_benchmark_reference": benchmark_reference,
            "evalanche_benchmark_fingerprint": fingerprint,
            "evalanche_scope": scope,
            "evalanche_case_count": len(samples),
            "evalanche_requests_per_minute": rpm,
        },
    )


@task
def dpd_replay(
    benchmark_reference: str = DEFAULT_BENCHMARK_REFERENCE,
    root: str = ".",
) -> Task:
    context = load_dpd_context(root, benchmark_reference)
    records = load_dpd_replay_records(context)
    fingerprint = context.fingerprint["compatibility_sha256"]
    samples = [
        Sample(
            id=record["sample_id"],
            input=record["input"],
            target=record["expected_output"],
            metadata=json_safe(
                {
                    **record,
                    "saved_model_output": record["model_output"],
                    "evaluation_type": "json",
                    "benchmark_fingerprint": fingerprint,
                }
            ),
        )
        for record in records
    ]
    return Task(
        dataset=MemoryDataset(
            samples=samples,
            name="evalanche_dpd_saved_output_replay",
            location=str(context.root / "reports/hc_dpd_census/0.2.0"),
        ),
        solver=evalanche_saved_output(),
        scorer=evalanche_dpd_scorer(root, benchmark_reference),
        model="mockllm/model",
        name="evalanche_dpd_replay",
        version=context.benchmark.version,
        metadata={
            "evalanche_benchmark_reference": benchmark_reference,
            "evalanche_benchmark_fingerprint": fingerprint,
            "evalanche_scope": "saved_output_replay",
            "evalanche_case_count": len(samples),
        },
    )


def build_dpd_historical_task(
    *,
    context: DpdContext,
    records: list[dict[str, Any]],
    model_id: str,
    source_path: str,
    source_sha256: str,
    release_id: str,
    run_info: dict[str, Any],
    benchmark_reference: str = DEFAULT_BENCHMARK_REFERENCE,
) -> Task:
    if not records:
        raise ValueError(f"No historical records selected for {model_id}.")
    observed_models = {
        str(record["historical_model_id"]) for record in records
    }
    if observed_models != {model_id}:
        raise ValueError(
            f"Historical task for {model_id} received records for "
            f"{sorted(observed_models)}."
        )
    fingerprint = context.fingerprint["compatibility_sha256"]
    metadata_keys = (
        "candidate_model",
        "generated_at_utc",
        "generation_seconds",
        "generation_prompt_tokens",
        "generation_cached_prompt_tokens",
        "generation_completion_tokens",
        "generation_total_tokens",
        "generation_cost_usd",
        "historical_model_id",
        "historical_run_id",
        "expected_metric_passed",
        "expected_output_is_json",
        "expected_field_score",
        "expected_mismatched_fields",
    )
    samples = []
    for record in records:
        extra_metadata = {
            key: record.get(key)
            for key in metadata_keys
            if key in record
        }
        extra_metadata.update(
            {
                "execution_mode": "historical_import",
                "saved_model_output": record["model_output"],
            }
        )
        samples.append(
            _case_sample(
                record,
                system_message=context.benchmark.prompt.system,
                prompt_template=context.benchmark.prompt.template,
                fingerprint=fingerprint,
                extra_metadata=extra_metadata,
            )
        )
    return Task(
        dataset=MemoryDataset(
            samples=samples,
            name=(
                f"{context.benchmark.benchmark_id}_"
                f"{context.benchmark.version}_historical_{model_id}"
            ),
            location=source_path,
        ),
        solver=evalanche_saved_output(),
        scorer=evalanche_dpd_scorer(
            str(context.root), benchmark_reference
        ),
        model="mockllm/model",
        name=f"evalanche_dpd_historical_{model_id}",
        version=context.benchmark.version,
        metadata={
            "evalanche_benchmark_reference": benchmark_reference,
            "evalanche_benchmark_fingerprint": fingerprint,
            "evalanche_scope": "historical_full_import",
            "evalanche_case_count": len(samples),
            "evalanche_model_id": model_id,
            "evalanche_execution_mode": "historical_import",
            "evalanche_historical_release_id": release_id,
            "evalanche_historical_run_id": run_info["run_id"],
            "evalanche_historical_source_path": source_path,
            "evalanche_historical_source_sha256": source_sha256,
            "evalanche_published_compatibility": run_info[
                "published_compatibility"
            ],
        },
    )


@task
def dpd_historical_import(
    model_id: str,
    source_path: str = DEFAULT_HISTORICAL_OUTPUTS_PATH,
    benchmark_reference: str = DEFAULT_BENCHMARK_REFERENCE,
    root: str = ".",
    limit: int | None = None,
) -> Task:
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive.")
    context = load_dpd_context(root, benchmark_reference)
    historical = load_dpd_historical_import(context, source_path)
    if model_id not in historical.model_ids:
        raise ValueError(
            f"Historical release has no model {model_id!r}."
        )
    selected = historical.records.loc[
        historical.records["historical_model_id"].astype(str) == model_id
    ]
    if limit is not None:
        selected = selected.head(limit)
    return build_dpd_historical_task(
        context=context,
        records=selected.to_dict(orient="records"),
        model_id=model_id,
        source_path=str(historical.source_path),
        source_sha256=historical.source_sha256,
        release_id=historical.release_id,
        run_info=historical.published_runs[model_id],
        benchmark_reference=benchmark_reference,
    )

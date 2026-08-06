from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from evalanche.benchmark_experiment import (
    build_benchmark_experiment_summary,
    discover_benchmark_evaluations,
)
from evalanche.benchmark_runner import (
    build_run_plan,
    run_experimental_benchmark,
)
from evalanche.benchmark_tiers import (
    materialize_tier_cohort,
    resolve_tier,
    tier_is_ancestor,
)
from evalanche.config import load_generation_config
from evalanche.generation import preflight_generation
from evalanche.pricing import (
    estimate_endpoint_cost_usd,
    load_endpoint_pricing_catalog,
    resolve_endpoint_price,
)
from evalanche.registry import (
    BenchmarkManifest,
    LoadedRegistry,
    ModelManifest,
    benchmark_fingerprint,
    sha256_file,
    sha256_json,
)


CAMPAIGN_SCHEMA_VERSION = "1.0"
DEFAULT_CAMPAIGN_ROOT = Path("data/generated/benchmark_campaigns")


def compatible_model_ids(
    *,
    registry: LoadedRegistry,
    benchmark: BenchmarkManifest,
) -> list[str]:
    required = set(benchmark.required_capabilities)
    return sorted(
        model_id
        for model_id, model in registry.models.items()
        if required.issubset(model.capabilities)
    )


def _validate_models(
    *,
    registry: LoadedRegistry,
    benchmark: BenchmarkManifest,
    model_ids: list[str],
) -> list[ModelManifest]:
    selected = list(dict.fromkeys(model_ids))
    if not selected:
        raise ValueError("No compatible models were selected")
    models = [registry.resolve_model(model_id) for model_id in selected]
    required = set(benchmark.required_capabilities)
    incompatible = [
        model.model_id
        for model in models
        if not required.issubset(model.capabilities)
    ]
    if incompatible:
        raise ValueError(
            "Selected models do not satisfy benchmark capabilities: "
            f"{incompatible}"
        )
    return models


def _completed_models(
    *,
    registry: LoadedRegistry,
    benchmark: BenchmarkManifest,
    tier_name: str,
) -> set[str]:
    try:
        results, _ = discover_benchmark_evaluations(
            registry=registry,
            benchmark=benchmark,
            tier_name=tier_name,
        )
    except ValueError as error:
        if str(error).startswith("No completed compatible evaluations"):
            return set()
        raise
    return set(results["model_name"].astype(str))


def select_models_for_promotion(
    *,
    registry: LoadedRegistry,
    benchmark: BenchmarkManifest,
    source_tier: str,
    model_ids: list[str] | None = None,
    access_scope: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if model_ids is not None and not model_ids:
        raise ValueError(
            "No access-confirmed models satisfy the benchmark capabilities"
        )
    tier = resolve_tier(benchmark, source_tier)
    if tier.promotion is None:
        raise ValueError(
            f"Tier {source_tier!r} does not define promotion gates"
        )
    summary_result = build_benchmark_experiment_summary(
        registry=registry,
        benchmark=benchmark,
        model_ids=model_ids,
        tier_name=source_tier,
        access_scope=access_scope,
    )
    records: list[dict[str, Any]] = []
    selected: list[str] = []
    for row in summary_result["summary"].to_dict(orient="records"):
        requests = int(row.get("generation_requests", 0) or 0)
        failures = int(row.get("generation_errors", 0) or 0)
        failure_rate = failures / requests if requests else 1.0
        pass_rate = row.get("pass_rate")
        qualifies = (
            pass_rate is not None
            and not pd.isna(pass_rate)
            and float(pass_rate) >= tier.promotion.minimum_pass_rate
            and failure_rate
            <= tier.promotion.maximum_generation_failure_rate
        )
        model_id = str(row["model_name"])
        if qualifies:
            selected.append(model_id)
        records.append(
            {
                "model_id": model_id,
                "pass_rate": pass_rate,
                "minimum_pass_rate": tier.promotion.minimum_pass_rate,
                "generation_failure_rate": failure_rate,
                "maximum_generation_failure_rate": (
                    tier.promotion.maximum_generation_failure_rate
                ),
                "promoted": qualifies,
            }
        )
    return {
        "source_tier": source_tier,
        "selected_model_ids": sorted(selected),
        "decisions": pd.DataFrame(records),
        "summary": summary_result,
    }


def _campaign_directory(
    *,
    registry: LoadedRegistry,
    benchmark: BenchmarkManifest,
    tier_name: str,
    campaign_root: str | Path,
) -> Path:
    stem = f"{benchmark.benchmark_id}_{benchmark.version}".replace(".", "_")
    return registry.root / campaign_root / stem / tier_name


def _source_generation_rows(
    *,
    registry: LoadedRegistry,
    benchmark: BenchmarkManifest,
    model_id: str,
    source_tier: str,
) -> list[pd.DataFrame]:
    _, sources = discover_benchmark_evaluations(
        registry=registry,
        benchmark=benchmark,
        model_ids=[model_id],
        tier_name=source_tier,
    )
    frames: list[pd.DataFrame] = []
    for source in sources:
        plan_path = registry.root / source["run_plan_path"]
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        output_value = plan.get("paths", {}).get("generation_output")
        if not output_value:
            continue
        output_path = registry.root / output_value
        if not output_path.is_file():
            continue
        frame = pd.read_csv(output_path)
        frame = frame[
            (frame["model_name"].astype(str) == model_id)
            & (frame["generation_status"].astype(str) == "success")
        ].copy()
        if not frame.empty:
            frames.append(frame)
    return frames


def _endpoint_price(
    *,
    registry: LoadedRegistry,
    model: ModelManifest,
) -> Any:
    if model.pricing_catalog_path is None or model.pricing_id is None:
        raise ValueError(
            f"Cost-guarded tier runs require pricing for {model.model_id!r}"
        )
    catalog = load_endpoint_pricing_catalog(
        registry.root / model.pricing_catalog_path
    )
    return resolve_endpoint_price(
        catalog=catalog,
        pricing_id=model.pricing_id,
        model=model.provider_route,
    )


def _write_cost_sample(
    *,
    registry: LoadedRegistry,
    benchmark: BenchmarkManifest,
    tier_name: str,
    model: ModelManifest,
    destination: Path,
) -> dict[str, Any]:
    tier = resolve_tier(benchmark, tier_name)
    frames: list[pd.DataFrame] = []
    source = "initial_token_estimate"
    if tier.cost.sample_from is not None:
        if not tier_is_ancestor(
            benchmark,
            ancestor=tier.cost.sample_from,
            descendant=tier_name,
        ):
            raise ValueError(
                f"Tier {tier_name!r} cost sample {tier.cost.sample_from!r} "
                "must be an inherited tier"
            )
        frames = _source_generation_rows(
            registry=registry,
            benchmark=benchmark,
            model_id=model.model_id,
            source_tier=tier.cost.sample_from,
        )
        if frames:
            source = f"observed_{tier.cost.sample_from}_tier"

    required = [
        "model_name",
        "candidate_model",
        "generation_status",
        "generation_prompt_tokens",
        "generation_completion_tokens",
    ]
    if frames:
        sample = pd.concat(frames, ignore_index=True)
        missing = sorted(set(required) - set(sample.columns))
        if missing:
            raise ValueError(
                f"Observed cost sample for {model.model_id!r} is missing "
                f"columns: {missing}"
            )
        sample = sample[required].copy()
        sample["generation_prompt_tokens"] = pd.to_numeric(
            sample["generation_prompt_tokens"], errors="coerce"
        )
        sample["generation_completion_tokens"] = pd.to_numeric(
            sample["generation_completion_tokens"], errors="coerce"
        )
        sample = sample.dropna(
            subset=[
                "generation_prompt_tokens",
                "generation_completion_tokens",
            ]
        )
    if not frames or sample.empty:
        if (
            tier.cost.initial_prompt_tokens is None
            or tier.cost.initial_completion_tokens is None
        ):
            raise ValueError(
                f"No observed cost sample exists for {model.model_id!r} and "
                f"tier {tier_name!r} has no initial token fallback"
            )
        sample = pd.DataFrame(
            [
                {
                    "model_name": model.model_id,
                    "candidate_model": model.provider_route,
                    "generation_status": "success",
                    "generation_prompt_tokens": (
                        tier.cost.initial_prompt_tokens
                    ),
                    "generation_completion_tokens": (
                        tier.cost.initial_completion_tokens
                    ),
                }
            ]
        )
        source = "initial_token_estimate"
    if sample.empty:
        raise ValueError(
            f"No usable token observations exist for {model.model_id!r}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(destination, index=False, lineterminator="\n")
    price = _endpoint_price(registry=registry, model=model)
    costs = [
        estimate_endpoint_cost_usd(
            price,
            prompt_tokens=int(row["generation_prompt_tokens"]),
            completion_tokens=int(row["generation_completion_tokens"]),
            cached_prompt_tokens=0,
        )
        for row in sample.to_dict(orient="records")
    ]
    if any(value is None for value in costs):
        raise ValueError(
            f"Cost sample for {model.model_id!r} could not be priced"
        )
    numeric_costs = [float(value) for value in costs if value is not None]
    request_ceiling = (
        max(numeric_costs) * tier.cost.request_ceiling_multiplier
    )
    return {
        "path": destination,
        "source": source,
        "rows": int(len(sample)),
        "sha256": sha256_file(destination),
        "average_request_cost_usd": sum(numeric_costs) / len(numeric_costs),
        "maximum_sample_cost_usd": max(numeric_costs),
        "request_cost_ceiling_usd": request_ceiling,
    }


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _write_campaign_artifacts(
    *,
    directory: Path,
    payload: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    table_path = directory / "cost_preflight.csv"
    plan_path = directory / "campaign.json"
    pd.DataFrame(rows).to_csv(table_path, index=False, lineterminator="\n")
    payload["artifacts"] = {
        "cost_preflight": {
            "path": table_path.relative_to(directory).as_posix(),
            "sha256": sha256_file(table_path),
        }
    }
    plan_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {"campaign": plan_path, "cost_preflight": table_path}


def build_tier_campaign_preflight(
    *,
    registry: LoadedRegistry,
    benchmark: BenchmarkManifest,
    tier_name: str,
    model_ids: list[str],
    maximum_cost_usd: float | None = None,
    campaign_root: str | Path = DEFAULT_CAMPAIGN_ROOT,
    access_scope: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Plan every pending call and price the campaign before execution."""

    if maximum_cost_usd is not None and maximum_cost_usd <= 0:
        raise ValueError("maximum_cost_usd must be positive")
    tier = resolve_tier(benchmark, tier_name)
    models = _validate_models(
        registry=registry,
        benchmark=benchmark,
        model_ids=model_ids,
    )
    materialized = materialize_tier_cohort(
        registry=registry,
        benchmark=benchmark,
        tier_name=tier_name,
    )
    completed = _completed_models(
        registry=registry,
        benchmark=benchmark,
        tier_name=tier_name,
    )
    if tier.inherits is not None:
        parent_completed = _completed_models(
            registry=registry,
            benchmark=benchmark,
            tier_name=tier.inherits,
        )
        missing_prerequisites = sorted(
            model.model_id
            for model in models
            if model.model_id not in completed
            and model.model_id not in parent_completed
        )
        if missing_prerequisites:
            raise ValueError(
                f"Tier {tier_name!r} requires completed {tier.inherits!r} "
                "results first for: "
                + ", ".join(missing_prerequisites)
            )

    directory = _campaign_directory(
        registry=registry,
        benchmark=benchmark,
        tier_name=tier_name,
        campaign_root=campaign_root,
    )
    rows: list[dict[str, Any]] = []
    model_plans: dict[str, dict[str, Any]] = {}
    for model in models:
        if model.model_id in completed:
            rows.append(
                {
                    "model_id": model.model_id,
                    "status": "complete",
                    "expected_requests": 0,
                    "checkpointed_requests": 0,
                    "pending_requests": 0,
                    "existing_budget_cost_usd": 0.0,
                    "projected_remaining_cost_usd": 0.0,
                    "projected_budgeted_total_usd": 0.0,
                    "cost_sample_source": "not_needed",
                    "request_cost_ceiling_usd": 0.0,
                }
            )
            continue

        sample = _write_cost_sample(
            registry=registry,
            benchmark=benchmark,
            tier_name=tier_name,
            model=model,
            destination=(
                directory / "cost_samples" / f"{model.model_id}.csv"
            ),
        )
        plan = build_run_plan(
            registry=registry,
            benchmark=benchmark,
            model=model,
            tier_name=tier_name,
            cost_preflight_sample_path=_relative(
                sample["path"], registry.root
            ),
            cost_safety_multiplier=tier.cost.safety_multiplier,
            request_cost_ceiling_usd=sample[
                "request_cost_ceiling_usd"
            ],
        )
        generation_path = registry.root / plan["paths"]["generation_config"]
        generation_config = load_generation_config(generation_path)
        preflight = preflight_generation(config=generation_config)
        row = {
            "model_id": model.model_id,
            "status": "pending",
            "expected_requests": preflight["expected_output_count"],
            "checkpointed_requests": preflight[
                "checkpointed_output_count"
            ],
            "pending_requests": preflight["pending_output_count"],
            "existing_budget_cost_usd": preflight[
                "existing_budget_cost_usd"
            ],
            "projected_remaining_cost_usd": preflight[
                "projected_remaining_cost_usd"
            ],
            "projected_budgeted_total_usd": preflight[
                "projected_budgeted_total_usd"
            ],
            "cost_safety_multiplier": tier.cost.safety_multiplier,
            "cost_sample_source": sample["source"],
            "cost_sample_rows": sample["rows"],
            "average_sample_cost_usd": sample[
                "average_request_cost_usd"
            ],
            "request_cost_ceiling_usd": sample[
                "request_cost_ceiling_usd"
            ],
            "run_plan": plan["paths"]["run_plan"],
        }
        rows.append(row)
        model_plans[model.model_id] = {
            "model": model,
            "plan": plan,
            "sample": sample,
            "preflight": preflight,
        }

    projected = sum(
        float(row["projected_budgeted_total_usd"]) for row in rows
    )
    pending_requests = sum(int(row["pending_requests"]) for row in rows)
    pending_models = sum(row["status"] != "complete" for row in rows)
    within_budget = (
        None if maximum_cost_usd is None else projected <= maximum_cost_usd
    )
    campaign_id = sha256_json(
        {
            "benchmark": benchmark_fingerprint(registry, benchmark),
            "tier_definition_sha256": materialized["plan"][
                "definition_sha256"
            ],
            "tier_case_ids_sha256": materialized["plan"]["cumulative"][
                "case_ids_sha256"
            ],
            "models": [model.model_id for model in models],
            "access_scope": access_scope,
        }
    )[:16]
    payload = {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "benchmark": f"{benchmark.benchmark_id}@{benchmark.version}",
        "benchmark_status": benchmark.status,
        "tier": tier_name,
        "tier_chain": materialized["plan"]["chain"],
        "cumulative_cases": materialized["plan"]["cumulative"]["cases"],
        "execution_cases": materialized["plan"]["execution"]["cases"],
        "models": [model.model_id for model in models],
        "access_scope": access_scope,
        "pending_requests": pending_requests,
        "pending_models": pending_models,
        "projected_budgeted_total_usd": projected,
        "maximum_cost_usd": maximum_cost_usd,
        "within_budget": within_budget,
        "status": "preflight_complete",
        "preflight": rows,
    }
    paths = _write_campaign_artifacts(
        directory=directory,
        payload=payload,
        rows=rows,
    )
    return {
        "campaign_id": campaign_id,
        "benchmark": benchmark,
        "tier": tier,
        "tier_name": tier_name,
        "models": models,
        "rows": rows,
        "model_plans": model_plans,
        "projected_budgeted_total_usd": projected,
        "maximum_cost_usd": maximum_cost_usd,
        "within_budget": within_budget,
        "pending_requests": pending_requests,
        "pending_models": pending_models,
        "payload": payload,
        "paths": paths,
        "directory": directory,
    }


def _generation_budget_cost(path: Path) -> float:
    if not path.is_file():
        return 0.0
    rows = pd.read_csv(path)
    for column in (
        "generation_budget_cost_usd",
        "generation_configured_cost_usd",
        "generation_cost_usd",
    ):
        if column in rows.columns:
            values = pd.to_numeric(rows[column], errors="coerce")
            if values.notna().any():
                return float(values.fillna(0).sum())
    raise ValueError(f"Generation output has no usable cost evidence: {path}")


def execute_tier_campaign(
    *,
    registry: LoadedRegistry,
    preflight: dict[str, Any],
) -> dict[str, Any]:
    maximum = preflight["maximum_cost_usd"]
    if maximum is None:
        raise ValueError("Tier execution requires maximum_cost_usd")
    if preflight["within_budget"] is not True:
        raise ValueError(
            "Aggregate cost preflight failed: projected budgeted cost "
            f"${preflight['projected_budgeted_total_usd']:.2f} exceeds "
            f"the ${maximum:.2f} campaign limit"
        )

    benchmark = preflight["benchmark"]
    tier = preflight["tier"]
    tier_name = preflight["tier_name"]
    rows_by_model = {row["model_id"]: row for row in preflight["rows"]}
    existing_budgeted = sum(
        float(row["existing_budget_cost_usd"])
        * float(row.get("cost_safety_multiplier", 1.0))
        for row in preflight["rows"]
    )
    completed_now: list[str] = []
    latest_summary = None
    payload = preflight["payload"]
    payload["status"] = "running"

    try:
        for model in preflight["models"]:
            row = rows_by_model[model.model_id]
            if row["status"] == "complete":
                continue
            plan_data = preflight["model_plans"][model.model_id]
            remaining = float(maximum) - existing_budgeted
            model_existing_budgeted = (
                float(row["existing_budget_cost_usd"])
                * tier.cost.safety_multiplier
            )
            if remaining <= 0 and row["pending_requests"]:
                raise ValueError(
                    "Campaign budget is exhausted before all pending models "
                    "could start"
                )
            model_limit = model_existing_budgeted + max(remaining, 0.000001)
            result = run_experimental_benchmark(
                registry=registry,
                benchmark=benchmark,
                model=model,
                tier_name=tier_name,
                cost_preflight_sample_path=_relative(
                    plan_data["sample"]["path"], registry.root
                ),
                maximum_estimated_cost_usd=model_limit,
                cost_safety_multiplier=tier.cost.safety_multiplier,
                request_cost_ceiling_usd=plan_data["sample"][
                    "request_cost_ceiling_usd"
                ],
            )
            generation_path = (
                registry.root
                / result["plan"]["paths"]["generation_output"]
            )
            actual_budgeted = (
                _generation_budget_cost(generation_path)
                * tier.cost.safety_multiplier
            )
            existing_budgeted = (
                existing_budgeted
                - model_existing_budgeted
                + actual_budgeted
            )
            if existing_budgeted > float(maximum):
                raise ValueError(
                    "Observed campaign cost exceeded the local budget after "
                    f"{model.model_id!r}; no additional model will start"
                )
            row["status"] = "complete"
            row["observed_budgeted_cost_usd"] = actual_budgeted
            completed_now.append(model.model_id)
            latest_summary = build_benchmark_experiment_summary(
                registry=registry,
                benchmark=benchmark,
                model_ids=[
                    str(selected["model_id"])
                    for selected in preflight["rows"]
                ],
                tier_name=tier_name,
                access_scope=preflight["payload"].get("access_scope"),
            )
            payload["status"] = "running"
            payload["completed_models"] = completed_now
            payload["observed_budgeted_cost_usd"] = existing_budgeted
            _write_campaign_artifacts(
                directory=preflight["directory"],
                payload=payload,
                rows=preflight["rows"],
            )
    except BaseException:
        payload["status"] = "stopped_safely"
        payload["completed_models"] = completed_now
        payload["observed_budgeted_cost_usd"] = existing_budgeted
        _write_campaign_artifacts(
            directory=preflight["directory"],
            payload=payload,
            rows=preflight["rows"],
        )
        raise

    payload["status"] = "complete"
    payload["completed_models"] = completed_now
    payload["observed_budgeted_cost_usd"] = existing_budgeted
    paths = _write_campaign_artifacts(
        directory=preflight["directory"],
        payload=payload,
        rows=preflight["rows"],
    )
    return {
        "campaign_id": preflight["campaign_id"],
        "completed_models": completed_now,
        "observed_budgeted_cost_usd": existing_budgeted,
        "summary": latest_summary,
        "paths": paths,
        "status": "complete",
    }

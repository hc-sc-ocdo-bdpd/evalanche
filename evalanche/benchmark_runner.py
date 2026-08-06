from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from evalanche.benchmark_tiers import materialize_tier_cohort
from evalanche.config import (
    load_evaluation_config,
    load_generation_config,
)
from evalanche.evaluation import evaluate_cases
from evalanche.generation import generate_outputs
from evalanche.io import load_eval_cases
from evalanche.leaderboard import (
    build_benchmark_index,
    build_leaderboard,
)
from evalanche.registry import (
    BenchmarkManifest,
    LoadedRegistry,
    ModelManifest,
    benchmark_fingerprint,
    dump_yaml,
)
from evalanche.result_bundle import register_result_bundle
from evalanche.routing import JUDGE

RUN_PLAN_SCHEMA_VERSION = "1.0"


def _paths(
    *,
    registry: LoadedRegistry,
    benchmark: BenchmarkManifest,
    model: ModelManifest,
    tier_name: str | None = None,
) -> dict[str, Path]:
    stem = (
        f"{benchmark.benchmark_id}_{benchmark.version}_{model.model_id}"
        .replace(".", "_")
    )
    if tier_name is not None:
        stem = f"{stem}_{tier_name}"
    plan_dir = registry.root / "data/generated/registry_plans" / stem
    output_dir = registry.root / "data/generated/registry_runs" / stem
    results_dir = registry.root / "results/registry_runs" / stem
    return {
        "plan_dir": plan_dir,
        "candidate_config": plan_dir / "candidate.yaml",
        "generation_config": plan_dir / "generation.yaml",
        "evaluation_config": plan_dir / "evaluation.yaml",
        "run_plan": plan_dir / "run_plan.json",
        "generation_output": output_dir / "outputs.csv",
        "evaluation_output": results_dir / "evaluation.csv",
    }


def build_run_plan(
    *,
    registry: LoadedRegistry,
    benchmark: BenchmarkManifest,
    model: ModelManifest,
    tier_name: str | None = None,
    cost_preflight_sample_path: str | Path | None = None,
    maximum_estimated_cost_usd: float | None = None,
    cost_safety_multiplier: float = 1.0,
    request_cost_ceiling_usd: float | None = None,
) -> dict[str, Any]:
    missing_capabilities = sorted(
        set(benchmark.required_capabilities) - set(model.capabilities)
    )
    if missing_capabilities:
        raise ValueError(
            f"Model {model.model_id!r} does not declare benchmark-required "
            f"capabilities: {missing_capabilities}"
        )

    tier_materialization = None
    cases_path = registry.root / benchmark.dataset.cases_path
    if tier_name is not None:
        tier_materialization = materialize_tier_cohort(
            registry=registry,
            benchmark=benchmark,
            tier_name=tier_name,
        )
        cases_path = tier_materialization["execution_path"]
    cases = pd.read_csv(cases_path)
    paths = _paths(
        registry=registry,
        benchmark=benchmark,
        model=model,
        tier_name=tier_name,
    )
    candidate = {
        "models": [
            {
                "name": model.model_id,
                "model": model.provider_route,
                "temperature": model.request.temperature,
                "reasoning_effort": model.request.reasoning_effort,
                "max_retries": model.request.max_retries,
                "max_completion_tokens": (
                    model.request.max_completion_tokens
                    or benchmark.runtime.max_completion_tokens
                ),
                "pricing_id": model.pricing_id,
                "provider_model_version": model.provider_model_version,
                "deployment_type": model.deployment_type,
                "resource_region": model.resource_region,
            }
        ]
    }
    candidate["models"][0] = {
        key: value
        for key, value in candidate["models"][0].items()
        if value is not None
    }
    generation = {
        "run": {
            "name": f"{benchmark.benchmark_id}_{model.model_id}",
            "input_path": cases_path.relative_to(registry.root).as_posix(),
            "output_path": paths["generation_output"]
            .relative_to(registry.root)
            .as_posix(),
        },
        "candidate_models_path": paths["candidate_config"]
        .relative_to(registry.root)
        .as_posix(),
        "prompt": benchmark.prompt.model_dump(mode="json"),
        "generation": {
            "continue_on_error": True,
            "request_api": benchmark.runtime.request_api,
            "max_completion_tokens": (
                model.request.max_completion_tokens
                or benchmark.runtime.max_completion_tokens
            ),
            "max_workers": benchmark.runtime.max_workers,
            "max_requests_per_minute": (
                benchmark.runtime.max_requests_per_minute
            ),
            "checkpoint_every": benchmark.runtime.checkpoint_every,
            "resume": benchmark.runtime.resume,
        },
    }
    if cost_preflight_sample_path is not None:
        generation["generation"]["cost_preflight_sample_path"] = Path(
            cost_preflight_sample_path
        ).as_posix()
    if maximum_estimated_cost_usd is not None:
        generation["generation"]["maximum_estimated_cost_usd"] = (
            maximum_estimated_cost_usd
        )
    if cost_safety_multiplier != 1.0:
        generation["generation"]["cost_safety_multiplier"] = (
            cost_safety_multiplier
        )
    if request_cost_ceiling_usd is not None:
        generation["generation"]["request_cost_ceiling_usd"] = (
            request_cost_ceiling_usd
        )
    generation["prompt"].pop("version")
    if model.pricing_catalog_path is not None:
        generation["endpoint_pricing_path"] = (
            model.pricing_catalog_path.as_posix()
        )

    evaluation = {
        "run": {
            "name": (
                f"evaluate_{benchmark.benchmark_id}_{model.model_id}"
            ),
            "input_path": paths["generation_output"]
            .relative_to(registry.root)
            .as_posix(),
            "output_path": paths["evaluation_output"]
            .relative_to(registry.root)
            .as_posix(),
        },
        "metrics": benchmark.scoring.metrics.model_dump(mode="json"),
        "judge": {
            "model": "${JUDGE_MODEL}",
            "temperature": 0,
            "max_retries": 3,
            "continue_on_error": True,
        },
        "task": {
            "name": benchmark.benchmark_id,
            "description": benchmark.description,
        },
        "scoring": {
            "score_min": 0,
            "score_max": 5,
            "pass_threshold": 0.75,
        },
        "selection": {
            "enabled": False,
            "weights": {
                "quality": 1.0,
                "cost": 0.0,
                "latency": 0.0,
                "reliability": 0.0,
            },
            "constraints": {
                "require_model_profile": False,
                "required_capabilities": [],
            },
            "model_profiles": [],
        },
        "criteria": [
            {
                "name": "correctness",
                "weight": 1.0,
                "description": (
                    "The output matches the benchmark reference. This "
                    "criterion is unused for deterministic cases."
                ),
            }
        ],
    }
    dump_yaml(paths["candidate_config"], candidate)
    dump_yaml(paths["generation_config"], generation)
    dump_yaml(paths["evaluation_config"], evaluation)
    fingerprint = benchmark_fingerprint(registry, benchmark)
    plan = {
        "schema_version": RUN_PLAN_SCHEMA_VERSION,
        "benchmark": (
            f"{benchmark.benchmark_id}@{benchmark.version}"
        ),
        "model_id": model.model_id,
        "benchmark_status": benchmark.status,
        "required_capabilities": benchmark.required_capabilities,
        "case_count": int(len(cases)),
        "planned_model_calls": int(len(cases)),
        "compatibility": fingerprint,
        "paths": {
            key: path.relative_to(registry.root).as_posix()
            for key, path in paths.items()
            if key != "plan_dir"
        },
        "execution": {
            "generation_complete": paths["generation_output"].is_file(),
            "evaluation_complete": paths["evaluation_output"].is_file(),
        },
    }
    if tier_materialization is not None:
        tier_plan = tier_materialization["plan"]
        plan["tier"] = {
            "name": tier_name,
            "inherits": tier_plan["inherits"],
            "chain": tier_plan["chain"],
            "definition_sha256": tier_plan["definition_sha256"],
            "cumulative_case_count": tier_plan["cumulative"]["cases"],
            "execution_case_count": tier_plan["execution"]["cases"],
            "cumulative_case_ids_sha256": tier_plan["cumulative"][
                "case_ids_sha256"
            ],
            "execution_case_ids_sha256": tier_plan["execution"][
                "case_ids_sha256"
            ],
        }
        plan["paths"]["tier_cohort"] = tier_materialization[
            "cohort_path"
        ].relative_to(registry.root).as_posix()
        plan["paths"]["tier_execution_cases"] = tier_materialization[
            "execution_path"
        ].relative_to(registry.root).as_posix()
        plan["paths"]["tier_plan"] = tier_materialization[
            "plan_path"
        ].relative_to(registry.root).as_posix()
    paths["run_plan"].parent.mkdir(parents=True, exist_ok=True)
    paths["run_plan"].write_text(
        json.dumps(plan, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return plan


def run_registered_benchmark(
    *,
    registry: LoadedRegistry,
    benchmark: BenchmarkManifest,
    model: ModelManifest,
    plan_only: bool = False,
) -> dict[str, Any]:
    plan = build_run_plan(
        registry=registry,
        benchmark=benchmark,
        model=model,
    )
    if plan_only:
        return {
            "plan": plan,
            "executed": False,
            "registered": False,
        }

    if benchmark.status not in {"ready", "frozen"}:
        raise ValueError(
            "Only ready or frozen benchmarks may execute. "
            f"{benchmark.benchmark_id}@{benchmark.version} is "
            f"{benchmark.status!r}. Use --plan-only while it is under "
            "review."
        )

    generation_config_path = registry.root / plan["paths"][
        "generation_config"
    ]
    evaluation_config_path = registry.root / plan["paths"][
        "evaluation_config"
    ]
    generation_config = load_generation_config(generation_config_path)
    generation_outputs, _, _ = generate_outputs(
        config=generation_config,
        config_path=generation_config_path,
    )
    run_status = generation_outputs.attrs.get("run_status", "completed")
    if run_status != "completed":
        raise ValueError(
            f"Generation did not complete, status={run_status}"
        )

    evaluation_config = load_evaluation_config(evaluation_config_path)
    evaluation_cases = load_eval_cases(
        evaluation_config.run.input_path
    )
    evaluation_results = evaluate_cases(
        evaluation_cases,
        evaluation_config,
    )
    evaluation_path = Path(evaluation_config.run.output_path)
    evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_results.to_csv(
        evaluation_path,
        index=False,
        lineterminator="\n",
    )
    bundle = register_result_bundle(
        registry=registry,
        benchmark=benchmark,
        model_id=model.model_id,
        results_path=evaluation_path,
    )
    leaderboard = build_leaderboard(
        registry=registry,
        benchmark=benchmark,
    )
    index_path = build_benchmark_index(registry=registry)
    return {
        "plan": plan,
        "executed": True,
        "registered": True,
        "bundle": bundle,
        "leaderboard": leaderboard,
        "index_path": index_path,
    }


def run_experimental_benchmark(
    *,
    registry: LoadedRegistry,
    benchmark: BenchmarkManifest,
    model: ModelManifest,
    tier_name: str | None = None,
    cost_preflight_sample_path: str | Path | None = None,
    maximum_estimated_cost_usd: float | None = None,
    cost_safety_multiplier: float = 1.0,
    request_cost_ceiling_usd: float | None = None,
) -> dict[str, Any]:
    """Run a local benchmark experiment without registration or publication."""

    plan = build_run_plan(
        registry=registry,
        benchmark=benchmark,
        model=model,
        tier_name=tier_name,
        cost_preflight_sample_path=cost_preflight_sample_path,
        maximum_estimated_cost_usd=maximum_estimated_cost_usd,
        cost_safety_multiplier=cost_safety_multiplier,
        request_cost_ceiling_usd=request_cost_ceiling_usd,
    )
    if benchmark.status == "retired":
        raise ValueError(
            f"Retired benchmark {_benchmark_reference(benchmark)} cannot run"
        )

    generation_config_path = registry.root / plan["paths"][
        "generation_config"
    ]
    evaluation_config_path = registry.root / plan["paths"][
        "evaluation_config"
    ]
    generation_config = load_generation_config(generation_config_path)
    generation_outputs, _, _ = generate_outputs(
        config=generation_config,
        config_path=generation_config_path,
    )
    run_status = generation_outputs.attrs.get("run_status", "completed")
    if run_status != "completed":
        raise ValueError(
            f"Generation did not complete, status={run_status}"
        )

    evaluation_config = load_evaluation_config(evaluation_config_path)
    evaluation_cases = load_eval_cases(evaluation_config.run.input_path)
    evaluation_results = evaluate_cases(
        evaluation_cases,
        evaluation_config,
    )
    evaluation_path = registry.root / plan["paths"]["evaluation_output"]
    evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_results.to_csv(
        evaluation_path,
        index=False,
        lineterminator="\n",
    )
    return {
        "plan": plan,
        "executed": True,
        "registered": False,
        "evaluation_path": evaluation_path,
        "evaluation_results": evaluation_results,
    }


def _benchmark_reference(benchmark: BenchmarkManifest) -> str:
    return f"{benchmark.benchmark_id}@{benchmark.version}"


def rescore_registered_benchmark(
    *,
    registry: LoadedRegistry,
    benchmark: BenchmarkManifest,
    model: ModelManifest,
) -> dict[str, Any]:
    """Re-evaluate saved model outputs without making provider calls.

    Rescoring is intentionally limited to deterministic benchmarks. A judge
    case would require a fresh model call and would make the command's offline
    behavior ambiguous.
    """

    plan = build_run_plan(
        registry=registry,
        benchmark=benchmark,
        model=model,
    )
    generation_path = registry.root / plan["paths"][
        "generation_output"
    ]
    if not generation_path.is_file():
        raise FileNotFoundError(
            "Saved generation output not found for rescoring: "
            f"{generation_path}"
        )

    evaluation_config_path = registry.root / plan["paths"][
        "evaluation_config"
    ]
    evaluation_config = load_evaluation_config(evaluation_config_path)
    evaluation_cases = load_eval_cases(generation_path)

    observed_models = set(evaluation_cases["model_name"].astype(str))
    if observed_models != {model.model_id}:
        raise ValueError(
            f"Saved outputs for {model.model_id!r} must contain exactly "
            f"that model, found {sorted(observed_models)}"
        )

    judge_cases = evaluation_cases[
        evaluation_cases["evaluation_type"] == JUDGE
    ]
    if not judge_cases.empty:
        raise ValueError(
            "Offline rescoring only supports deterministic exact/json "
            f"benchmarks; found {len(judge_cases)} judge case(s)."
        )

    evaluation_results = evaluate_cases(
        evaluation_cases,
        evaluation_config,
    )
    evaluation_path = registry.root / plan["paths"][
        "evaluation_output"
    ]
    evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_results.to_csv(
        evaluation_path,
        index=False,
        lineterminator="\n",
    )
    bundle = register_result_bundle(
        registry=registry,
        benchmark=benchmark,
        model_id=model.model_id,
        results_path=evaluation_path,
    )
    leaderboard = build_leaderboard(
        registry=registry,
        benchmark=benchmark,
    )
    index_path = build_benchmark_index(registry=registry)
    return {
        "plan": plan,
        "executed": True,
        "registered": True,
        "generation_calls": 0,
        "bundle": bundle,
        "leaderboard": leaderboard,
        "index_path": index_path,
    }

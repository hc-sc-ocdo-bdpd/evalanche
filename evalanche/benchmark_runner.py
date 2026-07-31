from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

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

RUN_PLAN_SCHEMA_VERSION = "1.0"


def _paths(
    *,
    registry: LoadedRegistry,
    benchmark: BenchmarkManifest,
    model: ModelManifest,
) -> dict[str, Path]:
    stem = (
        f"{benchmark.benchmark_id}_{benchmark.version}_{model.model_id}"
        .replace(".", "_")
    )
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
) -> dict[str, Any]:
    cases_path = registry.root / benchmark.dataset.cases_path
    cases = pd.read_csv(cases_path)
    paths = _paths(
        registry=registry,
        benchmark=benchmark,
        model=model,
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
            "input_path": benchmark.dataset.cases_path.as_posix(),
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

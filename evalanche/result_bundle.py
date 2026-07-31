from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from evalanche.registry import (
    BenchmarkManifest,
    LoadedRegistry,
    benchmark_fingerprint,
    sha256_file,
    sha256_json,
)

BUNDLE_SCHEMA_VERSION = "1.0"

CORE_SCORE_COLUMNS = [
    "case_id",
    "model_name",
    "final_passed",
    "final_score",
    "evaluation_source",
    "metric_status",
    "metric_passed",
    "output_is_json",
    "json_field_match_rate",
    "json_expected_field_count",
    "json_matching_field_count",
    "json_missing_fields",
    "json_extra_fields",
    "json_mismatched_fields",
    "generation_status",
    "generation_seconds",
    "generation_prompt_tokens",
    "generation_cached_prompt_tokens",
    "generation_completion_tokens",
    "generation_total_tokens",
    "generation_cost_usd",
]


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if pd.isna(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if pd.isna(value):
        return None
    return str(value)


def _read_results(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation results not found: {path}")
    results = pd.read_csv(path)
    required = {
        "case_id",
        "model_name",
        "final_passed",
        "final_score",
    }
    missing = required - set(results.columns)
    if missing:
        raise ValueError(
            f"Evaluation results are missing columns: {sorted(missing)}"
        )
    return results


def _select_compact_scores(
    *,
    results: pd.DataFrame,
    cases: pd.DataFrame,
    benchmark: BenchmarkManifest,
) -> pd.DataFrame:
    case_metadata = [
        "case_id",
        benchmark.group_key,
        *benchmark.slice_columns,
    ]
    case_metadata = list(dict.fromkeys(case_metadata))
    missing = set(case_metadata) - set(cases.columns)
    if missing:
        raise ValueError(
            "Benchmark cases are missing result metadata columns: "
            f"{sorted(missing)}"
        )

    compact = results.copy()
    metadata_missing = [
        column for column in case_metadata if column not in compact.columns
    ]
    if metadata_missing:
        compact = compact.merge(
            cases[case_metadata],
            on="case_id",
            how="left",
            validate="many_to_one",
        )

    columns = [
        *case_metadata,
        *(
            column
            for column in CORE_SCORE_COLUMNS
            if column not in case_metadata and column in compact.columns
        ),
    ]
    compact = compact[columns].copy()
    compact["case_id"] = compact["case_id"].astype(str)
    compact["model_name"] = compact["model_name"].astype(str)
    return compact.sort_values(
        ["case_id", "model_name"],
        kind="stable",
    ).reset_index(drop=True)


def _summary(
    compact: pd.DataFrame,
    benchmark: BenchmarkManifest,
) -> dict[str, Any]:
    scored = compact[compact["final_passed"].notna()]
    passed = int(scored["final_passed"].astype(bool).sum())
    generation_failures = (
        int(
            (
                compact["generation_status"]
                .fillna("success")
                .astype(str)
                .str.casefold()
                != "success"
            ).sum()
        )
        if "generation_status" in compact
        else 0
    )
    costs = (
        pd.to_numeric(compact["generation_cost_usd"], errors="coerce")
        if "generation_cost_usd" in compact
        else pd.Series(dtype=float)
    )
    latencies = (
        pd.to_numeric(compact["generation_seconds"], errors="coerce")
        if "generation_seconds" in compact
        else pd.Series(dtype=float)
    )

    slices: dict[str, dict[str, Any]] = {}
    for column in benchmark.slice_columns:
        records: dict[str, Any] = {}
        for value, group in compact.groupby(column, dropna=False):
            group_scored = group[group["final_passed"].notna()]
            group_passed = int(
                group_scored["final_passed"].astype(bool).sum()
            )
            records[str(value)] = {
                "cases": int(len(group)),
                "scored_cases": int(len(group_scored)),
                "passed_cases": group_passed,
                "pass_rate": (
                    group_passed / len(group_scored)
                    if len(group_scored)
                    else None
                ),
                "average_score": (
                    float(group_scored["final_score"].mean())
                    if len(group_scored)
                    else None
                ),
            }
        slices[column] = records

    return {
        "cases": int(len(compact)),
        "scored_cases": int(len(scored)),
        "passed_cases": passed,
        "failed_cases": int(len(scored) - passed),
        "pass_rate": passed / len(scored) if len(scored) else None,
        "average_score": (
            float(scored["final_score"].mean()) if len(scored) else None
        ),
        "generation_failures": generation_failures,
        "generation_failure_rate": (
            generation_failures / len(compact) if len(compact) else None
        ),
        "generation_total_cost_usd": (
            float(costs.sum()) if costs.notna().any() else None
        ),
        "generation_average_cost_usd": (
            float(costs.mean()) if costs.notna().any() else None
        ),
        "generation_median_seconds": (
            float(latencies.median()) if latencies.notna().any() else None
        ),
        "generation_p95_seconds": (
            float(latencies.quantile(0.95))
            if latencies.notna().any()
            else None
        ),
        "slices": slices,
    }


def register_result_bundle(
    *,
    registry: LoadedRegistry,
    benchmark: BenchmarkManifest,
    model_id: str,
    results_path: str | Path,
    provenance_path: str | Path | None = None,
    reports_root: str | Path = "reports/benchmarks",
) -> dict[str, Any]:
    model = registry.resolve_model(model_id)
    source_path = (
        Path(results_path)
        if Path(results_path).is_absolute()
        else registry.root / results_path
    )
    results = _read_results(source_path)
    observed_models = set(results["model_name"].astype(str))
    if observed_models != {model_id}:
        raise ValueError(
            f"Result bundle for {model_id!r} must contain exactly that "
            f"model, found {sorted(observed_models)}"
        )

    cases_path = registry.root / benchmark.dataset.cases_path
    cases = pd.read_csv(cases_path)
    expected_case_ids = set(cases["case_id"].astype(str))
    observed_case_ids = set(results["case_id"].astype(str))
    if observed_case_ids != expected_case_ids:
        missing = len(expected_case_ids - observed_case_ids)
        extra = len(observed_case_ids - expected_case_ids)
        raise ValueError(
            "Result cases do not match the benchmark membership: "
            f"{missing} missing, {extra} extra"
        )
    if results["case_id"].astype(str).duplicated().any():
        raise ValueError("Result bundle contains duplicate case_id values")

    fingerprint = benchmark_fingerprint(registry, benchmark)
    provenance_source = (
        source_path
        if provenance_path is None
        else Path(provenance_path)
        if Path(provenance_path).is_absolute()
        else registry.root / provenance_path
    )
    if not provenance_source.is_file():
        raise FileNotFoundError(
            f"Result provenance not found: {provenance_source}"
        )
    source_sha256 = sha256_file(provenance_source)
    compact = _select_compact_scores(
        results=results,
        cases=cases,
        benchmark=benchmark,
    )
    run_identity = {
        "compatibility_sha256": fingerprint["compatibility_sha256"],
        "model_id": model_id,
        "source_results_sha256": source_sha256,
    }
    run_id = sha256_json(run_identity)[:16]
    base = (
        registry.root
        / reports_root
        / benchmark.benchmark_id
        / benchmark.version
    )
    run_dir = base / "runs" / model_id / run_id
    score_path = run_dir / "case_scores.csv.gz"
    metadata_path = run_dir / "run.json"
    active_path = base / "active_runs" / f"{model_id}.json"

    run_dir.mkdir(parents=True, exist_ok=True)
    compact.to_csv(
        score_path,
        index=False,
        compression={"method": "gzip", "mtime": 0},
        lineterminator="\n",
    )
    result_created_at = None
    if "generated_at_utc" in results:
        generated = results["generated_at_utc"].dropna().astype(str)
        if not generated.empty:
            result_created_at = max(generated)
    metadata = {
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "run_id": run_id,
        "result_created_at_utc": result_created_at,
        "benchmark": {
            "benchmark_id": benchmark.benchmark_id,
            "version": benchmark.version,
            "title": benchmark.title,
        },
        "dataset": {
            "dataset_id": benchmark.dataset.dataset_id,
            "version": benchmark.dataset.version,
        },
        "model": {
            "model_id": model.model_id,
            "display_name": model.display_name,
            "provider_route": model.provider_route,
            "provider_model_version": model.provider_model_version,
            "deployment_type": model.deployment_type,
            "resource_region": model.resource_region,
        },
        "compatibility": fingerprint,
        "summary": _summary(compact, benchmark),
        "artifacts": {
            "case_scores": {
                "path": score_path.relative_to(registry.root).as_posix(),
                "sha256": sha256_file(score_path),
                "bytes": score_path.stat().st_size,
                "rows": int(len(compact)),
            },
            "source_results": {
                "path": (
                    provenance_source.relative_to(registry.root).as_posix()
                    if provenance_source.is_relative_to(registry.root)
                    else str(provenance_source)
                ),
                "sha256": source_sha256,
                "bytes": provenance_source.stat().st_size,
            },
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    active_path.parent.mkdir(parents=True, exist_ok=True)
    active = {
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "benchmark_id": benchmark.benchmark_id,
        "benchmark_version": benchmark.version,
        "model_id": model_id,
        "run_id": run_id,
        "run_metadata_path": metadata_path.relative_to(
            registry.root
        ).as_posix(),
        "run_metadata_sha256": sha256_file(metadata_path),
    }
    active_path.write_text(
        json.dumps(active, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "metadata_path": metadata_path,
        "case_scores_path": score_path,
        "active_path": active_path,
        "summary": metadata["summary"],
    }


def load_active_bundles(
    *,
    registry: LoadedRegistry,
    benchmark: BenchmarkManifest,
    reports_root: str | Path = "reports/benchmarks",
) -> list[dict[str, Any]]:
    base = (
        registry.root
        / reports_root
        / benchmark.benchmark_id
        / benchmark.version
    )
    active_dir = base / "active_runs"
    if not active_dir.is_dir():
        return []

    expected_fingerprint = benchmark_fingerprint(registry, benchmark)
    bundles: list[dict[str, Any]] = []
    for active_path in sorted(active_dir.glob("*.json")):
        active = json.loads(active_path.read_text(encoding="utf-8"))
        metadata_path = registry.root / active["run_metadata_path"]
        if not metadata_path.is_file():
            raise ValueError(
                f"Active run metadata is missing: {metadata_path}"
            )
        if sha256_file(metadata_path) != active["run_metadata_sha256"]:
            raise ValueError(
                f"Active run metadata hash mismatch: {metadata_path}"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        compatibility = metadata["compatibility"]["compatibility_sha256"]
        if compatibility != expected_fingerprint["compatibility_sha256"]:
            raise ValueError(
                f"Incompatible active run {active_path}: expected "
                f"{expected_fingerprint['compatibility_sha256']}, found "
                f"{compatibility}"
            )
        score_artifact = metadata["artifacts"]["case_scores"]
        score_path = registry.root / score_artifact["path"]
        if not score_path.is_file():
            raise ValueError(f"Case scores are missing: {score_path}")
        if sha256_file(score_path) != score_artifact["sha256"]:
            raise ValueError(f"Case-score hash mismatch: {score_path}")
        bundles.append(
            {
                "active": active,
                "metadata": metadata,
                "metadata_path": metadata_path,
                "score_path": score_path,
            }
        )
    return bundles

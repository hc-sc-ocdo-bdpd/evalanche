from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console
from rich.table import Table

from evalanche.access_sets import historical_results_scope
from evalanche.benchmark_tiers import materialize_tier_cohort, tier_chain
from evalanche.evaluation import build_evaluation_summary
from evalanche.registry import (
    BenchmarkManifest,
    LoadedRegistry,
    benchmark_fingerprint,
    sha256_file,
)
from evalanche.statistics import build_pairwise_comparisons


EXPERIMENT_SCHEMA_VERSION = "1.1"
DEFAULT_PLANS_ROOT = Path("data/generated/registry_plans")
DEFAULT_OUTPUT_ROOT = Path("results/benchmark_experiments")

console = Console()


def _benchmark_reference(benchmark: BenchmarkManifest) -> str:
    return f"{benchmark.benchmark_id}@{benchmark.version}"


def _inside_root(root: Path, relative_path: str | Path) -> Path:
    candidate = (root / relative_path).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(
            f"Experiment artifact path escapes the repository: {relative_path}"
        )
    return candidate


def _parse_optional_bool(value: Any) -> bool | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"Invalid final_passed value: {value!r}")


def _load_result(
    *,
    path: Path,
    model_id: str,
    expected_case_ids: set[str],
    allow_case_superset: bool = False,
) -> pd.DataFrame:
    results = pd.read_csv(path)
    required = {
        "case_id",
        "model_name",
        "final_passed",
        "final_score",
        "evaluation_source",
    }
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(
            f"Evaluation result {path} is missing columns: {missing}"
        )

    observed_models = set(results["model_name"].astype(str))
    if observed_models != {model_id}:
        raise ValueError(
            f"Evaluation result {path} must contain only {model_id!r}, "
            f"found {sorted(observed_models)}"
        )
    observed_case_ids = set(results["case_id"].astype(str))
    membership_valid = (
        expected_case_ids.issubset(observed_case_ids)
        if allow_case_superset
        else observed_case_ids == expected_case_ids
    )
    if not membership_valid:
        missing_cases = len(expected_case_ids - observed_case_ids)
        extra_cases = len(observed_case_ids - expected_case_ids)
        raise ValueError(
            f"Evaluation result {path} has incompatible case membership: "
            f"{missing_cases} missing, {extra_cases} extra"
        )
    if results["case_id"].astype(str).duplicated().any():
        raise ValueError(f"Evaluation result {path} has duplicate case IDs")

    results = results.copy()
    results["case_id"] = results["case_id"].astype(str)
    if allow_case_superset:
        results = results[results["case_id"].isin(expected_case_ids)].copy()
    results["model_name"] = results["model_name"].astype(str)
    results["final_passed"] = results["final_passed"].map(
        _parse_optional_bool
    )
    results["final_score"] = pd.to_numeric(
        results["final_score"],
        errors="coerce",
    )
    return results


def discover_benchmark_evaluations(
    *,
    registry: LoadedRegistry,
    benchmark: BenchmarkManifest,
    model_ids: list[str] | None = None,
    plans_root: str | Path = DEFAULT_PLANS_ROOT,
    tier_name: str | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Load every completed compatible local evaluation discovered by plan."""

    selected = None if model_ids is None else set(model_ids)
    if selected is not None:
        unknown = sorted(selected - set(registry.models))
        if unknown:
            raise ValueError(f"Unknown requested models: {unknown}")

    required_capabilities = set(benchmark.required_capabilities)
    compatible_models = {
        model_id
        for model_id, model in registry.models.items()
        if required_capabilities.issubset(model.capabilities)
    }
    if selected is not None:
        incompatible = sorted(selected - compatible_models)
        if incompatible:
            raise ValueError(
                "Requested models do not satisfy benchmark capabilities: "
                f"{incompatible}"
            )

    if tier_name is None:
        cases = pd.read_csv(registry.root / benchmark.dataset.cases_path)
        allowed_tiers: set[str] = set()
    else:
        materialized = materialize_tier_cohort(
            registry=registry,
            benchmark=benchmark,
            tier_name=tier_name,
        )
        cases = materialized["cohort"].cumulative_cases
        allowed_tiers = set(tier_chain(benchmark, tier_name))
    expected_case_ids = set(cases["case_id"].astype(str))
    expected_reference = _benchmark_reference(benchmark)
    expected_fingerprint = benchmark_fingerprint(
        registry,
        benchmark,
    )["compatibility_sha256"]
    plan_dir = _inside_root(registry.root, plans_root)

    candidates: dict[str, list[dict[str, Any]]] = {}
    if plan_dir.is_dir():
        plan_paths = sorted(plan_dir.rglob("run_plan.json"))
    else:
        plan_paths = []

    for plan_path in plan_paths:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan.get("benchmark") != expected_reference:
            continue
        compatibility = plan.get("compatibility", {})
        if compatibility.get("compatibility_sha256") != expected_fingerprint:
            continue

        model_id = str(plan.get("model_id", ""))
        if model_id not in compatible_models:
            continue
        if selected is not None and model_id not in selected:
            continue
        plan_tier_payload = plan.get("tier")
        plan_tier = (
            str(plan_tier_payload.get("name"))
            if isinstance(plan_tier_payload, dict)
            else None
        )
        if tier_name is None and plan_tier is not None:
            continue
        if tier_name is not None and plan_tier not in allowed_tiers | {None}:
            continue

        evaluation_value = plan.get("paths", {}).get("evaluation_output")
        if not evaluation_value:
            raise ValueError(
                f"Run plan has no evaluation output path: {plan_path}"
            )
        evaluation_path = _inside_root(registry.root, evaluation_value)
        if not evaluation_path.is_file():
            continue

        candidates.setdefault(model_id, []).append(
            {
                "plan": plan,
                "plan_tier": plan_tier,
                "plan_path": plan_path,
                "evaluation_path": evaluation_path,
            }
        )

    frames: list[pd.DataFrame] = []
    sources: list[dict[str, Any]] = []
    seen_models: set[str] = set()
    for model_id, model_candidates in sorted(candidates.items()):
        legacy = [
            candidate
            for candidate in model_candidates
            if candidate["plan_tier"] is None
        ]
        if len(legacy) > 1:
            raise ValueError(
                f"Multiple compatible legacy run plans found for {model_id!r}"
            )

        used: list[tuple[pd.DataFrame, dict[str, Any]]] = []
        if legacy:
            candidate = legacy[0]
            frame = _load_result(
                path=candidate["evaluation_path"],
                model_id=model_id,
                expected_case_ids=expected_case_ids,
                allow_case_superset=tier_name is not None,
            )
            used = [(frame, candidate)]
        else:
            for candidate in model_candidates:
                plan_tier = candidate["plan_tier"]
                if plan_tier is None:
                    continue
                materialized = materialize_tier_cohort(
                    registry=registry,
                    benchmark=benchmark,
                    tier_name=plan_tier,
                )
                fragment_ids = set(
                    materialized["cohort"].execution_cases[
                        "case_id"
                    ].astype(str)
                )
                frame = _load_result(
                    path=candidate["evaluation_path"],
                    model_id=model_id,
                    expected_case_ids=fragment_ids,
                )
                used.append((frame, candidate))

        if not used:
            continue
        combined_model = pd.concat(
            [frame for frame, _ in used],
            ignore_index=True,
        )
        if combined_model["case_id"].duplicated().any():
            raise ValueError(
                f"Tier evaluation fragments overlap for model {model_id!r}"
            )
        observed_ids = set(combined_model["case_id"].astype(str))
        if observed_ids != expected_case_ids:
            if selected is not None:
                missing_count = len(expected_case_ids - observed_ids)
                raise ValueError(
                    f"Completed tier evaluation for {model_id!r} is "
                    f"missing {missing_count} case(s)"
                )
            continue

        frames.append(combined_model)
        seen_models.add(model_id)
        for frame, candidate in used:
            plan_path = candidate["plan_path"]
            evaluation_path = candidate["evaluation_path"]
            sources.append(
                {
                    "model_id": model_id,
                    "tier": candidate["plan_tier"] or "legacy_full",
                    "run_plan_path": plan_path.relative_to(
                        registry.root
                    ).as_posix(),
                    "run_plan_sha256": sha256_file(plan_path),
                    "evaluation_path": evaluation_path.relative_to(
                        registry.root
                    ).as_posix(),
                    "evaluation_sha256": sha256_file(evaluation_path),
                    "rows": int(len(frame)),
                }
            )

    if selected is not None:
        missing_results = sorted(selected - seen_models)
        if missing_results:
            raise ValueError(
                "Completed compatible evaluations were not found for: "
                f"{missing_results}"
            )
    if not frames:
        raise ValueError(
            f"No completed compatible evaluations found for {expected_reference}"
        )

    combined = pd.concat(frames, ignore_index=True)
    return combined, sorted(
        sources,
        key=lambda item: (item["model_id"], item["tier"]),
    )


def build_slice_summary(
    *,
    results: pd.DataFrame,
    cases: pd.DataFrame,
    benchmark: BenchmarkManifest,
) -> pd.DataFrame:
    columns = [
        "model_name",
        "slice",
        "value",
        "cases",
        "scored_cases",
        "passed_cases",
        "pass_rate",
        "average_score",
    ]
    if not benchmark.slice_columns:
        return pd.DataFrame(columns=columns)

    metadata_columns = ["case_id", *benchmark.slice_columns]
    missing_metadata = [
        column for column in metadata_columns if column not in results.columns
    ]
    enriched = results.copy()
    if missing_metadata:
        enriched = enriched.merge(
            cases[metadata_columns],
            on="case_id",
            how="left",
            validate="many_to_one",
            suffixes=("", "_case"),
        )
        for column in benchmark.slice_columns:
            case_column = f"{column}_case"
            if column not in enriched and case_column in enriched:
                enriched[column] = enriched[case_column]

    records: list[dict[str, Any]] = []
    for model_name, model_rows in enriched.groupby(
        "model_name",
        dropna=False,
    ):
        for slice_name in benchmark.slice_columns:
            for value, group in model_rows.groupby(slice_name, dropna=False):
                scored = group[group["final_passed"].notna()]
                passed = int(scored["final_passed"].astype(bool).sum())
                records.append(
                    {
                        "model_name": str(model_name),
                        "slice": slice_name,
                        "value": str(value),
                        "cases": int(len(group)),
                        "scored_cases": int(len(scored)),
                        "passed_cases": passed,
                        "pass_rate": (
                            passed / len(scored) if len(scored) else None
                        ),
                        "average_score": (
                            float(scored["final_score"].mean())
                            if len(scored)
                            else None
                        ),
                    }
                )
    return pd.DataFrame(records, columns=columns).sort_values(
        ["slice", "value", "model_name"],
        kind="stable",
    )


def _diagnostic_fields(value: Any) -> set[str]:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return set()
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid field diagnostic JSON: {value!r}") from error
    if not isinstance(parsed, list):
        raise ValueError(f"Field diagnostic must be a JSON list: {value!r}")
    return {str(item) for item in parsed}


def build_field_summary(
    *,
    results: pd.DataFrame,
    benchmark: BenchmarkManifest,
) -> pd.DataFrame:
    columns = [
        "model_name",
        "field",
        "cases",
        "matched_cases",
        "field_match_rate",
    ]
    fields = benchmark.scoring.required_output_fields
    if not fields:
        return pd.DataFrame(columns=columns)
    required_diagnostics = {"json_missing_fields", "json_mismatched_fields"}
    if not required_diagnostics.issubset(results.columns):
        return pd.DataFrame(columns=columns)

    counts: dict[tuple[str, str], list[int]] = {}
    for row in results.to_dict(orient="records"):
        model_name = str(row["model_name"])
        missing = _diagnostic_fields(row.get("json_missing_fields"))
        mismatched = _diagnostic_fields(row.get("json_mismatched_fields"))
        output_is_json = _parse_optional_bool(row.get("output_is_json"))
        for field in fields:
            key = (model_name, field)
            total, matched = counts.get(key, [0, 0])
            total += 1
            if output_is_json is True and field not in missing | mismatched:
                matched += 1
            counts[key] = [total, matched]

    records = [
        {
            "model_name": model_name,
            "field": field,
            "cases": total,
            "matched_cases": matched,
            "field_match_rate": matched / total if total else None,
        }
        for (model_name, field), (total, matched) in sorted(counts.items())
    ]
    return pd.DataFrame(records, columns=columns)


def build_case_outcomes(
    *,
    results: pd.DataFrame,
    cases: pd.DataFrame,
    benchmark: BenchmarkManifest,
) -> pd.DataFrame:
    pivot = results.pivot(
        index="case_id",
        columns="model_name",
        values="final_passed",
    )
    model_names = sorted(str(column) for column in pivot.columns)
    pivot = pivot[model_names]
    records: list[dict[str, Any]] = []
    case_lookup = cases.set_index(cases["case_id"].astype(str))
    metadata_columns = list(
        dict.fromkeys(
            [benchmark.group_key, *benchmark.slice_columns]
        )
    )
    for case_id, row in pivot.iterrows():
        values = [_parse_optional_bool(row[model]) for model in model_names]
        scored = [value for value in values if value is not None]
        passed = sum(bool(value) for value in scored)
        failed = len(scored) - passed
        if len(scored) != len(model_names):
            outcome = "incomplete"
        elif len(model_names) == 1:
            outcome = "passed" if passed else "failed"
        elif passed == len(model_names):
            outcome = "all_passed"
        elif failed == len(model_names):
            outcome = "all_failed"
        else:
            outcome = "disagreement"
        record: dict[str, Any] = {
            "case_id": str(case_id),
            "outcome": outcome,
            "models_scored": len(scored),
            "models_passed": passed,
            "models_failed": failed,
        }
        if str(case_id) in case_lookup.index:
            case = case_lookup.loc[str(case_id)]
            for column in metadata_columns:
                if column in case.index:
                    record[column] = case[column]
        for model_name, value in zip(model_names, values, strict=True):
            record[f"passed__{model_name}"] = value
        records.append(record)
    return pd.DataFrame(records).sort_values("case_id", kind="stable")


def _format_percent(value: Any) -> str:
    return "Unknown" if value is None or pd.isna(value) else f"{value:.1%}"


def _format_float(value: Any, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "Unknown"
    return f"{float(value):.{digits}f}"


def _format_cost(value: Any) -> str:
    if value is None or pd.isna(value):
        return "Unknown"
    return f"${float(value):.4f}"


def _markdown_table(summary: pd.DataFrame) -> str:
    lines = [
        "| Observed order | Model | Passed | Pass rate | 95% range | "
        "Average score | Failures | Cost USD | Cost/request | Cost coverage | "
        "Tokens | p95 seconds |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        interval = (
            f"{row['pass_rate_ci_low']:.1%} to "
            f"{row['pass_rate_ci_high']:.1%}"
            if pd.notna(row["pass_rate_ci_low"])
            and pd.notna(row["pass_rate_ci_high"])
            else "Unknown"
        )
        tokens = row["generation_total_tokens"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["rank"]),
                    str(row["model_name"]),
                    f"{int(row['passed_cases'])}/{int(row['scored_cases'])}",
                    _format_percent(row["pass_rate"]),
                    interval,
                    _format_float(row["average_score"]),
                    (
                        f"{int(row['generation_errors'])}/"
                        f"{int(row['generation_requests'])}"
                    ),
                    _format_cost(row["generation_cost_usd"]),
                    _format_cost(row["generation_average_cost_usd"]),
                    _format_percent(row["generation_cost_coverage"]),
                    (
                        f"{int(tokens):,}"
                        if pd.notna(tokens)
                        else "Unknown"
                    ),
                    _format_float(row["generation_p95_seconds"], 2),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _pairwise_markdown(comparisons: pd.DataFrame) -> str:
    if comparisons.empty:
        return "At least two complete model evaluations are required."
    lines = [
        "| Model A | Model B | Paired cases | A only passed | B only passed | "
        "Both failed | Adjusted p | Clear winner |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in comparisons.iterrows():
        winner = row["clear_winner"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["model_a"]),
                    str(row["model_b"]),
                    str(int(row["paired_cases"])),
                    str(int(row["model_a_only_passed"])),
                    str(int(row["model_b_only_passed"])),
                    str(int(row["both_failed"])),
                    _format_float(row["holm_adjusted_p_value"], 4),
                    (
                        "None"
                        if winner is None or pd.isna(winner)
                        else str(winner)
                    ),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def _access_scope_markdown(scope: dict[str, Any]) -> str:
    if scope.get("access_confirmed") is True:
        if scope.get("mode") == "access_set":
            return (
                "Candidate access was confirmed through access set "
                f"`{scope.get('access_set_id')}`, dated "
                f"`{scope.get('confirmed_on')}`. Only compatible models "
                "from that set were included."
            )
        return (
            "Candidate access was confirmed for this comparison because "
            "each model was explicitly selected for the command."
        )
    return (
        "Current candidate access was not confirmed. This is a historical "
        "evidence summary of completed compatible results, not a decision "
        "shortlist."
    )


def build_benchmark_experiment_summary(
    *,
    registry: LoadedRegistry,
    benchmark: BenchmarkManifest,
    model_ids: list[str] | None = None,
    plans_root: str | Path = DEFAULT_PLANS_ROOT,
    output_dir: str | Path | None = None,
    tier_name: str | None = None,
    access_scope: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an unregistered comparison from compatible completed runs."""

    results, sources = discover_benchmark_evaluations(
        registry=registry,
        benchmark=benchmark,
        model_ids=model_ids,
        plans_root=plans_root,
        tier_name=tier_name,
    )
    if tier_name is None:
        cases = pd.read_csv(registry.root / benchmark.dataset.cases_path)
    else:
        cases = materialize_tier_cohort(
            registry=registry,
            benchmark=benchmark,
            tier_name=tier_name,
        )["cohort"].cumulative_cases
    summary = build_evaluation_summary(results)
    comparisons = build_pairwise_comparisons(results)
    slices = build_slice_summary(
        results=results,
        cases=cases,
        benchmark=benchmark,
    )
    fields = build_field_summary(results=results, benchmark=benchmark)
    outcomes = build_case_outcomes(
        results=results,
        cases=cases,
        benchmark=benchmark,
    )
    scope = dict(access_scope or historical_results_scope())
    scope["evaluated_model_ids"] = summary["model_name"].astype(str).tolist()

    if output_dir is None:
        destination = (
            registry.root
            / DEFAULT_OUTPUT_ROOT
            / benchmark.benchmark_id
            / benchmark.version
        )
        if tier_name is not None:
            destination = destination / tier_name
    else:
        destination = _inside_root(registry.root, output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "model_summary": destination / "model_summary.csv",
        "pairwise": destination / "pairwise_comparisons.csv",
        "slices": destination / "slice_summary.csv",
        "fields": destination / "field_summary.csv",
        "case_outcomes": destination / "case_outcomes.csv",
        "report": destination / "model_comparison.md",
        "metadata": destination / "model_comparison.json",
    }
    for frame, key in [
        (summary, "model_summary"),
        (comparisons, "pairwise"),
        (slices, "slices"),
        (fields, "fields"),
        (outcomes, "case_outcomes"),
    ]:
        frame.to_csv(paths[key], index=False, lineterminator="\n")

    publishable = benchmark.status in {"ready", "frozen"}
    report = "\n".join(
        [
            f"# {benchmark.title}, model comparison",
            "",
            f"Benchmark: `{_benchmark_reference(benchmark)}`",
            "",
            f"Benchmark status: `{benchmark.status}`",
            "",
            f"Tier: `{tier_name or 'full'}`",
            "",
            (
                "This is a local, unregistered comparison. It does not "
                "alter a published leaderboard."
            ),
            "",
            "## Candidate access boundary",
            "",
            _access_scope_markdown(scope),
            "",
            _markdown_table(summary),
            "",
            "## Paired evidence",
            "",
            _pairwise_markdown(comparisons),
            "",
            "## Interpretation boundary",
            "",
            (
                "Observed order is descriptive, not a universal model rank. "
                "Rows come from compatibility-checked run plans within the "
                "recorded scope. Rerunning the summary rebuilds every table "
                "without changing previous provider outputs."
            ),
            "",
            (
                "This benchmark is eligible for registered publication."
                if publishable
                else "This benchmark is draft, so these results remain "
                "experimental and unranked."
            ),
            "",
        ]
    )
    paths["report"].write_text(report, encoding="utf-8")

    artifacts = {
        key: {
            "path": path.relative_to(registry.root).as_posix(),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for key, path in paths.items()
        if key != "metadata"
    }
    metadata = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "benchmark": {
            "reference": _benchmark_reference(benchmark),
            "status": benchmark.status,
            "compatibility": benchmark_fingerprint(registry, benchmark),
        },
        "tier": tier_name,
        "comparison_status": (
            "local_unregistered" if publishable else "draft_experimental"
        ),
        "access_scope": scope,
        "models": summary["model_name"].astype(str).tolist(),
        "model_count": int(len(summary)),
        "case_count_per_model": int(summary["cases"].min()),
        "sources": sources,
        "summary": _records(summary),
        "artifacts": artifacts,
    }
    paths["metadata"].write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "results": results,
        "summary": summary,
        "comparisons": comparisons,
        "slices": slices,
        "fields": fields,
        "outcomes": outcomes,
        "sources": sources,
        "paths": paths,
        "comparison_status": metadata["comparison_status"],
        "tier": tier_name,
        "access_scope": scope,
    }


def print_benchmark_experiment_summary(result: dict[str, Any]) -> None:
    summary = result["summary"]
    tier_name = result.get("tier")
    title = "Local Benchmark Experiment"
    if tier_name is not None:
        title = f"Local Benchmark Experiment, {tier_name} tier"
    table = Table(title=title)
    table.add_column("Observed")
    table.add_column("Model")
    table.add_column("Passed")
    table.add_column("Pass rate")
    table.add_column("Avg score")
    table.add_column("Failures")
    table.add_column("Cost USD")
    table.add_column("Cost/request")
    table.add_column("Cost coverage")
    table.add_column("Tokens")
    table.add_column("p95 seconds")
    for _, row in summary.iterrows():
        tokens = row["generation_total_tokens"]
        table.add_row(
            str(row["rank"]),
            str(row["model_name"]),
            f"{int(row['passed_cases'])}/{int(row['scored_cases'])}",
            _format_percent(row["pass_rate"]),
            _format_float(row["average_score"]),
            f"{int(row['generation_errors'])}/{int(row['generation_requests'])}",
            _format_cost(row["generation_cost_usd"]),
            _format_cost(row["generation_average_cost_usd"]),
            _format_percent(row["generation_cost_coverage"]),
            f"{int(tokens):,}" if pd.notna(tokens) else "Unknown",
            _format_float(row["generation_p95_seconds"], 2),
        )
    console.print(table)
    scope = result.get("access_scope", {})
    if scope.get("access_confirmed") is True:
        console.print("Candidate access: confirmed for this comparison")
    else:
        console.print(
            "Candidate access: not confirmed, historical evidence only"
        )

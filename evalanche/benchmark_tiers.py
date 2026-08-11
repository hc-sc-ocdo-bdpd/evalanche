from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from evalanche.registry import (
    BenchmarkManifest,
    BenchmarkTier,
    LoadedRegistry,
    benchmark_fingerprint,
    sha256_file,
    sha256_json,
)


TIER_PLAN_SCHEMA_VERSION = "1.0"
DEFAULT_TIER_ROOT = Path("data/generated/benchmark_tiers")


@dataclass(frozen=True)
class TierCohort:
    name: str
    tier: BenchmarkTier
    cumulative_cases: pd.DataFrame
    execution_cases: pd.DataFrame
    cumulative_units: int
    execution_units: int
    inherited_from: str | None


def resolve_tier(
    benchmark: BenchmarkManifest,
    tier_name: str,
) -> BenchmarkTier:
    try:
        return benchmark.tiers[tier_name]
    except KeyError as error:
        available = ", ".join(sorted(benchmark.tiers)) or "none"
        raise ValueError(
            f"Benchmark does not define tier {tier_name!r}. "
            f"Available tiers: {available}"
        ) from error


def tier_chain(
    benchmark: BenchmarkManifest,
    tier_name: str,
) -> list[str]:
    """Return the inheritance chain from the first tier to the target."""

    resolve_tier(benchmark, tier_name)
    reversed_chain: list[str] = []
    current: str | None = tier_name
    while current is not None:
        reversed_chain.append(current)
        current = benchmark.tiers[current].inherits
    return list(reversed(reversed_chain))


def tier_is_ancestor(
    benchmark: BenchmarkManifest,
    *,
    ancestor: str,
    descendant: str,
) -> bool:
    return ancestor in tier_chain(benchmark, descendant)[:-1]


def _stable_order(seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _unit_members(
    cases: pd.DataFrame,
    *,
    benchmark: BenchmarkManifest,
    unit: str,
) -> dict[str, list[int]]:
    if unit == "case":
        return {
            str(row["case_id"]): [int(index)]
            for index, row in cases.reset_index(drop=True).iterrows()
        }

    group_key = benchmark.group_key
    if group_key not in cases.columns:
        raise ValueError(
            f"Tier group sampling requires benchmark group_key {group_key!r}"
        )
    members: dict[str, list[int]] = {}
    for index, value in enumerate(cases[group_key].astype(str)):
        members.setdefault(value, []).append(index)
    return members


def _distribution_distance(
    *,
    cases: pd.DataFrame,
    row_indices: set[int],
    columns: list[str],
) -> float:
    if not columns or not row_indices:
        return 0.0
    selected = cases.iloc[sorted(row_indices)]
    distance = 0.0
    for column in columns:
        full_counts = cases[column].fillna("<missing>").astype(str).value_counts()
        selected_counts = (
            selected[column].fillna("<missing>").astype(str).value_counts()
        )
        full_total = float(full_counts.sum())
        selected_total = float(selected_counts.sum())
        for value in full_counts.index:
            full_share = float(full_counts[value]) / full_total
            selected_share = float(selected_counts.get(value, 0)) / selected_total
            distance += abs(selected_share - full_share)
    return distance


def _select_balanced_units(
    *,
    cases: pd.DataFrame,
    members: dict[str, list[int]],
    inherited_rows: set[int],
    count: int,
    seed: int,
    stratify_by: list[str],
) -> set[str]:
    missing = sorted(set(stratify_by) - set(cases.columns))
    if missing:
        raise ValueError(f"Tier stratification columns are missing: {missing}")
    if count > len(members):
        raise ValueError(
            f"Tier requests {count} sampling units, but only "
            f"{len(members)} are available"
        )

    selected = {
        unit_id
        for unit_id, indices in members.items()
        if inherited_rows.intersection(indices)
    }
    if len(selected) > count:
        raise ValueError(
            "Inherited tier already contains more sampling units than the "
            "child tier allows"
        )

    values = {
        column: cases[column].fillna("<missing>").astype(str).tolist()
        for column in stratify_by
    }
    full_counts = {
        column: cases[column].fillna("<missing>").astype(str).value_counts().to_dict()
        for column in stratify_by
    }
    unit_counts = {
        unit_id: {
            column: Counter(values[column][index] for index in indices)
            for column in stratify_by
        }
        for unit_id, indices in members.items()
    }
    selected_counts = {
        column: Counter(
            values[column][index] for unit_id in selected for index in members[unit_id]
        )
        for column in stratify_by
    }
    selected_row_count = sum(len(members[unit_id]) for unit_id in selected)
    full_total = float(len(cases))
    while len(selected) < count:
        candidates: list[tuple[float, int, str]] = []
        for unit_id, indices in members.items():
            if unit_id in selected:
                continue
            trial_total = float(selected_row_count + len(indices))
            distance = 0.0
            for column in stratify_by:
                for value, full_count in full_counts[column].items():
                    trial_count = selected_counts[column].get(value, 0) + unit_counts[
                        unit_id
                    ][column].get(value, 0)
                    distance += abs(
                        (float(trial_count) / trial_total)
                        - (float(full_count) / full_total)
                    )
            candidates.append(
                (
                    distance,
                    _stable_order(seed, unit_id),
                    unit_id,
                )
            )
        _, _, chosen = min(candidates)
        selected.add(chosen)
        selected_row_count += len(members[chosen])
        for column in stratify_by:
            selected_counts[column].update(unit_counts[chosen][column])
    return selected


def build_tier_cohort(
    *,
    cases: pd.DataFrame,
    benchmark: BenchmarkManifest,
    tier_name: str,
) -> TierCohort:
    """Build one cumulative cohort and its call-only incremental delta."""

    if "case_id" not in cases.columns:
        raise ValueError("Benchmark tier cases must contain case_id")
    normalized = cases.reset_index(drop=True).copy()
    normalized["case_id"] = normalized["case_id"].astype(str)
    if normalized["case_id"].duplicated().any():
        raise ValueError("Benchmark tier case_id values must be unique")

    tier = resolve_tier(benchmark, tier_name)
    inherited_rows: set[int] = set()
    if tier.inherits is not None:
        parent = build_tier_cohort(
            cases=normalized,
            benchmark=benchmark,
            tier_name=tier.inherits,
        )
        inherited_ids = set(parent.cumulative_cases["case_id"].astype(str))
        inherited_rows = {
            int(index)
            for index, case_id in enumerate(normalized["case_id"])
            if case_id in inherited_ids
        }

    members = _unit_members(
        normalized,
        benchmark=benchmark,
        unit=tier.sampling.unit,
    )
    if tier.sampling.method == "all":
        selected_units = set(members)
    else:
        if tier.sampling.count is None:
            raise ValueError("Balanced tier is missing a sampling count")
        selected_units = _select_balanced_units(
            cases=normalized,
            members=members,
            inherited_rows=inherited_rows,
            count=tier.sampling.count,
            seed=tier.sampling.seed,
            stratify_by=tier.sampling.stratify_by,
        )

    selected_rows = {index for unit_id in selected_units for index in members[unit_id]}
    if not inherited_rows.issubset(selected_rows):
        raise ValueError(f"Tier {tier_name!r} does not contain every inherited case")
    execution_rows = selected_rows - inherited_rows
    cumulative = normalized.iloc[sorted(selected_rows)].copy()
    execution = normalized.iloc[sorted(execution_rows)].copy()
    execution_units = sum(
        1
        for unit_id in selected_units
        if set(members[unit_id]).intersection(execution_rows)
    )
    return TierCohort(
        name=tier_name,
        tier=tier,
        cumulative_cases=cumulative,
        execution_cases=execution,
        cumulative_units=len(selected_units),
        execution_units=execution_units,
        inherited_from=tier.inherits,
    )


def _tier_directory(
    registry: LoadedRegistry,
    benchmark: BenchmarkManifest,
    tier_name: str,
    tier_root: str | Path,
) -> Path:
    stem = f"{benchmark.benchmark_id}_{benchmark.version}".replace(".", "_")
    return registry.root / tier_root / stem / tier_name


def materialize_tier_cohort(
    *,
    registry: LoadedRegistry,
    benchmark: BenchmarkManifest,
    tier_name: str,
    tier_root: str | Path = DEFAULT_TIER_ROOT,
) -> dict[str, Any]:
    cases = pd.read_csv(registry.root / benchmark.dataset.cases_path)
    cohort = build_tier_cohort(
        cases=cases,
        benchmark=benchmark,
        tier_name=tier_name,
    )
    destination = _tier_directory(
        registry,
        benchmark,
        tier_name,
        tier_root,
    )
    destination.mkdir(parents=True, exist_ok=True)
    cohort_path = destination / "cohort.csv"
    execution_path = destination / "execution_cases.csv"
    plan_path = destination / "tier_plan.json"
    cohort.cumulative_cases.to_csv(
        cohort_path,
        index=False,
        lineterminator="\n",
    )
    cohort.execution_cases.to_csv(
        execution_path,
        index=False,
        lineterminator="\n",
    )
    payload = {
        "schema_version": TIER_PLAN_SCHEMA_VERSION,
        "benchmark": f"{benchmark.benchmark_id}@{benchmark.version}",
        "tier": tier_name,
        "inherits": cohort.inherited_from,
        "chain": tier_chain(benchmark, tier_name),
        "definition": cohort.tier.model_dump(mode="json"),
        "definition_sha256": sha256_json(cohort.tier.model_dump(mode="json")),
        "benchmark_compatibility": benchmark_fingerprint(
            registry,
            benchmark,
        ),
        "cumulative": {
            "cases": int(len(cohort.cumulative_cases)),
            "units": cohort.cumulative_units,
            "case_ids_sha256": sha256_json(
                cohort.cumulative_cases["case_id"].astype(str).tolist()
            ),
            "path": cohort_path.relative_to(registry.root).as_posix(),
            "sha256": sha256_file(cohort_path),
        },
        "execution": {
            "cases": int(len(cohort.execution_cases)),
            "units": cohort.execution_units,
            "case_ids_sha256": sha256_json(
                cohort.execution_cases["case_id"].astype(str).tolist()
            ),
            "path": execution_path.relative_to(registry.root).as_posix(),
            "sha256": sha256_file(execution_path),
        },
    }
    plan_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "cohort": cohort,
        "cohort_path": cohort_path,
        "execution_path": execution_path,
        "plan_path": plan_path,
        "plan": payload,
    }

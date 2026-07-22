from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from evalanche.config import SelectionConfig


SELECTION_COLUMNS = [
    "model_name",
    "selection_status",
    "selection_rank",
    "recommended",
    "decision_score",
    "quality_component",
    "cost_component",
    "latency_component",
    "reliability_component",
    "pass_rate",
    "pass_rate_ci_low",
    "generation_average_cost_usd",
    "generation_cost_sources",
    "generation_p95_seconds",
    "generation_failure_rate",
    "available",
    "capabilities",
    "constraint_failures",
    "missing_evidence",
]


def _empty_selection() -> pd.DataFrame:
    return pd.DataFrame(columns=SELECTION_COLUMNS)


def _number(row: pd.Series, column: str) -> float | None:
    if column not in row.index:
        return None
    value = row[column]
    if value is None or pd.isna(value):
        return None
    return float(value)


def _integer(row: pd.Series, column: str) -> int:
    value = _number(row, column)
    return int(value) if value is not None else 0


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _complete_metric(
    row: pd.Series,
    *,
    value_column: str,
    coverage_column: str,
) -> float | None:
    value = _number(row, value_column)
    coverage = _number(row, coverage_column)
    if value is None or coverage is None or coverage < 1.0:
        return None
    return value


def _average_generation_cost(row: pd.Series) -> float | None:
    requests = _integer(row, "generation_requests")
    total = _complete_metric(
        row,
        value_column="generation_cost_usd",
        coverage_column="generation_cost_coverage",
    )
    if requests <= 0 or total is None:
        return None
    return total / requests


def _bounded_inverse(value: float, maximum: float) -> float:
    return max(0.0, min(1.0, 1.0 - value / maximum))


def build_model_selection(
    summary: pd.DataFrame,
    config: SelectionConfig,
) -> pd.DataFrame:
    """Apply hard requirements and calculate configured decision scores."""
    if not config.enabled or summary.empty:
        return _empty_selection()

    profiles = {
        profile.name: profile
        for profile in config.model_profiles
    }
    required_capabilities = set(
        config.constraints.required_capabilities
    )
    total_weight = (
        config.weights.quality
        + config.weights.cost
        + config.weights.latency
        + config.weights.reliability
    )
    records: list[dict[str, Any]] = []

    for _, row in summary.iterrows():
        model_name = str(row["model_name"])
        profile = profiles.get(model_name)
        available = profile.available if profile is not None else True
        capabilities = (
            set(profile.capabilities)
            if profile is not None
            else set()
        )
        failures: list[str] = []
        missing: list[str] = []

        if config.constraints.require_model_profile and profile is None:
            missing.append("model profile")

        if not available:
            failures.append("model is not available for deployment")

        if required_capabilities:
            if profile is None:
                _append_unique(missing, "model profile")
            else:
                absent = sorted(required_capabilities - capabilities)
                if absent:
                    failures.append(
                        "missing required capabilities: "
                        + ", ".join(absent)
                    )

        pass_rate = _number(row, "pass_rate")
        pass_rate_ci_low = _number(row, "pass_rate_ci_low")
        unscored_cases = _integer(row, "unscored_cases")

        if unscored_cases > 0:
            missing.append("complete quality results")

        minimum_pass_rate = config.constraints.minimum_pass_rate
        if minimum_pass_rate is not None:
            if pass_rate is None:
                _append_unique(missing, "pass rate")
            elif pass_rate < minimum_pass_rate:
                failures.append(
                    f"pass rate {pass_rate:.1%} is below "
                    f"{minimum_pass_rate:.1%}"
                )

        minimum_ci_low = config.constraints.minimum_pass_rate_ci_low
        if minimum_ci_low is not None:
            if pass_rate_ci_low is None:
                _append_unique(missing, "pass-rate confidence bound")
            elif pass_rate_ci_low < minimum_ci_low:
                failures.append(
                    "95% pass-rate lower bound "
                    f"{pass_rate_ci_low:.1%} is below "
                    f"{minimum_ci_low:.1%}"
                )

        generation_requests = _integer(row, "generation_requests")
        generation_failure_rate = _number(
            row,
            "generation_failure_rate",
        )
        p95_latency = _complete_metric(
            row,
            value_column="generation_p95_seconds",
            coverage_column="generation_latency_coverage",
        )
        average_cost = _average_generation_cost(row)

        maximum_failure_rate = (
            config.constraints.maximum_generation_failure_rate
        )
        reliability_needed = (
            maximum_failure_rate is not None
            or config.weights.reliability > 0
        )
        if reliability_needed:
            if generation_requests <= 0 or generation_failure_rate is None:
                _append_unique(missing, "generation failure rate")
            elif (
                maximum_failure_rate is not None
                and generation_failure_rate > maximum_failure_rate
            ):
                failures.append(
                    "generation failure rate "
                    f"{generation_failure_rate:.1%} exceeds "
                    f"{maximum_failure_rate:.1%}"
                )

        maximum_latency = (
            config.constraints.maximum_p95_latency_seconds
        )
        latency_needed = (
            maximum_latency is not None
            or config.weights.latency > 0
        )
        if latency_needed:
            if p95_latency is None:
                _append_unique(missing, "complete generation latency")
            elif (
                maximum_latency is not None
                and p95_latency > maximum_latency
            ):
                failures.append(
                    f"p95 latency {p95_latency:.2f} s exceeds "
                    f"{maximum_latency:.2f} s"
                )

        maximum_cost = config.constraints.maximum_average_cost_usd
        cost_needed = maximum_cost is not None or config.weights.cost > 0
        if cost_needed:
            if average_cost is None:
                _append_unique(missing, "complete generation cost")
            elif maximum_cost is not None and average_cost > maximum_cost:
                failures.append(
                    f"average request cost ${average_cost:.6f} exceeds "
                    f"${maximum_cost:.6f}"
                )

        if config.weights.quality > 0 and pass_rate is None:
            _append_unique(missing, "pass rate")

        quality_component = pass_rate
        cost_component = (
            _bounded_inverse(average_cost, maximum_cost)
            if average_cost is not None and maximum_cost is not None
            else None
        )
        latency_component = (
            _bounded_inverse(p95_latency, maximum_latency)
            if p95_latency is not None and maximum_latency is not None
            else None
        )
        reliability_component = (
            1.0 - generation_failure_rate
            if generation_failure_rate is not None
            else None
        )

        if failures:
            status = "ineligible"
        elif missing:
            status = "unknown"
        else:
            status = "eligible"

        decision_score: float | None = None
        if status == "eligible":
            weighted_values = [
                (config.weights.quality, quality_component),
                (config.weights.cost, cost_component),
                (config.weights.latency, latency_component),
                (config.weights.reliability, reliability_component),
            ]
            if all(
                weight == 0 or component is not None
                for weight, component in weighted_values
            ):
                decision_score = sum(
                    weight * float(component)
                    for weight, component in weighted_values
                    if weight > 0 and component is not None
                ) / total_weight
            else:
                status = "unknown"
                _append_unique(missing, "weighted decision metric")

        records.append(
            {
                "model_name": model_name,
                "selection_status": status,
                "selection_rank": pd.NA,
                "recommended": False,
                "decision_score": decision_score,
                "quality_component": quality_component,
                "cost_component": cost_component,
                "latency_component": latency_component,
                "reliability_component": reliability_component,
                "pass_rate": pass_rate,
                "pass_rate_ci_low": pass_rate_ci_low,
                "generation_average_cost_usd": average_cost,
                "generation_cost_sources": (
                    str(row.get("generation_cost_sources", ""))
                    if pd.notna(row.get("generation_cost_sources", ""))
                    else ""
                ),
                "generation_p95_seconds": p95_latency,
                "generation_failure_rate": generation_failure_rate,
                "available": (
                    profile.available if profile is not None else None
                ),
                "capabilities": ", ".join(sorted(capabilities)),
                "constraint_failures": "; ".join(failures),
                "missing_evidence": "; ".join(missing),
            }
        )

    selection = pd.DataFrame(records)
    eligible_scores = selection["decision_score"].where(
        selection["selection_status"] == "eligible"
    )
    selection["selection_rank"] = eligible_scores.rank(
        method="min",
        ascending=False,
    ).astype("Int64")

    return selection[SELECTION_COLUMNS].sort_values(
        ["selection_rank", "model_name"],
        ascending=[True, True],
        na_position="last",
    )


def build_quality_decision(
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
) -> dict[str, Any]:
    if summary.empty:
        return {
            "status": "no_results",
            "top_ranked_models": [],
            "observed_leader": None,
            "recommended_model": None,
            "not_distinguished_from": [],
        }

    unscored_cases = (
        int(summary["unscored_cases"].sum())
        if "unscored_cases" in summary.columns
        else 0
    )
    if unscored_cases:
        ranked = summary[summary["rank"].notna()]
        top_models: list[str] = []
        if not ranked.empty:
            top_rank = ranked["rank"].min()
            top_models = (
                ranked.loc[
                    ranked["rank"] == top_rank,
                    "model_name",
                ]
                .astype(str)
                .tolist()
            )

        return {
            "status": "incomplete_evaluation",
            "top_ranked_models": top_models,
            "observed_leader": (
                top_models[0] if len(top_models) == 1 else None
            ),
            "recommended_model": None,
            "not_distinguished_from": [],
            "unscored_cases": unscored_cases,
        }

    top_rank = summary["rank"].min()
    top_models = (
        summary.loc[
            summary["rank"] == top_rank,
            "model_name",
        ]
        .astype(str)
        .tolist()
    )

    if len(summary) == 1:
        return {
            "status": "single_model",
            "top_ranked_models": top_models,
            "observed_leader": top_models[0],
            "recommended_model": None,
            "not_distinguished_from": [],
        }

    if len(top_models) > 1:
        return {
            "status": "observed_tie",
            "top_ranked_models": top_models,
            "observed_leader": None,
            "recommended_model": None,
            "not_distinguished_from": top_models,
        }

    observed_leader = top_models[0]
    other_models = sorted(
        set(summary["model_name"].astype(str)) - {observed_leader}
    )
    not_distinguished_from: list[str] = []

    for other_model in other_models:
        if (
            comparisons.empty
            or not {
                "model_a",
                "model_b",
                "clear_winner",
            }.issubset(comparisons.columns)
        ):
            not_distinguished_from.append(other_model)
            continue
        matching = comparisons[
            (
                (comparisons["model_a"] == observed_leader)
                & (comparisons["model_b"] == other_model)
            )
            | (
                (comparisons["model_a"] == other_model)
                & (comparisons["model_b"] == observed_leader)
            )
        ]
        if (
            len(matching) != 1
            or matching.iloc[0]["clear_winner"] != observed_leader
        ):
            not_distinguished_from.append(other_model)

    if not_distinguished_from:
        return {
            "status": "insufficient_evidence",
            "top_ranked_models": top_models,
            "observed_leader": observed_leader,
            "recommended_model": None,
            "not_distinguished_from": not_distinguished_from,
        }

    return {
        "status": "clear_leader",
        "top_ranked_models": top_models,
        "observed_leader": observed_leader,
        "recommended_model": observed_leader,
        "not_distinguished_from": [],
    }


def build_recommendation_decision(
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    selection: pd.DataFrame,
    config: SelectionConfig,
) -> dict[str, Any]:
    quality = build_quality_decision(summary, comparisons)
    if not config.enabled:
        return {
            "mode": "quality_only",
            "ranking_measure": "overall_pass_rate",
            **quality,
        }

    common = {
        "mode": "constraint_aware",
        "ranking_measure": "weighted_selection_score",
        "observed_leader": quality.get("observed_leader"),
        "eligible_models": [],
        "ineligible_models": [],
        "unknown_models": [],
        "not_distinguished_from": quality.get(
            "not_distinguished_from",
            [],
        ),
    }

    if quality["status"] in {"no_results", "incomplete_evaluation"}:
        return {**common, **quality}

    eligible = selection[
        selection["selection_status"] == "eligible"
    ]
    ineligible = selection[
        selection["selection_status"] == "ineligible"
    ]
    unknown = selection[
        selection["selection_status"] == "unknown"
    ]
    eligible_names = eligible["model_name"].astype(str).tolist()
    ineligible_names = ineligible["model_name"].astype(str).tolist()
    unknown_names = unknown["model_name"].astype(str).tolist()
    common.update(
        {
            "eligible_models": eligible_names,
            "ineligible_models": ineligible_names,
            "unknown_models": unknown_names,
        }
    )

    if len(summary) < 2:
        model_names = summary["model_name"].astype(str).tolist()
        return {
            **common,
            "status": "single_model",
            "top_ranked_models": model_names,
            "observed_leader": model_names[0] if model_names else None,
            "recommended_model": None,
        }

    if unknown_names:
        return {
            **common,
            "status": "insufficient_operational_data",
            "top_ranked_models": [],
            "recommended_model": None,
        }

    if eligible.empty:
        return {
            **common,
            "status": "no_eligible_models",
            "top_ranked_models": [],
            "recommended_model": None,
        }

    top_rank = eligible["selection_rank"].min()
    top = eligible[eligible["selection_rank"] == top_rank]
    top_names = top["model_name"].astype(str).tolist()

    if len(eligible) == 1:
        winner = top.iloc[0]
        return {
            **common,
            "status": "sole_eligible_model",
            "top_ranked_models": top_names,
            "recommended_model": str(winner["model_name"]),
            "top_selection_score": float(winner["decision_score"]),
            "score_margin": None,
        }

    if len(top) > 1:
        return {
            **common,
            "status": "policy_tie",
            "top_ranked_models": top_names,
            "recommended_model": None,
            "top_selection_score": float(
                top.iloc[0]["decision_score"]
            ),
            "score_margin": 0.0,
        }

    ordered = eligible.sort_values(
        ["decision_score", "model_name"],
        ascending=[False, True],
    )
    winner = ordered.iloc[0]
    runner_up = ordered.iloc[1]
    margin = float(
        winner["decision_score"] - runner_up["decision_score"]
    )
    winner_name = str(winner["model_name"])

    if margin < config.minimum_score_margin:
        return {
            **common,
            "status": "insufficient_policy_margin",
            "top_ranked_models": [winner_name],
            "recommended_model": None,
            "top_selection_score": float(winner["decision_score"]),
            "score_margin": margin,
        }

    operational_weight = (
        config.weights.cost
        + config.weights.latency
        + config.weights.reliability
    )
    if operational_weight == 0:
        eligible_quality = build_quality_decision(
            summary[
                summary["model_name"].astype(str).isin(eligible_names)
            ],
            comparisons,
        )
        common["not_distinguished_from"] = eligible_quality.get(
            "not_distinguished_from",
            [],
        )
        if eligible_quality.get("recommended_model") != winner_name:
            return {
                **common,
                "status": "insufficient_quality_evidence",
                "top_ranked_models": [winner_name],
                "recommended_model": None,
                "top_selection_score": float(
                    winner["decision_score"]
                ),
                "score_margin": margin,
            }
        status = "quality_leader"
    else:
        status = "policy_leader"

    return {
        **common,
        "status": status,
        "top_ranked_models": [winner_name],
        "recommended_model": winner_name,
        "top_selection_score": float(winner["decision_score"]),
        "score_margin": margin,
    }


def mark_recommended_model(
    selection: pd.DataFrame,
    decision: dict[str, Any],
) -> pd.DataFrame:
    marked = selection.copy()
    if marked.empty:
        return marked
    recommended = decision.get("recommended_model")
    marked["recommended"] = (
        marked["model_name"].astype(str) == str(recommended)
        if recommended is not None
        else False
    )
    return marked


def save_model_selection(
    selection: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    selection_path = output_path.with_name(
        output_path.stem + "_model_selection.csv"
    )
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection.to_csv(selection_path, index=False)
    return selection_path
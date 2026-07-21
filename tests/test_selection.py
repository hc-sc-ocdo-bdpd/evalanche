from pathlib import Path

import pandas as pd
import pytest

from evalanche.config import (
    ModelProfileConfig,
    SelectionConfig,
    SelectionConstraintsConfig,
    SelectionWeightsConfig,
)
from evalanche.selection import (
    build_model_selection,
    build_recommendation_decision,
    mark_recommended_model,
    save_model_selection,
)


def make_summary(
    records: list[dict[str, object]],
) -> pd.DataFrame:
    defaults: dict[str, object] = {
        "rank": 1,
        "cases": 10,
        "scored_cases": 10,
        "unscored_cases": 0,
        "passed_cases": 8,
        "failed_cases": 2,
        "pass_rate": 0.8,
        "pass_rate_ci_low": 0.6,
        "pass_rate_ci_high": 0.92,
        "generation_requests": 10,
        "generation_failure_rate": 0.0,
        "generation_p95_seconds": 2.0,
        "generation_latency_coverage": 1.0,
        "generation_cost_usd": 0.01,
        "generation_cost_coverage": 1.0,
    }
    return pd.DataFrame(
        [{**defaults, **record} for record in records]
    )


def test_selection_applies_constraints_before_weighted_scoring() -> None:
    summary = make_summary(
        [
            {
                "model_name": "model_a",
                "pass_rate": 0.9,
                "generation_cost_usd": 0.02,
            },
            {
                "model_name": "model_b",
                "pass_rate": 0.8,
                "generation_cost_usd": 0.20,
            },
        ]
    )
    config = SelectionConfig(
        enabled=True,
        weights=SelectionWeightsConfig(
            quality=0.5,
            cost=0.5,
        ),
        constraints=SelectionConstraintsConfig(
            minimum_pass_rate=0.75,
            maximum_average_cost_usd=0.01,
        ),
    )

    selection = build_model_selection(summary, config).set_index(
        "model_name"
    )

    assert selection.loc["model_a", "selection_status"] == "eligible"
    assert selection.loc["model_a", "cost_component"] == pytest.approx(
        0.8
    )
    assert selection.loc["model_a", "decision_score"] == pytest.approx(
        0.85
    )
    assert selection.loc["model_b", "selection_status"] == "ineligible"
    assert "average request cost" in selection.loc[
        "model_b",
        "constraint_failures",
    ]


def test_missing_required_cost_is_unknown_not_zero() -> None:
    summary = make_summary(
        [
            {
                "model_name": "model_a",
                "generation_cost_usd": None,
                "generation_cost_coverage": 0.0,
            }
        ]
    )
    config = SelectionConfig(
        enabled=True,
        weights=SelectionWeightsConfig(
            quality=0.5,
            cost=0.5,
        ),
        constraints=SelectionConstraintsConfig(
            maximum_average_cost_usd=0.01,
        ),
    )

    row = build_model_selection(summary, config).iloc[0]

    assert row["selection_status"] == "unknown"
    assert pd.isna(row["decision_score"])
    assert "complete generation cost" in row["missing_evidence"]


def test_required_capabilities_use_explicit_model_profiles() -> None:
    summary = make_summary(
        [
            {"model_name": "model_a"},
            {"model_name": "model_b"},
        ]
    )
    config = SelectionConfig(
        enabled=True,
        constraints=SelectionConstraintsConfig(
            required_capabilities=["bilingual", "approved-hosting"],
        ),
        model_profiles=[
            ModelProfileConfig(
                name="model_a",
                capabilities=["bilingual"],
            )
        ],
    )

    selection = build_model_selection(summary, config).set_index(
        "model_name"
    )

    assert selection.loc["model_a", "selection_status"] == "ineligible"
    assert "approved-hosting" in selection.loc[
        "model_a",
        "constraint_failures",
    ]
    assert selection.loc["model_b", "selection_status"] == "unknown"
    assert "model profile" in selection.loc[
        "model_b",
        "missing_evidence",
    ]


def test_required_model_profile_does_not_assume_availability() -> None:
    summary = make_summary([{"model_name": "model_a"}])
    config = SelectionConfig(
        enabled=True,
        constraints=SelectionConstraintsConfig(
            require_model_profile=True,
        ),
    )

    row = build_model_selection(summary, config).iloc[0]

    assert row["selection_status"] == "unknown"
    assert row["missing_evidence"] == "model profile"


def test_decision_recommends_only_model_that_meets_requirements() -> None:
    summary = make_summary(
        [
            {"model_name": "model_a", "pass_rate": 0.9, "rank": 1},
            {"model_name": "model_b", "pass_rate": 0.6, "rank": 2},
        ]
    )
    config = SelectionConfig(
        enabled=True,
        constraints=SelectionConstraintsConfig(
            minimum_pass_rate=0.75,
        ),
    )
    selection = build_model_selection(summary, config)

    decision = build_recommendation_decision(
        summary,
        pd.DataFrame(),
        selection,
        config,
    )
    marked = mark_recommended_model(selection, decision)

    assert decision["status"] == "sole_eligible_model"
    assert decision["recommended_model"] == "model_a"
    assert bool(
        marked.set_index("model_name").loc[
            "model_a",
            "recommended",
        ]
    ) is True


def test_weighted_policy_can_choose_best_operational_fit() -> None:
    summary = make_summary(
        [
            {
                "model_name": "quality_model",
                "pass_rate": 0.9,
                "rank": 1,
                "generation_p95_seconds": 9.0,
            },
            {
                "model_name": "fast_model",
                "pass_rate": 0.85,
                "rank": 2,
                "generation_p95_seconds": 1.0,
            },
        ]
    )
    config = SelectionConfig(
        enabled=True,
        minimum_score_margin=0.01,
        weights=SelectionWeightsConfig(
            quality=0.4,
            latency=0.6,
        ),
        constraints=SelectionConstraintsConfig(
            maximum_p95_latency_seconds=10.0,
        ),
    )
    selection = build_model_selection(summary, config)

    decision = build_recommendation_decision(
        summary,
        pd.DataFrame(),
        selection,
        config,
    )

    assert decision["status"] == "policy_leader"
    assert decision["recommended_model"] == "fast_model"


def test_policy_margin_prevents_near_tie_recommendation() -> None:
    summary = make_summary(
        [
            {
                "model_name": "model_a",
                "pass_rate": 0.81,
                "rank": 1,
            },
            {
                "model_name": "model_b",
                "pass_rate": 0.80,
                "rank": 2,
            },
        ]
    )
    config = SelectionConfig(
        enabled=True,
        minimum_score_margin=0.02,
        weights=SelectionWeightsConfig(
            quality=0.5,
            reliability=0.5,
        ),
    )
    selection = build_model_selection(summary, config)

    decision = build_recommendation_decision(
        summary,
        pd.DataFrame(),
        selection,
        config,
    )

    assert decision["status"] == "insufficient_policy_margin"
    assert decision["recommended_model"] is None


def test_quality_only_policy_keeps_paired_evidence_requirement() -> None:
    summary = make_summary(
        [
            {"model_name": "model_a", "pass_rate": 0.9, "rank": 1},
            {"model_name": "model_b", "pass_rate": 0.8, "rank": 2},
        ]
    )
    config = SelectionConfig(
        enabled=True,
        minimum_score_margin=0.01,
    )
    selection = build_model_selection(summary, config)

    decision = build_recommendation_decision(
        summary,
        pd.DataFrame(),
        selection,
        config,
    )

    assert decision["status"] == "insufficient_quality_evidence"
    assert decision["recommended_model"] is None


def test_quality_evidence_compares_only_eligible_models() -> None:
    summary = make_summary(
        [
            {"model_name": "model_a", "pass_rate": 0.95, "rank": 1},
            {"model_name": "model_b", "pass_rate": 0.85, "rank": 2},
            {"model_name": "model_c", "pass_rate": 0.70, "rank": 3},
        ]
    )
    comparisons = pd.DataFrame(
        [
            {
                "model_a": "model_a",
                "model_b": "model_b",
                "clear_winner": "model_a",
            },
            {
                "model_a": "model_a",
                "model_b": "model_c",
                "clear_winner": "model_a",
            },
            {
                "model_a": "model_b",
                "model_b": "model_c",
                "clear_winner": "model_b",
            },
        ]
    )
    config = SelectionConfig(
        enabled=True,
        model_profiles=[
            ModelProfileConfig(name="model_a", available=False),
            ModelProfileConfig(name="model_b"),
            ModelProfileConfig(name="model_c"),
        ],
    )
    selection = build_model_selection(summary, config)

    decision = build_recommendation_decision(
        summary,
        comparisons,
        selection,
        config,
    )

    assert decision["status"] == "quality_leader"
    assert decision["recommended_model"] == "model_b"


def test_save_model_selection_uses_reproducible_path(
    tmp_path: Path,
) -> None:
    selection = pd.DataFrame(
        [{"model_name": "model_a", "selection_status": "eligible"}]
    )

    path = save_model_selection(selection, tmp_path / "results.csv")

    assert path == tmp_path / "results_model_selection.csv"
    assert path.exists()
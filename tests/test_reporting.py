import pandas as pd

from evalanche.reporting import build_model_summary


def test_build_model_summary_ranks_models_and_averages_criteria() -> None:
    results = pd.DataFrame(
        [
            {
                "case_id": "case_001",
                "model_name": "model_a",
                "weighted_score": 1.0,
                "passed": True,
                "correctness_score": 5,
                "total_tokens": 100,
            },
            {
                "case_id": "case_002",
                "model_name": "model_a",
                "weighted_score": 0.8,
                "passed": True,
                "correctness_score": 4,
                "total_tokens": 120,
            },
            {
                "case_id": "case_001",
                "model_name": "model_b",
                "weighted_score": 0.5,
                "passed": False,
                "correctness_score": 2,
                "total_tokens": 80,
            },
        ]
    )

    summary = build_model_summary(results)

    assert summary["model_name"].tolist() == ["model_a", "model_b"]
    assert summary["rank"].tolist() == [1, 2]

    model_a = summary.iloc[0]
    assert model_a["cases"] == 2
    assert model_a["average_weighted_score"] == 0.9
    assert model_a["pass_rate"] == 1.0
    assert model_a["correctness_score"] == 4.5
    assert model_a["total_tokens"] == 220


def test_build_model_summary_assigns_same_rank_to_ties() -> None:
    results = pd.DataFrame(
        [
            {
                "case_id": "case_001",
                "model_name": "model_a",
                "weighted_score": 0.8,
                "passed": True,
            },
            {
                "case_id": "case_001",
                "model_name": "model_b",
                "weighted_score": 0.8,
                "passed": True,
            },
        ]
    )

    summary = build_model_summary(results)

    assert summary["rank"].tolist() == [1, 1]


def test_build_model_summary_handles_empty_results() -> None:
    assert build_model_summary(pd.DataFrame()).empty
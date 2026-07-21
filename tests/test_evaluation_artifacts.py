import json
from pathlib import Path

import pandas as pd

from evalanche.config import (
    CriterionConfig,
    EvaluationConfig,
    JudgeConfig,
    RunConfig,
    ScoringConfig,
    TaskConfig,
)
from evalanche.evaluation import (
    build_evaluation_summary,
    run_evaluation,
)
from evalanche.evaluation_artifacts import (
    build_evaluation_report,
)
from evalanche.statistics import (
    build_pairwise_comparisons,
)


def make_config(tmp_path: Path) -> EvaluationConfig:
    return EvaluationConfig(
        run=RunConfig(
            name="artifact-test",
            input_path=tmp_path / "inputs.csv",
            output_path=tmp_path / "results.csv",
        ),
        judge=JudgeConfig(
            model="azure/test-judge"
        ),
        task=TaskConfig(
            name="classification",
            description="Classify the text.",
        ),
        scoring=ScoringConfig(),
        criteria=[
            CriterionConfig(
                name="correctness",
                weight=1,
                description="Correctness",
            )
        ],
    )


def make_combined_results(
    model_names: list[str],
) -> pd.DataFrame:
    records = []

    for model_name in model_names:
        passed = model_name == "model_a"
        records.append(
            {
                "case_id": "case_001",
                "model_name": model_name,
                "evaluation_type": "exact",
                "evaluation_source": "deterministic",
                "final_passed": passed,
                "final_score": float(passed),
                "evaluation_reason": (
                    "Matched."
                    if passed
                    else "Did not match."
                ),
                "judge_total_tokens": None,
            }
        )

    return pd.DataFrame(records)


def test_single_model_report_does_not_claim_comparative_winner(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    results = make_combined_results(["model_a"])
    summary = build_evaluation_summary(results)
    comparisons = build_pairwise_comparisons(
        results
    )

    report = build_evaluation_report(
        config=config,
        results=results,
        summary=summary,
        comparisons=comparisons,
        case_results_path="results.csv",
        summary_path="summary.csv",
        comparison_path="comparisons.csv",
        metadata_path="metadata.json",
    )

    assert (
        "Comparative recommendation:** Not available"
        in report
    )
    assert "Only `model_a` was evaluated" in report


def test_small_multiple_model_report_does_not_overclaim_winner(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    results = make_combined_results(
        ["model_a", "model_b"]
    )
    summary = build_evaluation_summary(results)
    comparisons = build_pairwise_comparisons(
        results
    )

    report = build_evaluation_report(
        config=config,
        results=results,
        summary=summary,
        comparisons=comparisons,
        case_results_path="results.csv",
        summary_path="summary.csv",
        comparison_path="comparisons.csv",
        metadata_path="metadata.json",
    )

    assert (
        "Comparative recommendation:** "
        "No clear winner yet"
        in report
    )
    assert (
        "`model_a` had the highest observed result"
        in report
    )
    assert "model_b" in report
    assert "Did not match." in report
    assert "No clear difference" in report


def test_report_can_name_evidence_supported_leader(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    records = []

    for index in range(6):
        for model_name, passed in (
            ("model_a", True),
            ("model_b", False),
        ):
            records.append(
                {
                    "case_id": f"case_{index:03d}",
                    "model_name": model_name,
                    "evaluation_type": "exact",
                    "evaluation_source": (
                        "deterministic"
                    ),
                    "final_passed": passed,
                    "final_score": float(passed),
                    "evaluation_reason": (
                        "Test result."
                    ),
                    "judge_total_tokens": None,
                }
            )

    results = pd.DataFrame(records)
    summary = build_evaluation_summary(results)
    comparisons = build_pairwise_comparisons(
        results
    )
    report = build_evaluation_report(
        config=config,
        results=results,
        summary=summary,
        comparisons=comparisons,
        case_results_path="results.csv",
        summary_path="summary.csv",
        comparison_path="comparisons.csv",
        metadata_path="metadata.json",
    )

    assert (
        "Evidence-supported leader:** `model_a`"
        in report
    )
    assert (
        "clearly outperformed each other "
        "evaluated model"
        in report
    )


def test_report_does_not_treat_judge_error_as_model_failure(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    results = pd.DataFrame(
        [
            {
                "case_id": "case_001",
                "model_name": "model_a",
                "evaluation_type": "judge",
                "evaluation_source": "judge_error",
                "final_passed": None,
                "final_score": None,
                "evaluation_reason": "Judge failed.",
                "judge_status": "error",
            },
            {
                "case_id": "case_001",
                "model_name": "model_b",
                "evaluation_type": "judge",
                "evaluation_source": "llm_judge",
                "final_passed": True,
                "final_score": 1.0,
                "evaluation_reason": "Passed.",
                "judge_status": "success",
            },
        ]
    )
    summary = build_evaluation_summary(results)
    comparisons = build_pairwise_comparisons(results)

    report = build_evaluation_report(
        config=config,
        results=results,
        summary=summary,
        comparisons=comparisons,
        case_results_path="results.csv",
        summary_path="summary.csv",
        comparison_path="comparisons.csv",
        metadata_path="metadata.json",
    )

    assert "Comparative recommendation:** Not available" in report
    assert "judge failed" in report.lower()
    assert "not counted as candidate-model failures" in report


def test_report_includes_generation_and_judge_operations(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    results = pd.DataFrame(
        [
            {
                "case_id": "case_exact",
                "model_name": "model_a",
                "evaluation_type": "exact",
                "evaluation_source": "deterministic",
                "final_passed": True,
                "final_score": 1.0,
                "evaluation_reason": "Matched.",
                "generation_status": "success",
                "generation_seconds": 1.0,
                "generation_prompt_tokens": 10,
                "generation_completion_tokens": 2,
                "generation_total_tokens": 12,
                "generation_cost_usd": 0.001,
                "generation_cost_source": (
                    "litellm_response_metadata"
                ),
                "judge_status": "not_requested",
            },
            {
                "case_id": "case_judge",
                "model_name": "model_a",
                "evaluation_type": "judge",
                "evaluation_source": "llm_judge",
                "final_passed": True,
                "final_score": 0.8,
                "evaluation_reason": "Good.",
                "generation_status": "success",
                "generation_seconds": 3.0,
                "generation_prompt_tokens": 20,
                "generation_completion_tokens": 4,
                "generation_total_tokens": 24,
                "generation_cost_usd": 0.002,
                "generation_cost_source": (
                    "litellm_response_metadata"
                ),
                "judge_status": "success",
                "judge_seconds": 2.0,
                "judge_prompt_tokens": 30,
                "judge_completion_tokens": 10,
                "judge_total_tokens": 40,
                "judge_cost_usd": 0.004,
                "judge_cost_source": (
                    "litellm_response_metadata"
                ),
            },
        ]
    )
    summary = build_evaluation_summary(results)
    comparisons = build_pairwise_comparisons(results)

    report = build_evaluation_report(
        config=config,
        results=results,
        summary=summary,
        comparisons=comparisons,
        case_results_path="results.csv",
        summary_path="summary.csv",
        comparison_path="comparisons.csv",
        metadata_path="metadata.json",
    )

    assert "### Candidate Generation" in report
    assert "2.90 s" in report
    assert "36" in report
    assert "$0.003000 USD" in report
    assert "### LLM Judge" in report
    assert "2.00 s" in report
    assert "$0.004000 USD" in report


def test_run_evaluation_writes_reproducible_artifacts(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)

    pd.DataFrame(
        [
            {
                "case_id": "case_001",
                "input": "Return the sentiment.",
                "expected_output": "negative",
                "evaluation_type": "exact",
                "model_name": "model_a",
                "model_output": "negative",
            },
            {
                "case_id": "case_001",
                "input": "Return the sentiment.",
                "expected_output": "negative",
                "evaluation_type": "exact",
                "model_name": "model_b",
                "model_output": "neutral",
            },
        ]
    ).to_csv(
        config.run.input_path,
        index=False,
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "run: artifact-test\n",
        encoding="utf-8",
    )

    (
        results,
        results_path,
        summary_path,
        comparison_path,
        metadata_path,
        report_path,
    ) = run_evaluation(
        config,
        config_path=config_path,
    )

    assert len(results) == 2
    assert results_path.exists()
    assert summary_path.exists()
    assert comparison_path.exists()
    assert metadata_path.exists()
    assert report_path.exists()

    metadata = json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )

    assert metadata["schema_version"] == "0.5"
    assert (
        metadata["results"]["deterministic_rows"]
        == 2
    )
    assert metadata["results"]["judge_rows"] == 0
    assert metadata["results"]["judge_errors"] == 0
    assert (
        metadata["recommendation"]["comparative"]
        is True
    )
    assert metadata["recommendation"][
        "top_ranked_models"
    ] == ["model_a"]
    assert metadata["recommendation"]["status"] == (
        "insufficient_evidence"
    )
    assert metadata["recommendation"][
        "recommended_model"
    ] is None
    assert metadata["statistics"][
        "pass_rate_interval"
    ] == "wilson_score"
    assert metadata["statistics"]["paired_test"] == (
        "two_sided_exact_mcnemar"
    )
    assert len(
        metadata["hashes"]["config_sha256"]
    ) == 64
    assert metadata["operations"]["generation"]["requests"] == 0
    assert metadata["operations"]["judge"]["requests"] == 0
    assert metadata["operations"]["cost_policy"] == (
        "litellm_response_metadata_only"
    )
    assert len(
        metadata["hashes"]["input_sha256"]
    ) == 64
    assert len(
        metadata["hashes"][
            "pairwise_comparisons_sha256"
        ]
    ) == 64

    report = report_path.read_text(
        encoding="utf-8"
    )
    assert (
        "Comparative recommendation:** "
        "No clear winner yet"
        in report
    )
    assert "Deterministic rows:** 2" in report
    assert "## Operational Performance" in report
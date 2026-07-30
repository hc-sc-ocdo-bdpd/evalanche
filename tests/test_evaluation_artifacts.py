import json
from pathlib import Path

import pandas as pd

from evalanche.config import (
    CriterionConfig,
    EvaluationConfig,
    JudgeConfig,
    RunConfig,
    ScoringConfig,
    SelectionConfig,
    SelectionConstraintsConfig,
    SelectionWeightsConfig,
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
from evalanche.selection import (
    build_model_selection,
    build_recommendation_decision,
    mark_recommended_model,
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
        selection=pd.DataFrame(),
        case_results_path="results.csv",
        summary_path="summary.csv",
        comparison_path="comparisons.csv",
        selection_path="selection.csv",
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
        selection=pd.DataFrame(),
        case_results_path="results.csv",
        summary_path="summary.csv",
        comparison_path="comparisons.csv",
        selection_path="selection.csv",
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
        selection=pd.DataFrame(),
        case_results_path="results.csv",
        summary_path="summary.csv",
        comparison_path="comparisons.csv",
        selection_path="selection.csv",
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
        selection=pd.DataFrame(),
        case_results_path="results.csv",
        summary_path="summary.csv",
        comparison_path="comparisons.csv",
        selection_path="selection.csv",
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
        selection=pd.DataFrame(),
        case_results_path="results.csv",
        summary_path="summary.csv",
        comparison_path="comparisons.csv",
        selection_path="selection.csv",
        metadata_path="metadata.json",
    )

    assert "### Candidate Generation" in report
    assert "2.90 s" in report
    assert "36" in report
    assert "$0.003000 USD" in report
    assert "litellm_response_metadata" in report
    assert "### LLM Judge" in report
    assert "2.00 s" in report
    assert "$0.004000 USD" in report


def test_report_can_apply_constraint_aware_selection(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    config.selection = SelectionConfig(
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
    results = pd.DataFrame(
        [
            {
                "case_id": "case_001",
                "model_name": "model_a",
                "evaluation_type": "exact",
                "evaluation_source": "deterministic",
                "final_passed": True,
                "final_score": 1.0,
                "evaluation_reason": "Matched.",
                "generation_status": "success",
                "generation_seconds": 9.0,
                "judge_status": "not_requested",
            },
            {
                "case_id": "case_001",
                "model_name": "model_b",
                "evaluation_type": "exact",
                "evaluation_source": "deterministic",
                "final_passed": True,
                "final_score": 1.0,
                "evaluation_reason": "Matched.",
                "generation_status": "success",
                "generation_seconds": 1.0,
                "judge_status": "not_requested",
            },
        ]
    )
    summary = build_evaluation_summary(results)
    comparisons = build_pairwise_comparisons(results)
    selection = build_model_selection(summary, config.selection)
    decision = build_recommendation_decision(
        summary,
        comparisons,
        selection,
        config.selection,
    )
    selection = mark_recommended_model(selection, decision)

    report = build_evaluation_report(
        config=config,
        results=results,
        summary=summary,
        comparisons=comparisons,
        selection=selection,
        case_results_path="results.csv",
        summary_path="summary.csv",
        comparison_path="comparisons.csv",
        selection_path="selection.csv",
        metadata_path="metadata.json",
    )

    assert "Constraint-aware recommendation:** `model_b`" in report
    assert "## Model Selection Policy" in report
    assert "weighted decision score" in report
    assert "| model_b | eligible | 1 |" in report


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
        selection_path,
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
    assert selection_path.exists()
    assert metadata_path.exists()
    assert report_path.exists()

    metadata = json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )

    assert metadata["schema_version"] == "0.9"
    assert metadata["metrics"]["unicode_normalization"] == "NFC"
    assert metadata["metrics"]["json_comparison"] == {
        "unordered_list_paths": [],
        "numeric_value_paths": [],
        "zero_pad_numeric_string_paths": {},
        "value_aliases": {},
    }
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
        "configured_endpoint_pricing_then_litellm_response_metadata"
    )
    assert metadata["endpoint_pricing"]["catalog_path"] is None
    assert metadata["endpoint_pricing"]["resolved_endpoints"] == []
    assert len(
        metadata["hashes"]["input_sha256"]
    ) == 64
    assert len(
        metadata["hashes"][
            "pairwise_comparisons_sha256"
        ]
    ) == 64
    assert len(
        metadata["hashes"]["model_selection_sha256"]
    ) == 64
    assert metadata["selection_policy"]["enabled"] is False

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


def test_run_evaluation_records_enabled_selection_policy(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    config.selection = SelectionConfig(
        enabled=True,
        constraints=SelectionConstraintsConfig(
            minimum_pass_rate=0.5,
        ),
    )
    pd.DataFrame(
        [
            {
                "case_id": "case_001",
                "input": "Return negative.",
                "expected_output": "negative",
                "evaluation_type": "exact",
                "model_name": "model_a",
                "model_output": "negative",
            },
            {
                "case_id": "case_001",
                "input": "Return negative.",
                "expected_output": "negative",
                "evaluation_type": "exact",
                "model_name": "model_b",
                "model_output": "neutral",
            },
        ]
    ).to_csv(config.run.input_path, index=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "selection:\n  enabled: true\n",
        encoding="utf-8",
    )

    (
        _,
        _,
        _,
        _,
        selection_path,
        metadata_path,
        report_path,
    ) = run_evaluation(config, config_path=config_path)

    selection = pd.read_csv(selection_path).set_index("model_name")
    assert selection.loc["model_a", "selection_status"] == "eligible"
    assert bool(selection.loc["model_a", "recommended"]) is True
    assert selection.loc["model_b", "selection_status"] == "ineligible"

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["selection_policy"]["enabled"] is True
    assert metadata["recommendation"]["status"] == "sole_eligible_model"
    assert metadata["recommendation"]["recommended_model"] == "model_a"

    report = report_path.read_text(encoding="utf-8")
    assert "Constraint-aware recommendation:** `model_a`" in report


def test_evaluation_metadata_preserves_observed_generation_price_snapshot(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    records = []
    for model_name, model_output in (
        ("model_a", "negative"),
        ("model_b", "neutral"),
    ):
        records.append(
            {
                "case_id": "case_001",
                "input": "Return negative.",
                "expected_output": "negative",
                "evaluation_type": "exact",
                "model_name": model_name,
                "model_output": model_output,
                "generation_status": "success",
                "generation_seconds": 0.1,
                "generation_prompt_tokens": 10,
                "generation_completion_tokens": 2,
                "generation_total_tokens": 12,
                "generation_cost_usd": 0.000018,
                "generation_cost_source": (
                    "configured_endpoint_pricing"
                ),
                "generation_configured_cost_usd": 0.000018,
                "generation_provider_reported_cost_usd": 0.000020,
                "generation_pricing_id": f"{model_name}_global",
                "generation_pricing_model": f"azure/{model_name}",
                "generation_pricing_currency": "USD",
                "generation_pricing_input_per_million_tokens": 1.0,
                "generation_pricing_cached_input_per_million_tokens": 0.25,
                "generation_pricing_output_per_million_tokens": 4.0,
                "generation_pricing_effective_date": "2026-07-01",
                "generation_pricing_source": "test rate card",
            }
        )
    pd.DataFrame(records).to_csv(config.run.input_path, index=False)

    _, _, _, _, _, metadata_path, _ = run_evaluation(config)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    observed = metadata["endpoint_pricing"][
        "observed_generation_endpoints"
    ]
    assert [entry["pricing_id"] for entry in observed] == [
        "model_a_global",
        "model_b_global",
    ]
    assert observed[0]["input_per_million_tokens"] == 1.0
    assert metadata["operations"]["generation"][
        "configured_cost_usd"
    ] == 0.000036
    assert metadata["operations"]["generation"][
        "provider_reported_cost_usd"
    ] == 0.00004
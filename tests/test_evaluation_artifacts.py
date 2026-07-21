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
from evalanche.evaluation_artifacts import build_evaluation_report


def make_config(tmp_path: Path) -> EvaluationConfig:
    return EvaluationConfig(
        run=RunConfig(
            name="artifact-test",
            input_path=tmp_path / "inputs.csv",
            output_path=tmp_path / "results.csv",
        ),
        judge=JudgeConfig(model="azure/test-judge"),
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


def make_combined_results(model_names: list[str]) -> pd.DataFrame:
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
                    "Matched." if passed else "Did not match."
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

    report = build_evaluation_report(
        config=config,
        results=results,
        summary=summary,
        case_results_path="results.csv",
        summary_path="summary.csv",
        metadata_path="metadata.json",
    )

    assert "Comparative recommendation:** Not available" in report
    assert "Only `model_a` was evaluated" in report


def test_multiple_model_report_names_top_ranked_model(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    results = make_combined_results(["model_a", "model_b"])
    summary = build_evaluation_summary(results)

    report = build_evaluation_report(
        config=config,
        results=results,
        summary=summary,
        case_results_path="results.csv",
        summary_path="summary.csv",
        metadata_path="metadata.json",
    )

    assert "Top-ranked model:** `model_a`" in report
    assert "performed best among the evaluated candidates" in report
    assert "model_b" in report
    assert "Did not match." in report


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
    ).to_csv(config.run.input_path, index=False)

    config_path = tmp_path / "config.yaml"
    config_path.write_text("run: artifact-test\n", encoding="utf-8")

    (
        results,
        results_path,
        summary_path,
        metadata_path,
        report_path,
    ) = run_evaluation(
        config,
        config_path=config_path,
    )

    assert len(results) == 2
    assert results_path.exists()
    assert summary_path.exists()
    assert metadata_path.exists()
    assert report_path.exists()

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    assert metadata["schema_version"] == "0.3"
    assert metadata["results"]["deterministic_rows"] == 2
    assert metadata["results"]["judge_rows"] == 0
    assert metadata["recommendation"]["comparative"] is True
    assert metadata["recommendation"]["top_ranked_models"] == [
        "model_a"
    ]
    assert len(metadata["hashes"]["config_sha256"]) == 64
    assert len(metadata["hashes"]["input_sha256"]) == 64

    report = report_path.read_text(encoding="utf-8")
    assert "Top-ranked model:** `model_a`" in report
    assert "Deterministic rows:** 2" in report
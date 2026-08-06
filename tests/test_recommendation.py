from pathlib import Path

import pandas as pd

from evalanche.config import (
    CriterionConfig,
    EvalConfig,
    JudgeConfig,
    RunConfig,
    ScoringConfig,
    TaskConfig,
)
from evalanche.recommendation import (
    build_comparison_report,
    save_comparison_report,
)


def _config(tmp_path: Path) -> EvalConfig:
    return EvalConfig(
        run=RunConfig(
            name="comparison-test",
            input_path=tmp_path / "input.csv",
            output_path=tmp_path / "case_results.csv",
        ),
        judge=JudgeConfig(model="test/judge"),
        task=TaskConfig(
            name="open-ended-test",
            description="Comparison report test.",
        ),
        scoring=ScoringConfig(),
        criteria=[
            CriterionConfig(
                name="correctness",
                weight=1.0,
                description="Correctness.",
            )
        ],
    )


def _results(scores: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "case_id": "case_1",
                "model_name": model_name,
                "weighted_score": score,
                "passed": score >= 0.75,
            }
            for model_name, score in scores.items()
        ]
    )


def test_comparison_report_never_selects_highest_score(
    tmp_path: Path,
) -> None:
    report = build_comparison_report(
        config=_config(tmp_path),
        results=_results({"model_a": 1.0, "model_b": 0.0}),
        model_summary_path="summary.csv",
        metadata_path="metadata.json",
    )

    assert report.startswith("# Evalanche Model Comparison")
    assert "Highest observed score:** `model_a`" in report
    assert "does not select a model automatically" in report
    assert "Recommended model" not in report


def test_comparison_report_preserves_a_tie(tmp_path: Path) -> None:
    report = build_comparison_report(
        config=_config(tmp_path),
        results=_results({"model_a": 1.0, "model_b": 1.0}),
        model_summary_path="summary.csv",
        metadata_path="metadata.json",
    )

    assert "Highest observed scores" in report
    assert "tied for the top rank" in report


def test_comparison_report_handles_empty_results(tmp_path: Path) -> None:
    report = build_comparison_report(
        config=_config(tmp_path),
        results=pd.DataFrame(),
        model_summary_path="summary.csv",
        metadata_path="metadata.json",
    )

    assert "No comparison could be generated" in report


def test_saved_report_uses_comparison_filename(tmp_path: Path) -> None:
    config = _config(tmp_path)
    path = save_comparison_report(
        config=config,
        results=_results({"model_a": 1.0}),
        case_results_path=config.run.output_path,
        model_summary_path="summary.csv",
        metadata_path="metadata.json",
    )

    assert path == tmp_path / "case_results_comparison.md"
    assert path.is_file()

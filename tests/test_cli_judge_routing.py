from pathlib import Path
import sys
import types
from typing import Any

import pandas as pd
import pytest

# Importing LiteLLM initializes network clients. This test exercises CLI routing,
# not the provider SDK, so replace that boundary before importing evalanche.cli.
litellm_stub = types.ModuleType("litellm")
litellm_stub.completion = lambda *args, **kwargs: None
sys.modules["litellm"] = litellm_stub

import evalanche.cli as cli
from evalanche.config import (
    CriterionConfig,
    EvalConfig,
    JudgeConfig,
    RunConfig,
    ScoringConfig,
    TaskConfig,
)


def make_eval_config(tmp_path: Path) -> EvalConfig:
    return EvalConfig(
        run=RunConfig(
            name="judge-routing-test",
            input_path=tmp_path / "input.csv",
            output_path=tmp_path / "results.csv",
        ),
        judge=JudgeConfig(model="azure/test-judge"),
        task=TaskConfig(
            name="test-task",
            description="Test judge routing.",
        ),
        scoring=ScoringConfig(),
        criteria=[
            CriterionConfig(
                name="correctness",
                weight=1.0,
                description="Correctness",
            )
        ],
    )


def make_cases() -> pd.DataFrame:
    return pd.DataFrame(
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
                "case_id": "case_002",
                "input": "Return JSON.",
                "expected_output": '{"name": "Jordan"}',
                "evaluation_type": "json",
                "model_name": "model_a",
                "model_output": '{"name": "Jordan"}',
            },
            {
                "case_id": "case_003",
                "input": "Rewrite clearly.",
                "expected_output": "A clear rewrite.",
                "evaluation_type": "judge",
                "model_name": "model_a",
                "model_output": "A different clear rewrite.",
            },
        ]
    )


def disable_report_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(
        cli,
        "save_model_summary",
        lambda results, output_path: tmp_path / "summary.csv",
    )
    monkeypatch.setattr(
        cli,
        "save_run_metadata",
        lambda **kwargs: tmp_path / "metadata.json",
    )
    monkeypatch.setattr(
        cli,
        "save_recommendation_report",
        lambda **kwargs: tmp_path / "recommendation.md",
    )
    monkeypatch.setattr(cli, "print_summary", lambda results: None)
    monkeypatch.setattr(cli, "print_model_leaderboard", lambda results: None)
    monkeypatch.setattr(cli, "print_failures", lambda results: None)


def test_run_judge_only_sends_judge_cases_to_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = make_eval_config(tmp_path)
    cases = make_cases()
    judged_case_ids: list[str] = []
    captured_results: dict[str, pd.DataFrame] = {}

    class StubJudge:
        def __init__(self, judge_config: EvalConfig) -> None:
            assert judge_config is config

        def judge_case(self, row: dict[str, Any]) -> dict[str, Any]:
            judged_case_ids.append(str(row["case_id"]))
            return {
                "case_id": row["case_id"],
                "model_name": row["model_name"],
                "weighted_score": 1.0,
                "passed": True,
                "overall_reason": "Correct.",
            }

    disable_report_outputs(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "load_config", lambda path: config)
    monkeypatch.setattr(cli, "load_eval_cases", lambda path: cases)
    monkeypatch.setattr(cli, "CriteriaJudge", StubJudge)
    monkeypatch.setattr(
        cli,
        "save_results",
        lambda results, path: captured_results.update(
            {"results": results.copy()}
        ),
    )

    cli.run_judge("config.yaml")

    assert judged_case_ids == ["case_003"]
    assert captured_results["results"]["case_id"].tolist() == ["case_003"]
    assert captured_results["results"]["evaluation_type"].tolist() == [
        "judge"
    ]


def test_run_judge_exits_cleanly_when_there_are_no_judge_cases(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = make_eval_config(tmp_path)
    cases = make_cases().query("evaluation_type != 'judge'").copy()

    class UnexpectedJudge:
        def __init__(self, judge_config: EvalConfig) -> None:
            raise AssertionError("Judge should not be created")

    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(cli, "load_config", lambda path: config)
    monkeypatch.setattr(cli, "load_eval_cases", lambda path: cases)
    monkeypatch.setattr(cli, "CriteriaJudge", UnexpectedJudge)

    cli.run_judge("config.yaml")

    output = capsys.readouterr().out
    assert "No cases were routed to LLM judge evaluation" in output
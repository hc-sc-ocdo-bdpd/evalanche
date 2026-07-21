from pathlib import Path
from typing import Any

import pandas as pd
import pytest

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
    evaluate_cases,
)


def make_config() -> EvaluationConfig:
    return EvaluationConfig(
        run=RunConfig(
            name="combined-test",
            input_path=Path("input.csv"),
            output_path=Path("output.csv"),
        ),
        judge=JudgeConfig(model="azure/test-judge"),
        task=TaskConfig(
            name="mixed-task",
            description="Evaluate the candidate response.",
        ),
        scoring=ScoringConfig(pass_threshold=0.75),
        criteria=[
            CriterionConfig(
                name="correctness",
                weight=0.8,
                description="Correctness",
            ),
            CriterionConfig(
                name="completeness",
                weight=0.2,
                description="Completeness",
            ),
        ],
    )


def make_cases() -> pd.DataFrame:
    case_rows = [
        {
            "case_id": "case_exact",
            "input": "Return the sentiment.",
            "expected_output": "negative",
            "evaluation_type": "exact",
            "outputs": {
                "model_a": "negative",
                "model_b": "neutral",
            },
        },
        {
            "case_id": "case_json",
            "input": "Extract the fields.",
            "expected_output": '{"name": "Jordan", "amount": 42.5}',
            "evaluation_type": "json",
            "outputs": {
                "model_a": '{"name": "Jordan", "amount": 40}',
                "model_b": '{"amount": 42.5, "name": "Jordan"}',
            },
        },
        {
            "case_id": "case_judge",
            "input": "Rewrite clearly.",
            "expected_output": "A clear rewrite.",
            "evaluation_type": "judge",
            "outputs": {
                "model_a": "A good clear rewrite.",
                "model_b": "An unclear rewrite.",
            },
        },
    ]

    records = []
    for case in case_rows:
        for model_name, model_output in case["outputs"].items():
            records.append(
                {
                    "case_id": case["case_id"],
                    "input": case["input"],
                    "expected_output": case["expected_output"],
                    "evaluation_type": case["evaluation_type"],
                    "model_name": model_name,
                    "model_output": model_output,
                }
            )

    return pd.DataFrame(records)


class StubJudge:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def judge_case(self, row: dict[str, Any]) -> dict[str, Any]:
        key = (str(row["case_id"]), str(row["model_name"]))
        self.calls.append(key)

        if row["model_name"] == "model_a":
            score = 0.8
            passed = True
            correctness = 4
            reason = "Clear and correct."
        else:
            score = 0.4
            passed = False
            correctness = 2
            reason = "Unclear."

        return {
            "case_id": row["case_id"],
            "model_name": row["model_name"],
            "weighted_score": score,
            "passed": passed,
            "overall_reason": reason,
            "raw_judge_result": "{}",
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "correctness_score": correctness,
            "correctness_reason": reason,
            "completeness_score": correctness,
            "completeness_reason": reason,
        }


def test_combined_evaluation_uses_authoritative_route() -> None:
    judge = StubJudge()

    results = evaluate_cases(
        make_cases(),
        make_config(),
        judge=judge,
    )

    assert judge.calls == [
        ("case_judge", "model_a"),
        ("case_judge", "model_b"),
    ]

    exact_a = results.query(
        "case_id == 'case_exact' and model_name == 'model_a'"
    ).iloc[0]
    json_a = results.query(
        "case_id == 'case_json' and model_name == 'model_a'"
    ).iloc[0]
    judge_a = results.query(
        "case_id == 'case_judge' and model_name == 'model_a'"
    ).iloc[0]

    assert exact_a["evaluation_source"] == "deterministic"
    assert exact_a["final_score"] == 1.0
    assert bool(exact_a["final_passed"]) is True

    assert json_a["evaluation_source"] == "deterministic"
    assert json_a["json_field_match_rate"] == 0.5
    assert json_a["final_score"] == 0.5
    assert bool(json_a["final_passed"]) is False

    assert judge_a["evaluation_source"] == "llm_judge"
    assert judge_a["final_score"] == 0.8
    assert bool(judge_a["final_passed"]) is True
    assert judge_a["judge_correctness_score"] == 4


def test_generation_error_is_failed_without_calling_judge() -> None:
    cases = make_cases()
    failed_mask = (
        (cases["case_id"] == "case_judge")
        & (cases["model_name"] == "model_a")
    )
    cases.loc[failed_mask, "generation_status"] = "error"
    cases.loc[failed_mask, "generation_error"] = "Provider timeout"

    judge = StubJudge()
    results = evaluate_cases(cases, make_config(), judge=judge)

    assert judge.calls == [("case_judge", "model_b")]

    failed = results.loc[failed_mask].iloc[0]
    assert failed["evaluation_source"] == "generation_error"
    assert failed["final_score"] == 0.0
    assert bool(failed["final_passed"]) is False
    assert failed["evaluation_reason"] == "Provider timeout"


def test_combined_summary_ranks_by_pass_rate() -> None:
    results = evaluate_cases(
        make_cases(),
        make_config(),
        judge=StubJudge(),
    )

    summary = build_evaluation_summary(results)

    assert summary["model_name"].tolist() == ["model_a", "model_b"]
    assert summary["rank"].tolist() == [1, 2]

    model_a = summary.iloc[0]
    assert model_a["cases"] == 3
    assert model_a["passed_cases"] == 2
    assert model_a["pass_rate"] == pytest.approx(2 / 3)
    assert model_a["average_score"] == pytest.approx(
        (1.0 + 0.5 + 0.8) / 3
    )
    assert model_a["deterministic_cases"] == 2
    assert model_a["judge_cases"] == 1


def test_combined_evaluation_requires_same_cases_for_every_model() -> None:
    cases = make_cases()
    cases = cases[
        ~(
            (cases["case_id"] == "case_json")
            & (cases["model_name"] == "model_b")
        )
    ].copy()

    with pytest.raises(ValueError, match="same case set.*model_b"):
        evaluate_cases(cases, make_config(), judge=StubJudge())


def test_empty_combined_summary_is_empty() -> None:
    assert build_evaluation_summary(pd.DataFrame()).empty
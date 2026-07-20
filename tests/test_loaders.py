from pathlib import Path

import pandas as pd
import pytest

from evalanche.generation import load_generation_cases
from evalanche.io import load_eval_cases
from evalanche.metrics.deterministic import load_metric_cases


def test_generation_loader_rejects_duplicate_case_ids(
    tmp_path: Path,
) -> None:
    path = tmp_path / "generation.csv"
    pd.DataFrame(
        [
            {
                "case_id": "case_001",
                "input": "First",
                "expected_output": "first",
                "evaluation_type": "exact",
            },
            {
                "case_id": "case_001",
                "input": "Second",
                "expected_output": "second",
                "evaluation_type": "exact",
            },
        ]
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="duplicate case_id"):
        load_generation_cases(path)


def test_evaluation_loader_normalizes_composite_keys(
    tmp_path: Path,
) -> None:
    path = tmp_path / "evaluation.csv"
    pd.DataFrame(
        [
            {
                "case_id": " case_001 ",
                "input": "Question",
                "expected_output": "Answer",
                "model_name": " model_a ",
                "model_output": "Answer",
            }
        ]
    ).to_csv(path, index=False)

    loaded = load_eval_cases(path)

    assert loaded.loc[0, "case_id"] == "case_001"
    assert loaded.loc[0, "model_name"] == "model_a"
    assert loaded.loc[0, "evaluation_type"] == "judge"


def test_metrics_loader_rejects_duplicate_case_model_pair(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metrics.csv"
    row = {
        "case_id": "case_001",
        "input": "Question",
        "expected_output": "Answer",
        "evaluation_type": "exact",
        "model_name": "model_a",
        "model_output": "Answer",
    }
    pd.DataFrame([row, row]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="duplicate.*case_id, model_name"):
        load_metric_cases(path)
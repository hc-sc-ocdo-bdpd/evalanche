import pandas as pd
import pytest

from evalanche.identity import (
    validate_evaluation_keys,
    validate_generation_case_ids,
)


def make_evaluation_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "case_id": "case_001",
                "input": "Classify this text.",
                "expected_output": "negative",
                "evaluation_type": "exact",
                "model_name": "model_a",
                "model_output": "negative",
            },
            {
                "case_id": "case_001",
                "input": "Classify this text.",
                "expected_output": "negative",
                "evaluation_type": "exact",
                "model_name": "model_b",
                "model_output": "neutral",
            },
        ]
    )


def test_generation_case_ids_are_normalized() -> None:
    original = pd.DataFrame(
        {
            "case_id": [" case_001 ", 2],
            "input": ["First", "Second"],
        }
    )

    validated = validate_generation_case_ids(
        original,
        source_name="generation cases",
    )

    assert validated["case_id"].tolist() == ["case_001", "2"]
    assert original["case_id"].tolist() == [" case_001 ", 2]


def test_generation_case_ids_must_be_unique() -> None:
    cases = pd.DataFrame(
        {
            "case_id": ["case_001", " case_001 "],
            "input": ["First", "Second"],
        }
    )

    with pytest.raises(ValueError, match="duplicate case_id.*case_001"):
        validate_generation_case_ids(
            cases,
            source_name="generation cases",
        )


@pytest.mark.parametrize("case_id", [None, "", "   "])
def test_generation_case_id_cannot_be_missing_or_blank(
    case_id: object,
) -> None:
    cases = pd.DataFrame(
        {
            "case_id": [case_id],
            "input": ["First"],
        }
    )

    with pytest.raises(ValueError, match="case_id"):
        validate_generation_case_ids(
            cases,
            source_name="generation cases",
        )


def test_evaluation_key_allows_same_case_for_different_models() -> None:
    validated = validate_evaluation_keys(
        make_evaluation_rows(),
        source_name="evaluation outputs",
    )

    assert len(validated) == 2


def test_evaluation_key_pair_must_be_unique() -> None:
    rows = make_evaluation_rows()
    rows.loc[1, "model_name"] = "model_a"

    with pytest.raises(
        ValueError,
        match=r"duplicate \(case_id, model_name\) pairs",
    ):
        validate_evaluation_keys(
            rows,
            source_name="evaluation outputs",
        )


def test_model_name_cannot_be_blank() -> None:
    rows = make_evaluation_rows()
    rows.loc[0, "model_name"] = "   "

    with pytest.raises(ValueError, match="blank model_name"):
        validate_evaluation_keys(
            rows,
            source_name="evaluation outputs",
        )


@pytest.mark.parametrize(
    "column",
    ["input", "expected_output", "evaluation_type"],
)
def test_case_id_cannot_represent_inconsistent_case_definition(
    column: str,
) -> None:
    rows = make_evaluation_rows()
    rows.loc[1, column] = "different value"

    with pytest.raises(
        ValueError,
        match=f"inconsistent case definitions.*{column}.*case_001",
    ):
        validate_evaluation_keys(
            rows,
            source_name="evaluation outputs",
        )
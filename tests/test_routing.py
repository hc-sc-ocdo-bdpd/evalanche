import pandas as pd
import pytest

from evalanche.routing import (
    EXACT,
    JSON,
    normalize_evaluation_type,
    validate_evaluation_types,
)


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("exact", EXACT),
        (" EXACT ", EXACT),
        ("Json", JSON),
        (" judge ", "judge"),
    ],
)
def test_normalize_evaluation_type(
    raw_value: str,
    expected: str,
) -> None:
    assert normalize_evaluation_type(raw_value) == expected


def test_normalize_evaluation_type_rejects_empty_value() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        normalize_evaluation_type(None)


def test_normalize_evaluation_type_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="Unsupported evaluation_type"):
        normalize_evaluation_type("semantic")


def test_validate_evaluation_types_returns_normalized_copy() -> None:
    original = pd.DataFrame(
        {
            "case_id": ["case_001", "case_002"],
            "evaluation_type": [" EXACT ", "Json"],
        }
    )

    validated = validate_evaluation_types(
        original,
        source_name="test data",
    )

    assert validated["evaluation_type"].tolist() == [EXACT, JSON]
    assert original["evaluation_type"].tolist() == [" EXACT ", "Json"]


def test_validate_evaluation_types_requires_column() -> None:
    with pytest.raises(ValueError, match="missing required column"):
        validate_evaluation_types(
            pd.DataFrame({"case_id": ["case_001"]}),
            source_name="test data",
        )
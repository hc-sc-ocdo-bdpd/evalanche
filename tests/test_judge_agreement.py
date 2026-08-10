import pytest

from evalanche.judges.agreement import (
    binary_confusion,
    cohen_kappa,
    krippendorff_alpha_nominal,
    pairwise_raw_agreement,
)


def test_cohen_kappa_reports_perfect_and_chance_corrected_agreement() -> None:
    assert cohen_kappa(["pass", "fail"], ["pass", "fail"]) == 1.0
    assert cohen_kappa(["pass", "pass"], ["fail", "fail"]) == 0.0
    assert cohen_kappa([], []) is None


def test_cohen_kappa_rejects_unpaired_sequences() -> None:
    with pytest.raises(ValueError, match="equally sized"):
        cohen_kappa(["pass"], ["pass", "fail"])


def test_krippendorff_alpha_supports_multiple_and_missing_raters() -> None:
    assert krippendorff_alpha_nominal(
        [["pass", "pass"], ["fail", "fail", "fail"], ["pass"]]
    ) == 1.0
    assert krippendorff_alpha_nominal([["pass"], ["fail"]]) is None


def test_pairwise_raw_agreement_counts_every_rater_pair() -> None:
    agreement, comparisons = pairwise_raw_agreement(
        [["pass", "pass", "fail"], ["fail", "fail"]]
    )

    assert comparisons == 4
    assert agreement == pytest.approx(0.5)


def test_binary_confusion_treats_failure_as_positive_class() -> None:
    metrics = binary_confusion(
        ["fail", "fail", "pass", "pass"],
        ["fail", "pass", "fail", "pass"],
        pass_label="pass",
        failure_label="fail",
    )

    assert metrics["true_failure"] == 1
    assert metrics["false_approval"] == 1
    assert metrics["false_failure"] == 1
    assert metrics["true_pass"] == 1
    assert metrics["failure_precision"] == 0.5
    assert metrics["failure_recall"] == 0.5
    assert metrics["false_approval_rate"] == 0.5


def test_binary_confusion_rejects_unknown_labels() -> None:
    with pytest.raises(ValueError, match="Unexpected binary labels"):
        binary_confusion(
            ["unknown"],
            ["pass"],
            pass_label="pass",
            failure_label="fail",
        )

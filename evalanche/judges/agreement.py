from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Iterable, Sequence


def cohen_kappa(
    labels_a: Sequence[str],
    labels_b: Sequence[str],
) -> float | None:
    """Return nominal Cohen's kappa for two complete label sequences."""
    if len(labels_a) != len(labels_b):
        raise ValueError("Cohen's kappa requires equally sized label sequences")
    if not labels_a:
        return None

    pairs = list(zip(labels_a, labels_b, strict=True))
    observed = sum(a == b for a, b in pairs) / len(pairs)
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    labels = set(counts_a) | set(counts_b)
    expected = sum(
        (counts_a[label] / len(pairs))
        * (counts_b[label] / len(pairs))
        for label in labels
    )

    if expected == 1:
        return 1.0 if observed == 1 else None
    return (observed - expected) / (1 - expected)


def krippendorff_alpha_nominal(
    labels_by_item: Iterable[Sequence[str]],
) -> float | None:
    """Return nominal Krippendorff's alpha for two or more raters.

    Items may have different numbers of non-missing ratings. Items with fewer
    than two ratings do not contribute to observed disagreement.
    """
    coincidence: Counter[tuple[str, str]] = Counter()

    for raw_labels in labels_by_item:
        labels = [str(label) for label in raw_labels if str(label).strip()]
        count = len(labels)
        if count < 2:
            continue
        for first_index, first in enumerate(labels):
            for second_index, second in enumerate(labels):
                if first_index == second_index:
                    continue
                coincidence[(first, second)] += 1 / (count - 1)

    total = float(sum(coincidence.values()))
    if total <= 1:
        return None

    observed_disagreement = (
        sum(
            value
            for (first, second), value in coincidence.items()
            if first != second
        )
        / total
    )

    marginals: Counter[str] = Counter()
    for (first, _), value in coincidence.items():
        marginals[first] += value

    expected_disagreement = (
        sum(
            first_count * second_count
            for first_label, first_count in marginals.items()
            for second_label, second_count in marginals.items()
            if first_label != second_label
        )
        / (total * (total - 1))
    )

    if expected_disagreement == 0:
        return 1.0 if observed_disagreement == 0 else None
    return 1 - observed_disagreement / expected_disagreement


def pairwise_raw_agreement(
    labels_by_item: Iterable[Sequence[str]],
) -> tuple[float | None, int]:
    """Return raw agreement across every within-item rater pair."""
    agreements = 0
    comparisons = 0

    for raw_labels in labels_by_item:
        labels = [str(label) for label in raw_labels if str(label).strip()]
        for first, second in combinations(labels, 2):
            comparisons += 1
            agreements += int(first == second)

    if comparisons == 0:
        return None, 0
    return agreements / comparisons, comparisons


def binary_confusion(
    reference: Sequence[str],
    predicted: Sequence[str],
    *,
    pass_label: str,
    failure_label: str,
) -> dict[str, int | float | None]:
    """Return pass/fail agreement metrics with failure as the positive class."""
    if len(reference) != len(predicted):
        raise ValueError("Binary confusion requires equally sized sequences")

    allowed = {pass_label, failure_label}
    unknown = (set(reference) | set(predicted)) - allowed
    if unknown:
        raise ValueError(f"Unexpected binary labels: {sorted(unknown)}")

    true_failure = sum(
        expected == failure_label and observed == failure_label
        for expected, observed in zip(reference, predicted, strict=True)
    )
    false_failure = sum(
        expected == pass_label and observed == failure_label
        for expected, observed in zip(reference, predicted, strict=True)
    )
    true_pass = sum(
        expected == pass_label and observed == pass_label
        for expected, observed in zip(reference, predicted, strict=True)
    )
    false_approval = sum(
        expected == failure_label and observed == pass_label
        for expected, observed in zip(reference, predicted, strict=True)
    )

    def safe_ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    total = len(reference)
    return {
        "true_failure": true_failure,
        "false_failure": false_failure,
        "true_pass": true_pass,
        "false_approval": false_approval,
        "accuracy": safe_ratio(true_failure + true_pass, total),
        "failure_precision": safe_ratio(
            true_failure,
            true_failure + false_failure,
        ),
        "failure_recall": safe_ratio(
            true_failure,
            true_failure + false_approval,
        ),
        "false_approval_rate": safe_ratio(
            false_approval,
            true_failure + false_approval,
        ),
        "false_failure_rate": safe_ratio(
            false_failure,
            true_pass + false_failure,
        ),
    }

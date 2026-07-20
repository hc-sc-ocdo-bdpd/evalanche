from __future__ import annotations

import math
from typing import Any

from evalanche.config import EvalConfig


class JudgeResponseValidationError(ValueError):
    """Raised when a judge response does not match the configured rubric."""


def validate_judge_response(
    response: dict[str, Any],
    config: EvalConfig,
) -> None:
    if not isinstance(response, dict):
        raise JudgeResponseValidationError(
            "Judge response must be a JSON object"
        )

    criterion_scores = response.get("criteria")
    if not isinstance(criterion_scores, dict):
        raise JudgeResponseValidationError(
            "Judge response field 'criteria' must be a JSON object"
        )

    expected_names = {criterion.name for criterion in config.criteria}
    actual_names = set(criterion_scores)

    missing_names = sorted(expected_names - actual_names)
    if missing_names:
        raise JudgeResponseValidationError(
            f"Judge response is missing criteria: {missing_names}"
        )

    unexpected_names = sorted(actual_names - expected_names)
    if unexpected_names:
        raise JudgeResponseValidationError(
            f"Judge response contains unexpected criteria: "
            f"{unexpected_names}"
        )

    for criterion in config.criteria:
        score_object = criterion_scores[criterion.name]
        if not isinstance(score_object, dict):
            raise JudgeResponseValidationError(
                f"Criterion '{criterion.name}' must be a JSON object"
            )

        score = score_object.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise JudgeResponseValidationError(
                f"Criterion '{criterion.name}' score must be a number"
            )

        if not math.isfinite(float(score)):
            raise JudgeResponseValidationError(
                f"Criterion '{criterion.name}' score must be finite"
            )

        if not config.scoring.score_min <= score <= config.scoring.score_max:
            raise JudgeResponseValidationError(
                f"Criterion '{criterion.name}' score must be between "
                f"{config.scoring.score_min} and "
                f"{config.scoring.score_max}"
            )

        reason = score_object.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise JudgeResponseValidationError(
                f"Criterion '{criterion.name}' reason must be a "
                "non-empty string"
            )

    overall_reason = response.get("overall_reason")
    if not isinstance(overall_reason, str) or not overall_reason.strip():
        raise JudgeResponseValidationError(
            "Judge response field 'overall_reason' must be a "
            "non-empty string"
        )
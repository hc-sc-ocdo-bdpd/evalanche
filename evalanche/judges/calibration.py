from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence

import pandas as pd

from evalanche import __version__
from evalanche.judges.agreement import (
    binary_confusion,
    cohen_kappa,
    krippendorff_alpha_nominal,
    pairwise_raw_agreement,
)
from evalanche.judges.protocol import (
    JudgeProtocol,
    achieved_level_from_gates,
    level_at_least,
    load_judge_protocol,
    protocol_contract_sha256,
)
from evalanche.statistics import wilson_score_interval


HUMAN_COLUMNS = {
    "case_id",
    "criterion",
    "reviewer_id",
    "annotation_stage",
    "label",
}
OBSERVATION_COLUMNS = {
    "case_id",
    "criterion",
    "judge_variant_id",
    "trial_id",
    "prompt_variant_id",
    "identity_condition",
    "presentation_order",
    "label",
    "status",
    "uncertainty",
}


@dataclass(frozen=True)
class JudgeValidationArtifacts:
    report_json_path: Path
    report_markdown_path: Path
    gates_path: Path
    disagreements_path: Path
    subgroups_path: Path
    achieved_level: str
    target_level: str
    target_met: bool


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_text_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Judge validation input not found: {path}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    *,
    source: Path,
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def _require_nonblank(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    source: Path,
) -> None:
    for column in columns:
        blank = frame[column].astype(str).str.strip() == ""
        if blank.any():
            rows = (blank[blank].index + 2).tolist()[:10]
            raise ValueError(
                f"{source} has blank {column!r} values on CSV rows {rows}"
            )
        frame[column] = frame[column].astype(str).str.strip()


def _resolve_data_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _load_inputs(
    protocol: JudgeProtocol,
    *,
    root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    paths = {
        "cases": _resolve_data_path(root, protocol.data.cases_path),
        "human_annotations": _resolve_data_path(
            root,
            protocol.data.human_annotations_path,
        ),
        "judge_observations": _resolve_data_path(
            root,
            protocol.data.judge_observations_path,
        ),
    }
    cases = _read_text_csv(paths["cases"])
    humans = _read_text_csv(paths["human_annotations"])
    observations = _read_text_csv(paths["judge_observations"])

    case_required = {
        "case_id",
        protocol.data.split_column,
        *protocol.data.slice_columns,
    }
    _require_columns(cases, case_required, source=paths["cases"])
    _require_nonblank(
        cases,
        sorted(case_required),
        source=paths["cases"],
    )
    if cases.empty:
        raise ValueError("Judge validation cases cannot be empty")
    if cases["case_id"].duplicated().any():
        duplicates = sorted(
            cases.loc[cases["case_id"].duplicated(False), "case_id"].unique()
        )
        raise ValueError(f"Judge validation case IDs must be unique: {duplicates}")

    _require_columns(humans, HUMAN_COLUMNS, source=paths["human_annotations"])
    _require_nonblank(
        humans,
        sorted(HUMAN_COLUMNS),
        source=paths["human_annotations"],
    )
    if humans.empty:
        raise ValueError("Human annotations cannot be empty")
    unknown_case_ids = sorted(set(humans["case_id"]) - set(cases["case_id"]))
    if unknown_case_ids:
        raise ValueError(
            "Human annotations reference unknown case IDs: "
            f"{unknown_case_ids[:10]}"
        )
    unknown_criteria = sorted(
        set(humans["criterion"]) - set(protocol.data.criteria)
    )
    if unknown_criteria:
        raise ValueError(
            f"Human annotations contain unknown criteria: {unknown_criteria}"
        )
    unknown_stages = sorted(
        set(humans["annotation_stage"]) - {"independent", "adjudicated"}
    )
    if unknown_stages:
        raise ValueError(
            f"Unknown human annotation stages: {unknown_stages}"
        )
    unknown_labels = sorted(set(humans["label"]) - set(protocol.data.labels))
    if unknown_labels:
        raise ValueError(f"Unknown human labels: {unknown_labels}")
    human_key = ["case_id", "criterion", "reviewer_id", "annotation_stage"]
    if humans.duplicated(human_key).any():
        raise ValueError(
            "Human annotations require one label per case, criterion, "
            "reviewer, and stage"
        )
    adjudicated = humans[humans["annotation_stage"] == "adjudicated"]
    if adjudicated.duplicated(["case_id", "criterion"]).any():
        raise ValueError(
            "Human annotations require at most one adjudicated label per "
            "case and criterion"
        )

    _require_columns(
        observations,
        OBSERVATION_COLUMNS,
        source=paths["judge_observations"],
    )
    _require_nonblank(
        observations,
        [
            "case_id",
            "criterion",
            "judge_variant_id",
            "trial_id",
            "prompt_variant_id",
            "identity_condition",
            "presentation_order",
            "status",
        ],
        source=paths["judge_observations"],
    )
    if observations.empty:
        raise ValueError("Judge observations cannot be empty")
    unknown_case_ids = sorted(
        set(observations["case_id"]) - set(cases["case_id"])
    )
    if unknown_case_ids:
        raise ValueError(
            "Judge observations reference unknown case IDs: "
            f"{unknown_case_ids[:10]}"
        )
    unknown_criteria = sorted(
        set(observations["criterion"]) - set(protocol.data.criteria)
    )
    if unknown_criteria:
        raise ValueError(
            f"Judge observations contain unknown criteria: {unknown_criteria}"
        )
    unknown_statuses = sorted(
        set(observations["status"]) - {"success", "error", "abstain"}
    )
    if unknown_statuses:
        raise ValueError(f"Unknown judge observation statuses: {unknown_statuses}")
    unknown_identities = sorted(
        set(observations["identity_condition"]) - {"blinded", "named"}
    )
    if unknown_identities:
        raise ValueError(f"Unknown identity conditions: {unknown_identities}")
    allowed_orders = (
        {"pointwise"}
        if protocol.contract.mode == "pointwise"
        else {"AB", "BA"}
    )
    unknown_orders = sorted(set(observations["presentation_order"]) - allowed_orders)
    if unknown_orders:
        raise ValueError(f"Unknown presentation orders: {unknown_orders}")

    successful = observations["status"] == "success"
    blank_labels = observations["label"].astype(str).str.strip() == ""
    if (successful & blank_labels).any():
        raise ValueError("Successful judge observations require a label")
    if ((~successful) & (~blank_labels)).any():
        raise ValueError("Error and abstain observations must leave label blank")
    unknown_labels = sorted(
        set(observations.loc[successful, "label"]) - set(protocol.data.labels)
    )
    if unknown_labels:
        raise ValueError(f"Unknown judge observation labels: {unknown_labels}")

    observations["uncertainty"] = observations["uncertainty"].str.strip()
    uncertainty_numeric = pd.to_numeric(
        observations["uncertainty"].replace("", pd.NA),
        errors="coerce",
    )
    invalid_uncertainty = (
        (observations["uncertainty"] != "")
        & (
            uncertainty_numeric.isna()
            | (uncertainty_numeric < 0)
            | (uncertainty_numeric > 1)
        )
    )
    if invalid_uncertainty.any():
        raise ValueError("Judge uncertainty must be blank or between 0 and 1")
    observations["uncertainty"] = uncertainty_numeric

    observation_key = [
        "case_id",
        "criterion",
        "judge_variant_id",
        "trial_id",
        "prompt_variant_id",
        "identity_condition",
        "presentation_order",
    ]
    if observations.duplicated(observation_key).any():
        raise ValueError(
            "Judge observations require one row per case, criterion, judge "
            "variant, trial, prompt variant, identity condition, and order"
        )

    return cases, humans, observations, paths


def _percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return ordered[lower_index] + fraction * (
        ordered[upper_index] - ordered[lower_index]
    )


def _bootstrap_kappa_interval(
    reference: Sequence[str],
    predicted: Sequence[str],
    *,
    samples: int,
    seed: int,
) -> tuple[float | None, float | None]:
    if len(reference) < 2:
        return None, None
    generator = random.Random(seed)
    values: list[float] = []
    for _ in range(samples):
        indexes = [generator.randrange(len(reference)) for _ in reference]
        value = cohen_kappa(
            [reference[index] for index in indexes],
            [predicted[index] for index in indexes],
        )
        if value is not None:
            values.append(value)
    return _percentile(values, 0.025), _percentile(values, 0.975)


def _proportion_interval(
    numerator: int,
    denominator: int,
) -> tuple[float | None, float | None]:
    if denominator == 0:
        return None, None
    return wilson_score_interval(numerator, denominator)


def _baseline_observations(
    observations: pd.DataFrame,
    protocol: JudgeProtocol,
    *,
    judge_variant_id: str | None = None,
    criterion: str | None = None,
) -> pd.DataFrame:
    controls = protocol.controls
    return observations[
        (observations["judge_variant_id"] == (
            judge_variant_id or controls.primary_judge_variant_id
        ))
        & (observations["criterion"] == (
            criterion or protocol.data.primary_criterion
        ))
        & (observations["trial_id"] == controls.baseline_trial_id)
        & (
            observations["prompt_variant_id"]
            == controls.baseline_prompt_variant_id
        )
        & (
            observations["identity_condition"]
            == controls.baseline_identity_condition
        )
        & (
            observations["presentation_order"]
            == controls.baseline_presentation_order
        )
    ].copy()


def _adjudicated_labels(
    humans: pd.DataFrame,
    *,
    criterion: str,
) -> pd.DataFrame:
    selected = humans[
        (humans["annotation_stage"] == "adjudicated")
        & (humans["criterion"] == criterion)
    ][["case_id", "label"]].copy()
    return selected.rename(columns={"label": "reference_label"})


def _comparison_frame(
    cases: pd.DataFrame,
    humans: pd.DataFrame,
    observations: pd.DataFrame,
    protocol: JudgeProtocol,
    *,
    judge_variant_id: str | None = None,
    criterion: str | None = None,
) -> pd.DataFrame:
    criterion = criterion or protocol.data.primary_criterion
    baseline = _baseline_observations(
        observations,
        protocol,
        judge_variant_id=judge_variant_id,
        criterion=criterion,
    )[["case_id", "label", "status", "uncertainty"]].rename(
        columns={"label": "judge_label"}
    )
    return (
        cases.merge(
            _adjudicated_labels(humans, criterion=criterion),
            on="case_id",
            how="left",
        )
        .merge(baseline, on="case_id", how="left")
        .assign(criterion=criterion)
    )


def _agreement_metrics(
    frame: pd.DataFrame,
    protocol: JudgeProtocol,
    *,
    bootstrap: bool = True,
) -> dict[str, Any]:
    total = int(len(frame))
    has_reference = frame["reference_label"].notna()
    has_observation = frame["status"].notna()
    success = has_observation & (frame["status"] == "success")
    comparable = has_reference & success & frame["judge_label"].notna()
    matched = frame.loc[comparable]
    reference = matched["reference_label"].astype(str).tolist()
    predicted = matched["judge_label"].astype(str).tolist()
    agreements = sum(
        expected == observed
        for expected, observed in zip(reference, predicted, strict=True)
    )
    raw_agreement = agreements / len(reference) if reference else None
    kappa = cohen_kappa(reference, predicted)
    if bootstrap:
        kappa_low, kappa_high = _bootstrap_kappa_interval(
            reference,
            predicted,
            samples=protocol.controls.bootstrap_samples,
            seed=protocol.controls.bootstrap_seed,
        )
    else:
        kappa_low, kappa_high = None, None
    accuracy_low, accuracy_high = _proportion_interval(
        agreements,
        len(reference),
    )
    missing = int((~has_observation).sum())
    errors = int((frame["status"] == "error").sum())
    abstentions = int((frame["status"] == "abstain").sum())
    unscored = missing + errors + abstentions

    metrics: dict[str, Any] = {
        "cases": total,
        "human_reference_cases": int(has_reference.sum()),
        "comparable_cases": int(len(reference)),
        "missing_observations": missing,
        "judge_errors": errors,
        "judge_abstentions": abstentions,
        "observation_failure_rate": unscored / total if total else None,
        "raw_agreement": raw_agreement,
        "accuracy_ci_low": accuracy_low,
        "accuracy_ci_high": accuracy_high,
        "cohen_kappa": kappa,
        "cohen_kappa_ci_low": kappa_low,
        "cohen_kappa_ci_high": kappa_high,
        "reported_uncertainty_coverage": (
            float(matched["uncertainty"].notna().mean())
            if not matched.empty
            else None
        ),
        "average_reported_uncertainty": (
            float(pd.to_numeric(matched["uncertainty"]).mean())
            if matched["uncertainty"].notna().any()
            else None
        ),
    }

    if protocol.data.pass_label is not None:
        confusion = binary_confusion(
            reference,
            predicted,
            pass_label=protocol.data.pass_label,
            failure_label=str(protocol.data.failure_label),
        )
        metrics.update(confusion)
        failure_total = int(confusion["true_failure"]) + int(
            confusion["false_approval"]
        )
        predicted_failure_total = int(confusion["true_failure"]) + int(
            confusion["false_failure"]
        )
        recall_low, recall_high = _proportion_interval(
            int(confusion["true_failure"]),
            failure_total,
        )
        precision_low, precision_high = _proportion_interval(
            int(confusion["true_failure"]),
            predicted_failure_total,
        )
        metrics.update(
            {
                "failure_precision_ci_low": precision_low,
                "failure_precision_ci_high": precision_high,
                "failure_recall_ci_low": recall_low,
                "failure_recall_ci_high": recall_high,
                "false_approval_rate_ci_low": (
                    1 - recall_high if recall_high is not None else None
                ),
                "false_approval_rate_ci_high": (
                    1 - recall_low if recall_low is not None else None
                ),
            }
        )
    return metrics


def _human_metrics(
    cases: pd.DataFrame,
    humans: pd.DataFrame,
    protocol: JudgeProtocol,
    *,
    split: str,
) -> dict[str, Any]:
    split_cases = set(
        cases.loc[
            cases[protocol.data.split_column] == split,
            "case_id",
        ]
    )
    independent = humans[
        (humans["annotation_stage"] == "independent")
        & (humans["criterion"] == protocol.data.primary_criterion)
        & (humans["case_id"].isin(split_cases))
    ]
    grouped = independent.groupby("case_id")["label"].apply(list).to_dict()
    label_sets = list(grouped.values())
    raw_agreement, pair_comparisons = pairwise_raw_agreement(label_sets)
    counts = independent.groupby("case_id")["reviewer_id"].nunique()
    adjudicated = humans[
        (humans["annotation_stage"] == "adjudicated")
        & (humans["criterion"] == protocol.data.primary_criterion)
        & (humans["case_id"].isin(split_cases))
    ]
    return {
        "split": split,
        "cases": len(split_cases),
        "cases_with_independent_labels": int(len(counts)),
        "independent_label_coverage": (
            len(counts) / len(split_cases) if split_cases else None
        ),
        "minimum_independent_raters": (
            int(counts.min()) if not counts.empty else 0
        ),
        "average_independent_raters": (
            float(counts.mean()) if not counts.empty else 0.0
        ),
        "pair_comparisons": pair_comparisons,
        "raw_pairwise_agreement": raw_agreement,
        "krippendorff_alpha": krippendorff_alpha_nominal(label_sets),
        "adjudicated_cases": int(adjudicated["case_id"].nunique()),
        "adjudication_coverage": (
            adjudicated["case_id"].nunique() / len(split_cases)
            if split_cases
            else None
        ),
        "human_disagreement_cases": int(
            sum(len(set(labels)) > 1 for labels in label_sets)
        ),
    }


def _condition_metrics(
    observations: pd.DataFrame,
    *,
    condition_column: str,
    expected_case_ids: set[str],
) -> dict[str, Any]:
    successful = observations[observations["status"] == "success"]
    values = sorted(successful[condition_column].unique().tolist())
    labels_by_case = successful.groupby("case_id")["label"].apply(list)
    conditions_by_case = successful.groupby("case_id")[
        condition_column
    ].nunique()
    complete_cases = int(
        sum(
            conditions_by_case.get(case_id, 0) == len(values)
            for case_id in expected_case_ids
        )
    ) if values else 0
    raw_agreement, pair_comparisons = pairwise_raw_agreement(labels_by_case)
    return {
        "condition": condition_column,
        "observed_values": values,
        "observed_value_count": len(values),
        "expected_cases": len(expected_case_ids),
        "cases_with_observations": int(successful["case_id"].nunique()),
        "cases_with_comparisons": int(
            sum(len(labels) >= 2 for labels in labels_by_case)
        ),
        "cases_with_all_conditions": complete_cases,
        "complete_condition_coverage": (
            complete_cases / len(expected_case_ids)
            if expected_case_ids
            else None
        ),
        "pair_comparisons": pair_comparisons,
        "raw_agreement": raw_agreement,
        "flip_rate": 1 - raw_agreement if raw_agreement is not None else None,
        "krippendorff_alpha": krippendorff_alpha_nominal(labels_by_case),
    }


def _control_frames(
    observations: pd.DataFrame,
    protocol: JudgeProtocol,
) -> dict[str, pd.DataFrame]:
    controls = protocol.controls
    primary = observations[
        (observations["judge_variant_id"] == controls.primary_judge_variant_id)
        & (observations["criterion"] == protocol.data.primary_criterion)
    ]
    return {
        "stability": primary[
            (
                primary["prompt_variant_id"]
                == controls.baseline_prompt_variant_id
            )
            & (
                primary["identity_condition"]
                == controls.baseline_identity_condition
            )
            & (
                primary["presentation_order"]
                == controls.baseline_presentation_order
            )
        ],
        "prompt": primary[
            (primary["trial_id"] == controls.baseline_trial_id)
            & (
                primary["identity_condition"]
                == controls.baseline_identity_condition
            )
            & (
                primary["presentation_order"]
                == controls.baseline_presentation_order
            )
        ],
        "identity": primary[
            (primary["trial_id"] == controls.baseline_trial_id)
            & (
                primary["prompt_variant_id"]
                == controls.baseline_prompt_variant_id
            )
            & (
                primary["presentation_order"]
                == controls.baseline_presentation_order
            )
        ],
        "position": primary[
            (primary["trial_id"] == controls.baseline_trial_id)
            & (
                primary["prompt_variant_id"]
                == controls.baseline_prompt_variant_id
            )
            & (
                primary["identity_condition"]
                == controls.baseline_identity_condition
            )
        ],
    }


def _position_metrics(
    observations: pd.DataFrame,
    protocol: JudgeProtocol,
    *,
    expected_case_ids: set[str],
) -> dict[str, Any]:
    metrics = _condition_metrics(
        observations,
        condition_column="presentation_order",
        expected_case_ids=expected_case_ids,
    )
    if protocol.contract.mode != "pairwise" or protocol.data.pairwise is None:
        metrics.update(
            {
                "applicable": False,
                "first_position_selection_rate": None,
                "position_bias": None,
            }
        )
        return metrics

    labels = protocol.data.pairwise
    successful = observations[observations["status"] == "success"].copy()

    def selected_position(row: pd.Series) -> str:
        if row["label"] == labels.tie_label:
            return "tie"
        if row["presentation_order"] == "AB":
            return (
                "first"
                if row["label"] == labels.candidate_a_label
                else "second"
            )
        return (
            "first"
            if row["label"] == labels.candidate_b_label
            else "second"
        )

    successful["selected_position"] = successful.apply(
        selected_position,
        axis=1,
    )
    decisive = successful[successful["selected_position"] != "tie"]
    first_rate = (
        float((decisive["selected_position"] == "first").mean())
        if not decisive.empty
        else None
    )
    metrics.update(
        {
            "applicable": True,
            "decisive_observations": int(len(decisive)),
            "first_position_selection_rate": first_rate,
            "position_bias": (
                abs(first_rate - 0.5) if first_rate is not None else None
            ),
        }
    )
    return metrics


def _self_preference_metrics(
    cases: pd.DataFrame,
    observations: pd.DataFrame,
) -> dict[str, Any]:
    if "self_preference_label" not in cases.columns:
        return {
            "applicable": False,
            "named_selection_rate": None,
            "blinded_selection_rate": None,
            "self_preference_shift": None,
        }
    joined = observations.merge(
        cases[["case_id", "self_preference_label"]],
        on="case_id",
        how="left",
    )
    joined = joined[
        (joined["status"] == "success")
        & (joined["self_preference_label"].astype(str).str.strip() != "")
    ]

    def selection_rate(condition: str) -> float | None:
        selected = joined[joined["identity_condition"] == condition]
        if selected.empty:
            return None
        return float(
            (selected["label"] == selected["self_preference_label"]).mean()
        )

    named = selection_rate("named")
    blinded = selection_rate("blinded")
    return {
        "applicable": not joined.empty,
        "named_selection_rate": named,
        "blinded_selection_rate": blinded,
        "self_preference_shift": (
            named - blinded if named is not None and blinded is not None else None
        ),
    }


def _variant_metrics(
    cases: pd.DataFrame,
    humans: pd.DataFrame,
    observations: pd.DataFrame,
    protocol: JudgeProtocol,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    split_column = protocol.data.split_column
    validation_split = protocol.data.validation_split
    for variant in sorted(observations["judge_variant_id"].unique()):
        frame = _comparison_frame(
            cases,
            humans,
            observations,
            protocol,
            judge_variant_id=variant,
        )
        selected = frame[frame[split_column] == validation_split]
        records.append(
            {
                "judge_variant_id": variant,
                **_agreement_metrics(selected, protocol, bootstrap=False),
            }
        )
    return records


def _criterion_metrics(
    cases: pd.DataFrame,
    humans: pd.DataFrame,
    observations: pd.DataFrame,
    protocol: JudgeProtocol,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for criterion in protocol.data.criteria:
        frame = _comparison_frame(
            cases,
            humans,
            observations,
            protocol,
            criterion=criterion,
        )
        for split in (
            protocol.data.calibration_split,
            protocol.data.validation_split,
        ):
            selected = frame[frame[protocol.data.split_column] == split]
            records.append(
                {
                    "criterion": criterion,
                    "split": split,
                    **_agreement_metrics(selected, protocol, bootstrap=False),
                }
            )
    return records


def _subgroup_metrics(
    comparison: pd.DataFrame,
    protocol: JudgeProtocol,
) -> list[dict[str, Any]]:
    validation = comparison[
        comparison[protocol.data.split_column] == protocol.data.validation_split
    ]
    records: list[dict[str, Any]] = []
    for column in protocol.data.slice_columns:
        for value, group in validation.groupby(column, dropna=False):
            records.append(
                {
                    "slice_column": column,
                    "slice_value": str(value),
                    **_agreement_metrics(group, protocol, bootstrap=False),
                }
            )
    return records


def _threshold_gate(
    *,
    gate_id: str,
    level: str,
    observed: Any,
    threshold: Any,
    comparator: Literal[">=", "<=", "=="],
    note: str,
) -> dict[str, Any]:
    if observed is None or threshold is None:
        passed = False
    elif comparator == ">=":
        passed = float(observed) >= float(threshold)
    elif comparator == "<=":
        passed = float(observed) <= float(threshold)
    else:
        passed = observed == threshold
    return {
        "gate_id": gate_id,
        "required_for_level": level,
        "observed": observed,
        "comparator": comparator,
        "threshold": threshold,
        "passed": bool(passed),
        "note": note,
    }


def _boolean_gate(
    *,
    gate_id: str,
    level: str,
    observed: bool,
    note: str,
) -> dict[str, Any]:
    return _threshold_gate(
        gate_id=gate_id,
        level=level,
        observed=bool(observed),
        threshold=True,
        comparator="==",
        note=note,
    )


def _build_gates(
    *,
    protocol: JudgeProtocol,
    human_validation: dict[str, Any],
    judge_validation: dict[str, Any],
    stability: dict[str, Any],
    prompt: dict[str, Any],
    identity: dict[str, Any],
    self_preference: dict[str, Any],
    position: dict[str, Any],
    subgroups: list[dict[str, Any]],
    variants: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    thresholds = protocol.thresholds
    controls = protocol.controls
    human = protocol.human_review
    gates = [
        _boolean_gate(
            gate_id="not_synthetic_fixture",
            level="calibrated",
            observed=not protocol.synthetic_fixture,
            note="Synthetic fixtures can demonstrate the workflow only.",
        ),
        _threshold_gate(
            gate_id="validation_case_count",
            level="calibrated",
            observed=judge_validation.get("cases"),
            threshold=thresholds.minimum_validation_cases,
            comparator=">=",
            note="The minimum is protocol policy, not a universal cutoff.",
        ),
        _threshold_gate(
            gate_id="independent_human_raters",
            level="calibrated",
            observed=human_validation.get("minimum_independent_raters"),
            threshold=max(2, human.minimum_independent_raters),
            comparator=">=",
            note=(
                "At least two independent raters are required on every "
                "held-out validation case."
            ),
        ),
        _threshold_gate(
            gate_id="independent_human_label_coverage",
            level="calibrated",
            observed=human_validation.get("independent_label_coverage"),
            threshold=1.0,
            comparator=">=",
            note="Every validation case needs independent human labels.",
        ),
        _threshold_gate(
            gate_id="adjudication_coverage",
            level="calibrated",
            observed=human_validation.get("adjudication_coverage"),
            threshold=1.0,
            comparator=">=",
            note="Every validation case needs an adjudicated reference label.",
        ),
        _threshold_gate(
            gate_id="human_krippendorff_alpha",
            level="calibrated",
            observed=human_validation.get("krippendorff_alpha"),
            threshold=thresholds.minimum_human_alpha,
            comparator=">=",
            note="Chance-corrected agreement among independent human raters.",
        ),
        _threshold_gate(
            gate_id="judge_human_cohen_kappa",
            level="calibrated",
            observed=judge_validation.get("cohen_kappa"),
            threshold=thresholds.minimum_judge_human_kappa,
            comparator=">=",
            note="Chance-corrected agreement with adjudicated labels.",
        ),
        _threshold_gate(
            gate_id="failure_precision",
            level="calibrated",
            observed=judge_validation.get("failure_precision"),
            threshold=thresholds.minimum_failure_precision,
            comparator=">=",
            note="Failure is treated as the positive class.",
        ),
        _threshold_gate(
            gate_id="failure_recall",
            level="calibrated",
            observed=judge_validation.get("failure_recall"),
            threshold=thresholds.minimum_failure_recall,
            comparator=">=",
            note="Low recall means the judge approves too many human failures.",
        ),
        _threshold_gate(
            gate_id="false_approval_rate",
            level="calibrated",
            observed=judge_validation.get("false_approval_rate"),
            threshold=thresholds.maximum_false_approval_rate,
            comparator="<=",
            note="Human failures incorrectly approved by the judge.",
        ),
        _threshold_gate(
            gate_id="observation_failure_rate",
            level="calibrated",
            observed=judge_validation.get("observation_failure_rate"),
            threshold=thresholds.maximum_observation_failure_rate,
            comparator="<=",
            note="Includes missing, errored, and abstained baseline judgments.",
        ),
        _boolean_gate(
            gate_id="runtime_candidate_identity_blinded",
            level="calibrated",
            observed=protocol.contract.candidate_identity_blinded,
            note="Candidate model identity is absent from the baseline prompt.",
        ),
        _boolean_gate(
            gate_id="independent_before_adjudication",
            level="decision_grade",
            observed=human.independent_before_adjudication,
            note="Independent labels were recorded before adjudication.",
        ),
        _boolean_gate(
            gate_id="human_candidate_identity_blinded",
            level="decision_grade",
            observed=human.candidate_identity_blinded,
            note="Human reviewers did not receive candidate identities.",
        ),
        _boolean_gate(
            gate_id="human_judge_identity_blinded",
            level="decision_grade",
            observed=human.judge_identity_blinded,
            note="Human reviewers did not receive judge identities.",
        ),
        _boolean_gate(
            gate_id="response_cache_disabled",
            level="decision_grade",
            observed=controls.response_cache_disabled,
            note="Repeated observations must be independent calls.",
        ),
        _threshold_gate(
            gate_id="independent_trial_count",
            level="decision_grade",
            observed=stability.get("observed_value_count"),
            threshold=max(3, controls.minimum_independent_trials),
            comparator=">=",
            note="At least three trials are required by this implementation.",
        ),
        _threshold_gate(
            gate_id="test_retest_alpha",
            level="decision_grade",
            observed=stability.get("krippendorff_alpha"),
            threshold=thresholds.minimum_stability_alpha,
            comparator=">=",
            note="Stability is necessary but does not establish validity.",
        ),
        _threshold_gate(
            gate_id="independent_trial_coverage",
            level="decision_grade",
            observed=stability.get("complete_condition_coverage"),
            threshold=1.0,
            comparator=">=",
            note="Every validation case has every declared independent trial.",
        ),
        _threshold_gate(
            gate_id="prompt_variant_count",
            level="decision_grade",
            observed=prompt.get("observed_value_count"),
            threshold=max(2, controls.minimum_prompt_variants),
            comparator=">=",
            note="Semantically equivalent prompt variants are compared.",
        ),
        _threshold_gate(
            gate_id="prompt_flip_rate",
            level="decision_grade",
            observed=prompt.get("flip_rate"),
            threshold=thresholds.maximum_prompt_flip_rate,
            comparator="<=",
            note="A flip is a changed label under a prompt variant.",
        ),
        _threshold_gate(
            gate_id="prompt_variant_coverage",
            level="decision_grade",
            observed=prompt.get("complete_condition_coverage"),
            threshold=1.0,
            comparator=">=",
            note="Every validation case has every declared prompt variant.",
        ),
        _boolean_gate(
            gate_id="identity_bias_test_required",
            level="decision_grade",
            observed=controls.require_identity_bias_test,
            note="Named and blinded conditions test model-name effects.",
        ),
    ]

    if controls.require_identity_bias_test:
        gates.extend(
            [
                _threshold_gate(
                    gate_id="identity_condition_count",
                    level="decision_grade",
                    observed=identity.get("observed_value_count"),
                    threshold=2,
                    comparator=">=",
                    note="Both blinded and named conditions are required.",
                ),
                _threshold_gate(
                    gate_id="identity_flip_rate",
                    level="decision_grade",
                    observed=identity.get("flip_rate"),
                    threshold=thresholds.maximum_identity_flip_rate,
                    comparator="<=",
                    note="Changed labels under named versus blinded prompts.",
                ),
                _threshold_gate(
                    gate_id="identity_condition_coverage",
                    level="decision_grade",
                    observed=identity.get("complete_condition_coverage"),
                    threshold=1.0,
                    comparator=">=",
                    note="Every validation case has both identity conditions.",
                ),
            ]
        )
        if self_preference.get("applicable"):
            shift = self_preference.get("self_preference_shift")
            gates.append(
                _threshold_gate(
                    gate_id="self_preference_shift",
                    level="decision_grade",
                    observed=abs(shift) if shift is not None else None,
                    threshold=thresholds.maximum_self_preference_shift,
                    comparator="<=",
                    note="Absolute named-versus-blinded self-selection shift.",
                )
            )

    if protocol.contract.mode == "pairwise":
        gates.extend(
            [
                _boolean_gate(
                    gate_id="position_swap_required",
                    level="decision_grade",
                    observed=controls.require_position_swap,
                    note="Pairwise protocols must declare A/B and B/A control.",
                ),
                _threshold_gate(
                    gate_id="position_order_count",
                    level="decision_grade",
                    observed=position.get("observed_value_count"),
                    threshold=2,
                    comparator=">=",
                    note="Both canonical response orders are present.",
                ),
                _threshold_gate(
                    gate_id="position_flip_rate",
                    level="decision_grade",
                    observed=position.get("flip_rate"),
                    threshold=thresholds.maximum_position_flip_rate,
                    comparator="<=",
                    note="Canonical winner changes after response-order reversal.",
                ),
                _threshold_gate(
                    gate_id="position_bias",
                    level="decision_grade",
                    observed=position.get("position_bias"),
                    threshold=thresholds.maximum_position_bias,
                    comparator="<=",
                    note="Absolute deviation from equal first-position selection.",
                ),
                _threshold_gate(
                    gate_id="position_order_coverage",
                    level="decision_grade",
                    observed=position.get("complete_condition_coverage"),
                    threshold=1.0,
                    comparator=">=",
                    note="Every validation case has both response orders.",
                ),
            ]
        )

    required_slices = controls.required_slice_values
    gates.append(
        _boolean_gate(
            gate_id="required_subgroups_declared",
            level="decision_grade",
            observed=bool(required_slices),
            note="At least one decision-relevant subgroup must be declared.",
        )
    )
    subgroup_lookup = {
        (row["slice_column"], row["slice_value"]): row for row in subgroups
    }
    for column, values in required_slices.items():
        for value in values:
            row = subgroup_lookup.get((column, value), {})
            gates.append(
                _threshold_gate(
                    gate_id=f"subgroup_cases:{column}={value}",
                    level="decision_grade",
                    observed=row.get("cases"),
                    threshold=thresholds.minimum_cases_per_required_slice,
                    comparator=">=",
                    note="Validation cases in the declared required subgroup.",
                )
            )

    governance = protocol.governance
    gates.extend(
        [
            _boolean_gate(
                gate_id="protocol_frozen",
                level="publishable",
                observed=governance.protocol_status == "frozen",
                note="The protocol was frozen before final evidence claims.",
            ),
            _boolean_gate(
                gate_id="heldout_not_used_for_development",
                level="publishable",
                observed=governance.heldout_not_used_for_development,
                note="The final validation split was not judge-development data.",
            ),
            _boolean_gate(
                gate_id="independent_review",
                level="publishable",
                observed=governance.independent_review,
                note="Independent review is documented.",
            ),
            _boolean_gate(
                gate_id="conflicts_documented",
                level="publishable",
                observed=governance.conflicts_documented,
                note="Reviewer and evaluator conflicts are documented.",
            ),
            _boolean_gate(
                gate_id="raw_artifacts_preserved",
                level="publishable",
                observed=governance.raw_artifacts_preserved,
                note="Raw human and judge observations remain auditable.",
            ),
            _boolean_gate(
                gate_id="judge_selection_documented",
                level="publishable",
                observed=(
                    len(variants) >= 2
                    or governance.judge_selection_rationale is not None
                ),
                note=(
                    "Multiple judge variants were compared or the frozen "
                    "protocol documents why one judge was selected."
                ),
            ),
        ]
    )
    return gates


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is pd.NA or (isinstance(value, float) and pd.isna(value)):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _format_metric(value: Any) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _markdown_table(
    records: list[dict[str, Any]],
    columns: list[tuple[str, str]],
) -> str:
    if not records:
        return "_No records._"
    header = "| " + " | ".join(label for _, label in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [header, separator]
    for record in records:
        cells = []
        for key, _ in columns:
            text = _format_metric(record.get(key)).replace("|", "\\|")
            cells.append(text)
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _build_markdown_report(
    report: dict[str, Any],
    *,
    paths: dict[str, Path],
) -> str:
    validation = report["judge_human_agreement"]["validation"]
    human = report["human_reference"]["validation"]
    controls = report["controls"]
    gates = report["gates"]
    blockers = [
        gate
        for gate in gates
        if not gate["passed"]
        and not level_at_least(
            report["achieved_level"],
            gate["required_for_level"],
        )
    ]
    target_text = "met" if report["target_met"] else "not met"
    confusion = [
        {
            "reference": "Human failure",
            "judge_failure": validation.get("true_failure"),
            "judge_pass": validation.get("false_approval"),
        },
        {
            "reference": "Human pass",
            "judge_failure": validation.get("false_failure"),
            "judge_pass": validation.get("true_pass"),
        },
    ]
    return f"""# Judge Validation Report

## Status

- **Protocol:** `{report['protocol']['protocol_id']}@{report['protocol']['version']}`
- **Target level:** `{report['target_level']}`
- **Achieved level:** `{report['achieved_level']}`
- **Target:** {target_text}
- **Task:** {report['scope']['task_name']}
- **Construct:** {report['scope']['measured_construct']}
- **Primary judge variant:** `{report['scope']['primary_judge_variant_id']}`

This status applies only to the exact task, rubric, prompt, judge model,
settings, data, and validation protocol identified by the contract hash. It is
not a general certification of the model as a judge. Numeric thresholds are
declared protocol policy, not universal research cutoffs.

## Human reference quality

| Metric | Validation result |
| --- | --- |
| Cases | {_format_metric(human.get('cases'))} |
| Minimum independent raters | {_format_metric(human.get('minimum_independent_raters'))} |
| Raw pairwise agreement | {_format_metric(human.get('raw_pairwise_agreement'))} |
| Krippendorff alpha | {_format_metric(human.get('krippendorff_alpha'))} |
| Adjudication coverage | {_format_metric(human.get('adjudication_coverage'))} |
| Human disagreement cases | {_format_metric(human.get('human_disagreement_cases'))} |

## Judge agreement with adjudicated humans

| Metric | Validation result |
| --- | --- |
| Comparable cases | {_format_metric(validation.get('comparable_cases'))} |
| Raw agreement | {_format_metric(validation.get('raw_agreement'))} |
| Cohen kappa | {_format_metric(validation.get('cohen_kappa'))} |
| 95% bootstrap kappa interval | {_format_metric(validation.get('cohen_kappa_ci_low'))} to {_format_metric(validation.get('cohen_kappa_ci_high'))} |
| Failure precision | {_format_metric(validation.get('failure_precision'))} |
| Failure recall | {_format_metric(validation.get('failure_recall'))} |
| False approval rate | {_format_metric(validation.get('false_approval_rate'))} |
| Observation failure rate | {_format_metric(validation.get('observation_failure_rate'))} |
| Reported uncertainty coverage | {_format_metric(validation.get('reported_uncertainty_coverage'))} |

### Confusion matrix

{_markdown_table(confusion, [('reference', 'Human reference'), ('judge_failure', 'Judge failure'), ('judge_pass', 'Judge pass')])}

Failure is the positive class. A false approval is a human failure that the
judge marked as passing.

## Stability and bias controls

{_markdown_table([
    {
        'control': 'Repeated trials',
        'conditions': controls['stability'].get('observed_value_count'),
        'agreement': controls['stability'].get('raw_agreement'),
        'flip_rate': controls['stability'].get('flip_rate'),
        'alpha': controls['stability'].get('krippendorff_alpha'),
        'coverage': controls['stability'].get('complete_condition_coverage'),
    },
    {
        'control': 'Prompt variants',
        'conditions': controls['prompt'].get('observed_value_count'),
        'agreement': controls['prompt'].get('raw_agreement'),
        'flip_rate': controls['prompt'].get('flip_rate'),
        'alpha': controls['prompt'].get('krippendorff_alpha'),
        'coverage': controls['prompt'].get('complete_condition_coverage'),
    },
    {
        'control': 'Identity conditions',
        'conditions': controls['identity'].get('observed_value_count'),
        'agreement': controls['identity'].get('raw_agreement'),
        'flip_rate': controls['identity'].get('flip_rate'),
        'alpha': controls['identity'].get('krippendorff_alpha'),
        'coverage': controls['identity'].get('complete_condition_coverage'),
    },
    {
        'control': 'Response order',
        'conditions': controls['position'].get('observed_value_count'),
        'agreement': controls['position'].get('raw_agreement'),
        'flip_rate': controls['position'].get('flip_rate'),
        'alpha': controls['position'].get('krippendorff_alpha'),
        'coverage': controls['position'].get('complete_condition_coverage'),
    },
], [('control', 'Control'), ('conditions', 'Conditions'), ('coverage', 'Full-case coverage'), ('agreement', 'Agreement'), ('flip_rate', 'Flip rate'), ('alpha', 'Alpha')])}

Repeatability does not establish validity. Position, identity, and prompt
effects are reported separately because a judge can be highly stable and
systematically biased.

## Judge variants

{_markdown_table(report['judge_variants'], [('judge_variant_id', 'Variant'), ('comparable_cases', 'Cases'), ('raw_agreement', 'Raw agreement'), ('cohen_kappa', 'Kappa'), ('failure_recall', 'Failure recall'), ('false_approval_rate', 'False approval')])}

## Required subgroup results

{_markdown_table(report['subgroups'], [('slice_column', 'Slice'), ('slice_value', 'Value'), ('cases', 'Cases'), ('raw_agreement', 'Agreement'), ('cohen_kappa', 'Kappa'), ('failure_recall', 'Failure recall'), ('false_approval_rate', 'False approval')])}

## Protocol gates

{_markdown_table(gates, [('required_for_level', 'Level'), ('gate_id', 'Gate'), ('observed', 'Observed'), ('comparator', 'Rule'), ('threshold', 'Threshold'), ('passed', 'Passed')])}

## Unmet higher-level requirements

{_markdown_table(blockers, [('required_for_level', 'Level'), ('gate_id', 'Gate'), ('note', 'Why it matters')])}

## Evidence files

- Protocol: `{paths['protocol']}`
- Cases: `{paths['cases']}`
- Human annotations: `{paths['human_annotations']}`
- Judge observations: `{paths['judge_observations']}`
- Machine-readable report: `{paths['report_json']}`
- Gate results: `{paths['gates']}`
- Disagreements: `{paths['disagreements']}`
- Subgroups: `{paths['subgroups']}`

## Interpretation limits

{chr(10).join(f'- {item}' for item in report['limitations'])}

The report establishes measured agreement and control behavior under this
protocol. It does not prove that the judge is correct outside this scope, that
the human reference is infallible, or that a model comparison using the judge
is automatically decision-grade or publishable.
"""


def validate_judge_protocol(
    protocol_path: str | Path,
    *,
    root_path: str | Path = ".",
    output_dir: str | Path | None = None,
) -> JudgeValidationArtifacts:
    protocol_path = Path(protocol_path)
    root = Path(root_path)
    protocol = load_judge_protocol(protocol_path)
    cases, humans, observations, source_paths = _load_inputs(
        protocol,
        root=root,
    )

    comparison = _comparison_frame(cases, humans, observations, protocol)
    split_column = protocol.data.split_column
    calibration_frame = comparison[
        comparison[split_column] == protocol.data.calibration_split
    ]
    validation_frame = comparison[
        comparison[split_column] == protocol.data.validation_split
    ]
    if calibration_frame.empty:
        raise ValueError("The declared calibration split has no cases")
    if validation_frame.empty:
        raise ValueError("The declared validation split has no cases")

    human_calibration = _human_metrics(
        cases,
        humans,
        protocol,
        split=protocol.data.calibration_split,
    )
    human_validation = _human_metrics(
        cases,
        humans,
        protocol,
        split=protocol.data.validation_split,
    )
    judge_calibration = _agreement_metrics(calibration_frame, protocol)
    judge_validation = _agreement_metrics(validation_frame, protocol)
    validation_case_ids = set(
        validation_frame["case_id"].astype(str).tolist()
    )
    validation_observations = observations[
        observations["case_id"].isin(validation_case_ids)
    ]
    control_frames = _control_frames(validation_observations, protocol)
    stability = _condition_metrics(
        control_frames["stability"],
        condition_column="trial_id",
        expected_case_ids=validation_case_ids,
    )
    prompt = _condition_metrics(
        control_frames["prompt"],
        condition_column="prompt_variant_id",
        expected_case_ids=validation_case_ids,
    )
    identity = _condition_metrics(
        control_frames["identity"],
        condition_column="identity_condition",
        expected_case_ids=validation_case_ids,
    )
    position = _position_metrics(
        control_frames["position"],
        protocol,
        expected_case_ids=validation_case_ids,
    )
    self_preference = _self_preference_metrics(
        cases,
        control_frames["identity"],
    )
    variants = _variant_metrics(cases, humans, observations, protocol)
    criteria = _criterion_metrics(cases, humans, observations, protocol)
    subgroups = _subgroup_metrics(comparison, protocol)
    gates = _build_gates(
        protocol=protocol,
        human_validation=human_validation,
        judge_validation=judge_validation,
        stability=stability,
        prompt=prompt,
        identity=identity,
        self_preference=self_preference,
        position=position,
        subgroups=subgroups,
        variants=variants,
    )
    achieved_level = achieved_level_from_gates(
        gates,
        synthetic_fixture=protocol.synthetic_fixture,
    )
    target_met = level_at_least(achieved_level, protocol.target_level)

    safe_id = re.sub(r"[^a-zA-Z0-9_.-]+", "_", protocol.protocol_id)
    safe_version = re.sub(r"[^a-zA-Z0-9_.-]+", "_", protocol.version)
    output = (
        Path(output_dir)
        if output_dir is not None
        else root / "results" / "judge_validation" / safe_id / safe_version
    )
    output.mkdir(parents=True, exist_ok=True)
    stem = f"{safe_id}_{safe_version}"
    report_json_path = output / f"{stem}_validation.json"
    report_markdown_path = output / f"{stem}_validation.md"
    gates_path = output / f"{stem}_gates.csv"
    disagreements_path = output / f"{stem}_disagreements.csv"
    subgroups_path = output / f"{stem}_subgroups.csv"

    disagreements = validation_frame[
        validation_frame["reference_label"].notna()
        & (
            (validation_frame["status"] != "success")
            | (
                validation_frame["reference_label"]
                != validation_frame["judge_label"]
            )
        )
    ].copy()
    disagreement_columns = [
        "case_id",
        split_column,
        *protocol.data.slice_columns,
        "criterion",
        "reference_label",
        "judge_label",
        "status",
        "uncertainty",
    ]
    disagreements[disagreement_columns].to_csv(disagreements_path, index=False)
    pd.DataFrame(gates).to_csv(gates_path, index=False)
    pd.DataFrame(subgroups).to_csv(subgroups_path, index=False)

    report = _json_safe(
        {
            "schema_version": "1.0",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "evalanche_version": __version__,
            "protocol": {
                "protocol_id": protocol.protocol_id,
                "version": protocol.version,
                "title": protocol.title,
                "protocol_path": str(protocol_path),
                "protocol_sha256": _sha256_file(protocol_path),
                "contract_sha256": protocol_contract_sha256(protocol),
                "synthetic_fixture": protocol.synthetic_fixture,
            },
            "scope": {
                "task_name": protocol.contract.task.name,
                "measured_construct": (
                    protocol.contract.task.measured_construct
                ),
                "intended_use": protocol.contract.task.intended_use,
                "languages": protocol.contract.task.languages,
                "mode": protocol.contract.mode,
                "primary_judge_variant_id": (
                    protocol.controls.primary_judge_variant_id
                ),
                "judge_model": protocol.contract.judge_model,
                "provider_model_version": (
                    protocol.contract.provider_model_version
                ),
                "prompt_id": protocol.contract.prompt_id,
                "prompt_version": protocol.contract.prompt_version,
                "rubric_id": protocol.contract.rubric_id,
                "rubric_version": protocol.contract.rubric_version,
                "reference_mode": protocol.contract.reference_mode,
            },
            "target_level": protocol.target_level,
            "achieved_level": achieved_level,
            "target_met": target_met,
            "level_meanings": {
                "exploratory": "Valid artifacts for development and failure discovery.",
                "calibrated": "Passed declared human-agreement and failure-detection gates on this task.",
                "decision_grade": "Also passed declared stability, bias, blinding, and subgroup controls.",
                "publishable": "Also passed the declared frozen-protocol and governance gates. External review is still required.",
            },
            "data": {
                "cases": int(len(cases)),
                "human_annotation_rows": int(len(humans)),
                "judge_observation_rows": int(len(observations)),
                "paths": {key: str(path) for key, path in source_paths.items()},
                "hashes": {
                    key: _sha256_file(path) for key, path in source_paths.items()
                },
            },
            "human_reference": {
                "calibration": human_calibration,
                "validation": human_validation,
            },
            "judge_human_agreement": {
                "calibration": judge_calibration,
                "validation": judge_validation,
            },
            "criterion_results": criteria,
            "judge_variants": variants,
            "controls": {
                "stability": stability,
                "prompt": prompt,
                "identity": identity,
                "self_preference": self_preference,
                "position": position,
            },
            "subgroups": subgroups,
            "gates": gates,
            "disagreement_cases": int(len(disagreements)),
            "research_basis": protocol.research_basis,
            "governance": protocol.governance.model_dump(mode="json"),
            "limitations": [
                *protocol.limitations,
                (
                    "Validation is task-, rubric-, prompt-, model-, version-, "
                    "language-, and configuration-specific."
                ),
                (
                    "Configured gate thresholds are stakeholder policy and "
                    "are not universal scientific constants."
                ),
                (
                    "Bootstrap intervals describe this sampled validation "
                    "set and do not cover provider drift or construct error."
                ),
                (
                    "Self-reported judge uncertainty is diagnostic and is not "
                    "assumed to be calibrated probability."
                ),
                (
                    "An achieved level describes this judge evidence package, "
                    "not the rigor of an entire model-selection decision."
                ),
            ],
            "artifacts": {
                "gates_path": str(gates_path),
                "gates_sha256": _sha256_file(gates_path),
                "disagreements_path": str(disagreements_path),
                "disagreements_sha256": _sha256_file(disagreements_path),
                "subgroups_path": str(subgroups_path),
                "subgroups_sha256": _sha256_file(subgroups_path),
            },
        }
    )
    report_json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    markdown_paths = {
        "protocol": protocol_path,
        **source_paths,
        "report_json": report_json_path,
        "gates": gates_path,
        "disagreements": disagreements_path,
        "subgroups": subgroups_path,
    }
    report_markdown_path.write_text(
        _build_markdown_report(report, paths=markdown_paths),
        encoding="utf-8",
    )

    return JudgeValidationArtifacts(
        report_json_path=report_json_path,
        report_markdown_path=report_markdown_path,
        gates_path=gates_path,
        disagreements_path=disagreements_path,
        subgroups_path=subgroups_path,
        achieved_level=achieved_level,
        target_level=protocol.target_level,
        target_met=target_met,
    )

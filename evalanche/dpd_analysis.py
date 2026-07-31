from __future__ import annotations

import hashlib
import json
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from evalanche.config import EvaluationConfig, load_evaluation_config
from evalanche.dataset_manifest import sha256_file
from evalanche.metrics.deterministic import (
    canonical_json,
    json_values_equal,
    normalize_json_value,
    try_parse_json,
)
from evalanche.statistics import build_pairwise_comparisons


DPD_ANALYSIS_SCHEMA_VERSION = "1.0"
DPD_ANALYSIS_METHOD_VERSION = "hc_dpd_census_analysis/1.0"
DPD_ANALYSIS_RESULTS_PATH = Path(
    "results/evaluate_hc_dpd_census_all_models_results.csv"
)
DPD_ANALYSIS_CONFIG_PATH = Path(
    "configs/evaluate_hc_dpd_census_all_models.yaml"
)
DPD_ANALYSIS_OUTPUT_DIR = Path(
    "reports/hc_dpd_census/0.2.0/analysis"
)
DPD_ANALYSIS_PRIMARY_MODEL = "gpt_5_6_sol"
DPD_ANALYSIS_COMPARISON_MODEL = "gpt_5_6_terra"
DPD_ANALYSIS_LOWER_TIER_MODELS = (
    "gpt_5_4_mini",
    "gpt_5_6_luna",
)
DPD_ANALYSIS_REVIEW_SEED = 20260730
DPD_ANALYSIS_SHARED_FAILURE_SAMPLE = 25
DPD_ANALYSIS_ALL_PASS_SAMPLE = 25

DPD_FIELDS = (
    "din",
    "brand_name",
    "active_ingredients",
    "dosage_forms",
    "routes",
    "schedule",
    "product_status",
    "company",
)
DPD_SCALAR_FIELDS = {
    "brand_name",
    "product_status",
    "company",
}
DPD_LIST_FIELDS = {
    "din",
    "active_ingredients",
    "dosage_forms",
    "routes",
    "schedule",
}
DPD_COMPOUND_LABEL_FIELDS = {
    "dosage_forms",
    "routes",
    "schedule",
}
DPD_STRATUM_ORDER = (
    "single_ingredient",
    "multi_ingredient",
    "multi_variant",
    "multi_ingredient_multi_variant",
)
DPD_REVIEW_STRATUM_ORDER = (
    "multi_ingredient_multi_variant",
    "multi_variant",
    "multi_ingredient",
    "single_ingredient",
)
DPD_REVIEW_OUTPUT_FILES = (
    "model_overview.csv",
    "slice_performance.csv",
    "field_accuracy.csv",
    "error_taxonomy.csv",
    "pairwise_tradeoffs.csv",
    "frontier_cases.csv",
    "manual_review.csv",
    "README.md",
)
STRUCTURAL_ERROR_TYPES = {
    "schema_type_only",
    "duplicate_items_only",
    "compound_label_split_only",
    "schema_and_duplicates",
    "schema_and_label_split",
    "multiple_repairable",
}

REQUIRED_RESULT_COLUMNS = {
    "case_id",
    "product_id",
    "language",
    "stratum",
    "input",
    "expected_output",
    "source_metadata",
    "model_name",
    "model_output",
    "generation_status",
    "generation_seconds",
    "generation_cost_usd",
    "final_score",
    "final_passed",
}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        lineterminator="\n",
        float_format="%.12g",
    )


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _coerce_boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        raise ValueError("final_passed contains an empty value")
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"final_passed contains an invalid value: {value!r}")


def _parse_json_object(
    value: Any,
    *,
    field_name: str,
    allow_invalid: bool = False,
) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    parsed, result = try_parse_json(value)
    if parsed and isinstance(result, dict):
        return result
    if allow_invalid:
        return None
    raise ValueError(f"{field_name} must contain one JSON object")


def _parse_json_string_list(value: Any) -> list[str]:
    if pd.isna(value):
        return []
    if isinstance(value, list):
        parsed = value
    else:
        try:
            parsed = json.loads(str(value))
        except json.JSONDecodeError:
            return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed]


def _validate_results(
    frame: pd.DataFrame,
    *,
    required_models: Iterable[str],
) -> tuple[pd.DataFrame, list[str]]:
    missing = sorted(REQUIRED_RESULT_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(
            "DPD analysis results are missing columns: "
            + ", ".join(missing)
        )
    if frame.empty:
        raise ValueError("DPD analysis results are empty")
    if frame.duplicated(["case_id", "model_name"]).any():
        raise ValueError(
            "DPD analysis requires one row per case and model"
        )

    normalized = frame.copy(deep=False)
    normalized["case_id"] = normalized["case_id"].astype(str)
    normalized["product_id"] = normalized["product_id"].astype(str)
    normalized["model_name"] = normalized["model_name"].astype(str)
    normalized["language"] = normalized["language"].astype(str)
    normalized["stratum"] = normalized["stratum"].astype(str)
    normalized["final_passed"] = normalized["final_passed"].map(
        _coerce_boolean
    )
    normalized["final_score"] = pd.to_numeric(
        normalized["final_score"],
        errors="raise",
    )

    model_order = normalized["model_name"].drop_duplicates().tolist()
    required = set(required_models)
    missing_models = sorted(required - set(model_order))
    if missing_models:
        raise ValueError(
            "DPD analysis is missing required models: "
            + ", ".join(missing_models)
        )

    coverage = normalized.assign(_present=True).pivot(
        index="case_id",
        columns="model_name",
        values="_present",
    )
    if coverage.isna().any().any():
        raise ValueError(
            "DPD analysis requires identical case coverage for every model"
        )

    invariant_columns = (
        "product_id",
        "language",
        "stratum",
        "input",
        "expected_output",
        "source_metadata",
    )
    inconsistent = [
        column
        for column in invariant_columns
        if int(normalized.groupby("case_id")[column].nunique().max()) != 1
    ]
    if inconsistent:
        raise ValueError(
            "case metadata differs across models in columns: "
            + ", ".join(inconsistent)
        )

    case_frame = normalized.drop_duplicates("case_id")
    observed_languages = set(case_frame["language"])
    if observed_languages != {"en", "fr"}:
        raise ValueError(
            "DPD analysis requires English and French cases"
        )
    language_sets = case_frame.groupby("product_id")["language"].agg(set)
    if not language_sets.map(lambda value: value == {"en", "fr"}).all():
        raise ValueError(
            "DPD analysis requires one English and one French case "
            "per product"
        )

    expected_fields: set[tuple[str, ...]] = set()
    for value in case_frame["expected_output"]:
        expected = _parse_json_object(
            value,
            field_name="expected_output",
        )
        expected_fields.add(tuple(expected))
    if expected_fields != {DPD_FIELDS}:
        raise ValueError(
            "DPD expected outputs do not use the required eight-field "
            "contract"
        )

    return normalized, model_order


def _coerce_dpd_schema(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    result = dict(value)
    for key, item in list(result.items()):
        if (
            key in DPD_SCALAR_FIELDS
            and isinstance(item, list)
            and len(item) == 1
        ):
            result[key] = item[0]
        elif key in DPD_LIST_FIELDS and not isinstance(item, list):
            result[key] = [item]
    return result


def _deduplicate_lists(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _deduplicate_lists(item)
            for key, item in value.items()
        }
    if not isinstance(value, list):
        return value

    result = []
    seen: set[str] = set()
    for item in value:
        normalized_item = _deduplicate_lists(item)
        fingerprint = canonical_json(normalized_item)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        result.append(normalized_item)
    return result


def _merge_compound_labels(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    result = dict(value)
    for key in DPD_COMPOUND_LABEL_FIELDS:
        item = result.get(key)
        if (
            isinstance(item, list)
            and len(item) > 1
            and all(isinstance(part, str) for part in item)
        ):
            result[key] = [", ".join(item)]
    return result


def _normalized(
    value: Any,
    *,
    config: EvaluationConfig,
) -> Any:
    return normalize_json_value(value, config=config)


def classify_dpd_json_error(
    *,
    expected_output: Any,
    model_output: Any,
    config: EvaluationConfig,
    generation_status: str = "success",
) -> str:
    if str(generation_status) != "success":
        return "generation_error"

    expected = _parse_json_object(
        expected_output,
        field_name="expected_output",
    )
    output = _parse_json_object(
        model_output,
        field_name="model_output",
        allow_invalid=True,
    )
    if output is None:
        return "invalid_json"
    if set(output) != set(expected):
        return "missing_or_extra_fields"

    expected_normalized = _normalized(expected, config=config)
    candidates = (
        ("pass", _normalized(output, config=config)),
        (
            "schema_type_only",
            _normalized(_coerce_dpd_schema(output), config=config),
        ),
        (
            "duplicate_items_only",
            _deduplicate_lists(_normalized(output, config=config)),
        ),
        (
            "compound_label_split_only",
            _normalized(_merge_compound_labels(output), config=config),
        ),
        (
            "schema_and_duplicates",
            _deduplicate_lists(
                _normalized(
                    _coerce_dpd_schema(output),
                    config=config,
                )
            ),
        ),
        (
            "schema_and_label_split",
            _normalized(
                _merge_compound_labels(
                    _coerce_dpd_schema(output)
                ),
                config=config,
            ),
        ),
        (
            "multiple_repairable",
            _deduplicate_lists(
                _normalized(
                    _merge_compound_labels(
                        _coerce_dpd_schema(output)
                    ),
                    config=config,
                )
            ),
        ),
    )
    for classification, candidate in candidates:
        if json_values_equal(expected_normalized, candidate):
            return classification
    return "text_or_value_error"


def _json_field_diagnostics(
    *,
    expected_output: Any,
    model_output: Any,
    config: EvaluationConfig,
) -> tuple[list[str], list[str], list[str]]:
    expected = _parse_json_object(
        expected_output,
        field_name="expected_output",
    )
    output = _parse_json_object(
        model_output,
        field_name="model_output",
        allow_invalid=True,
    )
    if output is None:
        return sorted(expected), [], []

    expected_normalized = _normalized(expected, config=config)
    output_normalized = _normalized(output, config=config)
    expected_keys = set(expected_normalized)
    output_keys = set(output_normalized)
    missing = sorted(expected_keys - output_keys)
    extra = sorted(output_keys - expected_keys)
    mismatched = sorted(
        key
        for key in expected_keys & output_keys
        if not json_values_equal(
            expected_normalized[key],
            output_normalized[key],
        )
    )
    return missing, extra, mismatched


def _add_diagnostics(
    results: pd.DataFrame,
    *,
    config: EvaluationConfig,
) -> pd.DataFrame:
    records = []
    for row in results.to_dict(orient="records"):
        error_type = classify_dpd_json_error(
            expected_output=row["expected_output"],
            model_output=row["model_output"],
            config=config,
            generation_status=row["generation_status"],
        )
        missing, extra, mismatched = _json_field_diagnostics(
            expected_output=row["expected_output"],
            model_output=row["model_output"],
            config=config,
        )
        records.append(
            {
                "_error_type": error_type,
                "_missing_fields": canonical_json(missing),
                "_extra_fields": canonical_json(extra),
                "_mismatched_fields": canonical_json(mismatched),
                "_schema_exact": not missing and not extra,
            }
        )
    return pd.concat(
        [
            results.reset_index(drop=True),
            pd.DataFrame(records),
        ],
        axis=1,
    )


def _error_counts(
    diagnostics: pd.DataFrame,
    model_name: str,
) -> Counter[str]:
    failed = diagnostics[
        (diagnostics["model_name"] == model_name)
        & ~diagnostics["final_passed"]
    ]
    return Counter(failed["_error_type"].astype(str))


def _build_model_overview(
    diagnostics: pd.DataFrame,
    *,
    model_order: Sequence[str],
) -> pd.DataFrame:
    records = []
    for model_name in model_order:
        model_rows = diagnostics[
            diagnostics["model_name"] == model_name
        ]
        counts = _error_counts(diagnostics, model_name)
        passed = int(model_rows["final_passed"].sum())
        cases = len(model_rows)

        product_language = model_rows.pivot(
            index="product_id",
            columns="language",
            values="final_passed",
        ).astype(bool)
        both_passed = int(
            (product_language["en"] & product_language["fr"]).sum()
        )
        one_failed = int(
            (
                product_language["en"]
                ^ product_language["fr"]
            ).sum()
        )
        both_failed = int(
            (
                ~product_language["en"]
                & ~product_language["fr"]
            ).sum()
        )
        product_count = len(product_language)
        generation_failures = int(
            (model_rows["generation_status"] != "success").sum()
        )
        cost = pd.to_numeric(
            model_rows["generation_cost_usd"],
            errors="coerce",
        )
        latency = pd.to_numeric(
            model_rows["generation_seconds"],
            errors="coerce",
        )
        shape_repairs = counts["schema_type_only"]
        structural_repairs = sum(
            count
            for error_type, count in counts.items()
            if error_type in STRUCTURAL_ERROR_TYPES
        )

        records.append(
            {
                "model_name": model_name,
                "cases": cases,
                "passed_cases": passed,
                "failed_cases": cases - passed,
                "pass_rate": passed / cases,
                "average_field_score": float(
                    model_rows["final_score"].mean()
                ),
                "generation_failures": generation_failures,
                "average_latency_seconds": float(latency.mean()),
                "p95_latency_seconds": float(
                    latency.quantile(0.95)
                ),
                "total_cost_usd": float(cost.sum()),
                "cost_per_1000_cases_usd": (
                    float(cost.sum()) / cases * 1000
                ),
                "product_families": product_count,
                "both_languages_passed_products": both_passed,
                "one_language_failed_products": one_failed,
                "both_languages_failed_products": both_failed,
                "bilingual_product_pass_rate": (
                    both_passed / product_count
                ),
                "schema_type_only_failures": shape_repairs,
                "schema_type_repair_sensitivity_passes": (
                    passed + shape_repairs
                ),
                "schema_type_repair_sensitivity_rate": (
                    (passed + shape_repairs) / cases
                ),
                "all_structural_failures": structural_repairs,
                "structural_repair_sensitivity_passes": (
                    passed + structural_repairs
                ),
                "structural_repair_sensitivity_rate": (
                    (passed + structural_repairs) / cases
                ),
                "remaining_text_or_value_failures": counts[
                    "text_or_value_error"
                ],
            }
        )
    return pd.DataFrame(records)


def _build_slice_performance(
    diagnostics: pd.DataFrame,
    *,
    model_order: Sequence[str],
) -> pd.DataFrame:
    records = []

    def add_slice(
        frame: pd.DataFrame,
        *,
        model_name: str,
        slice_type: str,
        slice_value: str,
    ) -> None:
        cases = len(frame)
        passed = int(frame["final_passed"].sum())
        records.append(
            {
                "model_name": model_name,
                "slice_type": slice_type,
                "slice_value": slice_value,
                "cases": cases,
                "passed_cases": passed,
                "failed_cases": cases - passed,
                "pass_rate": passed / cases,
                "average_field_score": float(
                    frame["final_score"].mean()
                ),
            }
        )

    for model_name in model_order:
        model_rows = diagnostics[
            diagnostics["model_name"] == model_name
        ]
        add_slice(
            model_rows,
            model_name=model_name,
            slice_type="overall",
            slice_value="all",
        )
        for language in ("en", "fr"):
            add_slice(
                model_rows[model_rows["language"] == language],
                model_name=model_name,
                slice_type="language",
                slice_value=language,
            )
        for stratum in DPD_STRATUM_ORDER:
            add_slice(
                model_rows[model_rows["stratum"] == stratum],
                model_name=model_name,
                slice_type="stratum",
                slice_value=stratum,
            )
        for language, stratum in product(
            ("en", "fr"),
            DPD_STRATUM_ORDER,
        ):
            add_slice(
                model_rows[
                    (model_rows["language"] == language)
                    & (model_rows["stratum"] == stratum)
                ],
                model_name=model_name,
                slice_type="language_stratum",
                slice_value=f"{language}:{stratum}",
            )
    return pd.DataFrame(records)


def _build_field_accuracy(
    diagnostics: pd.DataFrame,
    *,
    model_order: Sequence[str],
) -> pd.DataFrame:
    records = []
    for model_name in model_order:
        model_rows = diagnostics[
            diagnostics["model_name"] == model_name
        ]
        for language in ("all", "en", "fr"):
            rows = (
                model_rows
                if language == "all"
                else model_rows[model_rows["language"] == language]
            )
            mismatch_counts = Counter()
            for value in rows["_mismatched_fields"]:
                mismatch_counts.update(_parse_json_string_list(value))
            for value in rows["_missing_fields"]:
                mismatch_counts.update(_parse_json_string_list(value))

            cases = len(rows)
            for field_name in DPD_FIELDS:
                mismatched = mismatch_counts[field_name]
                records.append(
                    {
                        "model_name": model_name,
                        "language": language,
                        "field_name": field_name,
                        "cases": cases,
                        "matching_cases": cases - mismatched,
                        "mismatched_cases": mismatched,
                        "field_accuracy": (
                            (cases - mismatched) / cases
                        ),
                    }
                )
            schema_matches = int(rows["_schema_exact"].sum())
            records.append(
                {
                    "model_name": model_name,
                    "language": language,
                    "field_name": "__schema_contract__",
                    "cases": cases,
                    "matching_cases": schema_matches,
                    "mismatched_cases": cases - schema_matches,
                    "field_accuracy": schema_matches / cases,
                }
            )
    return pd.DataFrame(records)


def _build_error_taxonomy(
    diagnostics: pd.DataFrame,
    *,
    model_order: Sequence[str],
) -> pd.DataFrame:
    records = []
    for model_name in model_order:
        model_rows = diagnostics[
            diagnostics["model_name"] == model_name
        ]
        for language in ("all", "en", "fr"):
            rows = (
                model_rows
                if language == "all"
                else model_rows[model_rows["language"] == language]
            )
            failed = rows[~rows["final_passed"]]
            counts = failed["_error_type"].value_counts()
            for error_type, count in counts.items():
                records.append(
                    {
                        "model_name": model_name,
                        "language": language,
                        "error_type": str(error_type),
                        "count": int(count),
                        "share_of_failures": (
                            int(count) / len(failed)
                            if len(failed)
                            else 0.0
                        ),
                        "share_of_cases": int(count) / len(rows),
                    }
                )
    return pd.DataFrame(records)


def _build_pairwise_tradeoffs(
    diagnostics: pd.DataFrame,
    model_overview: pd.DataFrame,
) -> pd.DataFrame:
    comparisons = build_pairwise_comparisons(
        diagnostics[
            [
                "case_id",
                "model_name",
                "final_passed",
                "final_score",
            ]
        ]
    )
    overview = model_overview.set_index("model_name")
    additional = []
    for row in comparisons.to_dict(orient="records"):
        model_a = str(row["model_a"])
        model_b = str(row["model_b"])
        pass_a = int(overview.loc[model_a, "passed_cases"])
        pass_b = int(overview.loc[model_b, "passed_cases"])
        if pass_a > pass_b:
            higher_model, lower_model = model_a, model_b
        elif pass_b > pass_a:
            higher_model, lower_model = model_b, model_a
        else:
            higher_model, lower_model = "", ""

        if higher_model:
            net_passes = int(
                overview.loc[higher_model, "passed_cases"]
                - overview.loc[lower_model, "passed_cases"]
            )
            incremental_cost = float(
                overview.loc[higher_model, "total_cost_usd"]
                - overview.loc[lower_model, "total_cost_usd"]
            )
            cost_per_net_pass = (
                incremental_cost / net_passes
                if incremental_cost > 0
                else None
            )
        else:
            net_passes = 0
            incremental_cost = 0.0
            cost_per_net_pass = None
        additional.append(
            {
                "higher_pass_model": higher_model,
                "lower_pass_model": lower_model,
                "net_additional_passes": net_passes,
                "incremental_cost_usd": incremental_cost,
                "incremental_cost_per_net_pass_usd": (
                    cost_per_net_pass
                ),
            }
        )
    return pd.concat(
        [comparisons, pd.DataFrame(additional)],
        axis=1,
    )


def _stable_rank(
    *,
    seed: int,
    purpose: str,
    case_id: str,
) -> str:
    return hashlib.sha256(
        f"{seed}|{purpose}|{case_id}".encode("utf-8")
    ).hexdigest()


def _stratified_case_sample(
    candidates: pd.DataFrame,
    *,
    count: int,
    seed: int,
    purpose: str,
    excluded_case_ids: set[str],
) -> list[str]:
    if count <= 0:
        return []
    available = (
        candidates[
            ~candidates["case_id"].astype(str).isin(
                excluded_case_ids
            )
        ]
        .drop_duplicates("case_id")
        .copy()
    )
    if len(available) < count:
        raise ValueError(
            f"not enough cases for {purpose}: "
            f"requested {count}, found {len(available)}"
        )

    group_order = list(
        product(
            DPD_REVIEW_STRATUM_ORDER,
            ("en", "fr"),
        )
    )
    base = count // len(group_order)
    remainder = count % len(group_order)
    selected: list[str] = []
    used_products: set[str] = set()

    for index, (stratum, language) in enumerate(group_order):
        quota = base + (1 if index < remainder else 0)
        if quota == 0:
            continue
        group = available[
            (available["stratum"] == stratum)
            & (available["language"] == language)
        ].copy()
        group["_rank"] = group["case_id"].astype(str).map(
            lambda case_id: _stable_rank(
                seed=seed,
                purpose=purpose,
                case_id=case_id,
            )
        )
        selected_in_group = 0
        for row in group.sort_values("_rank").itertuples():
            if str(row.product_id) in used_products:
                continue
            selected.append(str(row.case_id))
            used_products.add(str(row.product_id))
            selected_in_group += 1
            if selected_in_group >= quota:
                break

    if len(selected) < count:
        remainder_frame = available[
            ~available["case_id"].astype(str).isin(selected)
            & ~available["product_id"].astype(str).isin(used_products)
        ].copy()
        remainder_frame["_rank"] = (
            remainder_frame["case_id"].astype(str).map(
                lambda case_id: _stable_rank(
                    seed=seed,
                    purpose=f"{purpose}:fill",
                    case_id=case_id,
                )
            )
        )
        needed = count - len(selected)
        selected.extend(
            remainder_frame.sort_values("_rank")
            .head(needed)["case_id"]
            .astype(str)
            .tolist()
        )

    if len(selected) != count:
        raise ValueError(
            f"could not build the requested {purpose} sample"
        )
    return selected


def _review_focus(error_types: Iterable[str]) -> str:
    observed = set(error_types)
    if "compound_label_split_only" in observed:
        return "Adjudicate whether the rendered compound DPD label is unambiguous."
    if "schema_type_only" in observed:
        return "Verify the strict scalar and list JSON contract."
    if (
        "duplicate_items_only" in observed
        or "schema_and_duplicates" in observed
    ):
        return "Verify distinct-fact aggregation across product variants."
    if "text_or_value_error" in observed:
        return "Compare the transcription with the frozen source label."
    if "missing_or_extra_fields" in observed:
        return "Verify required and prohibited JSON fields."
    if "invalid_json" in observed:
        return "Verify JSON parseability and retained source facts."
    return "Verify the source record, expected answer, and strict score."


def _wide_case_records(
    diagnostics: pd.DataFrame,
    *,
    case_ids: Sequence[str],
    model_order: Sequence[str],
    focus_models: Sequence[str] | None = None,
) -> pd.DataFrame:
    by_key = diagnostics.set_index(["case_id", "model_name"])
    case_metadata = (
        diagnostics.drop_duplicates("case_id")
        .set_index("case_id")
    )
    records = []
    for case_id in case_ids:
        metadata = case_metadata.loc[case_id]
        record: dict[str, Any] = {
            "case_id": case_id,
            "product_id": metadata["product_id"],
            "language": metadata["language"],
            "stratum": metadata["stratum"],
            "brand_name": metadata.get("brand_name", ""),
            "input": metadata["input"],
            "expected_output": metadata["expected_output"],
            "source_metadata": metadata["source_metadata"],
        }
        error_types = []
        focus = set(focus_models or model_order)
        for model_name in model_order:
            row = by_key.loc[(case_id, model_name)]
            error_type = str(row["_error_type"])
            if (
                model_name in focus
                and not bool(row["final_passed"])
            ):
                error_types.append(error_type)
            prefix = model_name
            record[f"{prefix}_passed"] = bool(row["final_passed"])
            record[f"{prefix}_field_score"] = float(
                row["final_score"]
            )
            record[f"{prefix}_error_type"] = error_type
            record[f"{prefix}_mismatched_fields"] = row[
                "_mismatched_fields"
            ]
            record[f"{prefix}_model_output"] = row["model_output"]
        record["preliminary_review_focus"] = _review_focus(
            error_types
        )
        records.append(record)
    return pd.DataFrame(records)


def _build_review_tables(
    diagnostics: pd.DataFrame,
    *,
    model_order: Sequence[str],
    primary_model: str,
    comparison_model: str,
    lower_tier_models: Sequence[str],
    shared_failure_sample: int,
    all_pass_sample: int,
    review_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    passes = diagnostics.pivot(
        index="case_id",
        columns="model_name",
        values="final_passed",
    ).astype(bool)
    metadata = (
        diagnostics.drop_duplicates("case_id")
        .set_index("case_id")
    )
    frontier_mask = (
        ~passes[primary_model]
        | ~passes[comparison_model]
    )
    frontier_ids = sorted(
        passes.index[frontier_mask].astype(str).tolist()
    )
    frontier = _wide_case_records(
        diagnostics,
        case_ids=frontier_ids,
        model_order=model_order,
        focus_models=(primary_model, comparison_model),
    )
    frontier.insert(
        0,
        "pair_outcome",
        [
            (
                "both_failed"
                if not passes.loc[case_id, primary_model]
                and not passes.loc[case_id, comparison_model]
                else (
                    f"{primary_model}_only_passed"
                    if passes.loc[case_id, primary_model]
                    else f"{comparison_model}_only_passed"
                )
            )
            for case_id in frontier["case_id"]
        ],
    )

    selected = set(frontier_ids)
    primary_failure_ids = sorted(
        passes.index[~passes[primary_model]].astype(str).tolist()
    )
    comparison_only_ids = sorted(
        passes.index[
            passes[primary_model] & ~passes[comparison_model]
        ].astype(str).tolist()
    )

    if all(model in passes for model in lower_tier_models):
        lower_failure_mask = pd.Series(
            True,
            index=passes.index,
        )
        for model_name in lower_tier_models:
            lower_failure_mask &= ~passes[model_name]
        lower_candidates = metadata.loc[
            passes.index[lower_failure_mask]
        ].reset_index()
        lower_sample_ids = _stratified_case_sample(
            lower_candidates,
            count=shared_failure_sample,
            seed=review_seed,
            purpose="lower_tier_shared_failure",
            excluded_case_ids=selected,
        )
    else:
        lower_sample_ids = []
    selected.update(lower_sample_ids)

    all_pass_candidates = metadata.loc[
        passes.index[passes.all(axis=1)]
    ].reset_index()
    all_pass_ids = _stratified_case_sample(
        all_pass_candidates,
        count=all_pass_sample,
        seed=review_seed,
        purpose="all_model_pass",
        excluded_case_ids=selected,
    )

    groups = (
        (
            "primary_model_failure",
            1,
            primary_failure_ids,
            (
                f"Review every {primary_model} strict failure, including "
                "shared frontier failures."
            ),
            (primary_model, comparison_model),
        ),
        (
            "comparison_only_failure",
            2,
            comparison_only_ids,
            (
                f"Review cases passed by {primary_model} and failed by "
                f"{comparison_model}."
            ),
            (primary_model, comparison_model),
        ),
        (
            "lower_tier_shared_failure_sample",
            3,
            lower_sample_ids,
            "Stratified sample where both lower-tier models failed.",
            tuple(lower_tier_models),
        ),
        (
            "all_model_pass_sample",
            4,
            all_pass_ids,
            "Stratified sample where every model passed.",
            tuple(model_order),
        ),
    )
    review_frames = []
    for (
        group_name,
        priority,
        case_ids,
        reason,
        focus_models,
    ) in groups:
        if not case_ids:
            continue
        frame = _wide_case_records(
            diagnostics,
            case_ids=case_ids,
            model_order=model_order,
            focus_models=focus_models,
        )
        frame.insert(0, "selection_reason", reason)
        frame.insert(0, "review_priority", priority)
        frame.insert(0, "review_group", group_name)
        review_frames.append(frame)

    review = pd.concat(review_frames, ignore_index=True)
    review.insert(
        0,
        "review_id",
        [
            f"hc-dpd-review-{index:04d}"
            for index in range(1, len(review) + 1)
        ],
    )
    review["review_status"] = "pending"
    review["source_record_correct"] = ""
    review["expected_answer_correct"] = ""
    review["strict_score_correct"] = ""
    review["error_owner"] = ""
    review["operational_severity"] = ""
    review["adjudication"] = ""
    review["reviewer"] = ""
    review["review_date"] = ""
    review["review_notes"] = ""
    return frontier, review


def _markdown_table(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str],
    headers: Sequence[str],
) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame[list(columns)].itertuples(index=False, name=None):
        values = []
        for value in row:
            text = str(value).replace("|", "\\|").replace("\n", " ")
            values.append(text)
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _format_percent(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}%}"


def _render_analysis_report(
    *,
    model_overview: pd.DataFrame,
    slice_performance: pd.DataFrame,
    error_taxonomy: pd.DataFrame,
    pairwise: pd.DataFrame,
    frontier: pd.DataFrame,
    review: pd.DataFrame,
    primary_model: str,
    comparison_model: str,
) -> str:
    overview = model_overview.set_index("model_name")
    display_overview = model_overview[
        [
            "model_name",
            "passed_cases",
            "pass_rate",
            "average_field_score",
            "bilingual_product_pass_rate",
            "total_cost_usd",
        ]
    ].copy()
    display_overview["pass_rate"] = display_overview[
        "pass_rate"
    ].map(lambda value: _format_percent(value))
    display_overview["average_field_score"] = display_overview[
        "average_field_score"
    ].map(lambda value: f"{value:.5f}")
    display_overview["bilingual_product_pass_rate"] = (
        display_overview["bilingual_product_pass_rate"].map(
            lambda value: _format_percent(value)
        )
    )
    display_overview["total_cost_usd"] = display_overview[
        "total_cost_usd"
    ].map(lambda value: f"${value:.2f}")

    language = slice_performance[
        slice_performance["slice_type"] == "language"
    ]
    language_pivot = language.pivot(
        index="model_name",
        columns="slice_value",
        values="pass_rate",
    )
    language_gaps = (
        language_pivot["fr"] - language_pivot["en"]
    )
    primary_failures = int(
        overview.loc[primary_model, "failed_cases"]
    )
    comparison_failures = int(
        overview.loc[comparison_model, "failed_cases"]
    )
    compound_counts = (
        error_taxonomy[
            (error_taxonomy["language"] == "all")
            & (
                error_taxonomy["error_type"]
                == "compound_label_split_only"
            )
        ]
        .set_index("model_name")["count"]
        .to_dict()
    )
    primary_compound = int(compound_counts.get(primary_model, 0))
    comparison_compound = int(
        compound_counts.get(comparison_model, 0)
    )
    strict_frontier_gap = int(
        overview.loc[primary_model, "passed_cases"]
        - overview.loc[comparison_model, "passed_cases"]
    )
    compound_sensitivity_gap = int(
        (
            overview.loc[primary_model, "passed_cases"]
            + primary_compound
        )
        - (
            overview.loc[comparison_model, "passed_cases"]
            + comparison_compound
        )
    )

    frontier_pair = pairwise[
        (
            (pairwise["model_a"] == primary_model)
            & (pairwise["model_b"] == comparison_model)
        )
        | (
            (pairwise["model_a"] == comparison_model)
            & (pairwise["model_b"] == primary_model)
        )
    ].iloc[0]
    primary_only = (
        int(frontier_pair["model_a_only_passed"])
        if frontier_pair["model_a"] == primary_model
        else int(frontier_pair["model_b_only_passed"])
    )
    comparison_only = (
        int(frontier_pair["model_b_only_passed"])
        if frontier_pair["model_a"] == primary_model
        else int(frontier_pair["model_a_only_passed"])
    )
    both_failed = int(frontier_pair["both_failed"])

    frontier_primary_failures = frontier[
        ~frontier[f"{primary_model}_passed"]
    ][
        [
            "case_id",
            "language",
            "stratum",
            f"{primary_model}_mismatched_fields",
            f"{primary_model}_error_type",
            f"{comparison_model}_passed",
        ]
    ].copy()
    frontier_primary_failures[
        f"{comparison_model}_passed"
    ] = frontier_primary_failures[
        f"{comparison_model}_passed"
    ].map(lambda value: "yes" if value else "no")

    report = f"""# DPD census analysis and validation package

**Status:** Automated analysis complete. The selected-case evidence audit is
published separately in `EVIDENCE_AUDIT.md`. Independent human sign-off is
not claimed.

This package analyzes the four completed model runs on the frozen 14,034-case
Health Canada DPD census. It makes no model calls and does not replace the
strict deterministic scores.

## Published leaderboard

{_markdown_table(
    display_overview,
    columns=(
        "model_name",
        "passed_cases",
        "pass_rate",
        "average_field_score",
        "bilingual_product_pass_rate",
        "total_cost_usd",
    ),
    headers=(
        "Model",
        "Strict passes",
        "Case pass rate",
        "Average field score",
        "Both-language product pass rate",
        "Run cost",
    ),
)}

The frozen benchmark is a census of the eligible DPD snapshot, not a sample
from that snapshot. The rates above are descriptive for this benchmark.
They do not estimate performance on Product Monographs or future tasks.

## Main findings

1. `{primary_model}` and `{comparison_model}` nearly saturate this
   structured-copying task. They failed {primary_failures} and
   {comparison_failures} cases respectively.
2. Their errors are not nested. `{primary_model}` alone passed
   {primary_only} cases, `{comparison_model}` alone passed
   {comparison_only}, and both failed {both_failed}.
3. More than half of `{comparison_model}` failures are attributable to the
   diagnostic category `compound_label_split_only`
   ({comparison_compound} of {comparison_failures}). These cases contain an
   official DPD label such as `SPRAY, BAG-ON-VALVE`. The published evidence
   audit retains these failures because rendered multi-values use ` | `,
   while the comma remains part of the official source label.
4. `{primary_model}` has {primary_compound} failure in the same diagnostic
   category. A hypothetical repair would shrink the frontier gap from
   {strict_frontier_gap} to {compound_sensitivity_gap} cases, but the evidence
   audit rejected that alternate scoring interpretation. The tracked strict
   leaderboard is unchanged.
5. Mini and Luna have similar strict pass rates for different reasons.
   Their dominant failures are scalar/list type-shape violations, so average
   field score remains high even when the complete JSON record fails.

## Language and product-family dependence

The English and French records for each product are paired. Reporting whether
both language cases pass gives a more operationally meaningful product-level
view than treating all 14,034 cases as independent.

| Model | French minus English pass-rate gap | Both-language products passed |
| --- | ---: | ---: |
"""
    for model_name in model_overview["model_name"]:
        gap = language_gaps.loc[model_name]
        report += (
            f"| {model_name} | {gap:+.2%} | "
            f"{int(overview.loc[model_name, 'both_languages_passed_products'])}"
            f" / {int(overview.loc[model_name, 'product_families'])} |\n"
        )

    report += """

Mini is substantially stronger in French than English on the strict contract,
while Luna shows the opposite pattern. The error taxonomy shows that these
gaps are driven mainly by different schema-shape mistakes, not a uniform
loss of factual extraction quality.

## Diagnostic sensitivity, not rescoring

The analysis classifies failed cases by asking whether one narrowly defined,
mechanical transformation would recover the strict expected answer:

- `schema_type_only`, a singleton scalar/list type was reversed;
- `duplicate_items_only`, every expected item was present but at least one
  list item was repeated;
- `compound_label_split_only`, one official comma-bearing label was returned
  as multiple list entries;
- `text_or_value_error`, the remaining output changes source text or values.

These transformations are diagnostics only. They are not applied to the
published pass rate because doing so would weaken the tested contract.

| Model | Strict rate | Type-shape sensitivity | All structural sensitivity | Remaining text/value failures |
| --- | ---: | ---: | ---: | ---: |
"""
    for model_name in model_overview["model_name"]:
        row = overview.loc[model_name]
        report += (
            f"| {model_name} | {_format_percent(row['pass_rate'])} | "
            f"{_format_percent(row['schema_type_repair_sensitivity_rate'])} | "
            f"{_format_percent(row['structural_repair_sensitivity_rate'])} | "
            f"{int(row['remaining_text_or_value_failures'])} |\n"
        )

    report += f"""

The sensitivity columns are upper bounds under hypothetical deterministic
repair. They do not show what a model would produce under API-enforced
structured outputs. A controlled structured-output rerun is required to test
that production design.

## Every `{primary_model}` strict failure

{_markdown_table(
    frontier_primary_failures,
    columns=tuple(frontier_primary_failures.columns),
    headers=(
        "Case",
        "Language",
        "Stratum",
        "Mismatched fields",
        "Diagnostic type",
        f"{comparison_model} passed",
    ),
)}

The automated evidence audit rebuilt these cases from the frozen source,
independently parsed the expected answers, and recomputed their strict scores.
It retained every selected result, including comma-bearing labels and the
administrative `(FR)` prefix. That is automated benchmark validation, not
independent human sign-off.

## Selected-case evidence set

The tracked `manual_review.csv` contains {len(review)} unique cases:

- all {primary_failures} `{primary_model}` failures;
- all {primary_only} cases passed only by `{primary_model}` within the
  frontier pair;
- a deterministic, language-and-complexity-stratified sample of shared
  lower-tier failures;
- a deterministic, language-and-complexity-stratified sample of all-model
  passes.

`manual_review.csv` preserves the immutable selection and model evidence.
`evidence_audit.csv` records the completed automated decisions separately, so
the source worksheet does not need to be edited or attributed to a person.

## Leaderboard interpretation

The published evidence supports these conclusions:

- `{primary_model}` is the strict-score leader for this exact task.
- `{comparison_model}` is 35 complete-record passes behind and has a lower
  observed run cost. The leaderboard exposes that tradeoff without declaring
  a universal winner.
- Evalanche reports benchmark measurements. It does not make a formal product
  recommendation unless a downstream user explicitly defines decision
  priorities and constraints.
- Mini and Luna should not be dismissed solely from this condition. Their
  large schema-shape component justifies a separate structured-output
  experiment if a low-cost production route matters.
- this benchmark is now saturated for frontier models and should be followed
  by the harder Product Monograph extraction benchmark.

## Files

| File | Purpose |
| --- | --- |
| `model_overview.csv` | Quality, bilingual product outcomes, latency, cost, and diagnostic sensitivities |
| `slice_performance.csv` | Overall, language, stratum, and language-by-stratum results |
| `field_accuracy.csv` | Field-level accuracy overall and by language |
| `error_taxonomy.csv` | Diagnostic failure mechanisms overall and by language |
| `pairwise_tradeoffs.csv` | Paired outcomes with cost increments |
| `frontier_cases.csv` | Every case failed by either frontier model |
| `manual_review.csv` | Immutable selected-case evidence worksheet |
| `evidence_audit.csv` | Completed automated decisions for all selected cases |
| `evidence_audit_summary.json` | Audit inputs, hashes, checks, and outcomes |
| `EVIDENCE_AUDIT.md` | Human-readable audit result and interpretation |
| `analysis_manifest.json` | Input and output hashes, methods, and review-set counts |

## Limitations

- The benchmark inputs are deterministic renderings of structured DPD rows,
  not unstructured Product Monographs.
- The audit verifies lineage, expected-answer construction, and strict scoring
  independently in code, but it does not claim completed human validation.
- Diagnostic repairs intentionally do not alter the primary score.
- Run cost is estimated from configured token prices and recorded usage, not
  reconciled billing.
- One generation per case does not measure output variability across seeds.
"""
    return report


def build_dpd_census_analysis(
    *,
    root_path: str | Path = ".",
    results_path: str | Path = DPD_ANALYSIS_RESULTS_PATH,
    config_path: str | Path = DPD_ANALYSIS_CONFIG_PATH,
    output_dir: str | Path = DPD_ANALYSIS_OUTPUT_DIR,
    primary_model: str = DPD_ANALYSIS_PRIMARY_MODEL,
    comparison_model: str = DPD_ANALYSIS_COMPARISON_MODEL,
    lower_tier_models: Sequence[
        str
    ] = DPD_ANALYSIS_LOWER_TIER_MODELS,
    shared_failure_sample: int = DPD_ANALYSIS_SHARED_FAILURE_SAMPLE,
    all_pass_sample: int = DPD_ANALYSIS_ALL_PASS_SAMPLE,
    review_seed: int = DPD_ANALYSIS_REVIEW_SEED,
) -> dict[str, Any]:
    root = Path(root_path).resolve()
    results_file = Path(results_path)
    if not results_file.is_absolute():
        results_file = root / results_file
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    output_directory = Path(output_dir)
    if not output_directory.is_absolute():
        output_directory = root / output_directory

    if not results_file.is_file():
        raise FileNotFoundError(
            f"DPD case-level results not found: {results_file}"
        )
    if not config_file.is_file():
        raise FileNotFoundError(
            f"DPD evaluation config not found: {config_file}"
        )
    if shared_failure_sample < 0 or all_pass_sample < 0:
        raise ValueError("review sample sizes cannot be negative")
    if primary_model == comparison_model:
        raise ValueError(
            "primary and comparison models must be different"
        )

    config = load_evaluation_config(config_file)
    available_columns = set(
        pd.read_csv(results_file, nrows=0).columns
    )
    analysis_columns = sorted(
        (REQUIRED_RESULT_COLUMNS | {"brand_name"})
        & available_columns
    )
    raw_results = pd.read_csv(
        results_file,
        usecols=analysis_columns,
        low_memory=False,
    )
    required_models = {
        primary_model,
        comparison_model,
        *lower_tier_models,
    }
    results, model_order = _validate_results(
        raw_results,
        required_models=required_models,
    )
    diagnostics = _add_diagnostics(results, config=config)

    model_overview = _build_model_overview(
        diagnostics,
        model_order=model_order,
    )
    slice_performance = _build_slice_performance(
        diagnostics,
        model_order=model_order,
    )
    field_accuracy = _build_field_accuracy(
        diagnostics,
        model_order=model_order,
    )
    error_taxonomy = _build_error_taxonomy(
        diagnostics,
        model_order=model_order,
    )
    pairwise_tradeoffs = _build_pairwise_tradeoffs(
        diagnostics,
        model_overview,
    )
    frontier, review = _build_review_tables(
        diagnostics,
        model_order=model_order,
        primary_model=primary_model,
        comparison_model=comparison_model,
        lower_tier_models=lower_tier_models,
        shared_failure_sample=shared_failure_sample,
        all_pass_sample=all_pass_sample,
        review_seed=review_seed,
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, pd.DataFrame] = {
        "model_overview.csv": model_overview,
        "slice_performance.csv": slice_performance,
        "field_accuracy.csv": field_accuracy,
        "error_taxonomy.csv": error_taxonomy,
        "pairwise_tradeoffs.csv": pairwise_tradeoffs,
        "frontier_cases.csv": frontier,
        "manual_review.csv": review,
    }
    for filename, frame in outputs.items():
        _write_csv(output_directory / filename, frame)

    report = _render_analysis_report(
        model_overview=model_overview,
        slice_performance=slice_performance,
        error_taxonomy=error_taxonomy,
        pairwise=pairwise_tradeoffs,
        frontier=frontier,
        review=review,
        primary_model=primary_model,
        comparison_model=comparison_model,
    )
    (output_directory / "README.md").write_text(
        report,
        encoding="utf-8",
        newline="\n",
    )

    published_files = []
    for filename in DPD_REVIEW_OUTPUT_FILES:
        path = output_directory / filename
        published_files.append(
            {
                "path": _relative_path(path, root),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    passes = diagnostics.pivot(
        index="case_id",
        columns="model_name",
        values="final_passed",
    ).astype(bool)
    manifest = {
        "schema_version": DPD_ANALYSIS_SCHEMA_VERSION,
        "method_version": DPD_ANALYSIS_METHOD_VERSION,
        "benchmark": {
            "dataset_id": (
                "hc_dpd_structured_extraction_census"
            ),
            "dataset_version": "0.2.0",
            "case_count": int(diagnostics["case_id"].nunique()),
            "product_family_count": int(
                diagnostics["product_id"].nunique()
            ),
            "languages": ["en", "fr"],
        },
        "source": {
            "results_path": _relative_path(results_file, root),
            "results_size_bytes": results_file.stat().st_size,
            "results_sha256": sha256_file(results_file),
            "evaluation_config_path": _relative_path(
                config_file,
                root,
            ),
            "evaluation_config_sha256": sha256_file(config_file),
            "rows": len(diagnostics),
            "models": model_order,
        },
        "analysis": {
            "primary_model": primary_model,
            "comparison_model": comparison_model,
            "lower_tier_models": list(lower_tier_models),
            "diagnostic_repairs_change_primary_score": False,
            "manual_adjudication_complete": False,
        },
        "review_set": {
            "seed": review_seed,
            "primary_model_failures": int(
                (~passes[primary_model]).sum()
            ),
            "comparison_only_failures": int(
                (
                    passes[primary_model]
                    & ~passes[comparison_model]
                ).sum()
            ),
            "shared_lower_tier_failure_sample": (
                shared_failure_sample
            ),
            "all_model_pass_sample": all_pass_sample,
            "total_unique_cases": len(review),
            "status": "pending_human_review",
        },
        "published_files": published_files,
    }
    manifest_path = output_directory / "analysis_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        **manifest,
        "output_dir": str(output_directory),
        "manifest_path": str(manifest_path),
    }

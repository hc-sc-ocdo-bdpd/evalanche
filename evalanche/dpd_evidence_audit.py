from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from evalanche.config import load_evaluation_config
from evalanche.dataset_manifest import sha256_file
from evalanche.dpd_benchmark import (
    SelectedProduct,
    build_candidate_products,
    build_case_records,
    load_marketed_tables,
)


DPD_EVIDENCE_AUDIT_SCHEMA_VERSION = "1.0"
DPD_EVIDENCE_AUDIT_METHOD_VERSION = "hc_dpd_evidence_audit/1.0"
DPD_EVIDENCE_AUDIT_DATE = "2026-07-30"
DPD_EVIDENCE_AUDIT_CASES_PATH = Path(
    "reports/hc_dpd_census/0.2.0/analysis/selected_cases.csv"
)
DPD_EVIDENCE_AUDIT_CONFIG_PATH = Path(
    "configs/evaluate_hc_dpd_census_all_models.yaml"
)
DPD_EVIDENCE_AUDIT_ARCHIVE_PATH = Path(
    "data/hc/dpd/snapshots/2026-07-02/raw/allfiles.zip"
)
DPD_EVIDENCE_AUDIT_OUTPUT_PATH = Path(
    "reports/hc_dpd_census/0.2.0/analysis/evidence_audit.csv"
)
DPD_EVIDENCE_AUDIT_SUMMARY_PATH = Path(
    "reports/hc_dpd_census/0.2.0/analysis/"
    "evidence_audit_summary.json"
)
DPD_EVIDENCE_AUDIT_REPORT_PATH = Path(
    "reports/hc_dpd_census/0.2.0/analysis/EVIDENCE_AUDIT.md"
)
DPD_EVIDENCE_AUDIT_ANALYSIS_MANIFEST_PATH = Path(
    "reports/hc_dpd_census/0.2.0/analysis/analysis_manifest.json"
)
DPD_EVIDENCE_AUDIT_RELEASE_MANIFEST_PATH = Path(
    "reports/hc_dpd_census/0.2.0/manifest.json"
)

_ANALYSIS_RELEASE_ROLES = {
    "README.md": "analysis_report",
    "analysis_manifest.json": "analysis_integrity_manifest",
    "model_overview.csv": "analysis_model_overview",
    "slice_performance.csv": "analysis_slice_performance",
    "field_accuracy.csv": "analysis_field_accuracy",
    "error_taxonomy.csv": "analysis_error_taxonomy",
    "pairwise_tradeoffs.csv": "analysis_pairwise_tradeoffs",
    "frontier_cases.csv": "analysis_frontier_cases",
    "selected_cases.csv": "selected_audit_cases",
    "evidence_audit.csv": "automated_evidence_audit",
    "evidence_audit_summary.json": "evidence_audit_summary",
    "EVIDENCE_AUDIT.md": "evidence_audit_report",
}
_ANALYSIS_RELEASE_ORDER = (
    "README.md",
    "analysis_manifest.json",
    "model_overview.csv",
    "slice_performance.csv",
    "field_accuracy.csv",
    "error_taxonomy.csv",
    "pairwise_tradeoffs.csv",
    "frontier_cases.csv",
    "selected_cases.csv",
    "evidence_audit.csv",
    "evidence_audit_summary.json",
    "EVIDENCE_AUDIT.md",
)

_MODEL_SUFFIX = "_passed"
_VARIANT_PATTERNS = {
    "en": re.compile(r"^VARIANT \d+$"),
    "fr": re.compile(r"^VARIANTE \d+$"),
}
_INGREDIENT_PATTERNS = {
    "en": re.compile(
        r"^- NAME=(.*); STRENGTH=([^;]+); UNIT=(.*)$"
    ),
    "fr": re.compile(
        r"^- NOM=(.*); CONCENTRATION=([^;]+); UNITÉ=(.*)$"
    ),
}
_FIELD_PREFIXES = {
    "en": {
        "DIN: ": "din",
        "BRAND_NAME: ": "brand_name",
        "DOSAGE_FORMS: ": "dosage_forms",
        "ROUTES: ": "routes",
        "SCHEDULES: ": "schedule",
        "PRODUCT_STATUS: ": "product_status",
        "COMPANY: ": "company",
    },
    "fr": {
        "DIN: ": "din",
        "NOM_COMMERCIAL: ": "brand_name",
        "FORMES_PHARMACEUTIQUES: ": "dosage_forms",
        "VOIES_ADMINISTRATION: ": "routes",
        "ANNEXES: ": "schedule",
        "ÉTAT_DU_PRODUIT: ": "product_status",
        "ENTREPRISE: ": "company",
    },
}
_IGNORED_PREFIXES = {
    "en": ("DRUG_CODE: ", "ACTIVE_INGREDIENTS:"),
    "fr": ("DRUG_CODE: ", "INGRÉDIENTS_ACTIFS:"),
}
_LIST_FIELDS = {"dosage_forms", "routes", "schedule"}
_OPERATIONALLY_MATERIAL_FIELDS = {
    "active_ingredients",
    "din",
    "dosage_forms",
    "product_status",
    "routes",
    "schedule",
}
_REVIEW_GROUP_FOCUS = {
    "primary_model_failure": (
        "gpt_5_6_sol",
        "gpt_5_6_terra",
    ),
    "comparison_only_failure": (
        "gpt_5_6_sol",
        "gpt_5_6_terra",
    ),
    "lower_tier_shared_failure_sample": (
        "gpt_5_4_mini",
        "gpt_5_6_luna",
    ),
}


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _artifact_record(
    path: Path,
    *,
    root: Path,
    role: str | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"published audit artifact not found: {path}")
    record: dict[str, Any] = {
        "path": _relative_path(path, root),
    }
    if role is not None:
        record["role"] = role
    record.update(
        {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    )
    return record


def _refresh_release_manifests(
    *,
    root: Path,
    output_file: Path,
    summary_file: Path,
    summary: dict[str, Any],
) -> bool:
    default_output = _resolve(
        root,
        DPD_EVIDENCE_AUDIT_OUTPUT_PATH,
    ).resolve()
    default_summary = _resolve(
        root,
        DPD_EVIDENCE_AUDIT_SUMMARY_PATH,
    ).resolve()
    if (
        output_file.resolve() != default_output
        or summary_file.resolve() != default_summary
    ):
        return False

    analysis_manifest_path = _resolve(
        root,
        DPD_EVIDENCE_AUDIT_ANALYSIS_MANIFEST_PATH,
    )
    release_manifest_path = _resolve(
        root,
        DPD_EVIDENCE_AUDIT_RELEASE_MANIFEST_PATH,
    )
    audit_report_path = _resolve(
        root,
        DPD_EVIDENCE_AUDIT_REPORT_PATH,
    )
    for required in (
        analysis_manifest_path,
        release_manifest_path,
        audit_report_path,
    ):
        if not required.is_file():
            raise FileNotFoundError(
                f"release manifest input not found: {required}"
            )

    analysis_manifest = _read_json_object(analysis_manifest_path)
    analysis_manifest.setdefault("analysis", {}).update(
        {
            "automated_evidence_audit_complete": True,
            "evidence_audit_method_version": (
                DPD_EVIDENCE_AUDIT_METHOD_VERSION
            ),
            "leaderboard_scores_changed_by_audit": False,
            "human_signoff_status": "not_claimed",
        }
    )
    analysis_manifest.setdefault("audit_set", {}).update(
        {"status": "automated_evidence_audit_complete"}
    )
    analysis_manifest["evidence_audit"] = {
        "status": summary["status"],
        "human_signoff_status": summary["human_signoff_status"],
        "verification": summary["verification"],
        "outcomes": summary["outcomes"],
    }

    existing_analysis_paths = [
        str(item["path"])
        for item in analysis_manifest.get("published_files", [])
        if Path(str(item["path"])).name
        not in {
            "evidence_audit.csv",
            "evidence_audit_summary.json",
            "EVIDENCE_AUDIT.md",
        }
    ]
    audit_paths = (
        DPD_EVIDENCE_AUDIT_OUTPUT_PATH.as_posix(),
        DPD_EVIDENCE_AUDIT_SUMMARY_PATH.as_posix(),
        DPD_EVIDENCE_AUDIT_REPORT_PATH.as_posix(),
    )
    analysis_manifest["published_files"] = [
        _artifact_record(_resolve(root, path), root=root)
        for path in (*existing_analysis_paths, *audit_paths)
    ]
    _write_json(analysis_manifest_path, analysis_manifest)

    release_manifest = _read_json_object(release_manifest_path)
    release_manifest.setdefault("analysis", {}).update(
        {
            "evidence_audit_method_version": (
                DPD_EVIDENCE_AUDIT_METHOD_VERSION
            ),
            "automated_evidence_audit_complete": True,
            "leaderboard_scores_changed_by_audit": False,
            "human_signoff_status": "not_claimed",
        }
    )
    validation = release_manifest.setdefault("validation", {})
    validation.update(
        {
            "selected_case_audit": "complete",
            "automated_evidence_audit": "complete",
            "human_signoff": "not_claimed",
            "error_adjudication": "complete_automated",
        }
    )
    analysis_directory = (
        DPD_EVIDENCE_AUDIT_ANALYSIS_MANIFEST_PATH.parent
    )
    non_analysis_files = [
        item
        for item in release_manifest.get("published_files", [])
        if not str(item["path"]).startswith(
            analysis_directory.as_posix() + "/"
        )
    ]
    refreshed_analysis_files = []
    for filename in _ANALYSIS_RELEASE_ORDER:
        path = _resolve(root, analysis_directory / filename)
        refreshed_analysis_files.append(
            _artifact_record(
                path,
                root=root,
                role=_ANALYSIS_RELEASE_ROLES[filename],
            )
        )
    release_manifest["published_files"] = [
        *non_analysis_files,
        *refreshed_analysis_files,
    ]
    _write_json(release_manifest_path, release_manifest)
    return True


def _text_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return re.sub(r"\s+", " ", normalized).strip().casefold()


def _number(value: str) -> int | float:
    try:
        number = Decimal(value.strip())
    except InvalidOperation as error:
        raise ValueError(
            f"rendered DPD strength is not numeric: {value!r}"
        ) from error
    if not number.is_finite():
        raise ValueError(
            f"rendered DPD strength is not finite: {value!r}"
        )
    if number == number.to_integral_value():
        return int(number)
    return float(format(number.normalize(), "f"))


def _one(values: Iterable[str], field: str) -> str:
    distinct = set(values)
    if len(distinct) != 1:
        raise ValueError(
            f"rendered DPD record has inconsistent {field}: "
            f"{sorted(distinct)}"
        )
    return next(iter(distinct))


def parse_rendered_dpd_record(
    text: str,
    *,
    language: str,
) -> dict[str, Any]:
    """Independently parse one rendered benchmark input.

    The parser intentionally does not call the benchmark rendering or
    expected-answer functions. It treats `` | `` as the multi-value
    delimiter, preserving commas embedded in official DPD labels.
    """

    if language not in _VARIANT_PATTERNS:
        raise ValueError(f"unsupported DPD audit language: {language}")

    variants: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _VARIANT_PATTERNS[language].fullmatch(line):
            if current is not None:
                variants.append(current)
            current = {"active_ingredients": []}
            continue
        if current is None:
            continue
        ingredient = _INGREDIENT_PATTERNS[language].fullmatch(line)
        if ingredient:
            current["active_ingredients"].append(
                {
                    "name": ingredient.group(1),
                    "strength": _number(ingredient.group(2)),
                    "unit": ingredient.group(3),
                }
            )
            continue
        if line.startswith(_IGNORED_PREFIXES[language]):
            continue
        matched = False
        for prefix, field in _FIELD_PREFIXES[language].items():
            if not line.startswith(prefix):
                continue
            value = line.removeprefix(prefix)
            current[field] = (
                value.split(" | ") if field in _LIST_FIELDS else value
            )
            matched = True
            break
        if not matched:
            raise ValueError(
                f"unrecognized rendered DPD line for {language}: {line!r}"
            )
    if current is not None:
        variants.append(current)
    if not variants:
        raise ValueError("rendered DPD record contains no variants")

    required = {
        "din",
        "brand_name",
        "active_ingredients",
        "dosage_forms",
        "routes",
        "schedule",
        "product_status",
        "company",
    }
    for index, variant in enumerate(variants, start=1):
        missing = required - set(variant)
        if missing:
            raise ValueError(
                f"rendered DPD variant {index} is missing "
                f"{sorted(missing)}"
            )

    ingredients: dict[tuple[str, int | float, str], dict[str, Any]] = {}
    for variant in variants:
        for ingredient in variant["active_ingredients"]:
            key = (
                ingredient["name"],
                ingredient["strength"],
                ingredient["unit"],
            )
            ingredients[key] = ingredient
    ordered_ingredients = [
        ingredients[key]
        for key in sorted(
            ingredients,
            key=lambda item: (
                _text_key(item[0]),
                Decimal(str(item[1])),
                _text_key(item[2]),
            ),
        )
    ]

    def distinct_list(field: str) -> list[str]:
        values = {
            value
            for variant in variants
            for value in variant[field]
        }
        return sorted(values, key=_text_key)

    return {
        "din": sorted(variant["din"] for variant in variants),
        "brand_name": _one(
            (variant["brand_name"] for variant in variants),
            "brand_name",
        ),
        "active_ingredients": ordered_ingredients,
        "dosage_forms": distinct_list("dosage_forms"),
        "routes": distinct_list("routes"),
        "schedule": distinct_list("schedule"),
        "product_status": _one(
            (variant["product_status"] for variant in variants),
            "product_status",
        ),
        "company": _one(
            (variant["company"] for variant in variants),
            "company",
        ),
    }


def _model_order(columns: Sequence[str]) -> list[str]:
    models = [
        column.removesuffix(_MODEL_SUFFIX)
        for column in columns
        if column.endswith(_MODEL_SUFFIX)
    ]
    if not models:
        raise ValueError("selected audit cases contain no model pass columns")
    return models


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"recorded pass flag is not boolean: {value!r}")


def _json_list(value: Any) -> list[str]:
    parsed = json.loads(str(value))
    if not isinstance(parsed, list):
        raise ValueError(f"expected a JSON list, got {value!r}")
    return [str(item) for item in parsed]


def _decode_pointer(pointer: str) -> tuple[str, ...]:
    if not pointer:
        return ()
    return tuple(
        token.replace("~1", "/").replace("~0", "~")
        for token in pointer.removeprefix("/").split("/")
    )


def _path_matches(pointer: str, path: tuple[str, ...]) -> bool:
    expected = _decode_pointer(pointer)
    return len(expected) == len(path) and all(
        wanted == "*" or wanted == actual
        for wanted, actual in zip(expected, path, strict=True)
    )


def _canonical_number(value: Any) -> dict[str, str] | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        source = value
    elif isinstance(value, (int, float)):
        source = str(value)
    else:
        return None
    try:
        number = Decimal(source)
    except InvalidOperation:
        return None
    if not number.is_finite():
        return None
    if number.is_zero():
        normalized = "0"
    else:
        normalized = format(number.normalize(), "f")
    return {"__evalanche_numeric__": normalized}


def _normalize_json(
    value: Any,
    *,
    settings: Any,
    path: tuple[str, ...] = (),
) -> Any:
    comparison = settings.json_comparison
    numeric_path = any(
        _path_matches(pointer, path)
        for pointer in comparison.numeric_value_paths
    )

    if isinstance(value, dict):
        return {
            key: _normalize_json(
                nested,
                settings=settings,
                path=(*path, str(key)),
            )
            for key, nested in value.items()
        }
    if isinstance(value, list):
        normalized = [
            _normalize_json(
                nested,
                settings=settings,
                path=(*path, str(index)),
            )
            for index, nested in enumerate(value)
        ]
        if any(
            _path_matches(pointer, path)
            for pointer in comparison.unordered_list_paths
        ):
            normalized.sort(
                key=lambda item: json.dumps(
                    item,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        return normalized
    if isinstance(value, str):
        normalized = unicodedata.normalize(
            settings.unicode_normalization,
            value,
        )
        if settings.trim_whitespace:
            normalized = normalized.strip()
        if settings.collapse_whitespace:
            normalized = re.sub(r"\s+", " ", normalized)
        if not settings.case_sensitive:
            normalized = normalized.casefold()
        for pointer, width in (
            comparison.zero_pad_numeric_string_paths.items()
        ):
            if (
                _path_matches(pointer, path)
                and normalized.isdecimal()
            ):
                normalized = normalized.zfill(int(width))
        if numeric_path:
            number = _canonical_number(normalized)
            if number is not None:
                return number
        return normalized
    if numeric_path:
        number = _canonical_number(value)
        if number is not None:
            return number
    return value


def _score_json(
    expected_text: str,
    output_text: str,
    *,
    settings: Any,
) -> tuple[bool, list[str]]:
    try:
        expected = json.loads(expected_text)
    except json.JSONDecodeError as error:
        raise ValueError(
            "evidence audit expected output is not valid JSON"
        ) from error
    try:
        output = json.loads(output_text)
    except json.JSONDecodeError:
        return False, []
    expected_normalized = _normalize_json(
        expected,
        settings=settings,
    )
    output_normalized = _normalize_json(
        output,
        settings=settings,
    )
    if not isinstance(expected_normalized, dict):
        raise ValueError("DPD expected output is not a JSON object")
    if not isinstance(output_normalized, dict):
        return False, []
    mismatched = sorted(
        key
        for key in expected_normalized.keys() & output_normalized.keys()
        if expected_normalized[key] != output_normalized[key]
    )
    missing_or_extra = (
        expected_normalized.keys() != output_normalized.keys()
    )
    return (
        not missing_or_extra
        and not mismatched,
        mismatched,
    )


def _rebuild_cases_from_source(
    root: Path,
    review: pd.DataFrame,
    archive_path: Path,
) -> pd.DataFrame:
    metadata = review["source_metadata"].map(json.loads)
    archive_hashes = {
        str(value["archive_sha256"]) for value in metadata
    }
    manifest_hashes = {
        str(value["source_manifest_sha256"]) for value in metadata
    }
    versions = {
        str(value["source_dataset_version"]) for value in metadata
    }
    if len(archive_hashes) != 1 or len(manifest_hashes) != 1:
        raise ValueError(
            "selected audit rows do not share one frozen source release"
        )
    if versions != {"2026.7.2"}:
        raise ValueError(
            f"unexpected DPD source versions in selected audit cases: {sorted(versions)}"
        )
    expected_archive_hash = next(iter(archive_hashes))
    actual_archive_hash = sha256_file(archive_path)
    if actual_archive_hash != expected_archive_hash:
        raise ValueError(
            "frozen marketed DPD archive hash does not match selected-case "
            "provenance"
        )

    tables = load_marketed_tables(archive_path)
    candidates, _ = build_candidate_products(
        tables,
        include_complex_families=True,
    )
    selected = [
        SelectedProduct(candidate=candidate, split="census")
        for candidate in sorted(
            candidates,
            key=lambda candidate: candidate.product_id,
        )
    ]
    rebuilt = pd.DataFrame(
        build_case_records(
            selected,
            source_manifest_sha256=next(iter(manifest_hashes)),
            source_archive_sha256=expected_archive_hash,
            benchmark_version="0.2.0",
        )
    ).set_index("case_id")
    missing = set(review["case_id"]) - set(rebuilt.index)
    if missing:
        raise ValueError(
            f"source rebuild is missing selected audit cases: {sorted(missing)}"
        )
    return rebuilt


def _validate_source_and_expected(
    review: pd.DataFrame,
    rebuilt: pd.DataFrame,
) -> None:
    for row in review.itertuples(index=False):
        rebuilt_row = rebuilt.loc[str(row.case_id)]
        for field in ("input", "expected_output", "source_metadata"):
            if str(getattr(row, field)) != str(rebuilt_row[field]):
                raise ValueError(
                    f"{row.case_id} {field} differs from the frozen "
                    "source rebuild"
                )
        parsed = parse_rendered_dpd_record(
            str(row.input),
            language=str(row.language),
        )
        expected = json.loads(str(row.expected_output))
        if parsed != expected:
            raise ValueError(
                f"{row.case_id} expected answer does not match an "
                "independent parse of the rendered source"
            )


def _validate_scores(
    review: pd.DataFrame,
    *,
    models: Sequence[str],
    config_path: Path,
) -> None:
    config = load_evaluation_config(config_path)
    for row in review.itertuples(index=False):
        values = row._asdict()
        for model in models:
            rescored_pass, rescored_fields = _score_json(
                str(row.expected_output),
                str(values[f"{model}_model_output"]),
                settings=config.metrics,
            )
            recorded_pass = _as_bool(values[f"{model}_passed"])
            if rescored_pass != recorded_pass:
                raise ValueError(
                    f"{row.case_id} {model} pass flag does not match "
                    "a fresh score"
                )
            recorded_fields = sorted(
                _json_list(values[f"{model}_mismatched_fields"])
            )
            if recorded_fields != rescored_fields:
                raise ValueError(
                    f"{row.case_id} {model} mismatch fields do not "
                    "match a fresh score"
                )


def _case_decision(
    row: pd.Series,
    *,
    models: Sequence[str],
) -> dict[str, str]:
    focus_models = _REVIEW_GROUP_FOCUS.get(
        str(row["selection_group"]),
        tuple(models),
    )
    missing_focus = set(focus_models) - set(models)
    if missing_focus:
        raise ValueError(
            "selected audit cases are missing focus models: "
            f"{sorted(missing_focus)}"
        )
    failed_models = [
        model
        for model in focus_models
        if not _as_bool(row[f"{model}_passed"])
    ]
    if not failed_models:
        return {
            "error_owner": "none",
            "operational_severity": "none",
            "audit_notes": (
                "The frozen-source rebuild, independent expected-answer "
                "parse, and fresh strict scoring all agree. This "
                "all-model pass remains a valid control result."
            ),
        }

    error_types = {
        str(row[f"{model}_error_type"])
        for model in failed_models
    }
    mismatched_fields = {
        field
        for model in failed_models
        for field in _json_list(
            row[f"{model}_mismatched_fields"]
        )
    }
    material_text_error = (
        "text_or_value_error" in error_types
        and bool(
            mismatched_fields & _OPERATIONALLY_MATERIAL_FIELDS
        )
    )
    severity = "medium" if material_text_error else "low"

    if error_types == {"compound_label_split_only"}:
        expected = json.loads(str(row["expected_output"]))
        labels = [
            label
            for field in sorted(mismatched_fields)
            for label in expected[field]
        ]
        notes = (
            "The frozen source contains one official comma-bearing "
            f"label ({'; '.join(labels)}). Rendered multi-values use "
            "' | ' as their delimiter, so splitting at the comma "
            "changes the source label. The strict failure is correct."
        )
    elif error_types <= {
        "schema_type_only",
        "duplicate_items_only",
        "schema_and_duplicates",
    }:
        notes = (
            "The source facts and expected JSON are correct. Failed "
            "outputs violate the declared scalar/list contract, the "
            "distinct-fact requirement, or both. The strict failures "
            "are correct."
        )
    else:
        fields = ", ".join(sorted(mismatched_fields))
        notes = (
            "The frozen-source rebuild and expected-answer parse agree. "
            f"Failed outputs alter source text or values in {fields}, "
            "and may also contain structural violations. The strict "
            "failures are correct."
        )
    return {
        "error_owner": "model",
        "operational_severity": severity,
        "audit_notes": notes,
    }


def build_dpd_evidence_audit(
    *,
    root_path: str | Path,
    cases_path: str | Path = DPD_EVIDENCE_AUDIT_CASES_PATH,
    config_path: str | Path = DPD_EVIDENCE_AUDIT_CONFIG_PATH,
    archive_path: str | Path = DPD_EVIDENCE_AUDIT_ARCHIVE_PATH,
    output_path: str | Path = DPD_EVIDENCE_AUDIT_OUTPUT_PATH,
    summary_path: str | Path = DPD_EVIDENCE_AUDIT_SUMMARY_PATH,
) -> dict[str, Any]:
    root = Path(root_path).resolve()
    cases_file = _resolve(root, cases_path)
    config_file = _resolve(root, config_path)
    archive_file = _resolve(root, archive_path)
    output_file = _resolve(root, output_path)
    summary_file = _resolve(root, summary_path)

    for path in (cases_file, config_file, archive_file):
        if not path.is_file():
            raise FileNotFoundError(f"evidence audit input not found: {path}")

    selected = pd.read_csv(cases_file, keep_default_na=False)
    if selected.empty or selected["case_id"].duplicated().any():
        raise ValueError(
            "selected audit cases must contain unique, non-empty cases"
        )
    models = _model_order(selected.columns)
    rebuilt = _rebuild_cases_from_source(
        root,
        selected,
        archive_file,
    )
    _validate_source_and_expected(selected, rebuilt)
    _validate_scores(
        selected,
        models=models,
        config_path=config_file,
    )

    records: list[dict[str, str]] = []
    for _, row in selected.iterrows():
        decision = _case_decision(row, models=models)
        records.append(
            {
                "audit_case_id": str(row["audit_case_id"]),
                "case_id": str(row["case_id"]),
                "audit_status": "complete",
                "audit_method": DPD_EVIDENCE_AUDIT_METHOD_VERSION,
                "source_rebuild_verified": "yes",
                "source_record_correct": "yes",
                "expected_independent_parse_verified": "yes",
                "expected_answer_correct": "yes",
                "strict_score_recomputed": "yes",
                "strict_score_correct": "yes",
                "error_owner": decision["error_owner"],
                "operational_severity": decision[
                    "operational_severity"
                ],
                "adjudication": "retain_result",
                "audit_date": DPD_EVIDENCE_AUDIT_DATE,
                "human_signoff_status": "not_claimed",
                "audit_notes": decision["audit_notes"],
            }
        )
    audit = pd.DataFrame(records)
    _write_csv(output_file, audit)

    severity_counts = Counter(audit["operational_severity"])
    owner_counts = Counter(audit["error_owner"])
    adjudication_counts = Counter(audit["adjudication"])
    summary: dict[str, Any] = {
        "schema_version": DPD_EVIDENCE_AUDIT_SCHEMA_VERSION,
        "method_version": DPD_EVIDENCE_AUDIT_METHOD_VERSION,
        "status": "automated_evidence_audit_complete",
        "human_signoff_status": "not_claimed",
        "audit_date": DPD_EVIDENCE_AUDIT_DATE,
        "inputs": {
            "selected_cases_path": _relative_path(cases_file, root),
            "selected_cases_sha256": sha256_file(cases_file),
            "config_path": _relative_path(config_file, root),
            "config_sha256": sha256_file(config_file),
            "archive_path": _relative_path(archive_file, root),
            "archive_sha256": sha256_file(archive_file),
        },
        "verification": {
            "audit_cases": len(audit),
            "source_rebuild_matches": len(audit),
            "independent_expected_parses_match": len(audit),
            "model_outputs_rescored": len(audit) * len(models),
            "strict_score_disagreements": 0,
        },
        "outcomes": {
            "error_owner_counts": dict(sorted(owner_counts.items())),
            "severity_counts": dict(sorted(severity_counts.items())),
            "adjudication_counts": dict(
                sorted(adjudication_counts.items())
            ),
            "leaderboard_scores_changed": False,
            "cases_excluded": 0,
            "expected_answers_corrected": 0,
            "scoring_rules_revised": 0,
        },
        "output": {
            "path": _relative_path(output_file, root),
            "sha256": sha256_file(output_file),
            "rows": len(audit),
        },
    }
    _write_json(summary_file, summary)
    manifests_refreshed = _refresh_release_manifests(
        root=root,
        output_file=output_file,
        summary_file=summary_file,
        summary=summary,
    )
    return {
        **summary,
        "summary_path": _relative_path(summary_file, root),
        "release_manifests_refreshed": manifests_refreshed,
    }

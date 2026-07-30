from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import shutil
import tempfile
import unicodedata
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from evalanche import __version__
from evalanche.dataset_manifest import (
    DatasetManifest,
    load_dataset_manifest,
    sha256_file,
    verify_dataset_manifest_file,
)
from evalanche.dpd_snapshot import DPD_ARCHIVES, validate_dpd_archive


DPD_BENCHMARK_SCHEMA_VERSION = "1.0"
DPD_BENCHMARK_PARSER_VERSION = "hc_dpd_benchmark/1.0"
DPD_BENCHMARK_DATASET_ID = "hc_dpd_structured_extraction_slice"
DPD_BENCHMARK_VERSION = "0.1.0"
DPD_BENCHMARK_SEED = 20260702
DPD_BENCHMARK_CREATED_AT_UTC = "2026-07-27T00:00:00Z"
DPD_SOURCE_DATASET_ID = "hc_dpd_source_snapshot"
DPD_SOURCE_VERSION = "2026.7.2"
DPD_SOURCE_DATE = "2026-07-02"
DPD_SOURCE_MANIFEST = Path(
    "configs/datasets/hc_dpd_2026-07-02_manifest.yaml"
)

STRATUM_ORDER = (
    "single_ingredient",
    "multi_ingredient",
    "multi_variant",
)
COMPLEX_STRATUM = "multi_ingredient_multi_variant"
ALL_STRATUM_ORDER = (*STRATUM_ORDER, COMPLEX_STRATUM)
DEFAULT_STRATUM_TARGETS = {
    "single_ingredient": 20,
    "multi_ingredient": 10,
    "multi_variant": 10,
}
DEFAULT_HELDOUT_TARGETS = {
    "single_ingredient": 5,
    "multi_ingredient": 2,
    "multi_variant": 3,
}
LANGUAGES = ("en", "fr")
REQUIRED_ROUTE_COVERAGE = {
    "oral": frozenset({"oral"}),
    "injectable": frozenset(
        {
            "intramuscular",
            "intravenous",
            "subcutaneous",
        }
    ),
    "topical": frozenset({"topical"}),
    "inhaled": frozenset({"inhalation"}),
    "ophthalmic": frozenset({"ophthalmic"}),
}

OUTPUT_FIELDS = (
    "din",
    "brand_name",
    "active_ingredients",
    "dosage_forms",
    "routes",
    "schedule",
    "product_status",
    "company",
)

TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "drug": (
        "drug_code",
        "product_categorization",
        "class",
        "din",
        "brand_name",
        "descriptor",
        "pediatric_flag",
        "accession_number",
        "number_of_ais",
        "last_update_date",
        "ai_group_no",
        "class_f",
        "brand_name_f",
        "descriptor_f",
    ),
    "comp": (
        "drug_code",
        "mfr_code",
        "company_code",
        "company_name",
        "company_type",
        "address_mailing_flag",
        "address_billing_flag",
        "address_notification_flag",
        "address_other",
        "suite_number",
        "street_name",
        "city_name",
        "province",
        "country",
        "postal_code",
        "post_office_box",
        "province_f",
        "country_f",
    ),
    "ingred": (
        "drug_code",
        "active_ingredient_code",
        "ingredient",
        "ingredient_supplied_ind",
        "strength",
        "strength_unit",
        "strength_type",
        "dosage_value",
        "base",
        "dosage_unit",
        "notes",
        "ingredient_f",
        "strength_unit_f",
        "strength_type_f",
        "dosage_unit_f",
    ),
    "form": (
        "drug_code",
        "pharm_form_code",
        "pharmaceutical_form",
        "pharmaceutical_form_f",
    ),
    "route": (
        "drug_code",
        "route_code",
        "route",
        "route_f",
    ),
    "schedule": (
        "drug_code",
        "schedule",
        "schedule_f",
    ),
    "status": (
        "drug_code",
        "current_status_flag",
        "status",
        "history_date",
        "status_f",
        "lot_number",
        "expiration_date",
    ),
}

REQUIRED_RELATED_FIELDS: dict[str, tuple[str, ...]] = {
    "ingred": (
        "active_ingredient_code",
        "ingredient",
        "strength",
        "strength_unit",
        "ingredient_f",
        "strength_unit_f",
    ),
    "form": (
        "pharm_form_code",
        "pharmaceutical_form",
        "pharmaceutical_form_f",
    ),
    "route": (
        "route_code",
        "route",
        "route_f",
    ),
    "schedule": (
        "schedule",
        "schedule_f",
    ),
}

CASE_FIELDS = (
    "case_id",
    "benchmark_version",
    "snapshot_date",
    "product_id",
    "drug_codes",
    "din_list",
    "brand_name",
    "language",
    "split",
    "stratum",
    "input",
    "expected_output",
    "evaluation_type",
    "source_metadata",
)

PRODUCT_FIELDS = (
    "product_id",
    "split",
    "stratum",
    "ingredient_group_id",
    "company_code",
    "company_name",
    "brand_name",
    "drug_codes",
    "din_list",
    "active_ingredient_codes",
    "variant_count",
    "language_case_ids",
    "source_rows",
)

_DIN_PATTERN = re.compile(r"^\d{8}$")
_VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")


@dataclass(frozen=True)
class ProductVariant:
    drug_code: str
    din: str
    brand_name: str
    brand_name_f: str
    company_code: str
    company_name: str
    status: str
    status_f: str
    ingredients: tuple[dict[str, Any], ...]
    forms: tuple[dict[str, Any], ...]
    routes: tuple[dict[str, Any], ...]
    schedules: tuple[dict[str, Any], ...]
    source_rows: dict[str, tuple[int, ...]]


@dataclass(frozen=True)
class CandidateProduct:
    product_id: str
    ingredient_group_id: str
    stratum: str
    company_code: str
    company_name: str
    brand_name: str
    ingredient_codes: tuple[str, ...]
    variants: tuple[ProductVariant, ...]


@dataclass(frozen=True)
class SelectedProduct:
    candidate: CandidateProduct
    split: str


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return re.sub(r"\s+", " ", normalized).strip()


def _comparison_text(value: str) -> str:
    return _normalize_text(value).casefold()


def _stable_id(prefix: str, *values: str) -> str:
    payload = "\x1f".join(values).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(payload).hexdigest()[:16]}"


def _stable_rank(seed: int, purpose: str, *values: str) -> str:
    payload = "\x1f".join(
        [str(seed), purpose, *values]
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _numeric_sort_key(value: str) -> tuple[int, str]:
    try:
        return int(value), value
    except ValueError:
        return 2**63 - 1, value


def _decimal_value(value: str) -> int | float:
    normalized = _normalize_text(value)
    try:
        number = Decimal(normalized)
    except InvalidOperation as error:
        raise ValueError(
            f"DPD strength is not numeric: {value!r}"
        ) from error
    if not number.is_finite():
        raise ValueError(f"DPD strength is not finite: {value!r}")
    if number == number.to_integral_value():
        return int(number)
    return float(format(number.normalize(), "f"))


def _json_text(value: Any, *, sort_keys: bool = False) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=sort_keys,
        separators=(",", ":"),
    )


def _utc_text(value: datetime) -> str:
    normalized = value.astimezone(timezone.utc).replace(microsecond=0)
    return normalized.isoformat().replace("+00:00", "Z")


def _read_table(
    archive: zipfile.ZipFile,
    table_name: str,
) -> list[dict[str, Any]]:
    columns = TABLE_COLUMNS[table_name]
    member_name = f"{table_name}.txt"
    try:
        raw = archive.read(member_name)
    except KeyError as error:
        raise ValueError(
            f"marketed archive is missing {member_name}"
        ) from error
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(
            f"{member_name} is not valid UTF-8"
        ) from error

    records: list[dict[str, Any]] = []
    try:
        rows = csv.reader(io.StringIO(text, newline=""), strict=True)
        for row_number, row in enumerate(rows, start=1):
            if not any(cell.strip() for cell in row):
                raise ValueError(
                    f"{member_name} contains a blank row at {row_number}"
                )
            if len(row) != len(columns):
                raise ValueError(
                    f"{member_name} row {row_number} has {len(row)} "
                    f"columns, expected {len(columns)}"
                )
            record: dict[str, Any] = dict(
                zip(columns, row, strict=True)
            )
            record["_row_number"] = row_number
            records.append(record)
    except csv.Error as error:
        raise ValueError(
            f"{member_name} is not valid CSV: {error}"
        ) from error
    return records


def load_marketed_tables(
    archive_path: str | Path,
) -> dict[str, list[dict[str, Any]]]:
    path = Path(archive_path)
    try:
        with zipfile.ZipFile(path) as archive:
            return {
                table_name: _read_table(archive, table_name)
                for table_name in TABLE_COLUMNS
            }
    except zipfile.BadZipFile as error:
        raise ValueError(
            f"marketed DPD archive is not a valid ZIP file: {path}"
        ) from error


def _index_related_tables(
    tables: Mapping[str, list[dict[str, Any]]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    indexed: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for table_name in (
        "comp",
        "ingred",
        "form",
        "route",
        "schedule",
        "status",
    ):
        by_drug_code: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in tables[table_name]:
            by_drug_code[row["drug_code"]].append(row)
        indexed[table_name] = dict(by_drug_code)
    return indexed


def _all_required_values_present(
    rows: Iterable[dict[str, Any]],
    fields: Iterable[str],
) -> bool:
    row_list = list(rows)
    return bool(row_list) and all(
        _normalize_text(str(row[field]))
        for row in row_list
        for field in fields
    )


def _current_status_rows(
    rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if _comparison_text(row["current_status_flag"]) == "y"
    ]


def _source_rows(
    *,
    drug_row: dict[str, Any],
    company_row: dict[str, Any],
    status_row: dict[str, Any],
    related_rows: Mapping[str, list[dict[str, Any]]],
) -> dict[str, tuple[int, ...]]:
    rows = {
        "drug.txt": (int(drug_row["_row_number"]),),
        "comp.txt": (int(company_row["_row_number"]),),
        "status.txt": (int(status_row["_row_number"]),),
    }
    for table_name in ("ingred", "form", "route", "schedule"):
        rows[f"{table_name}.txt"] = tuple(
            sorted(
                int(row["_row_number"])
                for row in related_rows[table_name]
            )
        )
    return rows


def _variant_from_row(
    drug_row: dict[str, Any],
    indexed: Mapping[str, Mapping[str, list[dict[str, Any]]]],
    exclusion_counts: Counter[str],
) -> ProductVariant | None:
    drug_code = _normalize_text(drug_row["drug_code"])
    if _comparison_text(drug_row["class"]) != "human":
        exclusion_counts["not_human"] += 1
        return None

    status_rows = _current_status_rows(
        indexed["status"].get(drug_code, [])
    )
    if len(status_rows) != 1:
        exclusion_counts["missing_or_ambiguous_current_status"] += 1
        return None
    status_row = status_rows[0]
    if _comparison_text(status_row["status"]) != "marketed":
        exclusion_counts["current_status_not_marketed"] += 1
        return None
    if not _normalize_text(status_row["status_f"]):
        exclusion_counts["missing_french_status"] += 1
        return None

    din = _normalize_text(drug_row["din"])
    if not _DIN_PATTERN.fullmatch(din):
        exclusion_counts["invalid_din"] += 1
        return None
    brand_name = _normalize_text(drug_row["brand_name"])
    if not brand_name:
        exclusion_counts["missing_brand_name"] += 1
        return None

    company_rows = [
        row
        for row in indexed["comp"].get(drug_code, [])
        if _comparison_text(row["company_type"]) == "din_owner"
    ]
    if len(company_rows) != 1:
        exclusion_counts["missing_or_ambiguous_din_owner"] += 1
        return None
    company_row = company_rows[0]
    company_code = _normalize_text(company_row["company_code"])
    company_name = _normalize_text(company_row["company_name"])
    if not company_code or not company_name:
        exclusion_counts["incomplete_din_owner"] += 1
        return None

    related_rows = {
        table_name: indexed[table_name].get(drug_code, [])
        for table_name in ("ingred", "form", "route", "schedule")
    }
    for table_name, fields in REQUIRED_RELATED_FIELDS.items():
        if not _all_required_values_present(
            related_rows[table_name],
            fields,
        ):
            exclusion_counts[f"incomplete_{table_name}_rows"] += 1
            return None

    try:
        for ingredient in related_rows["ingred"]:
            _decimal_value(ingredient["strength"])
    except ValueError:
        exclusion_counts["non_numeric_strength"] += 1
        return None

    ingredients = tuple(
        sorted(
            related_rows["ingred"],
            key=lambda row: (
                _numeric_sort_key(row["active_ingredient_code"]),
                Decimal(row["strength"]),
                _comparison_text(row["strength_unit"]),
            ),
        )
    )
    forms = tuple(
        sorted(
            related_rows["form"],
            key=lambda row: (
                _numeric_sort_key(row["pharm_form_code"]),
                _comparison_text(row["pharmaceutical_form"]),
            ),
        )
    )
    routes = tuple(
        sorted(
            related_rows["route"],
            key=lambda row: (
                _numeric_sort_key(row["route_code"]),
                _comparison_text(row["route"]),
            ),
        )
    )
    schedules = tuple(
        sorted(
            related_rows["schedule"],
            key=lambda row: _comparison_text(row["schedule"]),
        )
    )

    return ProductVariant(
        drug_code=drug_code,
        din=din,
        brand_name=brand_name,
        brand_name_f=(
            _normalize_text(drug_row["brand_name_f"]) or brand_name
        ),
        company_code=company_code,
        company_name=company_name,
        status=_normalize_text(status_row["status"]),
        status_f=_normalize_text(status_row["status_f"]),
        ingredients=ingredients,
        forms=forms,
        routes=routes,
        schedules=schedules,
        source_rows=_source_rows(
            drug_row=drug_row,
            company_row=company_row,
            status_row=status_row,
            related_rows=related_rows,
        ),
    )


def _classify_family(
    ingredient_codes: tuple[str, ...],
    variant_count: int,
) -> str:
    if len(ingredient_codes) == 1 and variant_count == 1:
        return "single_ingredient"
    if len(ingredient_codes) >= 2 and variant_count == 1:
        return "multi_ingredient"
    if len(ingredient_codes) == 1 and variant_count >= 2:
        return "multi_variant"
    return COMPLEX_STRATUM


def build_candidate_products(
    tables: Mapping[str, list[dict[str, Any]]],
    *,
    include_complex_families: bool = False,
) -> tuple[list[CandidateProduct], dict[str, Any]]:
    indexed = _index_related_tables(tables)
    drug_codes = [
        _normalize_text(row["drug_code"]) for row in tables["drug"]
    ]
    duplicates = sorted(
        code
        for code, count in Counter(drug_codes).items()
        if count > 1
    )
    if duplicates:
        raise ValueError(
            "drug.txt contains duplicate DRUG_CODE values: "
            + ", ".join(duplicates[:10])
        )

    exclusion_counts: Counter[str] = Counter()
    variants = [
        variant
        for row in tables["drug"]
        if (
            variant := _variant_from_row(
                row,
                indexed,
                exclusion_counts,
            )
        )
        is not None
    ]

    families: dict[
        tuple[str, str, tuple[str, ...]],
        list[ProductVariant],
    ] = defaultdict(list)
    for variant in variants:
        ingredient_codes = tuple(
            sorted(
                {
                    _normalize_text(
                        row["active_ingredient_code"]
                    )
                    for row in variant.ingredients
                },
                key=_numeric_sort_key,
            )
        )
        key = (
            variant.company_code,
            _comparison_text(variant.brand_name),
            ingredient_codes,
        )
        families[key].append(variant)

    family_exclusions: Counter[str] = Counter()
    candidates: list[CandidateProduct] = []
    for (
        company_code,
        normalized_brand,
        ingredient_codes,
    ), family_variants in sorted(
        families.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            item[0][2],
        ),
    ):
        ordered_variants = tuple(
            sorted(
                family_variants,
                key=lambda variant: (
                    variant.din,
                    _numeric_sort_key(variant.drug_code),
                ),
            )
        )
        company_names = {
            _comparison_text(variant.company_name)
            for variant in ordered_variants
        }
        french_brands = {
            _comparison_text(variant.brand_name_f)
            for variant in ordered_variants
        }
        if len(company_names) != 1:
            family_exclusions["inconsistent_company_name"] += 1
            continue
        if len(french_brands) != 1:
            family_exclusions["inconsistent_french_brand_name"] += 1
            continue

        stratum = _classify_family(
            ingredient_codes,
            len(ordered_variants),
        )
        if stratum == COMPLEX_STRATUM and not include_complex_families:
            family_exclusions[
                COMPLEX_STRATUM
            ] += 1
            continue

        product_id = _stable_id(
            "hc_dpd",
            company_code,
            normalized_brand,
            ",".join(ingredient_codes),
        )
        ingredient_group_id = _stable_id(
            "hc_ing",
            ",".join(ingredient_codes),
        )
        candidates.append(
            CandidateProduct(
                product_id=product_id,
                ingredient_group_id=ingredient_group_id,
                stratum=stratum,
                company_code=company_code,
                company_name=ordered_variants[0].company_name,
                brand_name=ordered_variants[0].brand_name,
                ingredient_codes=ingredient_codes,
                variants=ordered_variants,
            )
        )

    candidate_counts = Counter(
        candidate.stratum for candidate in candidates
    )
    reported_strata = (
        ALL_STRATUM_ORDER
        if include_complex_families
        else STRATUM_ORDER
    )
    statistics = {
        "source_drug_rows": len(tables["drug"]),
        "eligible_drug_rows": len(variants),
        "eligible_product_families": len(families),
        "candidate_product_families": len(candidates),
        "candidate_product_families_by_stratum": {
            stratum: candidate_counts[stratum]
            for stratum in reported_strata
        },
        "drug_row_exclusions_by_reason": dict(
            sorted(exclusion_counts.items())
        ),
        "product_family_exclusions_by_reason": dict(
            sorted(family_exclusions.items())
        ),
    }
    return candidates, statistics


def _validate_sampling_targets(
    stratum_targets: Mapping[str, int],
    heldout_targets: Mapping[str, int],
) -> None:
    expected = set(STRATUM_ORDER)
    if set(stratum_targets) != expected:
        raise ValueError(
            "stratum_targets must contain exactly: "
            + ", ".join(STRATUM_ORDER)
        )
    if set(heldout_targets) != expected:
        raise ValueError(
            "heldout_targets must contain exactly: "
            + ", ".join(STRATUM_ORDER)
        )
    for stratum in STRATUM_ORDER:
        target = stratum_targets[stratum]
        heldout = heldout_targets[stratum]
        if target <= 0:
            raise ValueError(
                f"{stratum} target must be greater than zero"
            )
        if heldout < 0 or heldout > target:
            raise ValueError(
                f"{stratum} heldout target must be between 0 and "
                "the stratum target"
            )
    total = sum(stratum_targets.values())
    total_heldout = sum(heldout_targets.values())
    if total_heldout == 0 or total_heldout == total:
        raise ValueError(
            "sampling must produce nonempty development and heldout splits"
        )


def _candidate_routes(candidate: CandidateProduct) -> set[str]:
    return {
        _comparison_text(row["route"])
        for variant in candidate.variants
        for row in variant.routes
    }


def _candidate_covers_route_group(
    candidate: CandidateProduct,
    accepted_routes: Iterable[str],
) -> bool:
    normalized_routes = {
        _comparison_text(route) for route in accepted_routes
    }
    return bool(_candidate_routes(candidate) & normalized_routes)


def select_products(
    candidates: Iterable[CandidateProduct],
    *,
    seed: int,
    stratum_targets: Mapping[str, int] = DEFAULT_STRATUM_TARGETS,
    heldout_targets: Mapping[str, int] = DEFAULT_HELDOUT_TARGETS,
    required_route_coverage: Mapping[
        str,
        Iterable[str],
    ] = REQUIRED_ROUTE_COVERAGE,
) -> list[SelectedProduct]:
    _validate_sampling_targets(stratum_targets, heldout_targets)
    by_stratum: dict[str, list[CandidateProduct]] = defaultdict(list)
    for candidate in candidates:
        by_stratum[candidate.stratum].append(candidate)

    used_ingredient_codes: set[str] = set()
    selected_ids: set[str] = set()
    selected_by_stratum: dict[str, list[CandidateProduct]] = {
        stratum: [] for stratum in STRATUM_ORDER
    }

    all_candidates = [
        candidate
        for stratum in STRATUM_ORDER
        for candidate in by_stratum[stratum]
    ]
    for route_group, accepted_routes in required_route_coverage.items():
        if any(
            _candidate_covers_route_group(candidate, accepted_routes)
            for candidates_in_stratum in selected_by_stratum.values()
            for candidate in candidates_in_stratum
        ):
            continue
        ranked_for_coverage = sorted(
            all_candidates,
            key=lambda candidate: _stable_rank(
                seed,
                "route_coverage",
                route_group,
                candidate.product_id,
            ),
        )
        chosen: CandidateProduct | None = None
        for candidate in ranked_for_coverage:
            if (
                candidate.product_id in selected_ids
                or len(selected_by_stratum[candidate.stratum])
                >= stratum_targets[candidate.stratum]
                or used_ingredient_codes.intersection(
                    candidate.ingredient_codes
                )
                or not _candidate_covers_route_group(
                    candidate,
                    accepted_routes,
                )
            ):
                continue
            chosen = candidate
            break
        if chosen is None:
            raise ValueError(
                "not enough eligible candidates to satisfy required "
                f"route coverage: {route_group}"
            )
        selected_by_stratum[chosen.stratum].append(chosen)
        selected_ids.add(chosen.product_id)
        used_ingredient_codes.update(chosen.ingredient_codes)

    for stratum in STRATUM_ORDER:
        ranked = sorted(
            by_stratum[stratum],
            key=lambda candidate: _stable_rank(
                seed,
                "select",
                stratum,
                candidate.product_id,
            ),
        )
        selected = selected_by_stratum[stratum]
        for candidate in ranked:
            if (
                candidate.product_id in selected_ids
                or used_ingredient_codes.intersection(
                    candidate.ingredient_codes
                )
            ):
                continue
            selected.append(candidate)
            selected_ids.add(candidate.product_id)
            used_ingredient_codes.update(candidate.ingredient_codes)
            if len(selected) == stratum_targets[stratum]:
                break
        if len(selected) != stratum_targets[stratum]:
            raise ValueError(
                f"not enough eligible {stratum} candidates after "
                "active-ingredient leakage controls: expected "
                f"{stratum_targets[stratum]}, found {len(selected)}"
            )

    result: list[SelectedProduct] = []
    for stratum in STRATUM_ORDER:
        split_ranked = sorted(
            selected_by_stratum[stratum],
            key=lambda candidate: _stable_rank(
                seed,
                "split",
                stratum,
                candidate.product_id,
            ),
        )
        heldout_ids = {
            candidate.product_id
            for candidate in split_ranked[
                : heldout_targets[stratum]
            ]
        }
        result.extend(
            SelectedProduct(
                candidate=candidate,
                split=(
                    "heldout"
                    if candidate.product_id in heldout_ids
                    else "development"
                ),
            )
            for candidate in selected_by_stratum[stratum]
        )
    return sorted(
        result,
        key=lambda selected: selected.candidate.product_id,
    )


def _localized_value(
    row: Mapping[str, Any],
    english_field: str,
    french_field: str,
    language: str,
) -> str:
    field = french_field if language == "fr" else english_field
    return _normalize_text(str(row[field]))


def _expected_output(
    candidate: CandidateProduct,
    language: str,
) -> dict[str, Any]:
    ingredients: dict[
        tuple[str, int | float, str],
        dict[str, Any],
    ] = {}
    forms: set[str] = set()
    routes: set[str] = set()
    schedules: set[str] = set()
    statuses: set[str] = set()
    brand_names: set[str] = set()

    for variant in candidate.variants:
        brand_names.add(
            variant.brand_name_f
            if language == "fr"
            else variant.brand_name
        )
        statuses.add(
            variant.status_f if language == "fr" else variant.status
        )
        for row in variant.ingredients:
            name = _localized_value(
                row,
                "ingredient",
                "ingredient_f",
                language,
            )
            unit = _localized_value(
                row,
                "strength_unit",
                "strength_unit_f",
                language,
            )
            strength = _decimal_value(row["strength"])
            ingredients[(name, strength, unit)] = {
                "name": name,
                "strength": strength,
                "unit": unit,
            }
        forms.update(
            _localized_value(
                row,
                "pharmaceutical_form",
                "pharmaceutical_form_f",
                language,
            )
            for row in variant.forms
        )
        routes.update(
            _localized_value(
                row,
                "route",
                "route_f",
                language,
            )
            for row in variant.routes
        )
        schedules.update(
            _localized_value(
                row,
                "schedule",
                "schedule_f",
                language,
            )
            for row in variant.schedules
        )

    if len(brand_names) != 1:
        raise ValueError(
            f"{candidate.product_id} has inconsistent {language} brands"
        )
    if len(statuses) != 1:
        raise ValueError(
            f"{candidate.product_id} has inconsistent {language} statuses"
        )

    ordered_ingredients = [
        ingredients[key]
        for key in sorted(
            ingredients,
            key=lambda item: (
                _comparison_text(item[0]),
                Decimal(str(item[1])),
                _comparison_text(item[2]),
            ),
        )
    ]
    return {
        "din": sorted(variant.din for variant in candidate.variants),
        "brand_name": next(iter(brand_names)),
        "active_ingredients": ordered_ingredients,
        "dosage_forms": sorted(forms, key=_comparison_text),
        "routes": sorted(routes, key=_comparison_text),
        "schedule": sorted(schedules, key=_comparison_text),
        "product_status": next(iter(statuses)),
        "company": candidate.company_name,
    }


def _variant_lines(
    variant: ProductVariant,
    *,
    language: str,
    number: int,
) -> list[str]:
    if language == "fr":
        lines = [
            f"VARIANTE {number}",
            f"DRUG_CODE: {variant.drug_code}",
            f"DIN: {variant.din}",
            f"NOM_COMMERCIAL: {variant.brand_name_f}",
            "INGRÉDIENTS_ACTIFS:",
        ]
        for row in variant.ingredients:
            lines.append(
                "- NOM="
                f"{_normalize_text(row['ingredient_f'])}; "
                f"CONCENTRATION={_normalize_text(row['strength'])}; "
                f"UNITÉ={_normalize_text(row['strength_unit_f'])}"
            )
        lines.extend(
            [
                "FORMES_PHARMACEUTIQUES: "
                + " | ".join(
                    _normalize_text(row["pharmaceutical_form_f"])
                    for row in variant.forms
                ),
                "VOIES_ADMINISTRATION: "
                + " | ".join(
                    _normalize_text(row["route_f"])
                    for row in variant.routes
                ),
                "ANNEXES: "
                + " | ".join(
                    _normalize_text(row["schedule_f"])
                    for row in variant.schedules
                ),
                f"ÉTAT_DU_PRODUIT: {variant.status_f}",
                f"ENTREPRISE: {variant.company_name}",
            ]
        )
        return lines

    lines = [
        f"VARIANT {number}",
        f"DRUG_CODE: {variant.drug_code}",
        f"DIN: {variant.din}",
        f"BRAND_NAME: {variant.brand_name}",
        "ACTIVE_INGREDIENTS:",
    ]
    for row in variant.ingredients:
        lines.append(
            "- NAME="
            f"{_normalize_text(row['ingredient'])}; "
            f"STRENGTH={_normalize_text(row['strength'])}; "
            f"UNIT={_normalize_text(row['strength_unit'])}"
        )
    lines.extend(
        [
            "DOSAGE_FORMS: "
            + " | ".join(
                _normalize_text(row["pharmaceutical_form"])
                for row in variant.forms
            ),
            "ROUTES: "
            + " | ".join(
                _normalize_text(row["route"])
                for row in variant.routes
            ),
            "SCHEDULES: "
            + " | ".join(
                _normalize_text(row["schedule"])
                for row in variant.schedules
            ),
            f"PRODUCT_STATUS: {variant.status}",
            f"COMPANY: {variant.company_name}",
        ]
    )
    return lines


def _case_input(
    candidate: CandidateProduct,
    language: str,
) -> str:
    if language == "fr":
        instruction = (
            "Extrayez les faits de produit pharmaceutique du dossier "
            "source de la Base de données sur les produits "
            "pharmaceutiques de Santé Canada ci-dessous. Retournez "
            "uniquement un objet JSON avec exactement les clés din, "
            "brand_name, active_ingredients, dosage_forms, routes, "
            "schedule, product_status et company. din, dosage_forms, "
            "routes et schedule doivent être des listes. "
            "active_ingredients doit être une liste d'objets ayant les "
            "clés name, strength et unit. Regroupez les faits distincts "
            "de toutes les variantes, sans rien déduire."
        )
        heading = "DOSSIER SOURCE DPD (FRANÇAIS)"
    else:
        instruction = (
            "Extract drug-product facts from the Health Canada Drug "
            "Product Database source record below. Return only one JSON "
            "object with exactly the keys din, brand_name, "
            "active_ingredients, dosage_forms, routes, schedule, "
            "product_status, and company. din, dosage_forms, routes, "
            "and schedule must be lists. active_ingredients must be a "
            "list of objects with name, strength, and unit keys. Combine "
            "distinct facts from every variant and do not infer values."
        )
        heading = "DPD SOURCE RECORD (ENGLISH)"

    blocks = [
        "\n".join(
            _variant_lines(
                variant,
                language=language,
                number=index,
            )
        )
        for index, variant in enumerate(
            candidate.variants,
            start=1,
        )
    ]
    return f"{instruction}\n\n{heading}\n\n" + "\n\n".join(blocks)


def _aggregate_source_rows(
    candidate: CandidateProduct,
) -> dict[str, list[int]]:
    rows: dict[str, set[int]] = defaultdict(set)
    for variant in candidate.variants:
        for member_name, row_numbers in variant.source_rows.items():
            rows[member_name].update(row_numbers)
    return {
        member_name: sorted(row_numbers)
        for member_name, row_numbers in sorted(rows.items())
    }


def build_case_records(
    selected_products: Iterable[SelectedProduct],
    *,
    source_manifest_sha256: str,
    source_archive_sha256: str,
    benchmark_version: str,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for selected in selected_products:
        candidate = selected.candidate
        source_metadata = {
            "source_dataset_id": DPD_SOURCE_DATASET_ID,
            "source_dataset_version": DPD_SOURCE_VERSION,
            "source_manifest_sha256": source_manifest_sha256,
            "archive_file_id": "dpd_marketed_archive",
            "archive_sha256": source_archive_sha256,
            "archive_members": _aggregate_source_rows(candidate),
            "drug_codes": [
                variant.drug_code for variant in candidate.variants
            ],
        }
        for language in LANGUAGES:
            cases.append(
                {
                    "case_id": f"{candidate.product_id}_{language}",
                    "benchmark_version": benchmark_version,
                    "snapshot_date": DPD_SOURCE_DATE,
                    "product_id": candidate.product_id,
                    "drug_codes": _json_text(
                        [
                            variant.drug_code
                            for variant in candidate.variants
                        ]
                    ),
                    "din_list": _json_text(
                        [
                            variant.din
                            for variant in candidate.variants
                        ]
                    ),
                    "brand_name": candidate.brand_name,
                    "language": language,
                    "split": selected.split,
                    "stratum": candidate.stratum,
                    "input": _case_input(candidate, language),
                    "expected_output": _json_text(
                        _expected_output(candidate, language)
                    ),
                    "evaluation_type": "json",
                    "source_metadata": _json_text(
                        {
                            **source_metadata,
                            "language": language,
                        },
                        sort_keys=True,
                    ),
                }
            )
    return sorted(cases, key=lambda row: row["case_id"])


def build_product_records(
    selected_products: Iterable[SelectedProduct],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for selected in selected_products:
        candidate = selected.candidate
        records.append(
            {
                "product_id": candidate.product_id,
                "split": selected.split,
                "stratum": candidate.stratum,
                "ingredient_group_id": candidate.ingredient_group_id,
                "company_code": candidate.company_code,
                "company_name": candidate.company_name,
                "brand_name": candidate.brand_name,
                "drug_codes": _json_text(
                    [
                        variant.drug_code
                        for variant in candidate.variants
                    ]
                ),
                "din_list": _json_text(
                    [variant.din for variant in candidate.variants]
                ),
                "active_ingredient_codes": _json_text(
                    list(candidate.ingredient_codes)
                ),
                "variant_count": len(candidate.variants),
                "language_case_ids": _json_text(
                    [
                        f"{candidate.product_id}_{language}"
                        for language in LANGUAGES
                    ]
                ),
                "source_rows": _json_text(
                    _aggregate_source_rows(candidate),
                    sort_keys=True,
                ),
            }
        )
    return sorted(records, key=lambda row: row["product_id"])


def _write_csv(
    path: Path,
    *,
    fieldnames: Iterable[str],
    records: Iterable[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(fieldnames),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(records)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        json.dump(
            value,
            file,
            indent=2,
            ensure_ascii=False,
            sort_keys=False,
        )
        file.write("\n")


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        yaml.safe_dump(
            value,
            file,
            sort_keys=False,
            allow_unicode=False,
            width=88,
        )


def _file_evidence(
    path: Path,
    *,
    file_id: str,
    role: str,
    media_type: str,
    record_count: int,
    record_count_method: str,
) -> dict[str, Any]:
    digest = sha256_file(path)
    if digest is None:
        raise ValueError(f"generated file does not exist: {path}")
    return {
        "file_id": file_id,
        "source_id": "evalanche_dpd_benchmark_builder",
        "role": role,
        "relative_path": "",
        "media_type": media_type,
        "language": None,
        "byte_size": path.stat().st_size,
        "sha256": digest,
        "record_count": record_count,
        "record_count_method": record_count_method,
        "parser_schema_version": DPD_BENCHMARK_PARSER_VERSION,
        "validation_status": "passed",
        "validation_error_count": 0,
    }


def _split_member_ids(
    selected_products: Iterable[SelectedProduct],
    split: str,
) -> list[str]:
    return sorted(
        selected.candidate.product_id
        for selected in selected_products
        if selected.split == split
    )


def _ingredient_codes_by_split(
    selected_products: Iterable[SelectedProduct],
) -> dict[str, set[str]]:
    values = {"development": set(), "heldout": set()}
    for selected in selected_products:
        values[selected.split].update(
            selected.candidate.ingredient_codes
        )
    return values


def _coverage(
    selected_products: Iterable[SelectedProduct],
) -> dict[str, Any]:
    forms: set[str] = set()
    routes: set[str] = set()
    schedules: set[str] = set()
    for selected in selected_products:
        for variant in selected.candidate.variants:
            forms.update(
                _normalize_text(row["pharmaceutical_form"])
                for row in variant.forms
            )
            routes.update(
                _normalize_text(row["route"])
                for row in variant.routes
            )
            schedules.update(
                _normalize_text(row["schedule"])
                for row in variant.schedules
            )
    return {
        "dosage_forms_en": sorted(forms, key=_comparison_text),
        "routes_en": sorted(routes, key=_comparison_text),
        "schedules_en": sorted(schedules, key=_comparison_text),
    }


def _build_report(
    *,
    selected_products: list[SelectedProduct],
    case_records: list[dict[str, Any]],
    population_statistics: dict[str, Any],
    source_manifest_relative: Path,
    source_manifest_sha256: str,
    source_archive_relative: Path,
    source_archive_sha256: str,
    benchmark_version: str,
    seed: int,
    stratum_targets: Mapping[str, int],
    heldout_targets: Mapping[str, int],
) -> dict[str, Any]:
    ingredient_codes = _ingredient_codes_by_split(selected_products)
    overlap = sorted(
        ingredient_codes["development"]
        & ingredient_codes["heldout"],
        key=_numeric_sort_key,
    )
    split_counts = Counter(
        selected.split for selected in selected_products
    )
    stratum_counts = Counter(
        selected.candidate.stratum
        for selected in selected_products
    )
    language_counts = Counter(
        record["language"] for record in case_records
    )
    coverage = _coverage(selected_products)
    normalized_routes = {
        _comparison_text(route) for route in coverage["routes_en"]
    }
    required_route_checks = {
        route_group: bool(
            normalized_routes
            & {
                _comparison_text(route)
                for route in accepted_routes
            }
        )
        for route_group, accepted_routes in REQUIRED_ROUTE_COVERAGE.items()
    }
    return {
        "report_schema_version": DPD_BENCHMARK_SCHEMA_VERSION,
        "dataset_id": DPD_BENCHMARK_DATASET_ID,
        "dataset_version": benchmark_version,
        "release_created_at_utc": DPD_BENCHMARK_CREATED_AT_UTC,
        "scope": "DPD-only bilingual structured extraction slice",
        "source": {
            "dataset_id": DPD_SOURCE_DATASET_ID,
            "dataset_version": DPD_SOURCE_VERSION,
            "source_date": DPD_SOURCE_DATE,
            "manifest_path": source_manifest_relative.as_posix(),
            "manifest_sha256": source_manifest_sha256,
            "marketed_archive_path": source_archive_relative.as_posix(),
            "marketed_archive_sha256": source_archive_sha256,
        },
        "population": population_statistics,
        "sampling": {
            "method": (
                "deterministic stratified hash ranking with required "
                "route coverage"
            ),
            "seed": seed,
            "stratum_targets": dict(stratum_targets),
            "heldout_targets": dict(heldout_targets),
            "selected_product_count": len(selected_products),
            "generated_case_count": len(case_records),
            "selected_products_by_stratum": {
                stratum: stratum_counts[stratum]
                for stratum in STRATUM_ORDER
            },
            "selected_products_by_split": {
                split: split_counts[split]
                for split in ("development", "heldout")
            },
            "generated_cases_by_language": {
                language: language_counts[language]
                for language in LANGUAGES
            },
        },
        "coverage": {
            **coverage,
            "required_route_groups": required_route_checks,
        },
        "quality_checks": {
            "unique_product_ids": (
                len(
                    {
                        selected.candidate.product_id
                        for selected in selected_products
                    }
                )
                == len(selected_products)
            ),
            "unique_case_ids": (
                len({record["case_id"] for record in case_records})
                == len(case_records)
            ),
            "two_language_cases_per_product": (
                len(case_records) == len(selected_products) * 2
                and all(language_counts[language] == len(selected_products)
                        for language in LANGUAGES)
            ),
            "development_heldout_ingredient_overlap": overlap,
            "development_heldout_ingredient_overlap_count": len(overlap),
            "all_required_route_groups_present": all(
                required_route_checks.values()
            ),
            "all_expected_outputs_parse_as_json": all(
                isinstance(
                    json.loads(record["expected_output"]),
                    dict,
                )
                for record in case_records
            ),
            "manual_product_monograph_audit_status": "not_started",
        },
        "limitations": [
            (
                "Inputs are deterministic renderings of DPD relational "
                "records, not Product Monograph documents."
            ),
            (
                "Reference values have not yet been manually audited "
                "against English and French Product Monographs."
            ),
            (
                "This small slice validates acquisition, normalization, "
                "sampling, traceability, and evaluation plumbing; it is "
                "not evidence of production monograph performance."
            ),
            (
                "Product Monograph availability and document quality "
                "exclusions will be applied in a later release."
            ),
        ],
    }


def _manifest_source_from_parent(
    source_manifest: DatasetManifest,
) -> dict[str, Any]:
    source = next(
        source
        for source in source_manifest.sources
        if source.source_id == "hc_dpd_marketed"
    )
    return {
        "source_id": source.source_id,
        "name": source.name,
        "source_url": source.source_url,
        "retrieved_at_utc": _utc_text(source.retrieved_at_utc),
        "source_modified_date": (
            source.source_modified_date.isoformat()
            if source.source_modified_date is not None
            else None
        ),
        "license_or_terms": source.license_or_terms,
        "snapshot_notes": (
            "Inherited from frozen source dataset "
            f"{DPD_SOURCE_DATASET_ID} {DPD_SOURCE_VERSION}; only the "
            "marketed archive contributes rows to this benchmark slice."
        ),
    }


def _build_manifest(
    *,
    source_manifest: DatasetManifest,
    benchmark_version: str,
    output_relative: Path,
    file_evidence: list[dict[str, Any]],
    selected_products: list[SelectedProduct],
    seed: int,
    stratum_targets: Mapping[str, int],
) -> dict[str, Any]:
    development_ids = _split_member_ids(
        selected_products,
        "development",
    )
    heldout_ids = _split_member_ids(
        selected_products,
        "heldout",
    )
    evidence_by_id = {
        evidence["file_id"]: evidence for evidence in file_evidence
    }
    relative_names = {
        "dpd_benchmark_cases": "cases.csv",
        "dpd_benchmark_products": "products.csv",
        "dpd_benchmark_build_report": "build_report.json",
    }
    for file_id, filename in relative_names.items():
        evidence_by_id[file_id]["relative_path"] = (
            output_relative / filename
        ).as_posix()

    manifest = {
        "schema_version": "1.0",
        "release": {
            "dataset_id": DPD_BENCHMARK_DATASET_ID,
            "version": benchmark_version,
            "release_type": "benchmark",
            "title": (
                "Health Canada DPD bilingual structured extraction slice"
            ),
            "description": (
                "A reproducible 40-product, 80-case DPD-only benchmark "
                "slice for bilingual structured-extraction plumbing."
            ),
            "created_at_utc": DPD_BENCHMARK_CREATED_AT_UTC,
            "status": "frozen",
            "immutable": True,
            "license_or_terms": (
                "Open Government Licence - Canada; attribute Health "
                "Canada, Drug Product Database; "
                "https://open.canada.ca/en/open-government-licence-canada"
            ),
            "intended_use": (
                "Validate Health Canada benchmark acquisition, "
                "normalization, deterministic sampling, source "
                "traceability, bilingual case generation, and Evalanche "
                "execution plumbing before Product Monographs are added."
            ),
            "limitations": [
                (
                    "Inputs render DPD relational records and are not "
                    "Product Monograph documents."
                ),
                (
                    "Reference labels have not yet been manually audited "
                    "against Product Monographs."
                ),
                (
                    "The 40-product slice is too small and too narrow for "
                    "a general Health Canada model recommendation."
                ),
                (
                    "Product Monograph availability and document-quality "
                    "exclusions have not yet been applied."
                ),
            ],
            "languages": list(LANGUAGES),
            "task_types": [
                "bilingual_structured_extraction",
                "dpd_reference_normalization",
            ],
        },
        "sources": [
            _manifest_source_from_parent(source_manifest),
            {
                "source_id": "evalanche_dpd_benchmark_builder",
                "name": "Evalanche DPD benchmark slice builder",
                "source_url": (
                    "https://github.com/hc-sc-ocdo-bdpd/evalanche"
                ),
                "retrieved_at_utc": DPD_BENCHMARK_CREATED_AT_UTC,
                "source_modified_date": None,
                "license_or_terms": "Evalanche repository terms.",
                "snapshot_notes": (
                    "Generated cases.csv, products.csv, and "
                    "build_report.json deterministically from the frozen "
                    "marketed DPD archive."
                ),
            },
        ],
        "files": [
            evidence_by_id["dpd_benchmark_cases"],
            evidence_by_id["dpd_benchmark_products"],
            evidence_by_id["dpd_benchmark_build_report"],
        ],
        "sampling": {
            "method": "stratified_random",
            "unit": "product_family",
            "population_description": (
                "Eligible marketed human-drug product families in the "
                "frozen 2026-07-02 DPD marketed archive, grouped by DIN "
                "owner, normalized brand, and active-ingredient codes."
            ),
            "target_count": sum(stratum_targets.values()),
            "membership_file_id": "dpd_benchmark_products",
            "member_id_column": "product_id",
            "seed": seed,
            "inclusion_criteria": [
                "DPD class is Human.",
                "Exactly one current status row is MARKETED.",
                "DIN is an eight-digit numeric identifier.",
                (
                    "Exactly one DIN_OWNER and complete English and "
                    "French ingredient, form, route, schedule, and status "
                    "labels are available."
                ),
                (
                    "Strength values are numeric and every selected "
                    "product can produce paired English and French cases."
                ),
                (
                    "The release collectively covers oral, injectable, "
                    "topical, inhaled, and ophthalmic routes."
                ),
            ],
            "exclusion_criteria": [
                (
                    "Veterinary, disinfectant, and radiopharmaceutical "
                    "DPD classes."
                ),
                (
                    "Cancelled, dormant, approved-only, missing, or "
                    "ambiguous current status."
                ),
                (
                    "Invalid DIN, missing reference fields, ambiguous DIN "
                    "owner, or nonnumeric strength."
                ),
                (
                    "Multi-ingredient families that also contain multiple "
                    "variants, deferred from this first slice."
                ),
                (
                    "Any candidate whose active-ingredient code overlaps "
                    "a product already selected for this release."
                ),
            ],
            "strata": list(STRATUM_ORDER),
        },
        "splits": [
            {
                "name": "development",
                "purpose": (
                    "Prompt development, integration checks, and visible "
                    "benchmark diagnostics."
                ),
                "unit": "product_family",
                "target_count": len(development_ids),
                "group_key": "active_ingredient_codes",
                "selection_policy": (
                    "Seeded hash assignment after stratified selection; "
                    "no selected active-ingredient code appears in the "
                    "heldout split."
                ),
                "member_ids": development_ids,
            },
            {
                "name": "heldout",
                "purpose": (
                    "Product families reserved from prompt development "
                    "for later blind comparison."
                ),
                "unit": "product_family",
                "target_count": len(heldout_ids),
                "group_key": "active_ingredient_codes",
                "selection_policy": (
                    "Seeded hash assignment after stratified selection; "
                    "no selected active-ingredient code appears in the "
                    "development split."
                ),
                "member_ids": heldout_ids,
            },
        ],
        "lineage": {
            "parent_dataset_version": DPD_SOURCE_VERSION,
            "code_version": (
                f"evalanche {__version__}; "
                f"{DPD_BENCHMARK_PARSER_VERSION}"
            ),
            "transformations": [
                (
                    "Verified the frozen parent manifest and all parent "
                    "file hashes before reading source rows."
                ),
                (
                    "Validated the marketed ZIP member schema and parsed "
                    "relational tables with explicit columns and no "
                    "headers."
                ),
                (
                    "Joined DPD tables by DRUG_CODE and retained complete "
                    "marketed human-drug records."
                ),
                (
                    "Grouped product families by DIN owner, normalized "
                    "brand, and active-ingredient code set."
                ),
                (
                    "Applied seeded deterministic stratified selection "
                    "with required route coverage and active-ingredient "
                    "isolation across splits."
                ),
                (
                    "Generated paired English and French JSON extraction "
                    "cases and recorded exact archive-member row numbers."
                ),
            ],
        },
    }
    DatasetManifest.model_validate(manifest)
    return manifest


def _resolve_source_manifest(
    root: Path,
    source_manifest_path: str | Path,
) -> tuple[Path, Path]:
    requested = Path(source_manifest_path)
    resolved = requested if requested.is_absolute() else root / requested
    try:
        relative = resolved.resolve().relative_to(root)
    except (OSError, ValueError) as error:
        raise ValueError(
            "source manifest must be inside the repository root"
        ) from error
    return resolved, relative


def _validate_source_manifest(
    source_manifest: DatasetManifest,
) -> None:
    if source_manifest.release.dataset_id != DPD_SOURCE_DATASET_ID:
        raise ValueError(
            "source manifest dataset_id must be "
            f"{DPD_SOURCE_DATASET_ID}"
        )
    if source_manifest.release.version != DPD_SOURCE_VERSION:
        raise ValueError(
            "source manifest version must be "
            f"{DPD_SOURCE_VERSION}"
        )
    if source_manifest.release.release_type != "source_snapshot":
        raise ValueError("source manifest must be a source_snapshot")
    if (
        source_manifest.release.status != "frozen"
        or not source_manifest.release.immutable
    ):
        raise ValueError(
            "source manifest must be frozen and immutable"
        )


def _marketed_archive_entry(
    source_manifest: DatasetManifest,
) -> Any:
    matches = [
        file
        for file in source_manifest.files
        if file.role == "raw_dpd_marketed_archive"
    ]
    if len(matches) != 1:
        raise ValueError(
            "source manifest must contain exactly one "
            "raw_dpd_marketed_archive"
        )
    return matches[0]


def _output_paths(
    root: Path,
    benchmark_version: str,
) -> dict[str, Path]:
    output_relative = (
        Path("data")
        / "hc"
        / "benchmarks"
        / "dpd_structured_extraction"
        / benchmark_version
    )
    manifest_relative = (
        Path("configs")
        / "datasets"
        / f"hc_dpd_structured_extraction_{benchmark_version}_manifest.yaml"
    )
    return {
        "output_relative": output_relative,
        "output": root / output_relative,
        "manifest_relative": manifest_relative,
        "manifest": root / manifest_relative,
    }


def create_dpd_benchmark_slice(
    *,
    root_path: str | Path,
    source_manifest_path: str | Path = DPD_SOURCE_MANIFEST,
    benchmark_version: str = DPD_BENCHMARK_VERSION,
    seed: int = DPD_BENCHMARK_SEED,
    stratum_targets: Mapping[str, int] = DEFAULT_STRATUM_TARGETS,
    heldout_targets: Mapping[str, int] = DEFAULT_HELDOUT_TARGETS,
) -> dict[str, Any]:
    if not _VERSION_PATTERN.fullmatch(benchmark_version):
        raise ValueError(
            "benchmark_version must use MAJOR.MINOR.PATCH format"
        )
    _validate_sampling_targets(stratum_targets, heldout_targets)

    root = Path(root_path).resolve()
    source_manifest_file, source_manifest_relative = (
        _resolve_source_manifest(root, source_manifest_path)
    )
    paths = _output_paths(root, benchmark_version)
    output_path = paths["output"]
    manifest_path = paths["manifest"]
    if output_path.exists():
        raise ValueError(
            "benchmark output path already exists and will not be "
            f"overwritten: {output_path}"
        )
    if manifest_path.exists():
        raise ValueError(
            "benchmark manifest already exists and will not be "
            f"overwritten: {manifest_path}"
        )

    source_manifest = load_dataset_manifest(source_manifest_file)
    _validate_source_manifest(source_manifest)
    source_verification = verify_dataset_manifest_file(
        source_manifest_file,
        root_path=root,
    )
    if not source_verification["valid"]:
        raise ValueError(
            "frozen DPD source manifest failed file verification"
        )
    source_manifest_digest = sha256_file(source_manifest_file)
    if source_manifest_digest is None:
        raise ValueError("source manifest does not exist")

    marketed_entry = _marketed_archive_entry(source_manifest)
    archive_path = root / marketed_entry.relative_path
    archive_result = validate_dpd_archive(
        archive_path,
        DPD_ARCHIVES[0],
        source_date=DPD_SOURCE_DATE,
    )
    if archive_result["sha256"] != marketed_entry.sha256:
        raise ValueError(
            "marketed archive hash changed after manifest verification"
        )

    tables = load_marketed_tables(archive_path)
    candidates, population_statistics = build_candidate_products(tables)
    selected_products = select_products(
        candidates,
        seed=seed,
        stratum_targets=stratum_targets,
        heldout_targets=heldout_targets,
    )
    case_records = build_case_records(
        selected_products,
        source_manifest_sha256=source_manifest_digest,
        source_archive_sha256=marketed_entry.sha256,
        benchmark_version=benchmark_version,
    )
    product_records = build_product_records(selected_products)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = Path(
        tempfile.mkdtemp(
            prefix=f".{benchmark_version}-",
            dir=output_path.parent,
        )
    )
    manifest_temp: Path | None = None
    published_output = False
    published_manifest = False

    try:
        cases_path = staging_path / "cases.csv"
        products_path = staging_path / "products.csv"
        report_path = staging_path / "build_report.json"
        _write_csv(
            cases_path,
            fieldnames=CASE_FIELDS,
            records=case_records,
        )
        _write_csv(
            products_path,
            fieldnames=PRODUCT_FIELDS,
            records=product_records,
        )
        report = _build_report(
            selected_products=selected_products,
            case_records=case_records,
            population_statistics=population_statistics,
            source_manifest_relative=source_manifest_relative,
            source_manifest_sha256=source_manifest_digest,
            source_archive_relative=marketed_entry.relative_path,
            source_archive_sha256=marketed_entry.sha256,
            benchmark_version=benchmark_version,
            seed=seed,
            stratum_targets=stratum_targets,
            heldout_targets=heldout_targets,
        )
        _write_json(report_path, report)

        file_evidence = [
            _file_evidence(
                cases_path,
                file_id="dpd_benchmark_cases",
                role="benchmark_cases",
                media_type="text/csv",
                record_count=len(case_records),
                record_count_method="csv_rows",
            ),
            _file_evidence(
                products_path,
                file_id="dpd_benchmark_products",
                role="sampling_membership",
                media_type="text/csv",
                record_count=len(product_records),
                record_count_method="csv_rows",
            ),
            _file_evidence(
                report_path,
                file_id="dpd_benchmark_build_report",
                role="benchmark_build_report",
                media_type="application/json",
                record_count=1,
                record_count_method="declared",
            ),
        ]
        manifest = _build_manifest(
            source_manifest=source_manifest,
            benchmark_version=benchmark_version,
            output_relative=paths["output_relative"],
            file_evidence=file_evidence,
            selected_products=selected_products,
            seed=seed,
            stratum_targets=stratum_targets,
        )

        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, manifest_temp_text = tempfile.mkstemp(
            prefix=f".{manifest_path.name}.",
            dir=manifest_path.parent,
        )
        os.close(descriptor)
        manifest_temp = Path(manifest_temp_text)
        _write_yaml(manifest_temp, manifest)

        os.replace(staging_path, output_path)
        published_output = True
        os.replace(manifest_temp, manifest_path)
        published_manifest = True
        manifest_temp = None

        verification = verify_dataset_manifest_file(
            manifest_path,
            root_path=root,
        )
        if not verification["valid"]:
            raise ValueError(
                "materialized DPD benchmark manifest failed verification"
            )

        return {
            "dataset_id": DPD_BENCHMARK_DATASET_ID,
            "dataset_version": benchmark_version,
            "source_dataset_id": DPD_SOURCE_DATASET_ID,
            "source_dataset_version": DPD_SOURCE_VERSION,
            "output_path": output_path,
            "manifest_path": manifest_path,
            "product_count": len(product_records),
            "case_count": len(case_records),
            "development_product_count": len(
                _split_member_ids(selected_products, "development")
            ),
            "heldout_product_count": len(
                _split_member_ids(selected_products, "heldout")
            ),
            "verification": verification,
        }
    except Exception:
        if staging_path.exists():
            shutil.rmtree(staging_path)
        if manifest_temp is not None and manifest_temp.exists():
            manifest_temp.unlink()
        if published_output and output_path.exists():
            shutil.rmtree(output_path)
        if published_manifest and manifest_path.exists():
            manifest_path.unlink()
        raise

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from evalanche.dataset_manifest import DatasetManifest, sha256_file


DPD_BENCHMARK_SCHEMA_VERSION = "1.0"
DPD_BENCHMARK_PARSER_VERSION = "hc_dpd_benchmark/1.0"
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

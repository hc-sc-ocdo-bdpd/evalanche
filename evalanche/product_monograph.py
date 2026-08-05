from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import pandas as pd
import yaml
from pypdf import PdfReader, __version__ as pypdf_version

from evalanche import __version__
from evalanche.dataset_manifest import (
    DatasetManifest,
    verify_dataset_manifest_file,
)

logging.getLogger("pypdf").setLevel(logging.ERROR)

PM_DATASET_ID = "hc_product_monograph_structured_extraction"
PM_BENCHMARK_ID = "hc_product_monograph_structured_extraction"
PM_VERSION = "0.1.0"
PM_COHORT_PATH = Path(
    "configs/product_monograph/0.1.0/cohort.yaml"
)
PM_OVERRIDES_PATH = Path(
    "configs/product_monograph/0.1.0/label_overrides.yaml"
)
PM_OUTPUT_DIR = Path(
    "data/hc/benchmarks/product_monograph_structured_extraction/0.1.0"
)
PM_SOURCES_PATH = PM_OUTPUT_DIR / "source_documents.csv"
PM_EXCLUSIONS_PATH = PM_OUTPUT_DIR / "screening_exclusions.csv"
PM_RAW_DIR = Path("data/hc/product_monographs/0.1.0/raw")
PM_MANIFEST_PATH = Path(
    "configs/datasets/"
    "hc_product_monograph_structured_extraction_0.1.0_manifest.yaml"
)
PM_MAX_EVIDENCE_PAGES = 8
PM_RETRIEVED_AT_UTC = "2026-07-30T22:00:00Z"
PM_RELEASE_CREATED_AT_UTC = "2026-07-30T23:00:00Z"

PM_NATIVE_PDF_DATASET_ID = "hc_product_monograph_native_pdf_extraction"
PM_NATIVE_PDF_BENCHMARK_ID = PM_NATIVE_PDF_DATASET_ID
PM_NATIVE_PDF_VERSION = "0.1.0"
PM_NATIVE_PDF_OUTPUT_DIR = Path(
    "data/hc/benchmarks/product_monograph_native_pdf_extraction/0.1.0"
)
PM_NATIVE_PDF_CASES_PATH = PM_NATIVE_PDF_OUTPUT_DIR / "cases.csv.gz"
PM_NATIVE_PDF_REVIEW_PATH = PM_NATIVE_PDF_OUTPUT_DIR / "label_review.csv"
PM_NATIVE_PDF_REPORT_PATH = PM_NATIVE_PDF_OUTPUT_DIR / "build_report.json"
PM_NATIVE_PDF_MANIFEST_PATH = Path(
    "configs/datasets/"
    "hc_product_monograph_native_pdf_extraction_0.1.0_manifest.yaml"
)
PM_NATIVE_PDF_RELEASE_CREATED_AT_UTC = "2026-08-04T13:30:00Z"
PM_NATIVE_PDF_REVIEW_STATUSES = {
    "pending",
    "approved",
    "rejected",
    "needs_correction",
}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def _normalize(value: Any) -> str:
    text = str(value).translate(
        {
            ord("™"): " ",
            ord("®"): " ",
            ord("©"): " ",
        }
    )
    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        character for character in text if not unicodedata.combining(character)
    )
    text = text.casefold()
    text = re.sub(r"\b(?:pr|mc|tm)\b", " ", text)
    text = re.sub(r"[^\w\s%]", " ", text)
    text = text.replace("%", " % ")
    return re.sub(r"\s+", " ", text).strip()


def _base_ingredient_name(value: str) -> str:
    return re.sub(r"\s*\([^)]*\)\s*", "", value).strip()


def _document_filename(row: pd.Series | dict[str, Any]) -> str:
    return (
        f"{row['product_id']}_{row['language']}_"
        f"{row['monograph_id']}.pdf"
    )


def _extract_pdf_pages(path: Path, expected_sha256: str) -> list[str]:
    cache_path = path.with_suffix(".pages.json")
    if cache_path.is_file():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cache = {}
        if (
            cache.get("pdf_sha256") == expected_sha256
            and isinstance(cache.get("pages"), list)
        ):
            return [str(page) for page in cache["pages"]]

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    cache_path.write_text(
        json.dumps(
            {
                "pdf_sha256": expected_sha256,
                "pages": pages,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    return pages


def acquire_product_monographs(
    *,
    root_path: str | Path = ".",
    sources_path: str | Path = PM_SOURCES_PATH,
    raw_dir: str | Path = PM_RAW_DIR,
    timeout: float = 120.0,
) -> dict[str, Any]:
    root = Path(root_path).resolve()
    source_file = root / sources_path
    if not source_file.is_file():
        raise FileNotFoundError(
            f"Frozen source inventory not found: {source_file}"
        )
    sources = pd.read_csv(source_file, dtype=str)
    required = {
        "product_id",
        "language",
        "monograph_id",
        "source_url",
        "sha256",
        "byte_size",
        "page_count",
    }
    missing = required - set(sources.columns)
    if missing:
        raise ValueError(
            f"Source inventory is missing columns: {sorted(missing)}"
        )

    destination = root / raw_dir
    destination.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    reused = 0
    for row in sources.to_dict(orient="records"):
        path = destination / _document_filename(row)
        expected_hash = str(row["sha256"])
        if path.is_file() and _sha256(path) == expected_hash:
            reused += 1
            continue

        request = Request(
            str(row["source_url"]),
            headers={
                "User-Agent": (
                    "Evalanche benchmark acquisition "
                    "(reproducible public-source research)"
                )
            },
        )
        with urlopen(request, timeout=timeout) as response:
            content = response.read()
        actual_hash = hashlib.sha256(content).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(
                f"Source hash changed for {row['product_id']} "
                f"{row['language']}: expected {expected_hash}, "
                f"found {actual_hash}"
            )
        path.write_bytes(content)
        downloaded += 1

    verification = verify_product_monograph_sources(
        root_path=root,
        sources_path=sources_path,
        raw_dir=raw_dir,
    )
    return {
        **verification,
        "downloaded": downloaded,
        "reused": reused,
        "raw_dir": destination,
    }


def verify_product_monograph_sources(
    *,
    root_path: str | Path = ".",
    sources_path: str | Path = PM_SOURCES_PATH,
    raw_dir: str | Path = PM_RAW_DIR,
) -> dict[str, Any]:
    root = Path(root_path).resolve()
    sources = pd.read_csv(root / sources_path, dtype=str)
    issues: list[str] = []
    hashes: list[str] = []
    languages = Counter()
    for row in sources.to_dict(orient="records"):
        path = root / raw_dir / _document_filename(row)
        if not path.is_file():
            issues.append(f"Missing PDF: {path}")
            continue
        actual_hash = _sha256(path)
        hashes.append(actual_hash)
        if actual_hash != row["sha256"]:
            issues.append(f"Hash mismatch: {path}")
        if path.stat().st_size != int(row["byte_size"]):
            issues.append(f"Byte-size mismatch: {path}")
        try:
            page_count = len(PdfReader(path).pages)
        except Exception as error:
            issues.append(f"Unreadable PDF {path}: {error}")
            continue
        if page_count != int(row["page_count"]):
            issues.append(f"Page-count mismatch: {path}")
        languages[str(row["language"])] += 1

    if len(sources) != 80:
        issues.append(f"Expected 80 documents, found {len(sources)}")
    if len(set(hashes)) != len(hashes):
        issues.append("Document SHA-256 values are not unique")
    if languages != {"en": 40, "fr": 40}:
        issues.append(
            f"Expected 40 English and 40 French documents, found "
            f"{dict(languages)}"
        )
    return {
        "documents": int(len(sources)),
        "languages": dict(languages),
        "unique_hashes": len(set(hashes)),
        "issues": issues,
        "valid": not issues,
    }


def _apply_expected_overrides(
    *,
    source_expected: dict[str, Any],
    case_id: str,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    active = []
    case_overrides = overrides.get("cases", {}).get(case_id, {})
    name_overrides = case_overrides.get(
        "active_ingredient_names",
        {},
    )
    for ingredient in source_expected["active_ingredients"]:
        name = _base_ingredient_name(str(ingredient["name"]))
        name = name_overrides.get(name, name)
        active.append(
            {
                "name": name,
                "strength": ingredient["strength"],
                "unit": ingredient["unit"],
            }
        )
    expected = {
        "brand_name": case_overrides.get(
            "brand_name",
            source_expected["brand_name"],
        ),
        "active_ingredients": active,
        "dosage_forms": case_overrides.get(
            "dosage_forms",
            source_expected["dosage_forms"],
        ),
        "routes": case_overrides.get(
            "routes",
            source_expected["routes"],
        ),
    }
    return expected


def _alias_candidates(
    *,
    field: str,
    value: str,
    overrides: dict[str, Any],
) -> list[str]:
    candidates = [value]
    aliases = overrides.get("normalization_aliases", {})
    alias_group = (
        aliases.get("dosage_forms", {})
        if field == "dosage_forms"
        else aliases.get("units", {})
        if field == "unit"
        else {}
    )
    normalized_value = _normalize(value)
    for source, target in alias_group.items():
        if _normalize(target) == normalized_value:
            candidates.append(str(source))
    return list(dict.fromkeys(candidates))


def _number_candidates(value: Any) -> list[str]:
    text = str(value)
    values = [text]
    if text.endswith(".0"):
        values.append(text[:-2])
    values.extend(
        [
            text.replace(".", ","),
            text.replace(".", " "),
        ]
    )
    if re.fullmatch(r"\d{4,}", text):
        values.extend(
            [
                f"{int(text):,}",
                f"{int(text):,}".replace(",", " "),
            ]
        )
    return list(dict.fromkeys(values))


def _find_pages(
    *,
    pages: list[str],
    values: list[str],
) -> list[int]:
    normalized_values = [_normalize(value) for value in values]
    return [
        index
        for index, page in enumerate(pages, start=1)
        if any(
            value and value in _normalize(page)
            for value in normalized_values
        )
    ]


def _build_evidence(
    *,
    case_id: str,
    expected: dict[str, Any],
    pages: list[str],
    overrides: dict[str, Any],
) -> tuple[list[dict[str, Any]], set[int]]:
    evidence: list[dict[str, Any]] = []
    selected_pages: set[int] = {1}

    def record(
        field: str,
        item: str,
        candidates: list[str],
    ) -> None:
        matches = _find_pages(pages=pages, values=candidates)
        if not matches:
            raise ValueError(
                f"No source evidence found for {case_id} "
                f"{field}={item!r}"
            )
        page = matches[0]
        selected_pages.add(page)
        evidence.append(
            {
                "case_id": case_id,
                "field": field,
                "expected_item": item,
                "source_page": page,
                "support_method": (
                    "normalized_exact"
                    if _normalize(item) in _normalize(pages[page - 1])
                    else "declared_alias"
                ),
            }
        )

    record(
        "brand_name",
        str(expected["brand_name"]),
        [str(expected["brand_name"])],
    )
    for ingredient in expected["active_ingredients"]:
        name = str(ingredient["name"])
        unit = str(ingredient["unit"])
        name_pages = set(_find_pages(pages=pages, values=[name]))
        strength_values = [
            f"{number} {unit_value}"
            for number in _number_candidates(ingredient["strength"])
            for unit_value in _alias_candidates(
                field="unit",
                value=unit,
                overrides=overrides,
            )
        ]
        strength_pages = set(
            _find_pages(pages=pages, values=strength_values)
        )
        matches = sorted(name_pages & strength_pages)
        if not matches:
            matches = sorted(name_pages or strength_pages)
        if not matches:
            raise ValueError(
                f"No source evidence found for {case_id} ingredient "
                f"{ingredient!r}"
            )
        page = matches[0]
        selected_pages.add(page)
        evidence.append(
            {
                "case_id": case_id,
                "field": "active_ingredients",
                "expected_item": json.dumps(
                    ingredient,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "source_page": page,
                "support_method": (
                    "normalized_name_and_strength"
                    if page in name_pages and page in strength_pages
                    else "normalized_name_or_strength"
                ),
            }
        )
    for value in expected["dosage_forms"]:
        record(
            "dosage_forms",
            str(value),
            _alias_candidates(
                field="dosage_forms",
                value=str(value),
                overrides=overrides,
            ),
        )
    for value in expected["routes"]:
        record("routes", str(value), [str(value)])

    if len(selected_pages) > PM_MAX_EVIDENCE_PAGES:
        raise ValueError(
            f"{case_id} requires {len(selected_pages)} evidence pages, "
            f"above the limit of {PM_MAX_EVIDENCE_PAGES}"
        )
    return evidence, selected_pages


def _render_input(
    *,
    language: str,
    pages: list[str],
    selected_pages: set[int],
) -> str:
    contract = (
        "Extract facts from the official Health Canada Product Monograph "
        "evidence excerpt below. Return only one JSON object with exactly "
        "the keys brand_name, active_ingredients, dosage_forms, and routes. "
        "active_ingredients must be a list of objects with name, strength, "
        "and unit keys. dosage_forms and routes must be lists. Preserve the "
        "source language, omit trademark symbols, use numeric JSON values "
        "for strengths, combine distinct product variants, and do not infer "
        "facts that are absent from the excerpt."
    )
    excerpts = []
    for page_number in sorted(selected_pages):
        text = pages[page_number - 1].strip()
        excerpts.append(
            f"[PDF PAGE {page_number}]\n{text}"
        )
    language_name = "ENGLISH" if language == "en" else "FRENCH"
    return (
        f"{contract}\n\nPRODUCT MONOGRAPH EVIDENCE ({language_name})\n\n"
        + "\n\n".join(excerpts)
    )


def _document_complexity(page_count: int) -> str:
    if page_count <= 30:
        return "short"
    if page_count <= 60:
        return "medium"
    return "long"


def _manifest_file(
    *,
    root: Path,
    path: Path,
    file_id: str,
    role: str,
    record_count: int,
    media_type: str,
) -> dict[str, Any]:
    absolute = root / path
    return {
        "file_id": file_id,
        "source_id": "evalanche_pm_builder",
        "role": role,
        "relative_path": path.as_posix(),
        "media_type": media_type,
        "language": None,
        "byte_size": absolute.stat().st_size,
        "sha256": _sha256(absolute),
        "record_count": record_count,
        "record_count_method": (
            "csv_rows" if "csv" in media_type or path.suffix == ".gz"
            else "declared"
        ),
        "page_count": None,
        "parser_schema_version": "hc_pm/1.0",
        "validation_status": "passed",
        "validation_error_count": 0,
    }


def _build_manifest(
    *,
    root: Path,
    products: pd.DataFrame,
    cases: pd.DataFrame,
    evidence: pd.DataFrame,
    build_report_path: Path,
) -> dict[str, Any]:
    product_ids = products["product_id"].astype(str).tolist()
    development = products.loc[
        products["split"] == "development",
        "product_id",
    ].astype(str).tolist()
    heldout = products.loc[
        products["split"] == "heldout",
        "product_id",
    ].astype(str).tolist()
    files = [
        _manifest_file(
            root=root,
            path=PM_OUTPUT_DIR / "cases.csv.gz",
            file_id="pm_cases",
            role="benchmark_cases",
            record_count=len(cases),
            media_type="application/gzip",
        ),
        _manifest_file(
            root=root,
            path=PM_OUTPUT_DIR / "products.csv",
            file_id="pm_products",
            role="sampling_membership",
            record_count=len(products),
            media_type="text/csv",
        ),
        _manifest_file(
            root=root,
            path=PM_EXCLUSIONS_PATH,
            file_id="pm_screening_exclusions",
            role="selection_audit",
            record_count=23,
            media_type="text/csv",
        ),
        _manifest_file(
            root=root,
            path=PM_SOURCES_PATH,
            file_id="pm_source_documents",
            role="source_lock",
            record_count=80,
            media_type="text/csv",
        ),
        _manifest_file(
            root=root,
            path=PM_OUTPUT_DIR / "field_evidence.csv.gz",
            file_id="pm_field_evidence",
            role="reference_evidence",
            record_count=len(evidence),
            media_type="application/gzip",
        ),
        _manifest_file(
            root=root,
            path=build_report_path,
            file_id="pm_build_report",
            role="benchmark_build_report",
            record_count=1,
            media_type="application/json",
        ),
    ]
    return {
        "schema_version": "1.0",
        "release": {
            "dataset_id": PM_DATASET_ID,
            "version": PM_VERSION,
            "release_type": "benchmark",
            "title": (
                "Health Canada Product Monograph structured extraction"
            ),
            "description": (
                "A bilingual 40-product, 80-document evidence-grounded "
                "structured-extraction pilot using official Product "
                "Monographs."
            ),
            "created_at_utc": PM_RELEASE_CREATED_AT_UTC,
            "status": "frozen",
            "immutable": True,
            "license_or_terms": (
                "Official Product Monographs are publicly posted by Health "
                "Canada. Copyright in individual monographs may remain with "
                "their sponsors. The benchmark stores hashes, citations, "
                "and extracted evidence, not source PDFs."
            ),
            "intended_use": (
                "Compare model extraction quality on bounded, auditable "
                "English and French Product Monograph evidence."
            ),
            "limitations": [
                "The pilot evaluates extraction from evidence-selected "
                "pages, not retrieval from an entire monograph.",
                "DIN, regulatory schedule, product status, and sponsor are "
                "alignment metadata and are not scored because they are not "
                "consistently stated in the monographs.",
                "Automated source-evidence checks are complete, but "
                "independent human label sign-off is not claimed.",
                "The 40-product purposive sample is not representative of "
                "all marketed drugs or all document formats.",
            ],
            "languages": ["en", "fr"],
            "task_types": [
                "bilingual_structured_extraction",
                "product_monograph_extraction",
            ],
        },
        "sources": [
            {
                "source_id": "hc_product_monographs",
                "name": "Health Canada Product Monograph PDF repository",
                "source_url": "https://pdf.hres.ca/dpd_pm/",
                "retrieved_at_utc": PM_RETRIEVED_AT_UTC,
                "source_modified_date": None,
                "license_or_terms": (
                    "Publicly posted official Product Monographs; individual "
                    "sponsor copyright may apply."
                ),
                "snapshot_notes": (
                    "Eighty official PDF URLs, hashes, byte sizes, and page "
                    "counts are frozen in source_documents.csv."
                ),
            },
            {
                "source_id": "evalanche_pm_builder",
                "name": "Evalanche Product Monograph benchmark builder",
                "source_url": (
                    "https://github.com/hc-sc-ocdo-bdpd/evalanche"
                ),
                "retrieved_at_utc": PM_RETRIEVED_AT_UTC,
                "source_modified_date": None,
                "license_or_terms": "Evalanche repository terms.",
                "snapshot_notes": (
                    "Builds cases and field evidence deterministically from "
                    "the frozen source lock and DPD alignment records."
                ),
            },
        ],
        "files": files,
        "sampling": {
            "method": "purposive",
            "unit": "product_family",
            "population_description": (
                "Marketed human-drug product families with usable, distinct "
                "official English and French Product Monographs and "
                "resolvable DPD alignment."
            ),
            "target_count": len(product_ids),
            "membership_file_id": "pm_products",
            "member_id_column": "product_id",
            "seed": 20260730,
            "inclusion_criteria": [
                "Official English and French PDFs are both posted.",
                "Both PDFs are text-extractable and language-correct.",
                "Monograph product scope aligns with the selected DPD family.",
                "Every scored reference item has source-page evidence.",
            ],
            "exclusion_criteria": [
                "A language PDF is missing or duplicates the other language.",
                "The monograph covers unresolved sibling products or strengths.",
                "The PDF is malformed, image-only, or lacks scored evidence.",
            ],
            "strata": [
                "single_ingredient",
                "multi_ingredient",
                "multi_variant",
            ],
        },
        "splits": [
            {
                "name": "development",
                "purpose": "Prompt development and visible analysis.",
                "unit": "product_family",
                "target_count": len(development),
                "group_key": "ingredient_group_id",
                "selection_policy": (
                    "Thirty products assigned before model evaluation."
                ),
                "member_ids": development,
            },
            {
                "name": "heldout",
                "purpose": "Final prompt-locked comparison.",
                "unit": "product_family",
                "target_count": len(heldout),
                "group_key": "ingredient_group_id",
                "selection_policy": (
                    "Ten products with no active-ingredient group shared "
                    "with development."
                ),
                "member_ids": heldout,
            },
        ],
        "lineage": {
            "parent_dataset_version": "0.2.0",
            "code_version": __version__,
            "transformations": [
                "Resolve frozen bilingual Product Monograph source lock.",
                "Verify PDF hashes, page counts, uniqueness, and text.",
                "Align product scope to frozen DPD families.",
                "Build source-language four-field reference JSON.",
                "Select and preserve exact source-evidence pages.",
            ],
        },
    }


def build_product_monograph_benchmark(
    *,
    root_path: str | Path = ".",
    cohort_path: str | Path = PM_COHORT_PATH,
    overrides_path: str | Path = PM_OVERRIDES_PATH,
    sources_path: str | Path = PM_SOURCES_PATH,
    raw_dir: str | Path = PM_RAW_DIR,
) -> dict[str, Any]:
    root = Path(root_path).resolve()
    cohort = _load_yaml(root / cohort_path)
    overrides = _load_yaml(root / overrides_path)
    source_verification = verify_product_monograph_sources(
        root_path=root,
        sources_path=sources_path,
        raw_dir=raw_dir,
    )
    if not source_verification["valid"]:
        raise ValueError(
            "Product Monograph source verification failed: "
            + "; ".join(source_verification["issues"])
        )

    dpd_products_path = (
        root / cohort["source_dpd_dataset"]["products_path"]
    )
    dpd_cases_path = root / cohort["source_dpd_dataset"]["cases_path"]
    dpd_products = pd.read_csv(dpd_products_path, dtype=str)
    dpd_cases = pd.read_csv(dpd_cases_path, dtype=str)
    sources = pd.read_csv(root / sources_path, dtype=str)
    exclusions = pd.read_csv(root / PM_EXCLUSIONS_PATH, dtype=str)
    if len(exclusions) != 23:
        raise ValueError(
            f"Expected 23 documented screening exclusions, "
            f"found {len(exclusions)}"
        )
    cohort_rows = pd.DataFrame(cohort["products"])
    products = cohort_rows.merge(
        dpd_products,
        on=["product_id", "stratum"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_dpd"),
    )
    products = products.drop(columns=["split_dpd"])
    if products["ingredient_group_id"].isna().any():
        missing = products.loc[
            products["ingredient_group_id"].isna(),
            "product_id",
        ].tolist()
        raise ValueError(f"Cohort products missing from DPD: {missing}")

    expected_counts = {
        ("development", "single_ingredient"): 15,
        ("development", "multi_ingredient"): 8,
        ("development", "multi_variant"): 7,
        ("heldout", "single_ingredient"): 5,
        ("heldout", "multi_ingredient"): 2,
        ("heldout", "multi_variant"): 3,
    }
    observed_counts = Counter(
        zip(products["split"], products["stratum"], strict=True)
    )
    if observed_counts != expected_counts:
        raise ValueError(
            f"Cohort quotas do not match: {dict(observed_counts)}"
        )
    development_groups = set(
        products.loc[
            products["split"] == "development",
            "ingredient_group_id",
        ]
    )
    heldout_groups = set(
        products.loc[
            products["split"] == "heldout",
            "ingredient_group_id",
        ]
    )
    if development_groups & heldout_groups:
        raise ValueError(
            "Ingredient groups overlap development and heldout splits"
        )

    case_records: list[dict[str, Any]] = []
    evidence_records: list[dict[str, Any]] = []
    for product in products.to_dict(orient="records"):
        product_sources = sources[
            sources["product_id"] == product["product_id"]
        ]
        if set(product_sources["language"]) != {"en", "fr"}:
            raise ValueError(
                f"Product lacks a bilingual source pair: "
                f"{product['product_id']}"
            )
        for source in product_sources.to_dict(orient="records"):
            language = source["language"]
            dpd_case_id = f"{product['product_id']}_{language}"
            matches = dpd_cases[
                (dpd_cases["product_id"] == product["product_id"])
                & (dpd_cases["language"] == language)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one DPD alignment case: {dpd_case_id}"
                )
            dpd_case = matches.iloc[0]
            expected = _apply_expected_overrides(
                source_expected=json.loads(dpd_case["expected_output"]),
                case_id=dpd_case_id,
                overrides=overrides,
            )
            pdf_path = root / raw_dir / _document_filename(source)
            pages = _extract_pdf_pages(
                pdf_path,
                str(source["sha256"]),
            )
            evidence, selected_pages = _build_evidence(
                case_id=dpd_case_id,
                expected=expected,
                pages=pages,
                overrides=overrides,
            )
            evidence_records.extend(evidence)
            source_metadata = {
                "source_type": "official_product_monograph",
                "source_url": source["source_url"],
                "monograph_id": source["monograph_id"],
                "monograph_sha256": source["sha256"],
                "monograph_page_count": int(source["page_count"]),
                "evidence_pages": sorted(selected_pages),
                "dpd_dataset_id": (
                    cohort["source_dpd_dataset"]["dataset_id"]
                ),
                "dpd_dataset_version": (
                    cohort["source_dpd_dataset"]["version"]
                ),
                "dpd_case_id": dpd_case_id,
                "drug_codes": json.loads(product["drug_codes"]),
                "din_list": json.loads(product["din_list"]),
                "unscored_alignment_fields": [
                    "din",
                    "schedule",
                    "product_status",
                    "company",
                ],
            }
            case_records.append(
                {
                    "case_id": (
                        f"hc_pm_{product['product_id'].removeprefix('hc_dpd_')}"
                        f"_{language}"
                    ),
                    "benchmark_version": PM_VERSION,
                    "product_id": product["product_id"],
                    "ingredient_group_id": product[
                        "ingredient_group_id"
                    ],
                    "drug_codes": product["drug_codes"],
                    "din_list": product["din_list"],
                    "brand_name": expected["brand_name"],
                    "language": language,
                    "split": product["split"],
                    "stratum": product["stratum"],
                    "document_complexity": _document_complexity(
                        int(source["page_count"])
                    ),
                    "monograph_id": source["monograph_id"],
                    "monograph_sha256": source["sha256"],
                    "source_page_count": int(source["page_count"]),
                    "evidence_page_count": len(selected_pages),
                    "input": _render_input(
                        language=language,
                        pages=pages,
                        selected_pages=selected_pages,
                    ),
                    "expected_output": json.dumps(
                        expected,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    "evaluation_type": "json",
                    "source_metadata": json.dumps(
                        source_metadata,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                }
            )

    cases = pd.DataFrame(case_records).sort_values("case_id")
    evidence = pd.DataFrame(evidence_records).sort_values(
        ["case_id", "field", "expected_item"]
    )
    products = products.sort_values("product_id")
    if len(cases) != 80 or cases["case_id"].duplicated().any():
        raise ValueError("Benchmark must contain 80 unique cases")
    if len(evidence) == 0:
        raise ValueError("Benchmark evidence cannot be empty")
    if set(cases["language"]) != {"en", "fr"}:
        raise ValueError("Benchmark must contain English and French cases")
    for output in cases["expected_output"]:
        parsed = json.loads(output)
        if set(parsed) != {
            "brand_name",
            "active_ingredients",
            "dosage_forms",
            "routes",
        }:
            raise ValueError("Expected output has the wrong schema")

    output_dir = root / PM_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    products_path = output_dir / "products.csv"
    cases_path = output_dir / "cases.csv.gz"
    evidence_path = output_dir / "field_evidence.csv.gz"
    report_path = output_dir / "build_report.json"
    products.to_csv(products_path, index=False, lineterminator="\n")
    cases.to_csv(
        cases_path,
        index=False,
        compression={"method": "gzip", "mtime": 0},
        lineterminator="\n",
    )
    evidence.to_csv(
        evidence_path,
        index=False,
        compression={"method": "gzip", "mtime": 0},
        lineterminator="\n",
    )
    report = {
        "report_schema_version": "1.0",
        "dataset_id": PM_DATASET_ID,
        "dataset_version": PM_VERSION,
        "source_documents": source_verification,
        "population": {
            "products": len(products),
            "cases": len(cases),
            "languages": cases["language"].value_counts().to_dict(),
            "splits": products["split"].value_counts().to_dict(),
            "strata": products["stratum"].value_counts().to_dict(),
            "document_complexity": (
                cases["document_complexity"].value_counts().to_dict()
            ),
        },
        "screening": {
            "documented_exclusions": len(exclusions),
            "by_stage": (
                exclusions["exclusion_stage"].value_counts().to_dict()
            ),
        },
        "evidence": {
            "field_items": len(evidence),
            "support_methods": (
                evidence["support_method"].value_counts().to_dict()
            ),
            "maximum_pages_per_case": int(
                cases["evidence_page_count"].max()
            ),
            "all_scored_items_have_source_pages": True,
            "automated_source_evidence_audit": "complete",
            "independent_human_signoff": "not_claimed",
        },
        "scoring_contract": {
            "scored_fields": [
                "brand_name",
                "active_ingredients",
                "dosage_forms",
                "routes",
            ],
            "unscored_alignment_fields": [
                "din",
                "schedule",
                "product_status",
                "company",
            ],
        },
        "software": {
            "evalanche_version": __version__,
            "pypdf_version": pypdf_version,
        },
        "artifacts": {},
    }
    report["artifacts"] = {
        "products": {
            "path": products_path.relative_to(root).as_posix(),
            "sha256": _sha256(products_path),
            "bytes": products_path.stat().st_size,
        },
        "cases": {
            "path": cases_path.relative_to(root).as_posix(),
            "sha256": _sha256(cases_path),
            "bytes": cases_path.stat().st_size,
        },
        "field_evidence": {
            "path": evidence_path.relative_to(root).as_posix(),
            "sha256": _sha256(evidence_path),
            "bytes": evidence_path.stat().st_size,
        },
        "source_documents": {
            "path": (root / sources_path).relative_to(root).as_posix(),
            "sha256": _sha256(root / sources_path),
            "bytes": (root / sources_path).stat().st_size,
        },
        "screening_exclusions": {
            "path": PM_EXCLUSIONS_PATH.as_posix(),
            "sha256": _sha256(root / PM_EXCLUSIONS_PATH),
            "bytes": (root / PM_EXCLUSIONS_PATH).stat().st_size,
        },
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest = _build_manifest(
        root=root,
        products=products,
        cases=cases,
        evidence=evidence,
        build_report_path=PM_OUTPUT_DIR / "build_report.json",
    )
    DatasetManifest.model_validate(manifest)
    manifest_path = root / PM_MANIFEST_PATH
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="\n") as file:
        yaml.safe_dump(
            manifest,
            file,
            sort_keys=False,
            allow_unicode=True,
        )
    verification = verify_dataset_manifest_file(
        manifest_path,
        root_path=root,
    )
    if not verification["valid"]:
        raise ValueError("Generated dataset manifest did not verify")
    return {
        "dataset_id": PM_DATASET_ID,
        "dataset_version": PM_VERSION,
        "product_count": int(len(products)),
        "case_count": int(len(cases)),
        "evidence_item_count": int(len(evidence)),
        "output_dir": output_dir,
        "manifest_path": manifest_path,
        "verification": verification,
    }


def _native_pdf_case_id(product_id: str, language: str) -> str:
    suffix = product_id.removeprefix("hc_dpd_")
    return f"hc_pm_pdf_{suffix}_{language}"


def _native_pdf_prompt(language: str) -> str:
    language_name = "English" if language == "en" else "French"
    return (
        "Read the complete attached official Health Canada Product "
        f"Monograph in {language_name}. Locate the requested product facts "
        "anywhere in the document. Return only one JSON object with exactly "
        "the keys brand_name, active_ingredients, dosage_forms, and routes. "
        "active_ingredients must be a list of objects with name, strength, "
        "and unit keys. dosage_forms and routes must be lists. Preserve the "
        "document language, omit trademark symbols, use numeric JSON values "
        "for strengths, combine distinct product variants covered by the "
        "monograph, use only the attached document, and do not explain your "
        "answer."
    )


def _review_item_id(
    *,
    source_case_id: str,
    field: str,
    expected_item: str,
    source_page: str,
) -> str:
    payload = json.dumps(
        [source_case_id, field, expected_item, source_page],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return "pm_review_" + hashlib.sha256(payload).hexdigest()[:16]


def _build_native_pdf_review_queue(
    *,
    root: Path,
    native_cases: pd.DataFrame,
    evidence: pd.DataFrame,
    sources: pd.DataFrame,
) -> pd.DataFrame:
    case_by_dpd_id: dict[str, dict[str, Any]] = {}
    for case in native_cases.to_dict(orient="records"):
        metadata = json.loads(str(case["source_metadata"]))
        case_by_dpd_id[str(metadata["dpd_case_id"])] = case

    source_by_product_language = {
        (str(row["product_id"]), str(row["language"])): row
        for row in sources.to_dict(orient="records")
    }
    records: list[dict[str, Any]] = []
    for item in evidence.to_dict(orient="records"):
        source_case_id = str(item["case_id"])
        case = case_by_dpd_id.get(source_case_id)
        if case is None:
            raise ValueError(
                "Evidence item does not map to a native-PDF case: "
                f"{source_case_id}"
            )
        source = source_by_product_language[
            (str(case["product_id"]), str(case["language"]))
        ]
        expected_item = str(item["expected_item"])
        source_page = str(item["source_page"])
        records.append(
            {
                "review_item_id": _review_item_id(
                    source_case_id=source_case_id,
                    field=str(item["field"]),
                    expected_item=expected_item,
                    source_page=source_page,
                ),
                "native_case_id": case["case_id"],
                "source_case_id": source_case_id,
                "product_id": case["product_id"],
                "language": case["language"],
                "split": case["split"],
                "field": item["field"],
                "expected_item": expected_item,
                "source_page": source_page,
                "automated_support_method": item["support_method"],
                "monograph_id": source["monograph_id"],
                "monograph_sha256": source["sha256"],
                "source_url": source["source_url"],
            }
        )

    queue = pd.DataFrame(records).sort_values(
        ["native_case_id", "field", "expected_item"],
        kind="stable",
    )
    if queue["review_item_id"].duplicated().any():
        raise ValueError("Native-PDF label review IDs must be unique")

    mutable_columns = [
        "human_review_status",
        "reviewer",
        "reviewed_at_utc",
        "notes",
    ]
    review_path = root / PM_NATIVE_PDF_REVIEW_PATH
    if review_path.is_file():
        previous = pd.read_csv(
            review_path,
            dtype=str,
            keep_default_na=False,
        )
        required = {"review_item_id", *mutable_columns}
        missing = required - set(previous.columns)
        if missing:
            raise ValueError(
                "Existing native-PDF label review is missing columns: "
                f"{sorted(missing)}"
            )
        if previous["review_item_id"].duplicated().any():
            raise ValueError(
                "Existing native-PDF label review has duplicate IDs"
            )
        if set(previous["review_item_id"]) != set(queue["review_item_id"]):
            raise ValueError(
                "Existing native-PDF label review does not match the "
                "current evidence inventory"
            )
        queue = queue.merge(
            previous[["review_item_id", *mutable_columns]],
            on="review_item_id",
            how="left",
            validate="one_to_one",
        )
    else:
        queue["human_review_status"] = "pending"
        queue["reviewer"] = ""
        queue["reviewed_at_utc"] = ""
        queue["notes"] = ""

    queue["human_review_status"] = (
        queue["human_review_status"]
        .fillna("pending")
        .replace("", "pending")
        .str.casefold()
    )
    invalid_statuses = sorted(
        set(queue["human_review_status"])
        - PM_NATIVE_PDF_REVIEW_STATUSES
    )
    if invalid_statuses:
        raise ValueError(
            "Native-PDF label review has invalid statuses: "
            f"{invalid_statuses}"
        )
    return queue


def _build_native_pdf_manifest(
    *,
    root: Path,
    products: pd.DataFrame,
    native_cases: pd.DataFrame,
    evidence: pd.DataFrame,
    review: pd.DataFrame,
) -> dict[str, Any]:
    development = products.loc[
        products["split"] == "development",
        "product_id",
    ].astype(str).tolist()
    heldout = products.loc[
        products["split"] == "heldout",
        "product_id",
    ].astype(str).tolist()
    files = [
        _manifest_file(
            root=root,
            path=PM_NATIVE_PDF_CASES_PATH,
            file_id="pm_native_pdf_cases",
            role="benchmark_cases",
            record_count=len(native_cases),
            media_type="application/gzip",
        ),
        _manifest_file(
            root=root,
            path=PM_OUTPUT_DIR / "products.csv",
            file_id="pm_native_pdf_products",
            role="sampling_membership",
            record_count=len(products),
            media_type="text/csv",
        ),
        _manifest_file(
            root=root,
            path=PM_SOURCES_PATH,
            file_id="pm_native_pdf_source_documents",
            role="source_lock",
            record_count=len(native_cases),
            media_type="text/csv",
        ),
        _manifest_file(
            root=root,
            path=PM_OUTPUT_DIR / "field_evidence.csv.gz",
            file_id="pm_native_pdf_field_evidence",
            role="automated_reference_evidence",
            record_count=len(evidence),
            media_type="application/gzip",
        ),
        _manifest_file(
            root=root,
            path=PM_NATIVE_PDF_REVIEW_PATH,
            file_id="pm_native_pdf_label_review",
            role="human_label_review_queue",
            record_count=len(review),
            media_type="text/csv",
        ),
        _manifest_file(
            root=root,
            path=PM_NATIVE_PDF_REPORT_PATH,
            file_id="pm_native_pdf_build_report",
            role="benchmark_build_report",
            record_count=1,
            media_type="application/json",
        ),
    ]
    return {
        "schema_version": "1.0",
        "release": {
            "dataset_id": PM_NATIVE_PDF_DATASET_ID,
            "version": PM_NATIVE_PDF_VERSION,
            "release_type": "benchmark",
            "title": (
                "Health Canada Product Monograph native-PDF extraction"
            ),
            "description": (
                "A draft bilingual benchmark that sends each complete "
                "official Product Monograph PDF to the model through a "
                "native file-input API."
            ),
            "created_at_utc": PM_NATIVE_PDF_RELEASE_CREATED_AT_UTC,
            "status": "draft",
            "immutable": False,
            "license_or_terms": (
                "Official Product Monographs are publicly posted by Health "
                "Canada. Copyright in individual monographs may remain with "
                "their sponsors. The repository stores source locks and "
                "descriptors, not source PDF bytes."
            ),
            "intended_use": (
                "Measure end-to-end full-document retrieval, visual reading, "
                "and structured extraction on English and French Product "
                "Monographs after the label review gate is complete."
            ),
            "limitations": [
                "This release is draft and must not produce a published "
                "leaderboard until human label review is complete.",
                "Reference labels originate in the frozen DPD alignment and "
                "have automated page-evidence checks, not independent human "
                "sign-off.",
                "Native PDF processing and tokenization can differ between "
                "providers, so results are comparable only within a declared "
                "input contract.",
                "DIN, regulatory schedule, product status, and sponsor are "
                "alignment metadata and are not scored.",
                "The 40-product purposive sample is not representative of "
                "every marketed drug or document format.",
            ],
            "languages": ["en", "fr"],
            "task_types": [
                "bilingual_structured_extraction",
                "native_pdf_document_understanding",
                "product_monograph_extraction",
            ],
        },
        "sources": [
            {
                "source_id": "hc_product_monographs",
                "name": "Health Canada Product Monograph PDF repository",
                "source_url": "https://pdf.hres.ca/dpd_pm/",
                "retrieved_at_utc": PM_RETRIEVED_AT_UTC,
                "source_modified_date": None,
                "license_or_terms": (
                    "Publicly posted official Product Monographs; individual "
                    "sponsor copyright may apply."
                ),
                "snapshot_notes": (
                    "Eighty official PDF URLs, SHA-256 values, byte sizes, "
                    "and page counts are frozen in source_documents.csv."
                ),
            },
            {
                "source_id": "evalanche_pm_builder",
                "name": "Evalanche Product Monograph benchmark builder",
                "source_url": (
                    "https://github.com/hc-sc-ocdo-bdpd/evalanche"
                ),
                "retrieved_at_utc": PM_RETRIEVED_AT_UTC,
                "source_modified_date": None,
                "license_or_terms": "Evalanche repository terms.",
                "snapshot_notes": (
                    "Derives full-PDF case descriptors and a human review "
                    "queue from the frozen evidence-window release."
                ),
            },
        ],
        "files": files,
        "sampling": {
            "method": "purposive",
            "unit": "product_family",
            "population_description": (
                "The same 40 bilingual marketed product families used by "
                "the evidence-window diagnostic, with one complete PDF per "
                "language."
            ),
            "target_count": len(products),
            "membership_file_id": "pm_native_pdf_products",
            "member_id_column": "product_id",
            "seed": 20260730,
            "inclusion_criteria": [
                "An exact hash-locked official English PDF is available.",
                "An exact hash-locked official French PDF is available.",
                "The monograph scope aligns with the selected DPD family.",
                "Every scored reference item has automated source-page "
                "evidence.",
            ],
            "exclusion_criteria": [
                "A language PDF is missing or duplicates the other language.",
                "The monograph covers unresolved sibling products or "
                "strengths.",
                "The PDF is malformed, image-only, or lacks scored evidence.",
            ],
            "strata": [
                "single_ingredient",
                "multi_ingredient",
                "multi_variant",
            ],
        },
        "splits": [
            {
                "name": "development",
                "purpose": "Prompt development and provider smoke tests.",
                "unit": "product_family",
                "target_count": len(development),
                "group_key": "ingredient_group_id",
                "selection_policy": (
                    "Inherited unchanged from the evidence-window release."
                ),
                "member_ids": development,
            },
            {
                "name": "heldout",
                "purpose": "Final prompt-locked comparison after promotion.",
                "unit": "product_family",
                "target_count": len(heldout),
                "group_key": "ingredient_group_id",
                "selection_policy": (
                    "Inherited unchanged with no ingredient-group overlap."
                ),
                "member_ids": heldout,
            },
        ],
        "lineage": {
            "parent_dataset_version": PM_VERSION,
            "code_version": __version__,
            "transformations": [
                "Reuse the frozen bilingual Product Monograph cohort and "
                "source lock.",
                "Replace label-selected text windows with hash-verified "
                "complete-PDF input descriptors.",
                "Preserve the existing four-field reference JSON and source "
                "evidence as provisional labels.",
                "Create a field-level human review queue required for "
                "benchmark promotion.",
            ],
        },
    }


def build_product_monograph_native_pdf_benchmark(
    *,
    root_path: str | Path = ".",
    source_cases_path: str | Path = PM_OUTPUT_DIR / "cases.csv.gz",
    sources_path: str | Path = PM_SOURCES_PATH,
    raw_dir: str | Path = PM_RAW_DIR,
) -> dict[str, Any]:
    """Build the draft full-document benchmark without bundling PDF bytes."""
    root = Path(root_path).resolve()
    parent_cases = pd.read_csv(root / source_cases_path, dtype=str)
    products = pd.read_csv(root / PM_OUTPUT_DIR / "products.csv", dtype=str)
    evidence = pd.read_csv(
        root / PM_OUTPUT_DIR / "field_evidence.csv.gz",
        dtype=str,
    )
    sources = pd.read_csv(root / sources_path, dtype=str)
    if len(parent_cases) != 80 or len(sources) != 80:
        raise ValueError(
            "Native-PDF benchmark requires 80 parent cases and 80 sources"
        )

    source_by_product_language = {
        (str(row["product_id"]), str(row["language"])): row
        for row in sources.to_dict(orient="records")
    }
    records: list[dict[str, Any]] = []
    for parent in parent_cases.to_dict(orient="records"):
        key = (str(parent["product_id"]), str(parent["language"]))
        source = source_by_product_language.get(key)
        if source is None:
            raise ValueError(
                "Parent case has no source-locked PDF: "
                f"{parent['case_id']}"
            )
        if int(source["byte_size"]) > 50 * 1024 * 1024:
            raise ValueError(
                f"Source PDF exceeds the 50 MB input limit: {source['source_url']}"
            )
        metadata = json.loads(str(parent["source_metadata"]))
        metadata.update(
            {
                "benchmark_input_mode": "native_full_pdf",
                "parent_evidence_window_case_id": parent["case_id"],
                "label_provenance": (
                    "dpd_alignment_with_automated_monograph_page_evidence"
                ),
                "human_label_review": "pending",
            }
        )
        file_path = Path(raw_dir) / _document_filename(source)
        record = dict(parent)
        record.update(
            {
                "case_id": _native_pdf_case_id(*key),
                "benchmark_version": PM_NATIVE_PDF_VERSION,
                "input_mode": "native_pdf",
                "input": _native_pdf_prompt(str(parent["language"])),
                "input_files": json.dumps(
                    [
                        {
                            "path": file_path.as_posix(),
                            "filename": file_path.name,
                            "media_type": "application/pdf",
                            "sha256": source["sha256"],
                        }
                    ],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "source_metadata": json.dumps(
                    metadata,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
        )
        records.append(record)

    native_cases = pd.DataFrame(records).sort_values("case_id", kind="stable")
    if len(native_cases) != 80 or native_cases["case_id"].duplicated().any():
        raise ValueError("Native-PDF benchmark must contain 80 unique cases")
    if native_cases["input"].str.contains("[PDF PAGE", regex=False).any():
        raise ValueError("Native-PDF prompts cannot contain evidence windows")

    output_dir = root / PM_NATIVE_PDF_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    native_cases.to_csv(
        root / PM_NATIVE_PDF_CASES_PATH,
        index=False,
        compression={"method": "gzip", "mtime": 0},
        lineterminator="\n",
    )
    review = _build_native_pdf_review_queue(
        root=root,
        native_cases=native_cases,
        evidence=evidence,
        sources=sources,
    )
    review.to_csv(
        root / PM_NATIVE_PDF_REVIEW_PATH,
        index=False,
        lineterminator="\n",
    )
    status_counts = review["human_review_status"].value_counts().to_dict()
    approved = int(status_counts.get("approved", 0))
    report = {
        "report_schema_version": "1.0",
        "dataset_id": PM_NATIVE_PDF_DATASET_ID,
        "dataset_version": PM_NATIVE_PDF_VERSION,
        "status": "draft",
        "population": {
            "products": int(len(products)),
            "cases": int(len(native_cases)),
            "languages": native_cases["language"].value_counts().to_dict(),
            "splits": native_cases["split"].value_counts().to_dict(),
        },
        "input_contract": {
            "mode": "native_full_pdf",
            "request_api": "responses",
            "files_per_case": 1,
            "pdfs_committed": False,
            "pdfs_required": int(len(native_cases)),
            "hash_verification_required_at_request_time": True,
        },
        "label_review": {
            "items": int(len(review)),
            "approved": approved,
            "status_counts": status_counts,
            "all_items_approved": approved == len(review),
        },
        "promotion_gates": {
            "human_label_review_complete": approved == len(review),
            "local_source_hash_verification_required": True,
            "provider_pdf_smoke_test_required": True,
            "heldout_prompt_lock_required": True,
        },
        "artifacts": {
            "cases": {
                "path": PM_NATIVE_PDF_CASES_PATH.as_posix(),
                "sha256": _sha256(root / PM_NATIVE_PDF_CASES_PATH),
                "bytes": (root / PM_NATIVE_PDF_CASES_PATH).stat().st_size,
            },
            "label_review": {
                "path": PM_NATIVE_PDF_REVIEW_PATH.as_posix(),
                "sha256": _sha256(root / PM_NATIVE_PDF_REVIEW_PATH),
                "bytes": (root / PM_NATIVE_PDF_REVIEW_PATH).stat().st_size,
            },
            "source_documents": {
                "path": Path(sources_path).as_posix(),
                "sha256": _sha256(root / sources_path),
                "bytes": (root / sources_path).stat().st_size,
            },
        },
    }
    (root / PM_NATIVE_PDF_REPORT_PATH).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest = _build_native_pdf_manifest(
        root=root,
        products=products,
        native_cases=native_cases,
        evidence=evidence,
        review=review,
    )
    DatasetManifest.model_validate(manifest)
    manifest_path = root / PM_NATIVE_PDF_MANIFEST_PATH
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="\n") as file:
        yaml.safe_dump(
            manifest,
            file,
            sort_keys=False,
            allow_unicode=True,
        )
    verification = verify_dataset_manifest_file(
        manifest_path,
        root_path=root,
    )
    if not verification["valid"]:
        raise ValueError("Generated native-PDF dataset manifest did not verify")
    return {
        "dataset_id": PM_NATIVE_PDF_DATASET_ID,
        "dataset_version": PM_NATIVE_PDF_VERSION,
        "status": "draft",
        "product_count": int(len(products)),
        "case_count": int(len(native_cases)),
        "review_item_count": int(len(review)),
        "approved_review_item_count": approved,
        "output_dir": output_dir,
        "manifest_path": manifest_path,
        "verification": verification,
    }

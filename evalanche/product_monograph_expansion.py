from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
import yaml
from pypdf import __version__ as pypdf_version

from evalanche import __version__
from evalanche.dataset_manifest import (
    DatasetManifest,
    verify_dataset_manifest_file,
)
from evalanche.product_monograph import (
    PM_MAX_EVIDENCE_PAGES,
    _apply_expected_overrides,
    _build_evidence,
    _document_complexity,
    _document_filename,
    _extract_pdf_pages,
    _load_yaml,
    _manifest_file,
    _native_pdf_prompt,
    _normalize,
    _render_input,
    _sha256,
)
from evalanche.registry import BenchmarkManifest


EXPANDED_PM_VERSION = "1.0.0"
EXPANSION_CONFIG_PATH = Path("configs/product_monograph/1.0.0/expansion.yaml")
EXPANSION_COHORT_PATH = Path("configs/product_monograph/1.0.0/cohort.yaml")
EXPANSION_OVERRIDES_PATH = Path("configs/product_monograph/1.0.0/label_overrides.yaml")
EXPANSION_OUTPUT_DIR = Path(
    "data/hc/benchmarks/product_monograph_structured_extraction/1.0.0"
)
EXPANSION_NATIVE_OUTPUT_DIR = Path(
    "data/hc/benchmarks/product_monograph_native_pdf_extraction/1.0.0"
)
EXPANSION_RAW_DIR = Path("data/hc/product_monographs/1.0.0/raw")
EXPANSION_SOURCE_PATH = EXPANSION_OUTPUT_DIR / "source_documents.csv"
EXPANSION_SCREENING_PATH = EXPANSION_OUTPUT_DIR / "screening_log.csv"
EXPANSION_FACT_AUDIT_PATH = EXPANSION_OUTPUT_DIR / "fact_audit.csv"
EXPANSION_PRODUCT_AUDIT_PATH = EXPANSION_OUTPUT_DIR / "product_audit.csv"
EXPANSION_AUDIT_SUMMARY_PATH = EXPANSION_OUTPUT_DIR / "audit_summary.json"
EXPANSION_AUDIT_GUIDE_PATH = EXPANSION_OUTPUT_DIR / "HUMAN_AUDIT_GUIDE.md"
EXPANSION_MANIFEST_PATH = Path(
    "configs/datasets/hc_product_monograph_structured_extraction_1.0.0_manifest.yaml"
)
EXPANSION_NATIVE_MANIFEST_PATH = Path(
    "configs/datasets/hc_product_monograph_native_pdf_extraction_1.0.0_manifest.yaml"
)
EXPANSION_BENCHMARK_PATH = Path(
    "configs/benchmarks/hc_product_monograph_structured_extraction_1.0.0.yaml"
)
EXPANSION_NATIVE_BENCHMARK_PATH = Path(
    "configs/benchmarks/hc_product_monograph_native_pdf_extraction_1.0.0.yaml"
)

AUDIT_STATUSES = {
    "pending",
    "approved",
    "needs_correction",
    "rejected",
}
PRODUCT_AUDIT_STATUS_COLUMNS = (
    "identity_review_status",
    "scope_review_status",
    "bilingual_review_status",
)
FACT_AUDIT_MUTABLE_COLUMNS = (
    "human_review_status",
    "reviewer",
    "reviewed_at_utc",
    "corrected_value",
    "notes",
)
PRODUCT_AUDIT_MUTABLE_COLUMNS = (
    *PRODUCT_AUDIT_STATUS_COLUMNS,
    "reviewer",
    "reviewed_at_utc",
    "notes",
)

_PDF_LINK = re.compile(
    r"https://pdf\.hres\.ca/dpd_pm/(?P<id>\d+)\.PDF",
    re.IGNORECASE,
)
_ENGLISH_MARKERS = (
    "product monograph",
    "health canada",
    "dosage and administration",
    "table of contents",
    "indications and clinical use",
)
_FRENCH_MARKERS = (
    "monographie de produit",
    "sante canada",
    "posologie et administration",
    "table des matieres",
    "indications et usage clinique",
)


@dataclass(frozen=True)
class ExpansionSpec:
    version: str
    seed: str
    target_products: int
    development_products: int
    heldout_products: int
    target_strata: dict[str, int]
    route_cycle: tuple[str, ...]
    source_dpd_products_path: Path
    source_dpd_cases_path: Path
    inherited_version: str
    inherited_products_path: Path
    inherited_sources_path: Path
    inherited_overrides_path: Path
    retrieved_at_utc: str
    request_timeout_seconds: float
    request_retries: int
    retry_backoff_seconds: float
    max_workers: int


@dataclass(frozen=True)
class FetchedDocument:
    content: bytes
    headers: dict[str, str]


@dataclass(frozen=True)
class ScreenedProduct:
    product: dict[str, Any]
    sources: list[dict[str, Any]]
    cases: list[dict[str, Any]]
    evidence: list[dict[str, Any]]


Fetch = Callable[[str, float], FetchedDocument]


def load_expansion_spec(
    path: str | Path = EXPANSION_CONFIG_PATH,
    *,
    root_path: str | Path = ".",
) -> ExpansionSpec:
    root = Path(root_path).resolve()
    config = _load_yaml(root / path)
    if config.get("schema_version") != "1.0":
        raise ValueError("Expansion config schema_version must be 1.0")
    version = str(config.get("version", ""))
    if version != EXPANDED_PM_VERSION:
        raise ValueError(f"Expansion config version must be {EXPANDED_PM_VERSION}")
    selection = config.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("Expansion config requires a selection mapping")
    target_strata = {
        str(key): int(value)
        for key, value in dict(selection.get("target_strata", {})).items()
    }
    expected_strata = {
        "single_ingredient",
        "multi_ingredient",
        "multi_variant",
        "multi_ingredient_multi_variant",
    }
    if set(target_strata) != expected_strata:
        raise ValueError(
            "Expansion target_strata must contain exactly "
            + ", ".join(sorted(expected_strata))
        )
    target_products = int(selection["target_products"])
    development_products = int(selection["development_products"])
    heldout_products = int(selection["heldout_products"])
    if target_products != development_products + heldout_products:
        raise ValueError("Development and held-out counts must equal target")
    if sum(target_strata.values()) != target_products:
        raise ValueError("Stratum quotas must equal the target product count")
    if target_products != 200 or development_products != 40:
        raise ValueError(
            "The 1.0.0 expansion contract requires 200 products, including "
            "the 40 exposed pilot products as development data"
        )

    source = config.get("source_dpd_dataset")
    inherited = config.get("inherit")
    network = config.get("network", {})
    if not isinstance(source, dict) or not isinstance(inherited, dict):
        raise ValueError("Expansion config requires source and inherit mappings")
    route_cycle = tuple(str(value) for value in selection["route_cycle"])
    if not route_cycle:
        raise ValueError("route_cycle cannot be empty")
    return ExpansionSpec(
        version=version,
        seed=str(selection["seed"]),
        target_products=target_products,
        development_products=development_products,
        heldout_products=heldout_products,
        target_strata=target_strata,
        route_cycle=route_cycle,
        source_dpd_products_path=Path(source["products_path"]),
        source_dpd_cases_path=Path(source["cases_path"]),
        inherited_version=str(inherited["version"]),
        inherited_products_path=Path(inherited["products_path"]),
        inherited_sources_path=Path(inherited["sources_path"]),
        inherited_overrides_path=Path(inherited["overrides_path"]),
        retrieved_at_utc=str(config["retrieved_at_utc"]),
        request_timeout_seconds=float(network.get("timeout_seconds", 120.0)),
        request_retries=int(network.get("retries", 3)),
        retry_backoff_seconds=float(network.get("retry_backoff_seconds", 1.0)),
        max_workers=int(network.get("max_workers", 8)),
    )


def _default_fetch(url: str, timeout: float) -> FetchedDocument:
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Evalanche Product Monograph benchmark expansion "
                "(reproducible public-source research)"
            )
        },
    )
    with urlopen(request, timeout=timeout) as response:
        return FetchedDocument(
            content=response.read(),
            headers={key.casefold(): value for key, value in response.headers.items()},
        )


def _fetch_with_retries(
    *,
    url: str,
    spec: ExpansionSpec,
    fetch: Fetch,
) -> FetchedDocument:
    last_error: Exception | None = None
    for attempt in range(spec.request_retries):
        try:
            return fetch(url, spec.request_timeout_seconds)
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            last_error = error
            if attempt + 1 < spec.request_retries:
                time.sleep(spec.retry_backoff_seconds * (2**attempt))
    raise OSError(f"Could not retrieve {url}: {last_error}") from last_error


def _cache_key(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def _fetch_cached_html(
    *,
    root: Path,
    url: str,
    spec: ExpansionSpec,
    fetch: Fetch,
) -> str:
    cache_dir = root / EXPANSION_RAW_DIR / "discovery"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{_cache_key(url)}.html"
    if cache_path.is_file():
        return cache_path.read_text(encoding="utf-8", errors="replace")
    document = _fetch_with_retries(url=url, spec=spec, fetch=fetch)
    text = document.content.decode("utf-8", errors="replace")
    cache_path.write_text(text, encoding="utf-8")
    return text


def _dpd_info_url(drug_code: str, language: str) -> str:
    query_language = "eng" if language == "en" else "fra"
    return (
        "https://health-products.canada.ca/dpd-bdpp/info?"
        f"lang={query_language}&code={drug_code}"
    )


def _discover_monograph(
    *,
    root: Path,
    drug_code: str,
    language: str,
    spec: ExpansionSpec,
    fetch: Fetch,
) -> tuple[str, str] | None:
    info_url = _dpd_info_url(drug_code, language)
    html = _fetch_cached_html(
        root=root,
        url=info_url,
        spec=spec,
        fetch=fetch,
    )
    matches = {
        (match.group("id"), match.group(0)) for match in _PDF_LINK.finditer(html)
    }
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError(f"DPD page lists multiple Product Monographs: {info_url}")
    return next(iter(matches))


def _seeded_order_key(seed: str, product_id: str) -> str:
    return hashlib.sha256(f"{seed}|{product_id}".encode("utf-8")).hexdigest()


def _route_group(expected_output: str) -> str:
    routes = {
        _normalize(route) for route in json.loads(expected_output).get("routes", [])
    }
    if routes & {
        "intravenous",
        "intramuscular",
        "subcutaneous",
        "intradermal",
        "intrathecal",
        "epidural",
        "intra arterial",
        "intra articular",
        "intravitreal",
        "block infiltration",
    }:
        return "parenteral"
    if routes & {"topical", "transdermal", "percutaneous"}:
        return "topical"
    if routes & {"inhalation", "nasal"}:
        return "inhaled_or_nasal"
    if routes & {"ophthalmic", "otic"}:
        return "ophthalmic_or_otic"
    if "oral" in routes:
        return "oral"
    return "other"


def _coverage_order(
    candidates: pd.DataFrame,
    *,
    seed: str,
    route_cycle: tuple[str, ...],
) -> list[dict[str, Any]]:
    buckets: dict[str, deque[dict[str, Any]]] = {}
    for route_group, group in candidates.groupby("route_group", sort=True):
        records = group.to_dict(orient="records")
        records.sort(key=lambda row: _seeded_order_key(seed, row["product_id"]))
        buckets[str(route_group)] = deque(records)
    ordered: list[dict[str, Any]] = []
    while any(buckets.values()):
        progress = False
        for route_group in route_cycle:
            bucket = buckets.get(route_group)
            if bucket:
                ordered.append(bucket.popleft())
                progress = True
        if not progress:
            remaining = [row for bucket in buckets.values() for row in bucket]
            remaining.sort(key=lambda row: _seeded_order_key(seed, row["product_id"]))
            ordered.extend(remaining)
            break
    return ordered


def _validate_document_language(pages: list[str], language: str) -> dict[str, Any]:
    text = _normalize("\n".join(pages[:12]))
    english_score = sum(marker in text for marker in _ENGLISH_MARKERS)
    french_score = sum(marker in text for marker in _FRENCH_MARKERS)
    expected_score = english_score if language == "en" else french_score
    other_score = french_score if language == "en" else english_score
    passed = expected_score >= 1 and expected_score >= other_score
    return {
        "passed": passed,
        "english_marker_count": english_score,
        "french_marker_count": french_score,
    }


def _evidence_excerpt(page_text: str, expected_item: str, field: str) -> str:
    needle = expected_item
    if field == "active_ingredients":
        try:
            needle = str(json.loads(expected_item)["name"])
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    compact = re.sub(r"\s+", " ", page_text).strip()
    index = compact.casefold().find(needle.casefold())
    if index < 0:
        return compact[:900]
    start = max(0, index - 300)
    end = min(len(compact), index + len(needle) + 600)
    return compact[start:end]


def _download_and_extract(
    *,
    root: Path,
    product_id: str,
    language: str,
    monograph_id: str,
    source_url: str,
    spec: ExpansionSpec,
    fetch: Fetch,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    raw_dir = root / EXPANSION_RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "product_id": product_id,
        "language": language,
        "monograph_id": monograph_id,
    }
    path = raw_dir / _document_filename(row)
    headers: dict[str, str] = {}
    if path.is_file() and (expected_sha256 is None or _sha256(path) == expected_sha256):
        content = path.read_bytes()
    else:
        document = _fetch_with_retries(url=source_url, spec=spec, fetch=fetch)
        content = document.content
        headers = document.headers
        if not content.startswith(b"%PDF"):
            raise ValueError(f"Source is not a PDF: {source_url}")
        path.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(
            f"Frozen source hash changed for {product_id} {language}: "
            f"expected {expected_sha256}, found {digest}"
        )
    pages = _extract_pdf_pages(path, digest)
    if not any(page.strip() for page in pages):
        raise ValueError("PDF contains no extractable text")
    language_check = _validate_document_language(pages, language)
    if not language_check["passed"]:
        raise ValueError(
            "PDF language could not be validated "
            f"(English markers={language_check['english_marker_count']}, "
            f"French markers={language_check['french_marker_count']})"
        )
    source = {
        "product_id": product_id,
        "language": language,
        "monograph_id": monograph_id,
        "source_url": source_url,
        "dpd_info_url": "",
        "retrieved_at_utc": spec.retrieved_at_utc,
        "source_last_modified": headers.get("last-modified", ""),
        "sha256": digest,
        "byte_size": len(content),
        "page_count": len(pages),
        "text_extraction_status": "passed",
        "language_validation_status": "passed_automated",
        "scope_alignment_status": "pending_human_audit",
    }
    return source, pages


def _expected_case(
    *,
    dpd_cases: pd.DataFrame,
    product_id: str,
    language: str,
    overrides: dict[str, Any],
) -> tuple[pd.Series, dict[str, Any]]:
    case_id = f"{product_id}_{language}"
    matches = dpd_cases[
        (dpd_cases["product_id"] == product_id) & (dpd_cases["language"] == language)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one DPD alignment case: {case_id}")
    row = matches.iloc[0]
    expected = _apply_expected_overrides(
        source_expected=json.loads(row["expected_output"]),
        case_id=case_id,
        overrides=overrides,
    )
    expected["active_ingredients"] = list(
        {
            json.dumps(item, ensure_ascii=False, sort_keys=True): item
            for item in expected["active_ingredients"]
        }.values()
    )
    for field in ("dosage_forms", "routes"):
        expected[field] = list(dict.fromkeys(expected[field]))
    return row, expected


def _build_case(
    *,
    product: dict[str, Any],
    source: dict[str, Any],
    pages: list[str],
    dpd_cases: pd.DataFrame,
    overrides: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    product_id = str(product["product_id"])
    language = str(source["language"])
    _, expected = _expected_case(
        dpd_cases=dpd_cases,
        product_id=product_id,
        language=language,
        overrides=overrides,
    )
    source_case_id = f"{product_id}_{language}"
    evidence, selected_pages = _build_evidence(
        case_id=source_case_id,
        expected=expected,
        pages=pages,
        overrides=overrides,
    )
    for item in evidence:
        page = int(item["source_page"])
        item["evidence_excerpt"] = _evidence_excerpt(
            pages[page - 1],
            str(item["expected_item"]),
            str(item["field"]),
        )
    route_group = str(product["route_group"])
    source_metadata = {
        "source_type": "official_product_monograph",
        "source_url": source["source_url"],
        "monograph_id": source["monograph_id"],
        "monograph_sha256": source["sha256"],
        "monograph_page_count": int(source["page_count"]),
        "evidence_pages": sorted(selected_pages),
        "dpd_dataset_id": "hc_dpd_structured_extraction_census",
        "dpd_dataset_version": "0.2.0",
        "dpd_case_id": source_case_id,
        "drug_codes": json.loads(str(product["drug_codes"])),
        "din_list": json.loads(str(product["din_list"])),
        "selection_seed": "20260811",
        "human_audit_status": "pending",
        "unscored_alignment_fields": [
            "din",
            "schedule",
            "product_status",
            "company",
        ],
    }
    case = {
        "case_id": (f"hc_pm_{product_id.removeprefix('hc_dpd_')}_{language}"),
        "benchmark_version": EXPANDED_PM_VERSION,
        "product_id": product_id,
        "ingredient_group_id": product["ingredient_group_id"],
        "drug_codes": product["drug_codes"],
        "din_list": product["din_list"],
        "brand_name": expected["brand_name"],
        "language": language,
        "split": product["split"],
        "stratum": product["stratum"],
        "route_group": route_group,
        "document_complexity": _document_complexity(int(source["page_count"])),
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
    return case, evidence


def _screen_candidate(
    *,
    root: Path,
    product: dict[str, Any],
    dpd_cases: pd.DataFrame,
    overrides: dict[str, Any],
    spec: ExpansionSpec,
    fetch: Fetch,
) -> ScreenedProduct:
    product_id = str(product["product_id"])
    drug_codes = [str(value) for value in json.loads(str(product["drug_codes"]))]
    if not drug_codes:
        raise ValueError("DPD family has no drug codes")
    discoveries: dict[str, tuple[str, str]] = {}
    for language in ("en", "fr"):
        links: set[tuple[str, str]] = set()
        missing_codes: list[str] = []
        for drug_code in drug_codes:
            result = _discover_monograph(
                root=root,
                drug_code=drug_code,
                language=language,
                spec=spec,
                fetch=fetch,
            )
            if result is None:
                missing_codes.append(drug_code)
            else:
                links.add(result)
        if missing_codes:
            raise ValueError(
                f"{language} Product Monograph missing for DPD codes "
                + ", ".join(missing_codes)
            )
        if len(links) != 1:
            raise ValueError(
                f"{language} DPD variants resolve to {len(links)} documents"
            )
        discoveries[language] = next(iter(links))
    if discoveries["en"][0] == discoveries["fr"][0]:
        raise ValueError("English and French pages resolve to the same PDF")

    sources: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    hashes: set[str] = set()
    for language in ("en", "fr"):
        monograph_id, source_url = discoveries[language]
        source, pages = _download_and_extract(
            root=root,
            product_id=product_id,
            language=language,
            monograph_id=monograph_id,
            source_url=source_url,
            spec=spec,
            fetch=fetch,
        )
        if source["sha256"] in hashes:
            raise ValueError("English and French PDFs have identical hashes")
        hashes.add(str(source["sha256"]))
        source["dpd_info_url"] = _dpd_info_url(drug_codes[0], language)
        case, case_evidence = _build_case(
            product=product,
            source=source,
            pages=pages,
            dpd_cases=dpd_cases,
            overrides=overrides,
        )
        sources.append(source)
        cases.append(case)
        evidence.extend(case_evidence)
    return ScreenedProduct(
        product=product,
        sources=sources,
        cases=cases,
        evidence=evidence,
    )


def _load_inherited_products(
    *,
    root: Path,
    spec: ExpansionSpec,
    dpd_products: pd.DataFrame,
    dpd_cases: pd.DataFrame,
    overrides: dict[str, Any],
    fetch: Fetch,
) -> list[ScreenedProduct]:
    inherited_products = pd.read_csv(root / spec.inherited_products_path, dtype=str)
    inherited_sources = pd.read_csv(root / spec.inherited_sources_path, dtype=str)

    def load_one(inherited: dict[str, Any]) -> ScreenedProduct:
        product_id = str(inherited["product_id"])
        matches = dpd_products[dpd_products["product_id"] == product_id]
        if len(matches) != 1:
            raise ValueError(f"Inherited product missing from DPD: {product_id}")
        product = matches.iloc[0].to_dict()
        en_case = dpd_cases[
            (dpd_cases["product_id"] == product_id) & (dpd_cases["language"] == "en")
        ].iloc[0]
        product.update(
            {
                "split": "development",
                "selection_status": "inherited_development",
                "route_group": _route_group(str(en_case["expected_output"])),
            }
        )
        sources: list[dict[str, Any]] = []
        cases: list[dict[str, Any]] = []
        evidence: list[dict[str, Any]] = []
        for locked in inherited_sources[
            inherited_sources["product_id"] == product_id
        ].to_dict(orient="records"):
            source, pages = _download_and_extract(
                root=root,
                product_id=product_id,
                language=str(locked["language"]),
                monograph_id=str(locked["monograph_id"]),
                source_url=str(locked["source_url"]),
                spec=spec,
                fetch=fetch,
                expected_sha256=str(locked["sha256"]),
            )
            source["dpd_info_url"] = str(locked["dpd_info_url"])
            source["retrieved_at_utc"] = str(locked["retrieved_at_utc"])
            source["scope_alignment_status"] = "pending_human_audit"
            case, case_evidence = _build_case(
                product=product,
                source=source,
                pages=pages,
                dpd_cases=dpd_cases,
                overrides=overrides,
            )
            sources.append(source)
            cases.append(case)
            evidence.extend(case_evidence)
        if len(sources) != 2:
            raise ValueError(f"Inherited product lacks a bilingual pair: {product_id}")
        return ScreenedProduct(
            product=product,
            sources=sources,
            cases=cases,
            evidence=evidence,
        )

    records = inherited_products.to_dict(orient="records")
    with ThreadPoolExecutor(max_workers=spec.max_workers) as executor:
        return list(executor.map(load_one, records))


def _audit_id(prefix: str, values: list[str]) -> str:
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    return prefix + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _merge_review_columns(
    *,
    current: pd.DataFrame,
    path: Path | None,
    id_column: str,
    mutable_columns: tuple[str, ...],
    defaults: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    for column, default in defaults.items():
        current[column] = default
    changes = {
        "preserved": 0,
        "reset_or_new": len(current),
        "removed": 0,
    }
    if path is None or not path.is_file():
        return current, changes
    previous = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {id_column, *mutable_columns}
    missing = required - set(previous.columns)
    if missing:
        raise ValueError(f"Existing audit file is missing columns: {sorted(missing)}")
    immutable_columns = [
        column for column in current.columns if column not in mutable_columns
    ]
    previous_by_id = previous.set_index(id_column, drop=False)
    preserved = 0
    for index, row in current.iterrows():
        item_id = str(row[id_column])
        if item_id not in previous_by_id.index:
            continue
        old = previous_by_id.loc[item_id]
        if isinstance(old, pd.DataFrame):
            immutable_match = (
                old[immutable_columns]
                .astype(str)
                .eq(row[immutable_columns].astype(str), axis="columns")
                .all(axis=1)
            )
            mutable_versions = old.loc[
                immutable_match, list(mutable_columns)
            ].drop_duplicates()
            if len(mutable_versions) != 1:
                raise ValueError(
                    "Existing duplicate audit rows disagree or no longer "
                    f"match: {item_id}"
                )
            old = old.loc[immutable_match].iloc[0]
        if all(
            str(old.get(column, "")) == str(row[column]) for column in immutable_columns
        ):
            for column in mutable_columns:
                current.at[index, column] = str(old[column])
            preserved += 1
    changes = {
        "preserved": preserved,
        "reset_or_new": len(current) - preserved,
        "removed": len(set(previous[id_column]) - set(current[id_column])),
    }
    return current, changes


def _build_fact_audit(
    *,
    evidence: pd.DataFrame,
    cases: pd.DataFrame,
    sources: pd.DataFrame,
    existing_path: Path | None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    case_by_source_id: dict[str, dict[str, Any]] = {}
    for case in cases.to_dict(orient="records"):
        metadata = json.loads(str(case["source_metadata"]))
        case_by_source_id[str(metadata["dpd_case_id"])] = case
    source_by_key = {
        (str(row["product_id"]), str(row["language"])): row
        for row in sources.to_dict(orient="records")
    }
    rows: list[dict[str, Any]] = []
    for item in evidence.to_dict(orient="records"):
        source_case_id = str(item["case_id"])
        case = case_by_source_id[source_case_id]
        source = source_by_key[(str(case["product_id"]), str(case["language"]))]
        rows.append(
            {
                "audit_item_id": _audit_id(
                    "pm_fact_",
                    [
                        source_case_id,
                        str(item["field"]),
                        str(item["expected_item"]),
                        str(item["source_page"]),
                        str(source["sha256"]),
                    ],
                ),
                "case_id": case["case_id"],
                "source_case_id": source_case_id,
                "product_id": case["product_id"],
                "brand_name": case["brand_name"],
                "language": case["language"],
                "split": case["split"],
                "stratum": case["stratum"],
                "route_group": case["route_group"],
                "field": item["field"],
                "expected_item": item["expected_item"],
                "source_page": item["source_page"],
                "evidence_excerpt": item["evidence_excerpt"],
                "automated_support_method": item["support_method"],
                "monograph_id": source["monograph_id"],
                "monograph_sha256": source["sha256"],
                "source_url": source["source_url"],
            }
        )
    audit = pd.DataFrame(rows).sort_values(
        ["product_id", "language", "field", "expected_item"],
        kind="stable",
    )
    return _merge_review_columns(
        current=audit,
        path=existing_path,
        id_column="audit_item_id",
        mutable_columns=FACT_AUDIT_MUTABLE_COLUMNS,
        defaults={
            "human_review_status": "pending",
            "reviewer": "",
            "reviewed_at_utc": "",
            "corrected_value": "",
            "notes": "",
        },
    )


def _build_product_audit(
    *,
    products: pd.DataFrame,
    cases: pd.DataFrame,
    sources: pd.DataFrame,
    existing_path: Path | None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    case_lookup = {
        (str(row["product_id"]), str(row["language"])): row
        for row in cases.to_dict(orient="records")
    }
    source_lookup = {
        (str(row["product_id"]), str(row["language"])): row
        for row in sources.to_dict(orient="records")
    }
    rows: list[dict[str, Any]] = []
    for product in products.to_dict(orient="records"):
        product_id = str(product["product_id"])
        en_case = case_lookup[(product_id, "en")]
        fr_case = case_lookup[(product_id, "fr")]
        en_source = source_lookup[(product_id, "en")]
        fr_source = source_lookup[(product_id, "fr")]
        rows.append(
            {
                "audit_product_id": _audit_id(
                    "pm_product_",
                    [
                        product_id,
                        str(en_source["sha256"]),
                        str(fr_source["sha256"]),
                        str(en_case["expected_output"]),
                        str(fr_case["expected_output"]),
                    ],
                ),
                "product_id": product_id,
                "brand_name": product["brand_name"],
                "company_name": product["company_name"],
                "split": product["split"],
                "stratum": product["stratum"],
                "route_group": product["route_group"],
                "drug_codes": product["drug_codes"],
                "din_list": product["din_list"],
                "expected_output_en": en_case["expected_output"],
                "expected_output_fr": fr_case["expected_output"],
                "en_monograph_id": en_source["monograph_id"],
                "en_sha256": en_source["sha256"],
                "en_source_url": en_source["source_url"],
                "fr_monograph_id": fr_source["monograph_id"],
                "fr_sha256": fr_source["sha256"],
                "fr_source_url": fr_source["source_url"],
                "automated_bilingual_pair_check": "passed",
                "automated_variant_link_check": "passed",
                "automated_fact_evidence_check": "passed",
            }
        )
    audit = pd.DataFrame(rows).sort_values("product_id", kind="stable")
    return _merge_review_columns(
        current=audit,
        path=existing_path,
        id_column="audit_product_id",
        mutable_columns=PRODUCT_AUDIT_MUTABLE_COLUMNS,
        defaults={
            "identity_review_status": "pending",
            "scope_review_status": "pending",
            "bilingual_review_status": "pending",
            "reviewer": "",
            "reviewed_at_utc": "",
            "notes": "",
        },
    )


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        yaml.safe_dump(value, file, sort_keys=False, allow_unicode=True)


def _write_cohort(
    *,
    root: Path,
    spec: ExpansionSpec,
    products: pd.DataFrame,
) -> None:
    cohort = {
        "schema_version": "1.0",
        "dataset_id": "hc_product_monograph_structured_extraction",
        "version": spec.version,
        "source_dpd_dataset": {
            "dataset_id": "hc_dpd_structured_extraction_census",
            "version": "0.2.0",
            "products_path": spec.source_dpd_products_path.as_posix(),
            "cases_path": spec.source_dpd_cases_path.as_posix(),
        },
        "selection": {
            "seed": spec.seed,
            "method": (
                "Inherit the 40 exposed pilot families as development data; "
                "select 160 new held-out families by declared stratum and "
                "deterministic route-coverage order; require distinct official "
                "English and French PDFs, one consistent document link across "
                "every DPD variant, language validation, extractable text, and "
                "automated page evidence for every provisional fact; represent "
                "each ingredient group exactly once."
            ),
            "product_count": spec.target_products,
            "language_case_count": spec.target_products * 2,
            "development_products": spec.development_products,
            "heldout_products": spec.heldout_products,
            "strata": spec.target_strata,
        },
        "products": [
            {
                "product_id": str(row["product_id"]),
                "split": str(row["split"]),
                "stratum": str(row["stratum"]),
                "route_group": str(row["route_group"]),
                "selection_status": str(row["selection_status"]),
            }
            for row in products.to_dict(orient="records")
        ],
    }
    _write_yaml(root / EXPANSION_COHORT_PATH, cohort)


def _write_overrides(root: Path, spec: ExpansionSpec) -> dict[str, Any]:
    target = root / EXPANSION_OVERRIDES_PATH
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / spec.inherited_overrides_path, target)
    return _load_yaml(target)


def _validate_release_tables(
    *,
    spec: ExpansionSpec,
    products: pd.DataFrame,
    sources: pd.DataFrame,
    cases: pd.DataFrame,
    evidence: pd.DataFrame,
) -> None:
    required_columns = {
        "products": {
            "product_id",
            "ingredient_group_id",
            "split",
            "stratum",
            "route_group",
        },
        "sources": {
            "product_id",
            "language",
            "monograph_id",
            "source_url",
            "sha256",
            "byte_size",
            "page_count",
        },
        "cases": {
            "case_id",
            "product_id",
            "language",
            "monograph_id",
            "monograph_sha256",
            "expected_output",
            "evaluation_type",
            "source_metadata",
        },
        "evidence": {
            "case_id",
            "field",
            "expected_item",
            "source_page",
            "support_method",
            "evidence_excerpt",
        },
    }
    for label, table in (
        ("products", products),
        ("sources", sources),
        ("cases", cases),
        ("evidence", evidence),
    ):
        if missing := sorted(required_columns[label] - set(table.columns)):
            raise ValueError(f"Expanded {label} are missing columns: {missing}")
    if len(products) != spec.target_products:
        raise ValueError(
            f"Expected {spec.target_products} products, found {len(products)}"
        )
    if len(sources) != spec.target_products * 2:
        raise ValueError("Expanded release must contain two sources per product")
    if len(cases) != spec.target_products * 2:
        raise ValueError("Expanded release must contain two cases per product")
    if products["product_id"].duplicated().any():
        raise ValueError("Expanded product IDs must be unique")
    if products["ingredient_group_id"].duplicated().any():
        raise ValueError(
            "Expanded ingredient groups must be unique across product families"
        )
    if cases["case_id"].duplicated().any():
        raise ValueError("Expanded case IDs must be unique")
    if sources["sha256"].duplicated().any():
        raise ValueError("Expanded source PDF hashes must be unique")
    if sources[["product_id", "language"]].duplicated().any():
        raise ValueError("Expanded source product-language pairs must be unique")
    if cases[["product_id", "language"]].duplicated().any():
        raise ValueError("Expanded case product-language pairs must be unique")
    if set(sources["language"]) != {"en", "fr"}:
        raise ValueError("Expanded sources must contain only English and French")
    if set(cases["language"]) != {"en", "fr"}:
        raise ValueError("Expanded cases must contain only English and French")
    pairs = sources.groupby("product_id")["language"].agg(set)
    if any(value != {"en", "fr"} for value in pairs):
        raise ValueError("Every expanded product must have one bilingual pair")
    case_pairs = cases.groupby("product_id")["language"].agg(set)
    if any(value != {"en", "fr"} for value in case_pairs):
        raise ValueError("Every expanded product must have two bilingual cases")
    product_ids = set(products["product_id"].astype(str))
    if set(sources["product_id"].astype(str)) != product_ids:
        raise ValueError("Expanded sources and products have different membership")
    if set(cases["product_id"].astype(str)) != product_ids:
        raise ValueError("Expanded cases and products have different membership")
    if not sources["sha256"].str.fullmatch(r"[0-9a-f]{64}").all():
        raise ValueError("Expanded source hashes must be lowercase SHA-256 values")
    if (
        not sources["source_url"]
        .map(lambda value: bool(_PDF_LINK.fullmatch(value)))
        .all()
    ):
        raise ValueError("Expanded sources must use official Health Canada PDF URLs")
    observed_strata = products["stratum"].value_counts().to_dict()
    if observed_strata != spec.target_strata:
        raise ValueError(f"Expanded stratum quotas differ: {observed_strata}")
    split_counts = products["split"].value_counts().to_dict()
    if split_counts != {
        "heldout": spec.heldout_products,
        "development": spec.development_products,
    }:
        raise ValueError(f"Expanded split quotas differ: {split_counts}")
    development_groups = set(
        products.loc[products["split"] == "development", "ingredient_group_id"]
    )
    heldout_groups = set(
        products.loc[products["split"] == "heldout", "ingredient_group_id"]
    )
    if development_groups & heldout_groups:
        raise ValueError("Ingredient groups cross development and held-out")
    if evidence.empty:
        raise ValueError("Expanded field evidence cannot be empty")
    if cases["evaluation_type"].astype(str).ne("json").any():
        raise ValueError("Expanded cases must use JSON evaluation")
    if int(cases["evidence_page_count"].max()) > PM_MAX_EVIDENCE_PAGES:
        raise ValueError("Expanded evidence page limit was exceeded")

    source_lookup = {
        (str(row["product_id"]), str(row["language"])): row
        for row in sources.to_dict(orient="records")
    }
    expected_facts: set[tuple[str, str, str]] = set()
    source_case_to_page_count: dict[str, int] = {}
    for case in cases.to_dict(orient="records"):
        key = (str(case["product_id"]), str(case["language"]))
        source = source_lookup[key]
        if str(case["monograph_id"]) != str(source["monograph_id"]):
            raise ValueError(
                f"Case monograph ID differs from source: {case['case_id']}"
            )
        if str(case["monograph_sha256"]) != str(source["sha256"]):
            raise ValueError(f"Case source hash differs: {case['case_id']}")
        metadata = json.loads(str(case["source_metadata"]))
        source_case_id = str(metadata["dpd_case_id"])
        if source_case_id in source_case_to_page_count:
            raise ValueError(f"Duplicate DPD source case ID: {source_case_id}")
        source_case_to_page_count[source_case_id] = int(source["page_count"])
        if str(metadata["monograph_sha256"]) != str(source["sha256"]):
            raise ValueError(f"Case metadata source hash differs: {case['case_id']}")
        output = json.loads(str(case["expected_output"]))
        if set(output) != {
            "brand_name",
            "active_ingredients",
            "dosage_forms",
            "routes",
        }:
            raise ValueError(f"Unexpected output fields: {case['case_id']}")
        flattened = {
            "brand_name": [str(output["brand_name"])],
            "active_ingredients": [
                json.dumps(item, ensure_ascii=False, separators=(",", ":"))
                for item in output["active_ingredients"]
            ],
            "dosage_forms": [str(item) for item in output["dosage_forms"]],
            "routes": [str(item) for item in output["routes"]],
        }
        for field, values in flattened.items():
            if not values:
                raise ValueError(f"Empty expected field {field}: {case['case_id']}")
            if len(values) != len(set(values)):
                raise ValueError(
                    f"Duplicate expected values in {field}: {case['case_id']}"
                )
            expected_facts.update((source_case_id, field, value) for value in values)

    observed_facts = [
        (str(row["case_id"]), str(row["field"]), str(row["expected_item"]))
        for row in evidence.to_dict(orient="records")
    ]
    if len(observed_facts) != len(set(observed_facts)):
        raise ValueError("Expanded field evidence contains duplicate fact rows")
    if set(observed_facts) != expected_facts:
        missing = len(expected_facts - set(observed_facts))
        extra = len(set(observed_facts) - expected_facts)
        raise ValueError(
            f"Expanded evidence differs from expected outputs: {missing} missing, "
            f"{extra} extra"
        )
    for row in evidence.to_dict(orient="records"):
        source_case_id = str(row["case_id"])
        source_page = int(row["source_page"])
        if not 1 <= source_page <= source_case_to_page_count[source_case_id]:
            raise ValueError(
                f"Evidence page is outside the source PDF: {source_case_id}"
            )
        if not str(row["evidence_excerpt"]).strip():
            raise ValueError(f"Evidence excerpt is blank: {source_case_id}")


def _dataset_file(
    *,
    root: Path,
    path: Path,
    file_id: str,
    role: str,
    records: int,
    media_type: str,
) -> dict[str, Any]:
    return _manifest_file(
        root=root,
        path=path,
        file_id=file_id,
        role=role,
        record_count=records,
        media_type=media_type,
    )


def _base_dataset_manifest(
    *,
    spec: ExpansionSpec,
    products: pd.DataFrame,
    files: list[dict[str, Any]],
    native_pdf: bool,
) -> dict[str, Any]:
    dataset_id = (
        "hc_product_monograph_native_pdf_extraction"
        if native_pdf
        else "hc_product_monograph_structured_extraction"
    )
    development = (
        products.loc[products["split"] == "development", "product_id"]
        .astype(str)
        .tolist()
    )
    heldout = (
        products.loc[products["split"] == "heldout", "product_id"].astype(str).tolist()
    )
    input_text = "complete native PDFs" if native_pdf else "label-selected pages"
    return {
        "schema_version": "1.0",
        "release": {
            "dataset_id": dataset_id,
            "version": spec.version,
            "release_type": "benchmark",
            "title": (
                "Health Canada Product Monograph native-PDF extraction"
                if native_pdf
                else "Health Canada Product Monograph evidence-window extraction"
            ),
            "description": (
                "A draft, human-audit-gated bilingual 200-family, 400-document "
                f"structured-extraction benchmark using {input_text}."
            ),
            "created_at_utc": spec.retrieved_at_utc,
            "status": "draft",
            "immutable": False,
            "license_or_terms": (
                "Official Product Monographs are publicly posted by Health "
                "Canada. Individual sponsor copyright may apply. The repository "
                "stores hashes, citations, and extracted evidence, not PDFs."
            ),
            "intended_use": (
                "Compare bilingual Product Monograph extraction after all "
                "provisional labels and document scope are human approved."
            ),
            "limitations": [
                "The release is draft and cannot support ranked claims until "
                "every fact and product-scope audit is approved.",
                "The purposive cohort is larger and more diverse but is not a "
                "probability sample of every marketed drug.",
                "Products must have distinct, text-extractable, automatically "
                "alignable English and French monographs, which creates "
                "selection bias.",
                "DIN, schedule, product status, and sponsor are alignment "
                "metadata and are not scored.",
            ],
            "languages": ["en", "fr"],
            "task_types": [
                "bilingual_structured_extraction",
                "native_pdf_document_understanding"
                if native_pdf
                else "evidence_window_extraction",
                "product_monograph_extraction",
            ],
        },
        "sources": [
            {
                "source_id": "hc_product_monographs",
                "name": "Health Canada Product Monograph PDF repository",
                "source_url": "https://pdf.hres.ca/dpd_pm/",
                "retrieved_at_utc": spec.retrieved_at_utc,
                "source_modified_date": None,
                "license_or_terms": (
                    "Publicly posted official Product Monographs; individual "
                    "sponsor copyright may apply."
                ),
                "snapshot_notes": (
                    "Four hundred official PDF URLs, hashes, byte sizes, page "
                    "counts, and DPD linkages are frozen in source_documents.csv."
                ),
            },
            {
                "source_id": "evalanche_pm_builder",
                "name": "Evalanche Product Monograph expansion builder",
                "source_url": "https://github.com/hc-sc-ocdo-bdpd/evalanche",
                "retrieved_at_utc": spec.retrieved_at_utc,
                "source_modified_date": None,
                "license_or_terms": "Evalanche repository terms.",
                "snapshot_notes": (
                    "Selects the cohort without model-result input and creates "
                    "fact-level and product-level human audit gates."
                ),
            },
        ],
        "files": files,
        "sampling": {
            "method": "purposive",
            "unit": "product_family",
            "population_description": (
                "Marketed human-drug DPD families with distinct official "
                "English and French Product Monographs and automatically "
                "locatable source evidence for every provisional scored item."
            ),
            "target_count": spec.target_products,
            "membership_file_id": (
                "pm_expanded_native_products" if native_pdf else "pm_expanded_products"
            ),
            "member_id_column": "product_id",
            "seed": int(spec.seed),
            "inclusion_criteria": [
                "A distinct official English and French PDF pair is posted.",
                "Every DPD variant resolves to the same PDF within a language.",
                "Both PDFs are text-extractable and language-correct.",
                "Every provisional scored fact has source-page evidence.",
                "The product belongs to a declared stratum and route group.",
                "The ingredient group is not represented by another family.",
            ],
            "exclusion_criteria": [
                "A language PDF is missing, duplicated, malformed, or image-only.",
                "DPD variants resolve to different monographs.",
                "A provisional fact cannot be located within the document.",
                "The candidate's ingredient group is already represented.",
            ],
            "strata": list(spec.target_strata),
        },
        "splits": [
            {
                "name": "development",
                "purpose": "Prompt development and visible failure analysis.",
                "unit": "product_family",
                "target_count": len(development),
                "group_key": "ingredient_group_id",
                "selection_policy": (
                    "All 40 previously exposed 0.1.0 families, with no new "
                    "families added after model outputs were inspected."
                ),
                "member_ids": development,
            },
            {
                "name": "heldout",
                "purpose": "Prompt-locked comparison after human audit.",
                "unit": "product_family",
                "target_count": len(heldout),
                "group_key": "ingredient_group_id",
                "selection_policy": (
                    "One hundred sixty new families selected without model "
                    "outputs and with no repeated ingredient group."
                ),
                "member_ids": heldout,
            },
        ],
        "lineage": {
            "parent_dataset_version": spec.inherited_version,
            "code_version": __version__,
            "transformations": [
                "Inherit all exposed pilot products as development data.",
                "Select 160 new held-out products with deterministic quotas.",
                "Discover and hash-lock official bilingual monographs.",
                "Build four-field provisional labels from the frozen DPD.",
                "Locate source-page evidence and create human audit queues.",
            ],
        },
    }


def _write_dataset_manifests(
    *,
    root: Path,
    spec: ExpansionSpec,
    products: pd.DataFrame,
    cases: pd.DataFrame,
    native_cases: pd.DataFrame,
    sources: pd.DataFrame,
    screening: pd.DataFrame,
    evidence: pd.DataFrame,
    fact_audit: pd.DataFrame,
    product_audit: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    common = [
        (
            EXPANSION_OUTPUT_DIR / "products.csv",
            "products",
            "sampling_membership",
            len(products),
            "text/csv",
        ),
        (EXPANSION_SOURCE_PATH, "sources", "source_lock", len(sources), "text/csv"),
        (
            EXPANSION_SCREENING_PATH,
            "screening",
            "selection_audit",
            len(screening),
            "text/csv",
        ),
        (
            EXPANSION_OUTPUT_DIR / "field_evidence.csv.gz",
            "evidence",
            "reference_evidence",
            len(evidence),
            "application/gzip",
        ),
        (
            EXPANSION_FACT_AUDIT_PATH,
            "fact_audit",
            "human_fact_audit",
            len(fact_audit),
            "text/csv",
        ),
        (
            EXPANSION_PRODUCT_AUDIT_PATH,
            "product_audit",
            "human_product_audit",
            len(product_audit),
            "text/csv",
        ),
        (
            EXPANSION_AUDIT_SUMMARY_PATH,
            "audit_summary",
            "human_audit_status",
            1,
            "application/json",
        ),
        (
            EXPANSION_AUDIT_GUIDE_PATH,
            "audit_guide",
            "human_audit_instructions",
            1,
            "text/markdown",
        ),
    ]
    evidence_files = [
        _dataset_file(
            root=root,
            path=EXPANSION_OUTPUT_DIR / "cases.csv.gz",
            file_id="pm_expanded_cases",
            role="benchmark_cases",
            records=len(cases),
            media_type="application/gzip",
        )
    ]
    native_files = [
        _dataset_file(
            root=root,
            path=EXPANSION_NATIVE_OUTPUT_DIR / "cases.csv.gz",
            file_id="pm_expanded_native_cases",
            role="benchmark_cases",
            records=len(native_cases),
            media_type="application/gzip",
        )
    ]
    for path, suffix, role, records, media_type in common:
        evidence_files.append(
            _dataset_file(
                root=root,
                path=path,
                file_id=f"pm_expanded_{suffix}",
                role=role,
                records=records,
                media_type=media_type,
            )
        )
        native_files.append(
            _dataset_file(
                root=root,
                path=path,
                file_id=(
                    "pm_expanded_native_products"
                    if suffix == "products"
                    else f"pm_expanded_native_{suffix}"
                ),
                role=role,
                records=records,
                media_type=media_type,
            )
        )
    evidence_files.append(
        _dataset_file(
            root=root,
            path=EXPANSION_OUTPUT_DIR / "build_report.json",
            file_id="pm_expanded_build_report",
            role="benchmark_build_report",
            records=1,
            media_type="application/json",
        )
    )
    native_files.append(
        _dataset_file(
            root=root,
            path=EXPANSION_NATIVE_OUTPUT_DIR / "build_report.json",
            file_id="pm_expanded_native_build_report",
            role="benchmark_build_report",
            records=1,
            media_type="application/json",
        )
    )
    evidence_manifest = _base_dataset_manifest(
        spec=spec,
        products=products,
        files=evidence_files,
        native_pdf=False,
    )
    native_manifest = _base_dataset_manifest(
        spec=spec,
        products=products,
        files=native_files,
        native_pdf=True,
    )
    for manifest, path in (
        (evidence_manifest, root / EXPANSION_MANIFEST_PATH),
        (native_manifest, root / EXPANSION_NATIVE_MANIFEST_PATH),
    ):
        DatasetManifest.model_validate(manifest)
        _write_yaml(path, manifest)
        verification = verify_dataset_manifest_file(path, root_path=root)
        if not verification["valid"]:
            raise ValueError(f"Generated dataset manifest did not verify: {path}")
    return {
        "evidence_window": verify_dataset_manifest_file(
            root / EXPANSION_MANIFEST_PATH, root_path=root
        ),
        "native_pdf": verify_dataset_manifest_file(
            root / EXPANSION_NATIVE_MANIFEST_PATH, root_path=root
        ),
    }


def _write_benchmark_manifests(root: Path) -> None:
    source_evidence = _load_yaml(
        root / "configs/benchmarks/"
        "hc_product_monograph_structured_extraction_0.1.0.yaml"
    )
    source_native = _load_yaml(
        root / "configs/benchmarks/"
        "hc_product_monograph_native_pdf_extraction_0.1.0.yaml"
    )
    for manifest, native in ((source_evidence, False), (source_native, True)):
        manifest["version"] = EXPANDED_PM_VERSION
        manifest["description"] = (
            "Draft bilingual extraction from 400 official English and French "
            "Product Monographs representing 200 marketed human-drug families. "
            + (
                "Each model receives the complete PDF through native file input."
                if native
                else "Each model receives label-selected source pages as text."
            )
        )
        manifest["status"] = "draft"
        manifest["dataset"]["version"] = EXPANDED_PM_VERSION
        manifest["dataset"]["manifest_path"] = (
            EXPANSION_NATIVE_MANIFEST_PATH if native else EXPANSION_MANIFEST_PATH
        ).as_posix()
        manifest["dataset"]["cases_path"] = (
            (EXPANSION_NATIVE_OUTPUT_DIR if native else EXPANSION_OUTPUT_DIR)
            .joinpath("cases.csv.gz")
            .as_posix()
        )
        manifest["slice_columns"] = [
            "language",
            "split",
            "stratum",
            "route_group",
            "document_complexity",
        ]
        manifest["limitations"] = [
            "This expanded release is draft until every fact and product audit "
            "item is approved.",
            "The 200-family purposive sample is not a probability sample of all "
            "marketed drugs.",
            "Bilingual availability, text extraction, and automated fact "
            "alignment are selection requirements.",
            "DIN, schedule, product status, and sponsor are not scored.",
        ]
        manifest["tiers"] = {
            "smoke": {
                "description": "Two bilingual product pairs for transport and parsing checks.",
                "sampling": {
                    "method": "balanced",
                    "unit": "group",
                    "count": 2,
                    "seed": 20260811,
                    "stratify_by": manifest["slice_columns"],
                },
                "cost": {
                    "initial_prompt_tokens": 150000 if native else 18000,
                    "initial_completion_tokens": 900,
                    "safety_multiplier": 2.0,
                    "request_ceiling_multiplier": 2.0,
                },
            },
            "screen": {
                "description": "Twenty-five cumulative bilingual pairs for screening.",
                "inherits": "smoke",
                "sampling": {
                    "method": "balanced",
                    "unit": "group",
                    "count": 25,
                    "seed": 20260811,
                    "stratify_by": manifest["slice_columns"],
                },
                "cost": {
                    "sample_from": "smoke",
                    "initial_prompt_tokens": 150000 if native else 18000,
                    "initial_completion_tokens": 900,
                    "safety_multiplier": 1.75,
                    "request_ceiling_multiplier": 2.0,
                },
                "promotion": {
                    "minimum_pass_rate": 0.50,
                    "maximum_generation_failure_rate": 0.05,
                },
            },
            "standard": {
                "description": (
                    "The complete 400-case expanded cohort. Earlier tier calls "
                    "are reused, so promoted models make at most 400 calls."
                ),
                "inherits": "screen",
                "sampling": {
                    "method": "all",
                    "unit": "group",
                    "seed": 20260811,
                    "stratify_by": manifest["slice_columns"],
                },
                "cost": {
                    "sample_from": "screen",
                    "initial_prompt_tokens": 150000 if native else 18000,
                    "initial_completion_tokens": 900,
                    "safety_multiplier": 1.5,
                    "request_ceiling_multiplier": 1.75,
                },
            },
        }
        BenchmarkManifest.model_validate(manifest)
        _write_yaml(
            root
            / (EXPANSION_NATIVE_BENCHMARK_PATH if native else EXPANSION_BENCHMARK_PATH),
            manifest,
        )


def _native_cases(cases: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for parent in cases.to_dict(orient="records"):
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
        product_id = str(parent["product_id"])
        language = str(parent["language"])
        suffix = product_id.removeprefix("hc_dpd_")
        filename = f"{product_id}_{language}_{parent['monograph_id']}.pdf"
        file_path = EXPANSION_RAW_DIR / filename
        record = dict(parent)
        record.update(
            {
                "case_id": f"hc_pm_pdf_{suffix}_{language}",
                "benchmark_version": EXPANDED_PM_VERSION,
                "input_mode": "native_pdf",
                "input": _native_pdf_prompt(language),
                "input_files": json.dumps(
                    [
                        {
                            "path": file_path.as_posix(),
                            "filename": filename,
                            "media_type": "application/pdf",
                            "sha256": parent["monograph_sha256"],
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
    return pd.DataFrame(records).sort_values("case_id", kind="stable")


def _audit_summary(
    fact_audit: pd.DataFrame,
    product_audit: pd.DataFrame,
) -> dict[str, Any]:
    fact_counts = (
        fact_audit["human_review_status"].value_counts().sort_index().to_dict()
    )
    product_counts = {
        column: product_audit[column].value_counts().sort_index().to_dict()
        for column in PRODUCT_AUDIT_STATUS_COLUMNS
    }
    fact_complete = fact_counts.get("approved", 0) == len(fact_audit)
    product_complete = all(
        counts.get("approved", 0) == len(product_audit)
        for counts in product_counts.values()
    )
    return {
        "schema_version": "1.0",
        "dataset_version": EXPANDED_PM_VERSION,
        "fact_items": len(fact_audit),
        "fact_status_counts": fact_counts,
        "products": len(product_audit),
        "product_status_counts": product_counts,
        "fact_audit_complete": fact_complete,
        "product_audit_complete": product_complete,
        "promotion_ready": fact_complete and product_complete,
    }


def _write_build_reports(
    *,
    root: Path,
    spec: ExpansionSpec,
    products: pd.DataFrame,
    sources: pd.DataFrame,
    cases: pd.DataFrame,
    native_cases: pd.DataFrame,
    evidence: pd.DataFrame,
    screening: pd.DataFrame,
    audit_summary: dict[str, Any],
) -> None:
    common = {
        "report_schema_version": "1.0",
        "dataset_version": spec.version,
        "status": "draft",
        "population": {
            "products": len(products),
            "cases": len(cases),
            "languages": cases["language"].value_counts().to_dict(),
            "splits": products["split"].value_counts().to_dict(),
            "strata": products["stratum"].value_counts().to_dict(),
            "route_groups": products["route_group"].value_counts().to_dict(),
            "document_complexity": (
                cases["document_complexity"].value_counts().to_dict()
            ),
        },
        "screening": {
            "records": len(screening),
            "statuses": screening["status"].value_counts().to_dict(),
            "exclusion_stages": (
                screening.loc[screening["status"] == "excluded", "stage"]
                .value_counts()
                .to_dict()
            ),
        },
        "source_documents": {
            "documents": len(sources),
            "unique_hashes": int(sources["sha256"].nunique()),
            "bytes": int(sources["byte_size"].astype(int).sum()),
            "pages": int(sources["page_count"].astype(int).sum()),
        },
        "evidence": {
            "field_items": len(evidence),
            "support_methods": evidence["support_method"].value_counts().to_dict(),
            "maximum_pages_per_case": int(cases["evidence_page_count"].max()),
            "all_scored_items_have_source_pages": True,
            "independent_human_signoff": "pending",
        },
        "human_audit": audit_summary,
        "software": {
            "evalanche_version": __version__,
            "pypdf_version": pypdf_version,
        },
    }
    evidence_report = {
        **common,
        "dataset_id": "hc_product_monograph_structured_extraction",
        "input_contract": "label_selected_text_pages",
    }
    native_report = {
        **common,
        "dataset_id": "hc_product_monograph_native_pdf_extraction",
        "population": {**common["population"], "cases": len(native_cases)},
        "input_contract": {
            "mode": "native_full_pdf",
            "request_api": "responses",
            "files_per_case": 1,
            "pdfs_committed": False,
            "pdfs_required": len(native_cases),
            "hash_verification_required_at_request_time": True,
        },
    }
    for path, report in (
        (EXPANSION_OUTPUT_DIR / "build_report.json", evidence_report),
        (EXPANSION_NATIVE_OUTPUT_DIR / "build_report.json", native_report),
    ):
        (root / path).write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def build_expanded_product_monograph_release(
    *,
    root_path: str | Path = ".",
    config_path: str | Path = EXPANSION_CONFIG_PATH,
    selected: list[ScreenedProduct],
    screening_records: list[dict[str, Any]],
) -> dict[str, Any]:
    root = Path(root_path).resolve()
    spec = load_expansion_spec(config_path, root_path=root)
    products = pd.DataFrame([item.product for item in selected]).sort_values(
        "product_id", kind="stable"
    )
    sources = pd.DataFrame(
        [source for item in selected for source in item.sources]
    ).sort_values(["product_id", "language"], kind="stable")
    cases = pd.DataFrame(
        [case for item in selected for case in item.cases]
    ).sort_values("case_id", kind="stable")
    evidence = pd.DataFrame(
        [fact for item in selected for fact in item.evidence]
    ).sort_values(["case_id", "field", "expected_item"], kind="stable")
    screening = pd.DataFrame(screening_records).sort_values(
        ["candidate_order", "product_id"], kind="stable"
    )
    _validate_release_tables(
        spec=spec,
        products=products,
        sources=sources,
        cases=cases,
        evidence=evidence,
    )
    for directory in (EXPANSION_OUTPUT_DIR, EXPANSION_NATIVE_OUTPUT_DIR):
        (root / directory).mkdir(parents=True, exist_ok=True)
    _write_cohort(root=root, spec=spec, products=products)
    products.to_csv(
        root / EXPANSION_OUTPUT_DIR / "products.csv", index=False, lineterminator="\n"
    )
    sources.to_csv(root / EXPANSION_SOURCE_PATH, index=False, lineterminator="\n")
    screening.to_csv(root / EXPANSION_SCREENING_PATH, index=False, lineterminator="\n")
    cases.to_csv(
        root / EXPANSION_OUTPUT_DIR / "cases.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
        lineterminator="\n",
    )
    evidence.to_csv(
        root / EXPANSION_OUTPUT_DIR / "field_evidence.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
        lineterminator="\n",
    )
    native_cases = _native_cases(cases)
    native_cases.to_csv(
        root / EXPANSION_NATIVE_OUTPUT_DIR / "cases.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
        lineterminator="\n",
    )
    fact_audit, fact_changes = _build_fact_audit(
        evidence=evidence,
        cases=cases,
        sources=sources,
        existing_path=root / EXPANSION_FACT_AUDIT_PATH,
    )
    product_audit, product_changes = _build_product_audit(
        products=products,
        cases=cases,
        sources=sources,
        existing_path=root / EXPANSION_PRODUCT_AUDIT_PATH,
    )
    fact_audit.to_csv(
        root / EXPANSION_FACT_AUDIT_PATH, index=False, lineterminator="\n"
    )
    product_audit.to_csv(
        root / EXPANSION_PRODUCT_AUDIT_PATH, index=False, lineterminator="\n"
    )
    audit_summary = _audit_summary(fact_audit, product_audit)
    (root / EXPANSION_AUDIT_SUMMARY_PATH).write_text(
        json.dumps(audit_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_build_reports(
        root=root,
        spec=spec,
        products=products,
        sources=sources,
        cases=cases,
        native_cases=native_cases,
        evidence=evidence,
        screening=screening,
        audit_summary=audit_summary,
    )
    verifications = _write_dataset_manifests(
        root=root,
        spec=spec,
        products=products,
        cases=cases,
        native_cases=native_cases,
        sources=sources,
        screening=screening,
        evidence=evidence,
        fact_audit=fact_audit,
        product_audit=product_audit,
    )
    _write_benchmark_manifests(root)
    return {
        "version": spec.version,
        "products": len(products),
        "cases": len(cases),
        "native_pdf_cases": len(native_cases),
        "facts": len(evidence),
        "screened_candidates": len(screening),
        "screening_exclusions": int((screening["status"] == "excluded").sum()),
        "fact_audit_changes": fact_changes,
        "product_audit_changes": product_changes,
        "audit_summary": audit_summary,
        "verifications": verifications,
        "output_dir": root / EXPANSION_OUTPUT_DIR,
    }


def expand_product_monograph_benchmark(
    *,
    root_path: str | Path = ".",
    config_path: str | Path = EXPANSION_CONFIG_PATH,
    max_candidates: int | None = None,
    fetch: Fetch = _default_fetch,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    root = Path(root_path).resolve()
    spec = load_expansion_spec(config_path, root_path=root)
    overrides = _write_overrides(root, spec)
    dpd_products = pd.read_csv(root / spec.source_dpd_products_path, dtype=str)
    dpd_cases = pd.read_csv(root / spec.source_dpd_cases_path, dtype=str)
    selected = _load_inherited_products(
        root=root,
        spec=spec,
        dpd_products=dpd_products,
        dpd_cases=dpd_cases,
        overrides=overrides,
        fetch=fetch,
    )
    if len(selected) != spec.development_products:
        raise ValueError(
            f"Expected {spec.development_products} inherited products, "
            f"found {len(selected)}"
        )
    inherited_groups = {str(item.product["ingredient_group_id"]) for item in selected}
    inherited_ids = {str(item.product["product_id"]) for item in selected}
    selected_counts = Counter(str(item.product["stratum"]) for item in selected)
    remaining = {
        stratum: target - selected_counts.get(stratum, 0)
        for stratum, target in spec.target_strata.items()
    }
    if any(value < 0 for value in remaining.values()):
        raise ValueError("Inherited cohort exceeds an expanded stratum quota")

    en_cases = dpd_cases[dpd_cases["language"] == "en"].set_index("product_id")
    candidates = dpd_products[
        ~dpd_products["product_id"].isin(inherited_ids)
        & ~dpd_products["ingredient_group_id"].isin(inherited_groups)
        & dpd_products["stratum"].isin(spec.target_strata)
    ].copy()
    candidates["route_group"] = [
        _route_group(str(en_cases.loc[product_id, "expected_output"]))
        for product_id in candidates["product_id"]
    ]
    ordered_by_stratum = {
        stratum: deque(
            _coverage_order(
                candidates[candidates["stratum"] == stratum],
                seed=f"{spec.seed}|{stratum}",
                route_cycle=spec.route_cycle,
            )
        )
        for stratum in spec.target_strata
    }
    screening_records: list[dict[str, Any]] = []
    selected_hashes = {
        str(source["sha256"]) for item in selected for source in item.sources
    }
    selected_groups = {str(item.product["ingredient_group_id"]) for item in selected}
    for order, item in enumerate(selected, start=1):
        screening_records.append(
            {
                "candidate_order": order,
                "product_id": item.product["product_id"],
                "brand_name": item.product["brand_name"],
                "stratum": item.product["stratum"],
                "route_group": item.product["route_group"],
                "status": "retained",
                "stage": "inherited_development",
                "reason": "Exposed 0.1.0 pilot product retained as development data.",
            }
        )
    attempted = 0
    candidate_order = len(screening_records)
    for stratum in spec.target_strata:
        bucket = ordered_by_stratum[stratum]
        while remaining[stratum] > 0:
            if not bucket:
                raise ValueError(
                    f"Candidate pool exhausted before filling {stratum} quota"
                )
            if max_candidates is not None and attempted >= max_candidates:
                raise ValueError(
                    f"Candidate limit reached before quotas were filled: {remaining}"
                )
            batch_size = min(
                spec.max_workers,
                remaining[stratum],
                len(bucket),
            )
            if max_candidates is not None:
                batch_size = min(batch_size, max_candidates - attempted)
            batch: list[tuple[int, dict[str, Any]]] = []
            for _ in range(batch_size):
                candidate = bucket.popleft()
                candidate_order += 1
                attempted += 1
                candidate.update(
                    {
                        "split": "heldout",
                        "selection_status": "new_heldout",
                    }
                )
                batch.append((candidate_order, candidate))
                if progress is not None:
                    progress(
                        f"Screening {candidate_order}: "
                        f"{candidate['product_id']} ({stratum}, "
                        f"need {remaining[stratum]})"
                    )

            def screen(
                entry: tuple[int, dict[str, Any]],
            ) -> tuple[int, dict[str, Any], ScreenedProduct | None, str]:
                order, candidate = entry
                try:
                    result = _screen_candidate(
                        root=root,
                        product=candidate,
                        dpd_cases=dpd_cases,
                        overrides=overrides,
                        spec=spec,
                        fetch=fetch,
                    )
                    return order, candidate, result, ""
                except (OSError, ValueError) as error:
                    return order, candidate, None, str(error)

            with ThreadPoolExecutor(max_workers=spec.max_workers) as executor:
                results = list(executor.map(screen, batch))
            for order, candidate, screened, reason in results:
                if screened is None:
                    lowered = reason.casefold()
                    stage = (
                        "document_availability"
                        if "missing" in lowered
                        else "variant_alignment"
                        if "resolve" in lowered
                        else "document_validation"
                        if "pdf" in lowered or "language" in lowered
                        else "fact_alignment"
                    )
                    screening_records.append(
                        {
                            "candidate_order": order,
                            "product_id": candidate["product_id"],
                            "brand_name": candidate["brand_name"],
                            "stratum": stratum,
                            "route_group": candidate["route_group"],
                            "status": "excluded",
                            "stage": stage,
                            "reason": reason,
                        }
                    )
                    continue
                ingredient_group = str(screened.product["ingredient_group_id"])
                if ingredient_group in selected_groups:
                    screening_records.append(
                        {
                            "candidate_order": order,
                            "product_id": candidate["product_id"],
                            "brand_name": candidate["brand_name"],
                            "stratum": stratum,
                            "route_group": candidate["route_group"],
                            "status": "excluded",
                            "stage": "duplicate_ingredient_group",
                            "reason": (
                                "An earlier selected family already represents "
                                f"ingredient group {ingredient_group}."
                            ),
                        }
                    )
                    continue
                candidate_hashes = {
                    str(source["sha256"]) for source in screened.sources
                }
                duplicate_hashes = sorted(candidate_hashes & selected_hashes)
                if duplicate_hashes:
                    screening_records.append(
                        {
                            "candidate_order": order,
                            "product_id": candidate["product_id"],
                            "brand_name": candidate["brand_name"],
                            "stratum": stratum,
                            "route_group": candidate["route_group"],
                            "status": "excluded",
                            "stage": "duplicate_document",
                            "reason": (
                                "A Product Monograph PDF is already represented "
                                "by an earlier selected family: "
                                + ", ".join(duplicate_hashes)
                            ),
                        }
                    )
                    continue
                selected.append(screened)
                selected_hashes.update(candidate_hashes)
                selected_groups.add(ingredient_group)
                remaining[stratum] -= 1
                screening_records.append(
                    {
                        "candidate_order": order,
                        "product_id": candidate["product_id"],
                        "brand_name": candidate["brand_name"],
                        "stratum": stratum,
                        "route_group": candidate["route_group"],
                        "status": "retained",
                        "stage": "new_heldout",
                        "reason": (
                            "Bilingual source, variant-link, language, text, "
                            "and automated fact-evidence checks passed; human "
                            "audit pending."
                        ),
                    }
                )
    return build_expanded_product_monograph_release(
        root_path=root,
        config_path=config_path,
        selected=selected,
        screening_records=screening_records,
    )


def _read_release_tables(root: Path) -> dict[str, pd.DataFrame]:
    options = {"dtype": str, "keep_default_na": False}
    return {
        "products": pd.read_csv(
            root / EXPANSION_OUTPUT_DIR / "products.csv", **options
        ),
        "sources": pd.read_csv(root / EXPANSION_SOURCE_PATH, **options),
        "screening": pd.read_csv(root / EXPANSION_SCREENING_PATH, **options),
        "cases": pd.read_csv(root / EXPANSION_OUTPUT_DIR / "cases.csv.gz", **options),
        "native_cases": pd.read_csv(
            root / EXPANSION_NATIVE_OUTPUT_DIR / "cases.csv.gz", **options
        ),
        "evidence": pd.read_csv(
            root / EXPANSION_OUTPUT_DIR / "field_evidence.csv.gz", **options
        ),
        "fact_audit": pd.read_csv(root / EXPANSION_FACT_AUDIT_PATH, **options),
        "product_audit": pd.read_csv(root / EXPANSION_PRODUCT_AUDIT_PATH, **options),
    }


def _audit_integrity_issues(
    *,
    tables: dict[str, pd.DataFrame],
) -> list[str]:
    expected_fact, _ = _build_fact_audit(
        evidence=tables["evidence"],
        cases=tables["cases"],
        sources=tables["sources"],
        existing_path=None,
    )
    expected_product, _ = _build_product_audit(
        products=tables["products"],
        cases=tables["cases"],
        sources=tables["sources"],
        existing_path=None,
    )
    issues: list[str] = []
    for label, actual, expected, id_column, mutable_columns in (
        (
            "Fact audit",
            tables["fact_audit"],
            expected_fact,
            "audit_item_id",
            FACT_AUDIT_MUTABLE_COLUMNS,
        ),
        (
            "Product audit",
            tables["product_audit"],
            expected_product,
            "audit_product_id",
            PRODUCT_AUDIT_MUTABLE_COLUMNS,
        ),
    ):
        missing_columns = sorted(set(expected.columns) - set(actual.columns))
        extra_columns = sorted(set(actual.columns) - set(expected.columns))
        if missing_columns:
            issues.append(f"{label} is missing columns: {missing_columns}")
        if extra_columns:
            issues.append(f"{label} has unexpected columns: {extra_columns}")
        if missing_columns or id_column not in actual.columns:
            continue
        duplicate_ids = actual.loc[
            actual[id_column].duplicated(keep=False), id_column
        ].astype(str)
        if not duplicate_ids.empty:
            issues.append(
                f"{label} has duplicate IDs: "
                + ", ".join(sorted(set(duplicate_ids))[:5])
            )
            continue
        actual_ids = set(actual[id_column].astype(str))
        expected_ids = set(expected[id_column].astype(str))
        missing_ids = sorted(expected_ids - actual_ids)
        extra_ids = sorted(actual_ids - expected_ids)
        if missing_ids:
            issues.append(
                f"{label} is missing {len(missing_ids)} expected rows, "
                f"including {missing_ids[:3]}"
            )
        if extra_ids:
            issues.append(
                f"{label} has {len(extra_ids)} unexpected rows, "
                f"including {extra_ids[:3]}"
            )
        common_ids = sorted(actual_ids & expected_ids)
        if not common_ids:
            continue
        actual_by_id = actual.set_index(id_column).loc[common_ids]
        expected_by_id = expected.set_index(id_column).loc[common_ids]
        immutable_columns = [
            column
            for column in expected.columns
            if column != id_column and column not in mutable_columns
        ]
        changed = (
            actual_by_id[immutable_columns].astype(str)
            != expected_by_id[immutable_columns].astype(str)
        ).any(axis=1)
        if changed.any():
            changed_ids = changed.index[changed].astype(str).tolist()
            issues.append(
                f"{label} has immutable source or label changes in "
                f"{len(changed_ids)} rows, including {changed_ids[:3]}"
            )
    return issues


def _valid_utc(value: str) -> bool:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.utcoffset() == timedelta(0)


def _audit_metadata_issues(
    fact: pd.DataFrame,
    product: pd.DataFrame,
) -> list[str]:
    issues: list[str] = []
    required_fact = {
        "audit_item_id",
        *FACT_AUDIT_MUTABLE_COLUMNS,
    }
    required_product = {
        "audit_product_id",
        *PRODUCT_AUDIT_MUTABLE_COLUMNS,
    }
    if missing := sorted(required_fact - set(fact.columns)):
        issues.append(f"Fact audit is missing review columns: {missing}")
    if missing := sorted(required_product - set(product.columns)):
        issues.append(f"Product audit is missing review columns: {missing}")
    if issues:
        return issues
    invalid_fact = sorted(set(fact["human_review_status"]) - AUDIT_STATUSES)
    if invalid_fact:
        issues.append(f"Invalid fact audit statuses: {invalid_fact}")
    for column in PRODUCT_AUDIT_STATUS_COLUMNS:
        invalid = sorted(set(product[column]) - AUDIT_STATUSES)
        if invalid:
            issues.append(f"Invalid {column} values: {invalid}")
    for row in fact.to_dict(orient="records"):
        status = str(row["human_review_status"])
        if status == "pending":
            if row["reviewer"] or row["reviewed_at_utc"]:
                issues.append(
                    f"{row['audit_item_id']}: pending fact has review metadata"
                )
            if row["corrected_value"]:
                issues.append(f"{row['audit_item_id']}: pending fact has a correction")
            continue
        if not row["reviewer"] or not _valid_utc(str(row["reviewed_at_utc"])):
            issues.append(
                f"{row['audit_item_id']}: reviewed fact needs reviewer and UTC time"
            )
        if status in {"needs_correction", "rejected"} and not row["notes"]:
            issues.append(f"{row['audit_item_id']}: {status} requires notes")
        if status == "needs_correction" and not row["corrected_value"]:
            issues.append(
                f"{row['audit_item_id']}: correction requires corrected_value"
            )
        if status != "needs_correction" and row["corrected_value"]:
            issues.append(
                f"{row['audit_item_id']}: only a correction may set corrected_value"
            )
    for row in product.to_dict(orient="records"):
        statuses = {str(row[column]) for column in PRODUCT_AUDIT_STATUS_COLUMNS}
        if statuses == {"pending"}:
            if row["reviewer"] or row["reviewed_at_utc"]:
                issues.append(
                    f"{row['audit_product_id']}: pending product has review metadata"
                )
            continue
        if "pending" in statuses:
            issues.append(
                f"{row['audit_product_id']}: finish all three product checks together"
            )
        if not row["reviewer"] or not _valid_utc(str(row["reviewed_at_utc"])):
            issues.append(
                f"{row['audit_product_id']}: reviewed product needs reviewer and UTC time"
            )
        if statuses & {"needs_correction", "rejected"} and not row["notes"]:
            issues.append(f"{row['audit_product_id']}: nonapproval requires notes")
    return issues


def refresh_expanded_product_monograph_release(
    *,
    root_path: str | Path = ".",
    config_path: str | Path = EXPANSION_CONFIG_PATH,
) -> dict[str, Any]:
    """Refresh audit-derived reports and locks without network access."""
    root = Path(root_path).resolve()
    spec = load_expansion_spec(config_path, root_path=root)
    tables = _read_release_tables(root)
    _validate_release_tables(
        spec=spec,
        products=tables["products"],
        sources=tables["sources"],
        cases=tables["cases"],
        evidence=tables["evidence"],
    )
    expected_native = _native_cases(tables["cases"])
    try:
        pd.testing.assert_frame_equal(
            tables["native_cases"].reset_index(drop=True),
            expected_native.reset_index(drop=True),
            check_dtype=False,
        )
    except AssertionError as error:
        raise ValueError(
            "Native-PDF cases differ from the evidence-window source cases"
        ) from error
    issues = _audit_integrity_issues(tables=tables)
    issues.extend(_audit_metadata_issues(tables["fact_audit"], tables["product_audit"]))
    if issues:
        raise ValueError("Audit files are invalid: " + "; ".join(issues[:10]))
    audit_summary = _audit_summary(tables["fact_audit"], tables["product_audit"])
    _write_cohort(root=root, spec=spec, products=tables["products"])
    (root / EXPANSION_AUDIT_SUMMARY_PATH).write_text(
        json.dumps(audit_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_build_reports(
        root=root,
        spec=spec,
        products=tables["products"],
        sources=tables["sources"],
        cases=tables["cases"],
        native_cases=tables["native_cases"],
        evidence=tables["evidence"],
        screening=tables["screening"],
        audit_summary=audit_summary,
    )
    verifications = _write_dataset_manifests(
        root=root,
        spec=spec,
        products=tables["products"],
        cases=tables["cases"],
        native_cases=tables["native_cases"],
        sources=tables["sources"],
        screening=tables["screening"],
        evidence=tables["evidence"],
        fact_audit=tables["fact_audit"],
        product_audit=tables["product_audit"],
    )
    _write_benchmark_manifests(root)
    return {
        "version": spec.version,
        "products": len(tables["products"]),
        "cases": len(tables["cases"]),
        "facts": len(tables["evidence"]),
        "audit_summary": audit_summary,
        "verifications": verifications,
        "network_calls": 0,
        "model_calls": 0,
    }


def check_expanded_product_monograph_audit(
    *,
    root_path: str | Path = ".",
    verify_artifact_locks: bool = True,
) -> dict[str, Any]:
    root = Path(root_path).resolve()
    tables = _read_release_tables(root)
    fact = tables["fact_audit"]
    product = tables["product_audit"]
    issues = _audit_integrity_issues(tables=tables)
    issues.extend(_audit_metadata_issues(fact, product))
    verifications: dict[str, dict[str, Any]] = {}
    if verify_artifact_locks:
        for label, path in (
            ("evidence_window", EXPANSION_MANIFEST_PATH),
            ("native_pdf", EXPANSION_NATIVE_MANIFEST_PATH),
        ):
            verification = verify_dataset_manifest_file(root / path, root_path=root)
            verifications[label] = verification
            if not verification["valid"]:
                issues.append(f"{label} dataset manifest is out of date")
    summary = _audit_summary(fact, product)
    summary.update(
        {
            "issues": issues,
            "valid": not issues,
            "promotion_ready": summary["promotion_ready"] and not issues,
            "verifications": verifications,
            "fact_audit_path": root / EXPANSION_FACT_AUDIT_PATH,
            "product_audit_path": root / EXPANSION_PRODUCT_AUDIT_PATH,
        }
    )
    return summary

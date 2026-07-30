from __future__ import annotations

import csv
import gzip
import io
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from evalanche import __version__
from evalanche.dataset_manifest import (
    DatasetManifest,
    load_dataset_manifest,
    sha256_file,
    verify_dataset_manifest_file,
)
from evalanche.dpd_benchmark import (
    ALL_STRATUM_ORDER,
    CASE_FIELDS,
    DPD_SOURCE_DATASET_ID,
    DPD_SOURCE_DATE,
    DPD_SOURCE_MANIFEST,
    DPD_SOURCE_VERSION,
    PRODUCT_FIELDS,
    REQUIRED_ROUTE_COVERAGE,
    CandidateProduct,
    SelectedProduct,
    _candidate_covers_route_group,
    _file_evidence,
    _marketed_archive_entry,
    _resolve_source_manifest,
    _stable_rank,
    _validate_source_manifest,
    _write_csv,
    _write_json,
    _write_yaml,
    build_candidate_products,
    build_case_records,
    build_product_records,
    load_marketed_tables,
)
from evalanche.dpd_snapshot import DPD_ARCHIVES, validate_dpd_archive


DPD_CENSUS_SCHEMA_VERSION = "1.0"
DPD_CENSUS_PARSER_VERSION = "hc_dpd_census/1.0"
DPD_CENSUS_DATASET_ID = "hc_dpd_structured_extraction_census"
DPD_CENSUS_VERSION = "0.2.0"
DPD_CENSUS_CREATED_AT_UTC = "2026-07-27T00:00:00Z"
DPD_CENSUS_DEMO_SEED = 20260727
DPD_CENSUS_DEMO_TARGETS = {
    "single_ingredient": 3,
    "multi_ingredient": 3,
    "multi_variant": 3,
    "multi_ingredient_multi_variant": 3,
}


def _write_csv_gzip(
    path: Path,
    *,
    fieldnames: Iterable[str],
    records: Iterable[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw_file:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_file,
            compresslevel=9,
            mtime=0,
        ) as compressed:
            with io.TextIOWrapper(
                compressed,
                encoding="utf-8",
                newline="",
            ) as text_file:
                writer = csv.DictWriter(
                    text_file,
                    fieldnames=list(fieldnames),
                    extrasaction="raise",
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerows(records)


def _census_file_evidence(
    path: Path,
    *,
    file_id: str,
    role: str,
    media_type: str,
    record_count: int,
    record_count_method: str,
) -> dict[str, Any]:
    evidence = _file_evidence(
        path,
        file_id=file_id,
        role=role,
        media_type=media_type,
        record_count=record_count,
        record_count_method=record_count_method,
    )
    return {
        **evidence,
        "source_id": "evalanche_dpd_census_builder",
        "relative_path": path.name,
        "parser_schema_version": DPD_CENSUS_PARSER_VERSION,
    }


def _validate_demo_targets(
    targets: Mapping[str, int],
) -> None:
    if set(targets) != set(ALL_STRATUM_ORDER):
        raise ValueError(
            "demo targets must contain exactly: "
            + ", ".join(ALL_STRATUM_ORDER)
        )
    if any(target <= 0 for target in targets.values()):
        raise ValueError("every demo target must be greater than zero")


def select_demo_products(
    candidates: Iterable[CandidateProduct],
    *,
    seed: int = DPD_CENSUS_DEMO_SEED,
    stratum_targets: Mapping[
        str,
        int,
    ] = DPD_CENSUS_DEMO_TARGETS,
    required_route_coverage: Mapping[
        str,
        Iterable[str],
    ] = REQUIRED_ROUTE_COVERAGE,
) -> list[CandidateProduct]:
    _validate_demo_targets(stratum_targets)
    by_stratum: dict[str, list[CandidateProduct]] = {
        stratum: [] for stratum in ALL_STRATUM_ORDER
    }
    for candidate in candidates:
        by_stratum[candidate.stratum].append(candidate)

    selected_by_stratum: dict[str, list[CandidateProduct]] = {
        stratum: [] for stratum in ALL_STRATUM_ORDER
    }
    selected_ids: set[str] = set()
    used_ingredient_codes: set[str] = set()
    all_candidates = [
        candidate
        for stratum in ALL_STRATUM_ORDER
        for candidate in by_stratum[stratum]
    ]

    for route_group, accepted_routes in required_route_coverage.items():
        ranked = sorted(
            all_candidates,
            key=lambda candidate: _stable_rank(
                seed,
                "census_demo_route",
                route_group,
                candidate.product_id,
            ),
        )
        chosen: CandidateProduct | None = None
        for candidate in ranked:
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
                "not enough census candidates to satisfy demo route "
                f"coverage: {route_group}"
            )
        selected_by_stratum[chosen.stratum].append(chosen)
        selected_ids.add(chosen.product_id)
        used_ingredient_codes.update(chosen.ingredient_codes)

    for stratum in ALL_STRATUM_ORDER:
        ranked = sorted(
            by_stratum[stratum],
            key=lambda candidate: _stable_rank(
                seed,
                "census_demo_fill",
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
                f"not enough {stratum} candidates for the census demo: "
                f"expected {stratum_targets[stratum]}, found "
                f"{len(selected)}"
            )

    return sorted(
        [
            candidate
            for stratum in ALL_STRATUM_ORDER
            for candidate in selected_by_stratum[stratum]
        ],
        key=lambda candidate: candidate.product_id,
    )


def _route_coverage(
    candidates: Iterable[CandidateProduct],
) -> dict[str, bool]:
    candidate_list = list(candidates)
    return {
        route_group: any(
            _candidate_covers_route_group(
                candidate,
                accepted_routes,
            )
            for candidate in candidate_list
        )
        for route_group, accepted_routes in REQUIRED_ROUTE_COVERAGE.items()
    }


def _distribution(values: Iterable[int]) -> dict[str, int | float]:
    ordered = sorted(values)
    if not ordered:
        return {
            "minimum": 0,
            "median": 0,
            "p95": 0,
            "maximum": 0,
        }
    middle = len(ordered) // 2
    if len(ordered) % 2:
        median: int | float = ordered[middle]
    else:
        median = (ordered[middle - 1] + ordered[middle]) / 2
    p95_index = int(0.95 * (len(ordered) - 1))
    return {
        "minimum": ordered[0],
        "median": median,
        "p95": ordered[p95_index],
        "maximum": ordered[-1],
    }


def _build_report(
    *,
    selected_products: list[SelectedProduct],
    case_records: list[dict[str, Any]],
    demo_products: list[CandidateProduct],
    demo_case_records: list[dict[str, Any]],
    population_statistics: dict[str, Any],
    source_manifest_relative: Path,
    source_manifest_sha256: str,
    source_archive_relative: Path,
    source_archive_sha256: str,
) -> dict[str, Any]:
    product_counts = Counter(
        selected.candidate.stratum for selected in selected_products
    )
    demo_counts = Counter(
        candidate.stratum for candidate in demo_products
    )
    product_case_counts = Counter(
        row["product_id"] for row in case_records
    )
    language_counts = Counter(
        row["language"] for row in case_records
    )
    demo_language_counts = Counter(
        row["language"] for row in demo_case_records
    )
    represented_drug_rows = sum(
        len(selected.candidate.variants)
        for selected in selected_products
    )
    full_route_coverage = _route_coverage(
        selected.candidate for selected in selected_products
    )
    demo_route_coverage = _route_coverage(demo_products)
    input_lengths = [len(row["input"]) for row in case_records]
    expected_lengths = [
        len(row["expected_output"]) for row in case_records
    ]

    return {
        "report_schema_version": DPD_CENSUS_SCHEMA_VERSION,
        "dataset_id": DPD_CENSUS_DATASET_ID,
        "dataset_version": DPD_CENSUS_VERSION,
        "release_created_at_utc": DPD_CENSUS_CREATED_AT_UTC,
        "scope": (
            "DPD-only full eligible-population bilingual structured "
            "extraction census"
        ),
        "source": {
            "dataset_id": DPD_SOURCE_DATASET_ID,
            "dataset_version": DPD_SOURCE_VERSION,
            "source_date": DPD_SOURCE_DATE,
            "manifest_path": source_manifest_relative.as_posix(),
            "manifest_sha256": source_manifest_sha256,
            "marketed_archive_path": (
                source_archive_relative.as_posix()
            ),
            "marketed_archive_sha256": source_archive_sha256,
        },
        "population": {
            **population_statistics,
            "included_product_families": len(selected_products),
            "included_drug_rows": represented_drug_rows,
            "included_cases": len(case_records),
            "included_product_families_by_stratum": {
                stratum: product_counts[stratum]
                for stratum in ALL_STRATUM_ORDER
            },
            "included_cases_by_language": {
                language: language_counts[language]
                for language in ("en", "fr")
            },
        },
        "census": {
            "method": "full eligible population, no sampling",
            "product_count": len(selected_products),
            "case_count": len(case_records),
            "languages_per_product": 2,
            "strata": list(ALL_STRATUM_ORDER),
        },
        "demo_view": {
            "method": (
                "deterministic stratified hash selection with route "
                "coverage and ingredient diversity"
            ),
            "seed": DPD_CENSUS_DEMO_SEED,
            "stratum_targets": DPD_CENSUS_DEMO_TARGETS,
            "product_count": len(demo_products),
            "case_count": len(demo_case_records),
            "products_by_stratum": {
                stratum: demo_counts[stratum]
                for stratum in ALL_STRATUM_ORDER
            },
            "cases_by_language": {
                language: demo_language_counts[language]
                for language in ("en", "fr")
            },
            "product_ids": [
                candidate.product_id for candidate in demo_products
            ],
            "required_route_groups": demo_route_coverage,
        },
        "size_profile": {
            "input_characters_total": sum(input_lengths),
            "input_characters_per_case": _distribution(input_lengths),
            "expected_output_characters_total": sum(expected_lengths),
            "expected_output_characters_per_case": _distribution(
                expected_lengths
            ),
        },
        "coverage": {
            "required_route_groups": full_route_coverage,
            "all_required_route_groups_present": all(
                full_route_coverage.values()
            ),
        },
        "quality_checks": {
            "all_candidate_product_families_included": (
                len(selected_products)
                == population_statistics["candidate_product_families"]
            ),
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
                len({row["case_id"] for row in case_records})
                == len(case_records)
            ),
            "two_language_cases_per_product": (
                all(count == 2 for count in product_case_counts.values())
                and len(product_case_counts) == len(selected_products)
            ),
            "all_expected_outputs_parse_as_json": all(
                isinstance(
                    json.loads(row["expected_output"]),
                    dict,
                )
                for row in case_records
            ),
            "complex_families_included": (
                product_counts["multi_ingredient_multi_variant"] > 0
            ),
            "demo_has_all_strata": all(
                demo_counts[stratum]
                == DPD_CENSUS_DEMO_TARGETS[stratum]
                for stratum in ALL_STRATUM_ORDER
            ),
            "demo_has_all_required_route_groups": all(
                demo_route_coverage.values()
            ),
            "manual_product_monograph_audit_status": "not_started",
        },
        "limitations": [
            (
                "Inputs are deterministic renderings of DPD relational "
                "records, not Product Monograph documents."
            ),
            (
                "Thirteen otherwise eligible product families are "
                "excluded because their variants contain conflicting "
                "French brand names that cannot be represented by the "
                "current scalar brand_name field without adjudication."
            ),
            (
                "Reference values have not yet been manually audited "
                "against English and French Product Monographs."
            ),
            (
                "The census is a DPD extraction coverage and stress "
                "benchmark, not evidence of production Product "
                "Monograph extraction performance."
            ),
            (
                "Running every case against a hosted model requires "
                "14,034 calls per model; the 24-case demo view is "
                "provided for bounded live demonstrations."
            ),
        ],
    }


def _build_manifest(
    *,
    source_manifest: DatasetManifest,
    output_relative: Path,
    file_evidence: list[dict[str, Any]],
    selected_products: list[SelectedProduct],
) -> dict[str, Any]:
    marketed_source = next(
        source
        for source in source_manifest.sources
        if source.source_id == "hc_dpd_marketed"
    )
    member_ids = [
        selected.candidate.product_id
        for selected in selected_products
    ]
    manifest = {
        "schema_version": "1.0",
        "release": {
            "dataset_id": DPD_CENSUS_DATASET_ID,
            "version": DPD_CENSUS_VERSION,
            "release_type": "benchmark",
            "title": (
                "Health Canada DPD bilingual structured extraction "
                "census"
            ),
            "description": (
                "A reproducible full-population benchmark containing "
                "every marketed human-drug product family that satisfies "
                "the version 1.0 bilingual JSON case contract."
            ),
            "created_at_utc": DPD_CENSUS_CREATED_AT_UTC,
            "status": "frozen",
            "immutable": True,
            "license_or_terms": marketed_source.license_or_terms,
            "intended_use": (
                "Measure DPD structured-extraction coverage, model "
                "reliability, operational performance, and subgroup "
                "behavior across the complete eligible frozen "
                "population. Use the bounded demo view for live "
                "demonstrations."
            ),
            "limitations": [
                (
                    "Inputs render DPD relational records and are not "
                    "Product Monograph documents."
                ),
                (
                    "Thirteen source-ambiguous French-brand families "
                    "require manual adjudication and are excluded."
                ),
                (
                    "Reference labels have not yet been manually audited "
                    "against Product Monographs."
                ),
                (
                    "This release alone cannot support a general Health "
                    "Canada model recommendation."
                ),
            ],
            "languages": ["en", "fr"],
            "task_types": [
                "bilingual_structured_extraction",
                "dpd_reference_normalization",
                "full_population_stress_evaluation",
            ],
        },
        "sources": [
            {
                "source_id": marketed_source.source_id,
                "name": marketed_source.name,
                "source_url": marketed_source.source_url,
                "retrieved_at_utc": marketed_source.retrieved_at_utc,
                "source_modified_date": (
                    marketed_source.source_modified_date
                ),
                "license_or_terms": marketed_source.license_or_terms,
                "snapshot_notes": (
                    "Inherited from frozen source dataset "
                    f"{DPD_SOURCE_DATASET_ID} {DPD_SOURCE_VERSION}; only "
                    "the marketed archive contributes rows."
                ),
            },
            {
                "source_id": "evalanche_dpd_census_builder",
                "name": "Evalanche DPD census builder",
                "source_url": (
                    "https://github.com/hc-sc-ocdo-bdpd/evalanche"
                ),
                "retrieved_at_utc": DPD_CENSUS_CREATED_AT_UTC,
                "source_modified_date": None,
                "license_or_terms": "Evalanche repository terms.",
                "snapshot_notes": (
                    "Generated the compressed census cases, membership "
                    "file, bounded demo view, and build report "
                    "deterministically from the frozen marketed DPD "
                    "archive."
                ),
            },
        ],
        "files": [
            {
                **evidence,
                "relative_path": (
                    output_relative / evidence["relative_path"]
                ).as_posix(),
            }
            for evidence in file_evidence
        ],
        "sampling": {
            "method": "full_population",
            "unit": "product_family",
            "population_description": (
                "Every marketed human-drug product family in the frozen "
                "2026-07-02 DPD archive that can produce paired English "
                "and French cases under the version 1.0 JSON contract."
            ),
            "target_count": len(member_ids),
            "membership_file_id": "dpd_census_products",
            "member_id_column": "product_id",
            "seed": None,
            "inclusion_criteria": [
                "DPD class is Human.",
                "Exactly one current status row is MARKETED.",
                "DIN is an eight-digit numeric identifier.",
                (
                    "Exactly one DIN_OWNER and complete English and "
                    "French ingredient, form, route, schedule, and "
                    "status labels are available."
                ),
                (
                    "Strength values are numeric and the product family "
                    "can produce paired English and French JSON cases."
                ),
                (
                    "Single-ingredient, multi-ingredient, "
                    "multi-variant, and multi-ingredient multi-variant "
                    "families are all included."
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
                    "Invalid DIN, missing reference fields, ambiguous "
                    "DIN owner, or nonnumeric strength."
                ),
                (
                    "Product families with conflicting French brand "
                    "names across variants, pending manual adjudication."
                ),
            ],
            "strata": list(ALL_STRATUM_ORDER),
        },
        "splits": [
            {
                "name": "census",
                "purpose": (
                    "Full eligible-population coverage and stress "
                    "evaluation. This is not a prompt-development or "
                    "hidden-test split."
                ),
                "unit": "product_family",
                "target_count": len(member_ids),
                "group_key": "product_id",
                "selection_policy": (
                    "All eligible product-family IDs are included; no "
                    "sampling or split assignment is applied."
                ),
                "member_ids": member_ids,
            }
        ],
        "lineage": {
            "parent_dataset_version": DPD_SOURCE_VERSION,
            "code_version": (
                f"evalanche {__version__}; "
                f"{DPD_CENSUS_PARSER_VERSION}"
            ),
            "transformations": [
                (
                    "Verified the frozen parent manifest and every "
                    "parent file hash before reading source rows."
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
                    "English brand, and active-ingredient code set."
                ),
                (
                    "Included every representable family across all four "
                    "complexity strata without population sampling."
                ),
                (
                    "Generated paired English and French JSON cases with "
                    "exact archive-member row traceability."
                ),
                (
                    "Compressed full-population CSV artifacts with "
                    "deterministic gzip metadata and generated a fixed "
                    "24-case live-demo view."
                ),
            ],
        },
    }
    DatasetManifest.model_validate(manifest)
    return manifest


def _output_paths(root: Path) -> dict[str, Path]:
    output_relative = (
        Path("data")
        / "hc"
        / "benchmarks"
        / "dpd_structured_extraction_census"
        / DPD_CENSUS_VERSION
    )
    manifest_relative = (
        Path("configs")
        / "datasets"
        / (
            "hc_dpd_structured_extraction_census_"
            f"{DPD_CENSUS_VERSION}_manifest.yaml"
        )
    )
    return {
        "output_relative": output_relative,
        "output": root / output_relative,
        "manifest_relative": manifest_relative,
        "manifest": root / manifest_relative,
    }


def create_dpd_benchmark_census(
    *,
    root_path: str | Path,
    source_manifest_path: str | Path = DPD_SOURCE_MANIFEST,
) -> dict[str, Any]:
    root = Path(root_path).resolve()
    source_manifest_file, source_manifest_relative = (
        _resolve_source_manifest(root, source_manifest_path)
    )
    paths = _output_paths(root)
    output_path = paths["output"]
    manifest_path = paths["manifest"]
    if output_path.exists():
        raise ValueError(
            "census output path already exists and will not be "
            f"overwritten: {output_path}"
        )
    if manifest_path.exists():
        raise ValueError(
            "census manifest already exists and will not be "
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
    candidates, population_statistics = build_candidate_products(
        tables,
        include_complex_families=True,
    )
    selected_products = [
        SelectedProduct(candidate=candidate, split="census")
        for candidate in sorted(
            candidates,
            key=lambda candidate: candidate.product_id,
        )
    ]
    case_records = build_case_records(
        selected_products,
        source_manifest_sha256=source_manifest_digest,
        source_archive_sha256=marketed_entry.sha256,
        benchmark_version=DPD_CENSUS_VERSION,
    )
    product_records = build_product_records(selected_products)
    demo_products = select_demo_products(candidates)
    demo_product_ids = {
        candidate.product_id for candidate in demo_products
    }
    demo_case_records = [
        row
        for row in case_records
        if row["product_id"] in demo_product_ids
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = Path(
        tempfile.mkdtemp(
            prefix=f".{DPD_CENSUS_VERSION}-",
            dir=output_path.parent,
        )
    )
    manifest_temp: Path | None = None
    published_output = False
    published_manifest = False

    try:
        cases_path = staging_path / "cases.csv.gz"
        products_path = staging_path / "products.csv.gz"
        demo_path = staging_path / "demo_cases.csv"
        report_path = staging_path / "build_report.json"
        _write_csv_gzip(
            cases_path,
            fieldnames=CASE_FIELDS,
            records=case_records,
        )
        _write_csv_gzip(
            products_path,
            fieldnames=PRODUCT_FIELDS,
            records=product_records,
        )
        _write_csv(
            demo_path,
            fieldnames=CASE_FIELDS,
            records=demo_case_records,
        )
        report = _build_report(
            selected_products=selected_products,
            case_records=case_records,
            demo_products=demo_products,
            demo_case_records=demo_case_records,
            population_statistics=population_statistics,
            source_manifest_relative=source_manifest_relative,
            source_manifest_sha256=source_manifest_digest,
            source_archive_relative=marketed_entry.relative_path,
            source_archive_sha256=marketed_entry.sha256,
        )
        _write_json(report_path, report)

        file_evidence = [
            _census_file_evidence(
                cases_path,
                file_id="dpd_census_cases",
                role="benchmark_cases",
                media_type="application/gzip",
                record_count=len(case_records),
                record_count_method="csv_rows",
            ),
            _census_file_evidence(
                products_path,
                file_id="dpd_census_products",
                role="sampling_membership",
                media_type="application/gzip",
                record_count=len(product_records),
                record_count_method="csv_rows",
            ),
            _census_file_evidence(
                demo_path,
                file_id="dpd_census_demo_cases",
                role="bounded_demo_view",
                media_type="text/csv",
                record_count=len(demo_case_records),
                record_count_method="csv_rows",
            ),
            _census_file_evidence(
                report_path,
                file_id="dpd_census_build_report",
                role="benchmark_build_report",
                media_type="application/json",
                record_count=1,
                record_count_method="declared",
            ),
        ]
        manifest = _build_manifest(
            source_manifest=source_manifest,
            output_relative=paths["output_relative"],
            file_evidence=file_evidence,
            selected_products=selected_products,
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
                "materialized DPD census manifest failed verification"
            )

        return {
            "dataset_id": DPD_CENSUS_DATASET_ID,
            "dataset_version": DPD_CENSUS_VERSION,
            "source_dataset_id": DPD_SOURCE_DATASET_ID,
            "source_dataset_version": DPD_SOURCE_VERSION,
            "output_path": output_path,
            "manifest_path": manifest_path,
            "product_count": len(product_records),
            "case_count": len(case_records),
            "demo_product_count": len(demo_products),
            "demo_case_count": len(demo_case_records),
            "excluded_ambiguous_family_count": (
                population_statistics[
                    "product_family_exclusions_by_reason"
                ].get("inconsistent_french_brand_name", 0)
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
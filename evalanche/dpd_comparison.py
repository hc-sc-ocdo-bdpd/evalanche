from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from evalanche.dataset_manifest import sha256_file


DPD_COMPARISON_SCHEMA_VERSION = "1.0"
DPD_COMPARISON_DATASET_ID = "hc_dpd_model_comparison"
DPD_COMPARISON_VERSION = "1.0.0"
DPD_COMPARISON_CREATED_AT_UTC = "2026-07-28T00:00:00Z"
DPD_COMPARISON_SEED = 20260728
DPD_COMPARISON_PRODUCT_COUNT = 250
DPD_COMPARISON_CASE_COUNT = DPD_COMPARISON_PRODUCT_COUNT * 2

DPD_CENSUS_CASES_PATH = Path(
    "data/hc/benchmarks/dpd_structured_extraction_census/"
    "0.2.0/cases.csv.gz"
)
DPD_COMPARISON_DIRECTORY = Path(
    "data/hc/demos/dpd_model_comparison/1.0.0"
)
DPD_COMPARISON_CASES_PATH = (
    DPD_COMPARISON_DIRECTORY / "cases.csv"
)
DPD_COMPARISON_REPORT_PATH = (
    DPD_COMPARISON_DIRECTORY / "build_report.json"
)

DPD_COMPARISON_MINI_OUTPUT_PATH = Path(
    "data/generated/"
    "hc_dpd_comparison_500_gpt_5_4_mini_outputs.csv"
)
DPD_COMPARISON_COMBINED_OUTPUT_PATH = Path(
    "data/generated/hc_dpd_comparison_500_outputs.csv"
)
DPD_COMPARISON_ASSEMBLY_METADATA_PATH = Path(
    "data/generated/"
    "hc_dpd_comparison_500_outputs_assembly_metadata.json"
)

MODEL_OUTPUT_SOURCES = (
    (
        "gpt_5_4_mini",
        Path(
            "data/generated/"
            "hc_dpd_census_gpt_5_4_mini_outputs.csv"
        ),
        True,
    ),
    (
        "gpt_5_6_luna",
        Path(
            "data/generated/"
            "hc_dpd_comparison_500_gpt_5_6_luna_outputs.csv"
        ),
        False,
    ),
    (
        "gpt_5_6_terra",
        Path(
            "data/generated/"
            "hc_dpd_comparison_500_gpt_5_6_terra_outputs.csv"
        ),
        False,
    ),
    (
        "gpt_5_6_sol",
        Path(
            "data/generated/"
            "hc_dpd_comparison_500_gpt_5_6_sol_outputs.csv"
        ),
        False,
    ),
)

STRATUM_ORDER = (
    "single_ingredient",
    "multi_ingredient",
    "multi_variant",
    "multi_ingredient_multi_variant",
)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _stable_rank(
    seed: int,
    stratum: str,
    product_id: str,
) -> str:
    material = (
        f"{seed}|dpd_model_comparison|{stratum}|{product_id}"
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _allocate_proportionally(
    population: Counter[str],
    *,
    target: int,
) -> dict[str, int]:
    if target <= 0:
        raise ValueError("comparison product target must be positive")

    total = sum(population.values())
    if total < target:
        raise ValueError(
            "comparison product target exceeds the census population"
        )

    raw = {
        stratum: target * population[stratum] / total
        for stratum in STRATUM_ORDER
    }
    allocation = {
        stratum: int(raw[stratum])
        for stratum in STRATUM_ORDER
    }
    remaining = target - sum(allocation.values())
    ranked_remainders = sorted(
        STRATUM_ORDER,
        key=lambda stratum: (
            -(raw[stratum] - allocation[stratum]),
            STRATUM_ORDER.index(stratum),
        ),
    )
    for stratum in ranked_remainders[:remaining]:
        allocation[stratum] += 1

    if any(
        allocation[stratum] > population[stratum]
        for stratum in STRATUM_ORDER
    ):
        raise ValueError(
            "comparison allocation exceeds a stratum population"
        )
    if sum(allocation.values()) != target:
        raise ValueError(
            "comparison allocation does not match the requested target"
        )
    return allocation


def _validate_bilingual_census(cases: pd.DataFrame) -> pd.DataFrame:
    required = {
        "case_id",
        "product_id",
        "language",
        "stratum",
        "input",
        "expected_output",
        "evaluation_type",
    }
    missing = required - set(cases.columns)
    if missing:
        raise ValueError(
            "DPD census is missing comparison fields: "
            f"{sorted(missing)}"
        )
    if cases["case_id"].duplicated().any():
        raise ValueError("DPD census contains duplicate case IDs")

    product_rows: list[dict[str, str]] = []
    for product_id, product_cases in cases.groupby(
        "product_id",
        sort=False,
    ):
        languages = set(product_cases["language"].astype(str))
        strata = set(product_cases["stratum"].astype(str))
        if len(product_cases) != 2 or languages != {"en", "fr"}:
            raise ValueError(
                "every comparison product must have exactly one English "
                f"and one French case: {product_id}"
            )
        if len(strata) != 1:
            raise ValueError(
                "comparison product has inconsistent strata: "
                f"{product_id}"
            )
        product_rows.append(
            {
                "product_id": str(product_id),
                "stratum": next(iter(strata)),
            }
        )

    products = pd.DataFrame(product_rows)
    unknown = set(products["stratum"]) - set(STRATUM_ORDER)
    if unknown:
        raise ValueError(
            f"DPD census contains unknown strata: {sorted(unknown)}"
        )
    return products


def _route_coverage(cases: pd.DataFrame) -> dict[str, bool]:
    route_groups = {
        "oral": {"oral"},
        "injectable": {
            "intramuscular",
            "intravenous",
            "subcutaneous",
        },
        "topical": {"topical"},
        "inhaled": {"inhalation"},
        "ophthalmic": {"ophthalmic"},
    }
    observed: set[str] = set()
    for expected in cases["expected_output"]:
        parsed = json.loads(str(expected))
        observed.update(
            str(route).casefold()
            for route in parsed.get("routes", [])
        )
    return {
        name: bool(observed.intersection(routes))
        for name, routes in route_groups.items()
    }


def build_dpd_comparison_sample(
    *,
    root_path: str | Path = ".",
    seed: int = DPD_COMPARISON_SEED,
    product_count: int = DPD_COMPARISON_PRODUCT_COUNT,
) -> dict[str, Any]:
    root = Path(root_path)
    census_path = root / DPD_CENSUS_CASES_PATH
    if not census_path.exists():
        raise FileNotFoundError(
            f"DPD census cases not found: {census_path}"
        )

    cases = pd.read_csv(census_path)
    products = _validate_bilingual_census(cases)
    population = Counter(products["stratum"].astype(str))
    allocation = _allocate_proportionally(
        population,
        target=product_count,
    )

    selected_ids: set[str] = set()
    for stratum in STRATUM_ORDER:
        stratum_products = products[
            products["stratum"] == stratum
        ].copy()
        stratum_products["_rank"] = stratum_products[
            "product_id"
        ].map(
            lambda product_id, selected_stratum=stratum: _stable_rank(
                seed,
                selected_stratum,
                str(product_id),
            )
        )
        chosen = stratum_products.sort_values(
            ["_rank", "product_id"],
            kind="stable",
        ).head(allocation[stratum])
        selected_ids.update(chosen["product_id"].astype(str))

    selected = cases[
        cases["product_id"].astype(str).isin(selected_ids)
    ].copy()
    language_order = pd.Categorical(
        selected["language"],
        categories=["en", "fr"],
        ordered=True,
    )
    selected = (
        selected.assign(_language_order=language_order)
        .sort_values(
            ["product_id", "_language_order"],
            kind="stable",
        )
        .drop(columns=["_language_order"])
        .reset_index(drop=True)
    )

    if len(selected_ids) != product_count:
        raise ValueError(
            "comparison selection produced the wrong product count"
        )
    if len(selected) != product_count * 2:
        raise ValueError(
            "comparison selection produced the wrong case count"
        )

    output_path = root / DPD_COMPARISON_CASES_PATH
    report_path = root / DPD_COMPARISON_REPORT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_path, index=False, lineterminator="\n")

    product_counts = Counter(
        selected.drop_duplicates("product_id")["stratum"].astype(str)
    )
    case_counts = Counter(selected["stratum"].astype(str))
    language_counts = Counter(selected["language"].astype(str))
    route_coverage = _route_coverage(selected)
    report = {
        "report_schema_version": DPD_COMPARISON_SCHEMA_VERSION,
        "dataset_id": DPD_COMPARISON_DATASET_ID,
        "dataset_version": DPD_COMPARISON_VERSION,
        "release_created_at_utc": DPD_COMPARISON_CREATED_AT_UTC,
        "source": {
            "dataset_id": "hc_dpd_structured_extraction_census",
            "dataset_version": "0.2.0",
            "cases_path": DPD_CENSUS_CASES_PATH.as_posix(),
            "cases_sha256": sha256_file(census_path),
        },
        "selection": {
            "method": (
                "deterministic proportional stratified hash sample of "
                "product families, retaining paired English and French "
                "cases"
            ),
            "seed": seed,
            "product_count": product_count,
            "case_count": len(selected),
            "products_by_stratum": {
                stratum: product_counts[stratum]
                for stratum in STRATUM_ORDER
            },
            "cases_by_stratum": {
                stratum: case_counts[stratum]
                for stratum in STRATUM_ORDER
            },
            "cases_by_language": {
                language: language_counts[language]
                for language in ("en", "fr")
            },
        },
        "coverage": {
            "required_route_groups": route_coverage,
            "all_required_route_groups_present": all(
                route_coverage.values()
            ),
        },
        "artifacts": {
            "cases_path": DPD_COMPARISON_CASES_PATH.as_posix(),
            "cases_sha256": sha256_file(output_path),
        },
        "quality_checks": {
            "unique_case_ids": bool(selected["case_id"].is_unique),
            "unique_product_count": (
                int(selected["product_id"].nunique())
                == product_count
            ),
            "paired_languages_per_product": all(
                set(group["language"].astype(str)) == {"en", "fr"}
                and len(group) == 2
                for _, group in selected.groupby("product_id")
            ),
            "proportional_stratum_allocation": (
                dict(product_counts) == allocation
            ),
            "all_expected_outputs_parse_as_json": all(
                _is_json(value)
                for value in selected["expected_output"]
            ),
        },
        "limitations": [
            (
                "The 500 cases are a sample of the frozen census, so small "
                "performance differences should be interpreted with their "
                "uncertainty intervals and paired tests."
            ),
            (
                "Inputs are deterministic renderings of DPD records, not "
                "Product Monograph documents."
            ),
            (
                "Recorded token costs use public Global Standard rates and "
                "are estimates, not reconciled Azure invoices."
            ),
        ],
    }
    _write_json(report_path, report)
    return {
        **report["selection"],
        "output_path": str(output_path),
        "report_path": str(report_path),
        "cases_sha256": report["artifacts"]["cases_sha256"],
    }


def _is_json(value: Any) -> bool:
    try:
        json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def _validate_model_output(
    *,
    frame: pd.DataFrame,
    sample: pd.DataFrame,
    model_name: str,
    source_path: Path,
    filter_to_sample: bool,
) -> pd.DataFrame:
    required = {
        "case_id",
        "model_name",
        "candidate_model",
        "input",
        "expected_output",
        "evaluation_type",
        "generation_status",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"{source_path} is missing output fields: {sorted(missing)}"
        )

    expected_case_ids = set(sample["case_id"].astype(str))
    if filter_to_sample:
        frame = frame[
            frame["case_id"].astype(str).isin(expected_case_ids)
        ].copy()

    observed_models = set(frame["model_name"].astype(str))
    if observed_models != {model_name}:
        raise ValueError(
            f"{source_path} has unexpected model names: "
            f"{sorted(observed_models)}"
        )
    if frame["case_id"].astype(str).duplicated().any():
        raise ValueError(
            f"{source_path} contains duplicate case IDs"
        )

    observed_case_ids = set(frame["case_id"].astype(str))
    if observed_case_ids != expected_case_ids:
        missing_ids = sorted(expected_case_ids - observed_case_ids)
        extra_ids = sorted(observed_case_ids - expected_case_ids)
        raise ValueError(
            f"{source_path} does not contain the complete frozen "
            f"comparison sample; missing={len(missing_ids)}, "
            f"extra={len(extra_ids)}"
        )

    reference = sample.set_index("case_id")
    candidate = frame.set_index("case_id")
    for column in ("input", "expected_output", "evaluation_type"):
        aligned = candidate.loc[reference.index, column].astype(str)
        expected = reference[column].astype(str)
        if not aligned.equals(expected):
            raise ValueError(
                f"{source_path} does not match the frozen sample in "
                f"column {column!r}"
            )

    order = {
        case_id: index
        for index, case_id in enumerate(sample["case_id"].astype(str))
    }
    frame["_case_order"] = frame["case_id"].astype(str).map(order)
    return (
        frame.sort_values("_case_order", kind="stable")
        .drop(columns="_case_order")
        .reset_index(drop=True)
    )


def assemble_dpd_comparison_outputs(
    *,
    root_path: str | Path = ".",
    required_models: Iterable[str] | None = None,
) -> dict[str, Any]:
    root = Path(root_path)
    sample_path = root / DPD_COMPARISON_CASES_PATH
    if not sample_path.exists():
        build_dpd_comparison_sample(root_path=root)
    sample = pd.read_csv(sample_path)

    required = set(required_models or ())
    known_models = {item[0] for item in MODEL_OUTPUT_SOURCES}
    unknown_required = required - known_models
    if unknown_required:
        raise ValueError(
            "unknown required comparison models: "
            f"{sorted(unknown_required)}"
        )

    available: list[pd.DataFrame] = []
    sources: list[dict[str, Any]] = []
    missing_required: list[str] = []
    for model_name, relative_path, filter_to_sample in (
        MODEL_OUTPUT_SOURCES
    ):
        source_path = root / relative_path
        if not source_path.exists():
            if model_name in required:
                missing_required.append(model_name)
            continue

        validated = _validate_model_output(
            frame=pd.read_csv(source_path),
            sample=sample,
            model_name=model_name,
            source_path=source_path,
            filter_to_sample=filter_to_sample,
        )
        if model_name == "gpt_5_4_mini":
            mini_path = root / DPD_COMPARISON_MINI_OUTPUT_PATH
            mini_path.parent.mkdir(parents=True, exist_ok=True)
            validated.to_csv(
                mini_path,
                index=False,
                lineterminator="\n",
            )

        available.append(validated)
        sources.append(
            {
                "model_name": model_name,
                "source_path": relative_path.as_posix(),
                "source_sha256": sha256_file(source_path),
                "rows": len(validated),
            }
        )

    if missing_required:
        raise FileNotFoundError(
            "required comparison outputs are missing: "
            + ", ".join(missing_required)
        )
    if not available:
        raise FileNotFoundError(
            "no comparison model outputs were found; the completed "
            "gpt-5.4-mini census output is the first required source"
        )

    combined = pd.concat(available, ignore_index=True)
    model_order = {
        model_name: index
        for index, (model_name, _, _) in enumerate(
            MODEL_OUTPUT_SOURCES
        )
    }
    case_order = {
        case_id: index
        for index, case_id in enumerate(sample["case_id"].astype(str))
    }
    combined["_model_order"] = (
        combined["model_name"].astype(str).map(model_order)
    )
    combined["_case_order"] = (
        combined["case_id"].astype(str).map(case_order)
    )
    combined = (
        combined.sort_values(
            ["_model_order", "_case_order"],
            kind="stable",
        )
        .drop(columns=["_model_order", "_case_order"])
        .reset_index(drop=True)
    )

    output_path = root / DPD_COMPARISON_COMBINED_OUTPUT_PATH
    metadata_path = (
        root / DPD_COMPARISON_ASSEMBLY_METADATA_PATH
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False, lineterminator="\n")
    metadata = {
        "schema_version": "1.0",
        "comparison_dataset_id": DPD_COMPARISON_DATASET_ID,
        "comparison_dataset_version": DPD_COMPARISON_VERSION,
        "comparison_cases_path": DPD_COMPARISON_CASES_PATH.as_posix(),
        "comparison_cases_sha256": sha256_file(sample_path),
        "model_count": len(available),
        "case_count_per_model": len(sample),
        "combined_row_count": len(combined),
        "models": sources,
        "output_path": DPD_COMPARISON_COMBINED_OUTPUT_PATH.as_posix(),
        "output_sha256": sha256_file(output_path),
    }
    _write_json(metadata_path, metadata)
    return {
        **metadata,
        "available_models": [
            source["model_name"] for source in sources
        ],
        "metadata_path": str(metadata_path),
    }

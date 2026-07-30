from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from evalanche.dataset_manifest import sha256_file


DPD_CENSUS_CASES_PATH = Path(
    "data/hc/benchmarks/dpd_structured_extraction_census/"
    "0.2.0/cases.csv.gz"
)
DPD_CENSUS_CASE_COUNT = 14_034
DPD_CENSUS_COMBINED_OUTPUT_PATH = Path(
    "data/generated/hc_dpd_census_all_models_outputs.csv"
)
DPD_CENSUS_ASSEMBLY_METADATA_PATH = Path(
    "data/generated/"
    "hc_dpd_census_all_models_outputs_assembly_metadata.json"
)

MODEL_OUTPUT_SOURCES = (
    (
        "gpt_5_4_mini",
        Path(
            "data/generated/"
            "hc_dpd_census_gpt_5_4_mini_outputs.csv"
        ),
    ),
    (
        "gpt_5_6_luna",
        Path(
            "data/generated/"
            "hc_dpd_census_gpt_5_6_luna_outputs.csv"
        ),
    ),
    (
        "gpt_5_6_terra",
        Path(
            "data/generated/"
            "hc_dpd_census_gpt_5_6_terra_outputs.csv"
        ),
    ),
    (
        "gpt_5_6_sol",
        Path(
            "data/generated/"
            "hc_dpd_census_gpt_5_6_sol_outputs.csv"
        ),
    ),
)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _validate_model_output(
    *,
    frame: pd.DataFrame,
    census: pd.DataFrame,
    model_name: str,
    source_path: Path,
) -> pd.DataFrame:
    required = {
        "case_id",
        "model_name",
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

    expected_case_ids = set(census["case_id"].astype(str))
    observed_case_ids = set(frame["case_id"].astype(str))
    if observed_case_ids != expected_case_ids:
        missing_ids = expected_case_ids - observed_case_ids
        extra_ids = observed_case_ids - expected_case_ids
        raise ValueError(
            f"{source_path} does not contain the complete frozen "
            f"census; missing={len(missing_ids)}, "
            f"extra={len(extra_ids)}"
        )

    reference = census.set_index("case_id")
    candidate = frame.set_index("case_id")
    for column in ("input", "expected_output", "evaluation_type"):
        aligned = candidate.loc[reference.index, column].astype(str)
        expected = reference[column].astype(str)
        if not aligned.equals(expected):
            raise ValueError(
                f"{source_path} does not match the frozen census in "
                f"column {column!r}"
            )

    order = {
        case_id: index
        for index, case_id in enumerate(
            census["case_id"].astype(str)
        )
    }
    frame = frame.copy()
    frame["_case_order"] = (
        frame["case_id"].astype(str).map(order)
    )
    return (
        frame.sort_values("_case_order", kind="stable")
        .drop(columns="_case_order")
        .reset_index(drop=True)
    )


def assemble_dpd_census_comparison_outputs(
    *,
    root_path: str | Path = ".",
    required_models: Iterable[str] | None = None,
) -> dict[str, Any]:
    root = Path(root_path)
    census_path = root / DPD_CENSUS_CASES_PATH
    if not census_path.exists():
        raise FileNotFoundError(
            f"frozen DPD census not found: {census_path}"
        )
    census = pd.read_csv(census_path)
    if len(census) != DPD_CENSUS_CASE_COUNT:
        raise ValueError(
            "frozen DPD census has an unexpected case count: "
            f"{len(census)}"
        )
    if census["case_id"].astype(str).duplicated().any():
        raise ValueError(
            "frozen DPD census contains duplicate case IDs"
        )

    required = set(required_models or ())
    known_models = {
        model_name for model_name, _ in MODEL_OUTPUT_SOURCES
    }
    unknown_required = required - known_models
    if unknown_required:
        raise ValueError(
            "unknown required census models: "
            f"{sorted(unknown_required)}"
        )

    available: list[pd.DataFrame] = []
    sources: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    missing_required: list[str] = []
    for model_name, relative_path in MODEL_OUTPUT_SOURCES:
        source_path = root / relative_path
        if not source_path.exists():
            if model_name in required:
                missing_required.append(model_name)
            continue

        try:
            validated = _validate_model_output(
                frame=pd.read_csv(source_path),
                census=census,
                model_name=model_name,
                source_path=source_path,
            )
        except ValueError as error:
            if model_name in required:
                raise
            skipped.append(
                {
                    "model_name": model_name,
                    "source_path": relative_path.as_posix(),
                    "reason": str(error),
                }
            )
            continue

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
            "required census outputs are missing: "
            + ", ".join(missing_required)
        )
    if not available:
        raise FileNotFoundError(
            "no complete full-census model outputs were found"
        )

    combined = pd.concat(available, ignore_index=True)
    model_order = {
        model_name: index
        for index, (model_name, _) in enumerate(
            MODEL_OUTPUT_SOURCES
        )
    }
    case_order = {
        case_id: index
        for index, case_id in enumerate(
            census["case_id"].astype(str)
        )
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

    output_path = root / DPD_CENSUS_COMBINED_OUTPUT_PATH
    metadata_path = root / DPD_CENSUS_ASSEMBLY_METADATA_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False, lineterminator="\n")
    metadata = {
        "schema_version": "1.0",
        "dataset_id": "hc_dpd_structured_extraction_census",
        "dataset_version": "0.2.0",
        "census_cases_path": DPD_CENSUS_CASES_PATH.as_posix(),
        "census_cases_sha256": sha256_file(census_path),
        "model_count": len(available),
        "case_count_per_model": len(census),
        "combined_row_count": len(combined),
        "models": sources,
        "skipped_incomplete_models": skipped,
        "output_path": DPD_CENSUS_COMBINED_OUTPUT_PATH.as_posix(),
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

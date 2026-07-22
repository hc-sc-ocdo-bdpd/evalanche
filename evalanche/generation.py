from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

from evalanche import __version__
from evalanche.config import (
    CandidateModelConfig,
    GenerationConfig,
    load_candidate_models,
)
from evalanche.identity import validate_generation_case_ids
from evalanche.llm import LLMClient, operational_from_error
from evalanche.operational import (
    COST_CURRENCY,
    COST_POLICY,
    summarize_stage,
)
from evalanche.pricing import (
    EndpointPriceConfig,
    EndpointPricingCatalogConfig,
    build_pricing_snapshot,
    load_optional_endpoint_pricing_catalog_with_hash,
    resolve_endpoint_price,
)
from evalanche.routing import validate_evaluation_types


REQUIRED_GENERATION_COLUMNS = {
    "case_id",
    "input",
    "expected_output",
    "evaluation_type",
}


class SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def sha256_file(path: str | Path | None) -> str | None:
    if path is None:
        return None

    file_path = Path(path)

    if not file_path.exists() or not file_path.is_file():
        return None

    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_generation_cases(path: str | Path) -> pd.DataFrame:
    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Generation input file not found: {input_path}"
        )

    cases = pd.read_csv(input_path)

    missing_columns = REQUIRED_GENERATION_COLUMNS - set(cases.columns)

    if missing_columns:
        raise ValueError(
            "Generation input is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    cases = validate_evaluation_types(
        cases,
        source_name=str(input_path),
    )

    return validate_generation_case_ids(
        cases,
        source_name=str(input_path),
    )


def render_prompt(
    template: str,
    row: dict[str, Any],
) -> str:
    values = SafeFormatDict(
        {
            key: "" if pd.isna(value) else str(value)
            for key, value in row.items()
        }
    )

    return template.format_map(values)


def build_messages(
    config: GenerationConfig,
    row: dict[str, Any],
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": config.prompt.system,
        },
        {
            "role": "user",
            "content": render_prompt(
                config.prompt.template,
                row,
            ),
        },
    ]


def generate_one(
    *,
    config: GenerationConfig,
    candidate: CandidateModelConfig,
    row: dict[str, Any],
    endpoint_price: EndpointPriceConfig | None = None,
) -> dict[str, Any]:
    client = LLMClient(
        model=candidate.model,
        temperature=candidate.temperature,
        max_retries=candidate.max_retries,
        endpoint_price=endpoint_price,
    )

    max_completion_tokens = (
        candidate.max_completion_tokens
        if candidate.max_completion_tokens is not None
        else config.generation.max_completion_tokens
    )

    started_at = perf_counter()

    response = client.complete_text(
        messages=build_messages(config, row),
        max_completion_tokens=max_completion_tokens,
    )

    finished_at = datetime.now(timezone.utc)
    usage = response.get("usage", {})
    operational = response.get("operational", {})

    record = dict(row)

    record.update(
        {
            "model_name": candidate.name,
            "candidate_model": candidate.model,
            "model_output": response.get("content", ""),
            "generation_status": "success",
            "generation_error": "",
            "generated_at_utc": finished_at.isoformat(),
            "generation_seconds": operational.get(
                "latency_seconds",
                perf_counter() - started_at,
            ),
            "generation_api_seconds": operational.get(
                "api_seconds"
            ),
            "generation_attempts": operational.get("attempts"),
            "generation_failed_attempts": operational.get(
                "failed_attempts"
            ),
            "generation_prompt_tokens": usage.get(
                "prompt_tokens"
            ),
            "generation_cached_prompt_tokens": usage.get(
                "cached_prompt_tokens"
            ),
            "generation_completion_tokens": usage.get(
                "completion_tokens"
            ),
            "generation_total_tokens": usage.get(
                "total_tokens"
            ),
            "generation_cost_usd": operational.get("cost_usd"),
            "generation_cost_source": operational.get(
                "cost_source"
            ),
            "generation_configured_cost_usd": operational.get(
                "configured_cost_usd"
            ),
            "generation_provider_reported_cost_usd": operational.get(
                "provider_reported_cost_usd"
            ),
            "generation_pricing_id": operational.get(
                "pricing_id"
            ),
            "generation_pricing_model": operational.get(
                "pricing_model"
            ),
            "generation_pricing_currency": operational.get(
                "pricing_currency"
            ),
            "generation_pricing_input_per_million_tokens": operational.get(
                "pricing_input_per_million_tokens"
            ),
            "generation_pricing_cached_input_per_million_tokens": operational.get(
                "pricing_cached_input_per_million_tokens"
            ),
            "generation_pricing_output_per_million_tokens": operational.get(
                "pricing_output_per_million_tokens"
            ),
            "generation_pricing_effective_date": operational.get(
                "pricing_effective_date"
            ),
            "generation_pricing_source": operational.get(
                "pricing_source"
            ),
        }
    )

    return record


def generate_outputs(
    *,
    config: GenerationConfig,
    config_path: str | Path,
) -> tuple[pd.DataFrame, Path, Path]:
    cases = load_generation_cases(config.run.input_path)

    candidate_config = load_candidate_models(
        config.candidate_models_path
    )
    pricing_catalog, pricing_catalog_sha256 = (
        load_optional_endpoint_pricing_catalog_with_hash(
            config.endpoint_pricing_path
        )
    )
    endpoint_prices = {
        candidate.name: resolve_endpoint_price(
            catalog=pricing_catalog,
            pricing_id=candidate.pricing_id,
            model=candidate.model,
        )
        for candidate in candidate_config.models
    }

    records: list[dict[str, Any]] = []

    work_items = [
        (row, candidate)
        for row in cases.to_dict(orient="records")
        for candidate in candidate_config.models
    ]

    for row, candidate in tqdm(
        work_items,
        desc="Generating outputs",
    ):
        try:
            record = generate_one(
                config=config,
                candidate=candidate,
                row=row,
                endpoint_price=endpoint_prices[candidate.name],
            )

            records.append(record)

        except Exception as error:
            if not config.generation.continue_on_error:
                raise

            operational = operational_from_error(error)
            error_record = dict(row)

            error_record.update(
                {
                    "model_name": candidate.name,
                    "candidate_model": candidate.model,
                    "model_output": "",
                    "generation_status": "error",
                    "generation_error": repr(error),
                    "generated_at_utc": (
                        datetime.now(timezone.utc).isoformat()
                    ),
                    "generation_seconds": operational.get(
                        "latency_seconds"
                    ),
                    "generation_api_seconds": operational.get(
                        "api_seconds"
                    ),
                    "generation_attempts": operational.get(
                        "attempts"
                    ),
                    "generation_failed_attempts": operational.get(
                        "failed_attempts"
                    ),
                    "generation_prompt_tokens": operational.get(
                        "prompt_tokens"
                    ),
                    "generation_cached_prompt_tokens": operational.get(
                        "cached_prompt_tokens"
                    ),
                    "generation_completion_tokens": operational.get(
                        "completion_tokens"
                    ),
                    "generation_total_tokens": operational.get(
                        "total_tokens"
                    ),
                    "generation_cost_usd": operational.get(
                        "cost_usd"
                    ),
                    "generation_cost_source": operational.get(
                        "cost_source"
                    ),
                    "generation_configured_cost_usd": operational.get(
                        "configured_cost_usd"
                    ),
                    "generation_provider_reported_cost_usd": operational.get(
                        "provider_reported_cost_usd"
                    ),
                    "generation_pricing_id": operational.get(
                        "pricing_id"
                    ),
                    "generation_pricing_model": operational.get(
                        "pricing_model"
                    ),
                    "generation_pricing_currency": operational.get(
                        "pricing_currency"
                    ),
                    "generation_pricing_input_per_million_tokens": operational.get(
                        "pricing_input_per_million_tokens"
                    ),
                    (
                        "generation_pricing_cached_"
                        "input_per_million_tokens"
                    ): operational.get(
                        "pricing_cached_input_per_million_tokens"
                    ),
                    "generation_pricing_output_per_million_tokens": operational.get(
                        "pricing_output_per_million_tokens"
                    ),
                    "generation_pricing_effective_date": operational.get(
                        "pricing_effective_date"
                    ),
                    "generation_pricing_source": operational.get(
                        "pricing_source"
                    ),
                }
            )

            records.append(error_record)

    outputs = pd.DataFrame(records)
    outputs["generation_pricing_catalog_version"] = (
        pricing_catalog.catalog_version
        if pricing_catalog is not None
        else None
    )
    outputs["generation_pricing_catalog_sha256"] = (
        pricing_catalog_sha256
    )

    output_path = Path(config.run.output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs.to_csv(
        output_path,
        index=False,
    )

    metadata_path = save_generation_metadata(
        config=config,
        config_path=config_path,
        candidates=candidate_config.models,
        cases=cases,
        outputs=outputs,
        output_path=output_path,
        pricing_catalog=pricing_catalog,
        pricing_catalog_sha256=pricing_catalog_sha256,
    )

    return outputs, output_path, metadata_path


def save_generation_metadata(
    *,
    config: GenerationConfig,
    config_path: str | Path,
    candidates: list[CandidateModelConfig],
    cases: pd.DataFrame,
    outputs: pd.DataFrame,
    output_path: str | Path,
    pricing_catalog: EndpointPricingCatalogConfig | None,
    pricing_catalog_sha256: str | None,
) -> Path:
    output_path = Path(output_path)

    metadata_path = output_path.with_name(
        output_path.stem
        + "_generation_metadata.json"
    )

    successful_outputs = outputs[
        outputs["generation_status"] == "success"
    ]

    failed_outputs = outputs[
        outputs["generation_status"] == "error"
    ]
    operation_summary = summarize_stage(
        outputs,
        "generation",
    )
    operations_by_model = {
        str(model_name): summarize_stage(
            group,
            "generation",
        )
        for model_name, group in outputs.groupby(
            "model_name",
            dropna=False,
        )
    }
    pricing_snapshot = build_pricing_snapshot(
        path=config.endpoint_pricing_path,
        catalog=pricing_catalog,
        catalog_sha256=pricing_catalog_sha256,
        references=[
            {
                "name": candidate.name,
                "model": candidate.model,
                "pricing_id": candidate.pricing_id,
            }
            for candidate in candidates
        ],
    )

    metadata = {
        "schema_version": "0.4",
        "created_at_utc": (
            datetime.now(timezone.utc).isoformat()
        ),
        "evalanche_version": __version__,
        "run": {
            "name": config.run.name,
            "config_path": str(config_path),
            "input_path": str(config.run.input_path),
            "output_path": str(output_path),
            "candidate_models_path": str(
                config.candidate_models_path
            ),
            "endpoint_pricing_path": (
                str(config.endpoint_pricing_path)
                if config.endpoint_pricing_path is not None
                else None
            ),
        },
        "hashes": {
            "config_sha256": sha256_file(config_path),
            "input_sha256": sha256_file(
                config.run.input_path
            ),
            "candidate_models_sha256": sha256_file(
                config.candidate_models_path
            ),
            "endpoint_pricing_sha256": pricing_catalog_sha256,
            "output_sha256": sha256_file(output_path),
        },
        "prompt": {
            "system": config.prompt.system,
            "template": config.prompt.template,
        },
        "generation": {
            "continue_on_error": (
                config.generation.continue_on_error
            ),
            "max_completion_tokens": (
                config.generation.max_completion_tokens
            ),
        },
        "candidate_models": [
            {
                "name": candidate.name,
                "model": candidate.model,
                "temperature": candidate.temperature,
                "max_retries": candidate.max_retries,
                "max_completion_tokens": (
                    candidate.max_completion_tokens
                ),
                "pricing_id": candidate.pricing_id,
            }
            for candidate in candidates
        ],
        "input_data": {
            "rows": int(len(cases)),
            "columns": list(cases.columns),
            "case_count": int(
                cases["case_id"].nunique()
            ),
            "evaluation_types": sorted(
                cases["evaluation_type"]
                .astype(str)
                .unique()
                .tolist()
            ),
        },
        "outputs": {
            "rows": int(len(outputs)),
            "success_count": int(
                len(successful_outputs)
            ),
            "error_count": int(
                len(failed_outputs)
            ),
            "models": sorted(
                outputs["model_name"]
                .astype(str)
                .unique()
                .tolist()
            ),
            "failure_rate": operation_summary[
                "failure_rate"
            ],
            "prompt_tokens": operation_summary[
                "prompt_tokens"
            ],
            "completion_tokens": operation_summary[
                "completion_tokens"
            ],
            "total_tokens": operation_summary["total_tokens"],
            "average_generation_seconds": operation_summary[
                "average_seconds"
            ],
            "p95_generation_seconds": operation_summary[
                "p95_seconds"
            ],
            "cost_usd": operation_summary["cost_usd"],
            "cost_coverage": operation_summary[
                "cost_coverage"
            ],
            "configured_cost_usd": operation_summary[
                "configured_cost_usd"
            ],
            "configured_cost_coverage": operation_summary[
                "configured_cost_coverage"
            ],
            "provider_reported_cost_usd": operation_summary[
                "provider_reported_cost_usd"
            ],
            "provider_reported_cost_coverage": operation_summary[
                "provider_reported_cost_coverage"
            ],
        },
        "endpoint_pricing": pricing_snapshot,
        "operations": {
            "cost_currency": COST_CURRENCY,
            "cost_policy": COST_POLICY,
            "generation": operation_summary,
            "by_model": operations_by_model,
        },
        "limitations": [
            (
                "Generated outputs are specific to the "
                "candidate models, prompts, inputs, and "
                "API configuration used in this run."
            ),
            (
                "Generated outputs should be inspected "
                "before important evaluations."
            ),
            (
                "Using the same model as both candidate "
                "and judge may introduce evaluation bias."
            ),
            (
                "Configured token costs are estimates from recorded usage "
                "and declared rates, not provider invoices."
            ),
        ],
    }

    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return metadata_path
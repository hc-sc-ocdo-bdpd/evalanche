from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
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
from evalanche.llm import LLMClient
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


def sha256_file(path: str | Path) -> str | None:
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
) -> dict[str, Any]:
    client = LLMClient(
        model=candidate.model,
        temperature=candidate.temperature,
        max_retries=candidate.max_retries,
    )

    max_completion_tokens = (
        candidate.max_completion_tokens
        if candidate.max_completion_tokens is not None
        else config.generation.max_completion_tokens
    )

    started_at = datetime.now(timezone.utc)

    response = client.complete_text(
        messages=build_messages(config, row),
        max_completion_tokens=max_completion_tokens,
    )

    finished_at = datetime.now(timezone.utc)
    usage = response.get("usage", {})

    record = dict(row)

    record.update(
        {
            "model_name": candidate.name,
            "candidate_model": candidate.model,
            "model_output": response.get("content", ""),
            "generation_status": "success",
            "generation_error": "",
            "generated_at_utc": finished_at.isoformat(),
            "generation_seconds": (
                finished_at - started_at
            ).total_seconds(),
            "generation_prompt_tokens": usage.get(
                "prompt_tokens"
            ),
            "generation_completion_tokens": usage.get(
                "completion_tokens"
            ),
            "generation_total_tokens": usage.get(
                "total_tokens"
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
            )

            records.append(record)

        except Exception as error:
            if not config.generation.continue_on_error:
                raise

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
                    "generation_seconds": None,
                    "generation_prompt_tokens": None,
                    "generation_completion_tokens": None,
                    "generation_total_tokens": None,
                }
            )

            records.append(error_record)

    outputs = pd.DataFrame(records)

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

    metadata = {
        "schema_version": "0.2",
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
        },
        "hashes": {
            "config_sha256": sha256_file(config_path),
            "input_sha256": sha256_file(
                config.run.input_path
            ),
            "candidate_models_sha256": sha256_file(
                config.candidate_models_path
            ),
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
            "total_tokens": (
                int(
                    outputs[
                        "generation_total_tokens"
                    ]
                    .fillna(0)
                    .sum()
                )
                if "generation_total_tokens"
                in outputs.columns
                else None
            ),
            "average_generation_seconds": (
                float(
                    successful_outputs[
                        "generation_seconds"
                    ].mean()
                )
                if not successful_outputs.empty
                else None
            ),
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
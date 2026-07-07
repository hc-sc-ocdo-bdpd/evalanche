from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from evalanche import __version__
from evalanche.config import EvalConfig
from evalanche.reporting import build_model_summary


def sha256_file(path: str | Path) -> str | None:
    file_path = Path(path)

    if not file_path.exists() or not file_path.is_file():
        return None

    digest = hashlib.sha256()

    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def build_run_metadata(
    *,
    config_path: str | Path,
    config: EvalConfig,
    cases: pd.DataFrame,
    results: pd.DataFrame,
    case_results_path: str | Path,
    model_summary_path: str | Path,
) -> dict[str, Any]:
    model_summary = build_model_summary(results)

    model_names = (
        sorted(results["model_name"].dropna().astype(str).unique().tolist())
        if "model_name" in results.columns
        else []
    )

    metadata: dict[str, Any] = {
        "schema_version": "0.1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evalanche_version": __version__,
        "runtime": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "run": {
            "name": config.run.name,
            "config_path": str(config_path),
            "input_path": str(config.run.input_path),
            "case_results_path": str(case_results_path),
            "model_summary_path": str(model_summary_path),
        },
        "hashes": {
            "config_sha256": sha256_file(config_path),
            "input_sha256": sha256_file(config.run.input_path),
            "case_results_sha256": sha256_file(case_results_path),
            "model_summary_sha256": sha256_file(model_summary_path),
        },
        "judge": {
            "model": config.judge.model,
            "temperature": config.judge.temperature,
            "max_retries": config.judge.max_retries,
        },
        "task": {
            "name": config.task.name,
            "description": config.task.description,
        },
        "scoring": {
            "score_min": config.scoring.score_min,
            "score_max": config.scoring.score_max,
            "pass_threshold": config.scoring.pass_threshold,
        },
        "criteria": [
            {
                "name": criterion.name,
                "weight": criterion.weight,
                "description": criterion.description,
            }
            for criterion in config.criteria
        ],
        "input_data": {
            "rows": int(len(cases)),
            "columns": list(cases.columns),
            "case_count": (
                int(cases["case_id"].nunique())
                if "case_id" in cases.columns
                else None
            ),
            "model_count": len(model_names),
            "model_names": model_names,
        },
        "results": {
            "rows": int(len(results)),
            "average_weighted_score": (
                float(results["weighted_score"].mean())
                if "weighted_score" in results.columns and len(results) > 0
                else None
            ),
            "pass_rate": (
                float(results["passed"].mean())
                if "passed" in results.columns and len(results) > 0
                else None
            ),
            "total_tokens": (
                int(results["total_tokens"].fillna(0).sum())
                if "total_tokens" in results.columns
                else None
            ),
        },
        "model_summary": (
            model_summary.to_dict(orient="records")
            if not model_summary.empty
            else []
        ),
        "limitations": [
            "Results are specific to this input dataset, task description, rubric, judge model, and configuration.",
            "LLM-as-judge scores should be interpreted as evaluation signals, not objective truth.",
            "High-stakes use cases should calibrate automated judge scores against human or expert review.",
        ],
    }

    return metadata


def save_run_metadata(
    *,
    config_path: str | Path,
    config: EvalConfig,
    cases: pd.DataFrame,
    results: pd.DataFrame,
    case_results_path: str | Path,
    model_summary_path: str | Path,
) -> Path:
    case_results_path = Path(case_results_path)
    metadata_path = case_results_path.with_name(
        case_results_path.stem + "_run_metadata.json"
    )

    metadata = build_run_metadata(
        config_path=config_path,
        config=config,
        cases=cases,
        results=results,
        case_results_path=case_results_path,
        model_summary_path=model_summary_path,
    )

    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return metadata_path
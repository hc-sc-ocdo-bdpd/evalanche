from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from evalanche.access_sets import AccessSet, resolve_access_set
from evalanche.cli import (
    _resolve_access_confirmed_models,
    build_parser,
    run_registry_validate,
)
from evalanche.registry import load_registry


def _write_yaml(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(value, sort_keys=False),
        encoding="utf-8",
    )


def _build_registry(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    pd.DataFrame(
        [
            {
                "case_id": "case_1",
                "group_id": "group_1",
                "input": "one",
                "expected_output": "one",
                "evaluation_type": "exact",
            }
        ]
    ).to_csv(data / "cases.csv", index=False)
    (data / "manifest.yaml").write_text(
        "schema_version: test\n",
        encoding="utf-8",
    )
    for model_id, capabilities in (
        ("arbitrary_model_a", ["text", "json"]),
        ("arbitrary_model_b", ["text"]),
    ):
        _write_yaml(
            tmp_path / "configs/models" / f"{model_id}.yaml",
            {
                "schema_version": "1.0",
                "model_id": model_id,
                "display_name": model_id,
                "provider_route": f"test/{model_id}",
                "capabilities": capabilities,
            },
        )
    _write_yaml(
        tmp_path / "configs/benchmarks/synthetic.yaml",
        {
            "schema_version": "1.0",
            "benchmark_id": "synthetic",
            "version": "1.0.0",
            "title": "Synthetic benchmark",
            "description": "Access-set test.",
            "status": "draft",
            "dataset": {
                "dataset_id": "synthetic_cases",
                "version": "1.0.0",
                "manifest_path": "data/manifest.yaml",
                "cases_path": "data/cases.csv",
            },
            "prompt": {
                "system": "Return JSON.",
                "template": "{input}",
                "version": "1.0",
            },
            "scoring": {
                "evaluation_type": "exact",
                "version": "1.0",
            },
            "group_key": "group_id",
            "required_capabilities": ["json"],
        },
    )


def _access_set(path: Path) -> None:
    _write_yaml(
        path,
        {
            "schema_version": "1.0",
            "access_set_id": "available_to_test_user",
            "description": "Synthetic confirmed routes.",
            "confirmed_on": "2026-08-06",
            "models": [
                {
                    "model_id": "arbitrary_model_a",
                    "access_confirmed": True,
                    "confirmation_basis": "Test route A is callable.",
                },
                {
                    "model_id": "arbitrary_model_b",
                    "access_confirmed": True,
                    "confirmation_basis": "Test route B is callable.",
                },
            ],
        },
    )


def test_access_set_filters_only_within_confirmed_models(
    tmp_path: Path,
) -> None:
    _build_registry(tmp_path)
    access_path = tmp_path / "configs/access_sets/available.yaml"
    _access_set(access_path)
    registry = load_registry(tmp_path)
    benchmark = registry.resolve_benchmark("synthetic@1.0.0")

    result = resolve_access_set(
        registry=registry,
        path=access_path,
        benchmark=benchmark,
    )

    assert result["compatible_model_ids"] == ["arbitrary_model_a"]
    assert result["incompatible_models"] == [
        {
            "model_id": "arbitrary_model_b",
            "missing_capabilities": ["json"],
        }
    ]
    assert result["scope"]["access_confirmed"] is True


def test_all_compatible_requires_an_access_set(tmp_path: Path) -> None:
    _build_registry(tmp_path)
    registry = load_registry(tmp_path)
    benchmark = registry.resolve_benchmark("synthetic@1.0.0")

    with pytest.raises(ValueError, match="requires --access-set"):
        _resolve_access_confirmed_models(
            registry=registry,
            benchmark=benchmark,
            model_ids=None,
            all_compatible=True,
            access_set_path=None,
        )


def test_registry_validation_checks_saved_access_sets(tmp_path: Path) -> None:
    _build_registry(tmp_path)
    access_path = tmp_path / "configs/access_sets/available.yaml"
    _access_set(access_path)

    result = run_registry_validate(root_path=str(tmp_path))

    assert result["valid"] is True
    assert result["access_sets"] == 1


def test_access_confirmation_cannot_be_false() -> None:
    with pytest.raises(ValidationError, match="access_confirmed"):
        AccessSet.model_validate(
            {
                "access_set_id": "invalid",
                "description": "Invalid confirmation.",
                "confirmed_on": "2026-08-06",
                "models": [
                    {
                        "model_id": "model_a",
                        "access_confirmed": False,
                        "confirmation_basis": "Not actually available.",
                    }
                ],
            }
        )


def test_summary_command_requires_an_explicit_scope() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "summarize-benchmark",
                "--benchmark",
                "synthetic@1.0.0",
            ]
        )

    args = parser.parse_args(
        [
            "summarize-benchmark",
            "--benchmark",
            "synthetic@1.0.0",
            "--all-results",
        ]
    )
    assert args.all_results is True

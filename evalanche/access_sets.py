from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from evalanche.config import load_yaml
from evalanche.registry import (
    BenchmarkManifest,
    LoadedRegistry,
    sha256_yaml_document,
)


ACCESS_SET_SCHEMA_VERSION = "1.0"
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def _identifier(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not _IDENTIFIER.fullmatch(normalized):
        raise ValueError(
            f"{field_name} must contain lowercase letters, numbers, "
            "dots, underscores, and hyphens"
        )
    return normalized


def _nonblank(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} cannot be blank")
    return normalized


class AccessSetModel(BaseModel):
    """One deployment route whose access was explicitly confirmed."""

    model_config = ConfigDict(extra="forbid")

    model_id: str
    access_confirmed: Literal[True]
    confirmation_basis: str
    notes: str | None = None

    @field_validator("model_id")
    @classmethod
    def validate_model_id(cls, value: str) -> str:
        return _identifier(value, "model_id")

    @field_validator("confirmation_basis")
    @classmethod
    def validate_confirmation_basis(cls, value: str) -> str:
        return _nonblank(value, "confirmation_basis")

    @field_validator("notes")
    @classmethod
    def validate_notes(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _nonblank(value, "notes")


class AccessSet(BaseModel):
    """A dated, reusable list of model deployments available to a user."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = ACCESS_SET_SCHEMA_VERSION
    access_set_id: str
    description: str
    confirmed_on: date
    models: list[AccessSetModel] = Field(min_length=1)

    @field_validator("access_set_id")
    @classmethod
    def validate_access_set_id(cls, value: str) -> str:
        return _identifier(value, "access_set_id")

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str) -> str:
        return _nonblank(value, "description")

    @model_validator(mode="after")
    def validate_unique_models(self) -> "AccessSet":
        model_ids = [model.model_id for model in self.models]
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("access set model IDs must be unique")
        return self

    @property
    def model_ids(self) -> list[str]:
        return [model.model_id for model in self.models]


def _resolve_path(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise ValueError(f"Access set does not exist: {path}")
    return resolved


def _display_path(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def load_access_set(path: str | Path) -> AccessSet:
    return AccessSet.model_validate(load_yaml(path))


def resolve_access_set(
    *,
    registry: LoadedRegistry,
    path: str | Path,
    benchmark: BenchmarkManifest | None = None,
) -> dict[str, Any]:
    """Validate an access set against registered deployment manifests."""

    resolved_path = _resolve_path(registry.root, path)
    access_set = load_access_set(resolved_path)
    unknown = sorted(set(access_set.model_ids) - set(registry.models))
    if unknown:
        raise ValueError(
            "Access set references unregistered model IDs: "
            + ", ".join(unknown)
        )

    compatible = list(access_set.model_ids)
    incompatible: list[dict[str, Any]] = []
    if benchmark is not None:
        required = set(benchmark.required_capabilities)
        compatible = []
        for model_id in access_set.model_ids:
            model = registry.resolve_model(model_id)
            missing = sorted(required - set(model.capabilities))
            if missing:
                incompatible.append(
                    {
                        "model_id": model_id,
                        "missing_capabilities": missing,
                    }
                )
            else:
                compatible.append(model_id)

    return {
        "access_set": access_set,
        "path": resolved_path,
        "compatible_model_ids": compatible,
        "incompatible_models": incompatible,
        "scope": {
            "mode": "access_set",
            "access_confirmed": True,
            "access_set_id": access_set.access_set_id,
            "confirmed_on": access_set.confirmed_on.isoformat(),
            "path": _display_path(registry.root, resolved_path),
            "sha256": sha256_yaml_document(resolved_path),
            "model_ids": list(access_set.model_ids),
        },
    }


def explicit_model_scope(model_ids: list[str]) -> dict[str, Any]:
    """Record that repeated model arguments confirmed access for one run."""

    selected = list(dict.fromkeys(model_ids))
    if not selected:
        raise ValueError("No models were explicitly selected")
    return {
        "mode": "explicit_models",
        "access_confirmed": True,
        "confirmation": (
            "Each model was explicitly selected for this command."
        ),
        "model_ids": selected,
    }


def historical_results_scope() -> dict[str, Any]:
    """Label a discovery report that does not assert current model access."""

    return {
        "mode": "historical_results",
        "access_confirmed": False,
        "confirmation": (
            "This report discovers compatible completed results and does "
            "not assert that the models are currently available to the user."
        ),
    }

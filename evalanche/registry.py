from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from evalanche import __version__
from evalanche.config import (
    DeterministicMetricsSettingsConfig,
    load_yaml,
)

REGISTRY_SCHEMA_VERSION = "1.0"
DEFAULT_MODELS_DIR = Path("configs/models")
DEFAULT_BENCHMARKS_DIR = Path("configs/benchmarks")

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_VERSION = re.compile(r"^\d+\.\d+\.\d+$")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_yaml_document(path: str | Path) -> str:
    with Path(path).open("r", encoding="utf-8") as file:
        document = yaml.safe_load(file)
    if document is None:
        raise ValueError(f"YAML document is empty: {path}")
    return sha256_json(document)


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


class RegistryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelRequestSettings(RegistryModel):
    temperature: float | None = 0
    reasoning_effort: Literal[
        "none",
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
    ] | None = None
    max_retries: int = Field(default=3, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)


class ModelManifest(RegistryModel):
    schema_version: Literal["1.0"] = REGISTRY_SCHEMA_VERSION
    model_id: str
    display_name: str
    provider_route: str
    request: ModelRequestSettings = Field(
        default_factory=ModelRequestSettings
    )
    pricing_catalog_path: Path | None = None
    pricing_id: str | None = None
    provider_model_version: str | None = None
    deployment_type: str | None = None
    resource_region: str | None = None
    capabilities: list[str] = Field(default_factory=list)
    notes: str | None = None

    @field_validator("model_id")
    @classmethod
    def validate_model_id(cls, value: str) -> str:
        return _identifier(value, "model_id")

    @field_validator("display_name", "provider_route")
    @classmethod
    def validate_required_text(cls, value: str, info: Any) -> str:
        return _nonblank(value, info.field_name)

    @field_validator("capabilities")
    @classmethod
    def validate_capabilities(cls, values: list[str]) -> list[str]:
        normalized = [
            _identifier(value.casefold(), "capability")
            for value in values
        ]
        if len(normalized) != len(set(normalized)):
            raise ValueError("capabilities must be unique")
        return normalized


class BenchmarkDataset(RegistryModel):
    dataset_id: str
    version: str
    manifest_path: Path
    cases_path: Path

    @field_validator("dataset_id")
    @classmethod
    def validate_dataset_id(cls, value: str) -> str:
        return _identifier(value, "dataset_id")

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        normalized = value.strip()
        if not _VERSION.fullmatch(normalized):
            raise ValueError("dataset version must use MAJOR.MINOR.PATCH")
        return normalized


class BenchmarkPrompt(RegistryModel):
    system: str
    template: str = "{input}"
    version: str

    @field_validator("system", "template", "version")
    @classmethod
    def validate_required_text(cls, value: str, info: Any) -> str:
        return _nonblank(value, info.field_name)


class BenchmarkScoring(RegistryModel):
    evaluation_type: Literal["exact", "json", "judge"]
    metrics: DeterministicMetricsSettingsConfig = Field(
        default_factory=DeterministicMetricsSettingsConfig
    )
    required_output_fields: list[str] = Field(default_factory=list)
    version: str

    @field_validator("required_output_fields")
    @classmethod
    def validate_output_fields(cls, values: list[str]) -> list[str]:
        normalized = [
            _nonblank(value, "required_output_fields") for value in values
        ]
        if len(normalized) != len(set(normalized)):
            raise ValueError("required_output_fields must be unique")
        return normalized

    @field_validator("version")
    @classmethod
    def validate_scoring_version(cls, value: str) -> str:
        return _nonblank(value, "scoring version")


class BenchmarkRuntime(RegistryModel):
    request_api: Literal["chat_completions", "responses"] = (
        "chat_completions"
    )
    max_completion_tokens: int = Field(default=900, ge=1)
    max_workers: int = Field(default=4, ge=1, le=64)
    max_requests_per_minute: float | None = Field(default=60, gt=0)
    checkpoint_every: int = Field(default=25, ge=1)
    resume: bool = True


class BenchmarkManifest(RegistryModel):
    schema_version: Literal["1.0"] = REGISTRY_SCHEMA_VERSION
    benchmark_id: str
    version: str
    title: str
    description: str
    status: Literal["draft", "ready", "frozen", "retired"]
    dataset: BenchmarkDataset
    prompt: BenchmarkPrompt
    scoring: BenchmarkScoring
    slice_columns: list[str] = Field(default_factory=list)
    group_key: str
    language_column: str | None = "language"
    required_capabilities: list[str] = Field(default_factory=list)
    runtime: BenchmarkRuntime = Field(default_factory=BenchmarkRuntime)
    limitations: list[str] = Field(default_factory=list)

    @field_validator("benchmark_id")
    @classmethod
    def validate_benchmark_id(cls, value: str) -> str:
        return _identifier(value, "benchmark_id")

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        normalized = value.strip()
        if not _VERSION.fullmatch(normalized):
            raise ValueError("benchmark version must use MAJOR.MINOR.PATCH")
        return normalized

    @field_validator("title", "description", "group_key")
    @classmethod
    def validate_required_text(cls, value: str, info: Any) -> str:
        return _nonblank(value, info.field_name)

    @field_validator("slice_columns")
    @classmethod
    def validate_slice_columns(cls, values: list[str]) -> list[str]:
        normalized = [_nonblank(value, "slice column") for value in values]
        if len(normalized) != len(set(normalized)):
            raise ValueError("slice_columns must be unique")
        return normalized

    @field_validator("required_capabilities")
    @classmethod
    def validate_required_capabilities(
        cls,
        values: list[str],
    ) -> list[str]:
        normalized = [
            _identifier(value.casefold(), "required capability")
            for value in values
        ]
        if len(normalized) != len(set(normalized)):
            raise ValueError("required_capabilities must be unique")
        return normalized


class LoadedRegistry:
    def __init__(
        self,
        *,
        root: Path,
        models: dict[str, ModelManifest],
        benchmarks: dict[str, BenchmarkManifest],
        model_paths: dict[str, Path],
        benchmark_paths: dict[str, Path],
    ) -> None:
        self.root = root
        self.models = models
        self.benchmarks = benchmarks
        self.model_paths = model_paths
        self.benchmark_paths = benchmark_paths

    def resolve_benchmark(self, reference: str) -> BenchmarkManifest:
        if "@" not in reference:
            matches = [
                benchmark
                for benchmark in self.benchmarks.values()
                if benchmark.benchmark_id == reference
            ]
            if len(matches) == 1:
                return matches[0]
            if not matches:
                raise ValueError(f"Unknown benchmark: {reference}")
            versions = ", ".join(
                sorted(benchmark.version for benchmark in matches)
            )
            raise ValueError(
                f"Benchmark {reference!r} has multiple versions: {versions}. "
                "Use benchmark_id@version."
            )

        benchmark_id, version = reference.rsplit("@", 1)
        key = f"{benchmark_id}@{version}"
        try:
            return self.benchmarks[key]
        except KeyError:
            raise ValueError(f"Unknown benchmark: {reference}") from None

    def resolve_model(self, model_id: str) -> ModelManifest:
        try:
            return self.models[model_id]
        except KeyError:
            raise ValueError(f"Unknown model: {model_id}") from None


def _load_manifests(
    directory: Path,
    manifest_type: type[ModelManifest] | type[BenchmarkManifest],
    key_builder: Any,
) -> tuple[dict[str, Any], dict[str, Path]]:
    manifests: dict[str, Any] = {}
    paths: dict[str, Path] = {}
    if not directory.is_dir():
        return manifests, paths

    for path in sorted(directory.glob("*.yaml")):
        manifest = manifest_type.model_validate(load_yaml(path))
        key = key_builder(manifest)
        if key in manifests:
            raise ValueError(
                f"Duplicate registry identifier {key!r}: "
                f"{paths[key]} and {path}"
            )
        manifests[key] = manifest
        paths[key] = path
    return manifests, paths


def load_registry(
    root: str | Path = ".",
    *,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    benchmarks_dir: str | Path = DEFAULT_BENCHMARKS_DIR,
) -> LoadedRegistry:
    root_path = Path(root).resolve()
    model_path = root_path / models_dir
    benchmark_path = root_path / benchmarks_dir
    models, model_paths = _load_manifests(
        model_path,
        ModelManifest,
        lambda model: model.model_id,
    )
    benchmarks, benchmark_paths = _load_manifests(
        benchmark_path,
        BenchmarkManifest,
        lambda benchmark: (
            f"{benchmark.benchmark_id}@{benchmark.version}"
        ),
    )
    return LoadedRegistry(
        root=root_path,
        models=models,
        benchmarks=benchmarks,
        model_paths=model_paths,
        benchmark_paths=benchmark_paths,
    )


def benchmark_fingerprint(
    registry: LoadedRegistry,
    benchmark: BenchmarkManifest,
) -> dict[str, Any]:
    cases_path = registry.root / benchmark.dataset.cases_path
    dataset_manifest_path = (
        registry.root / benchmark.dataset.manifest_path
    )
    if not cases_path.is_file():
        raise FileNotFoundError(f"Benchmark cases not found: {cases_path}")
    if not dataset_manifest_path.is_file():
        raise FileNotFoundError(
            f"Dataset manifest not found: {dataset_manifest_path}"
        )

    prompt_payload = benchmark.prompt.model_dump(mode="json")
    scoring_payload = benchmark.scoring.model_dump(mode="json")
    components = {
        "benchmark_id": benchmark.benchmark_id,
        "benchmark_version": benchmark.version,
        "dataset_id": benchmark.dataset.dataset_id,
        "dataset_version": benchmark.dataset.version,
        "cases_sha256": sha256_file(cases_path),
        "dataset_manifest_sha256": sha256_yaml_document(
            dataset_manifest_path
        ),
        "prompt_sha256": sha256_json(prompt_payload),
        "scoring_sha256": sha256_json(scoring_payload),
        "evaluator_version": __version__,
    }
    return {
        **components,
        "compatibility_sha256": sha256_json(components),
    }


def validate_registry(
    registry: LoadedRegistry,
) -> dict[str, Any]:
    issues: list[str] = []
    fingerprints: dict[str, dict[str, Any]] = {}

    if not registry.models:
        issues.append("No model manifests were found.")
    if not registry.benchmarks:
        issues.append("No benchmark manifests were found.")

    for model_id, model in registry.models.items():
        if model.pricing_catalog_path is not None:
            path = registry.root / model.pricing_catalog_path
            if not path.is_file():
                issues.append(
                    f"Model {model_id}: pricing catalog not found: {path}"
                )

    for key, benchmark in registry.benchmarks.items():
        try:
            fingerprints[key] = benchmark_fingerprint(
                registry,
                benchmark,
            )
        except (OSError, ValueError) as error:
            issues.append(f"Benchmark {key}: {error}")
            continue

        import pandas as pd

        cases_path = registry.root / benchmark.dataset.cases_path
        try:
            cases = pd.read_csv(cases_path)
        except Exception as error:
            issues.append(f"Benchmark {key}: cannot read cases: {error}")
            continue

        required = {
            "case_id",
            "input",
            "expected_output",
            "evaluation_type",
            benchmark.group_key,
            *benchmark.slice_columns,
        }
        missing = sorted(required - set(cases.columns))
        if missing:
            issues.append(
                f"Benchmark {key}: cases are missing columns {missing}"
            )
        if cases["case_id"].astype(str).duplicated().any():
            issues.append(f"Benchmark {key}: case_id values are not unique")
        observed_types = set(
            cases["evaluation_type"].astype(str).str.casefold()
        )
        expected_types = {benchmark.scoring.evaluation_type}
        if observed_types != expected_types:
            issues.append(
                f"Benchmark {key}: evaluation types {sorted(observed_types)} "
                f"do not match {sorted(expected_types)}"
            )

        if benchmark.scoring.evaluation_type == "json":
            for index, value in enumerate(cases["expected_output"]):
                try:
                    parsed = json.loads(str(value))
                except json.JSONDecodeError:
                    issues.append(
                        f"Benchmark {key}: expected_output row {index} "
                        "is not valid JSON"
                    )
                    break
                missing_fields = (
                    set(benchmark.scoring.required_output_fields)
                    - set(parsed)
                )
                if missing_fields:
                    issues.append(
                        f"Benchmark {key}: expected_output row {index} "
                        f"is missing fields {sorted(missing_fields)}"
                    )
                    break

    return {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "models": len(registry.models),
        "benchmarks": len(registry.benchmarks),
        "fingerprints": fingerprints,
        "issues": issues,
        "valid": not issues,
    }


def dump_yaml(path: str | Path, value: Any) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        yaml.safe_dump(
            value,
            file,
            sort_keys=False,
            allow_unicode=True,
        )
    return output_path

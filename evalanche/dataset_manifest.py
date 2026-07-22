from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Literal
from urllib.parse import urlparse

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from evalanche import __version__

MANIFEST_SCHEMA_VERSION = "1.0"
VERIFICATION_SCHEMA_VERSION = "1.0"

_IDENTIFIER_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_SEMANTIC_VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: str | Path) -> str | None:
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return None

    digest = hashlib.sha256()
    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ManifestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _require_nonblank(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} cannot be blank")
    return normalized


def _validate_identifier(value: str, field_name: str) -> str:
    normalized = _require_nonblank(value, field_name)
    if not _IDENTIFIER_PATTERN.fullmatch(normalized):
        raise ValueError(
            f"{field_name} must contain only lowercase letters, numbers, "
            "dots, underscores, and hyphens"
        )
    return normalized


def _validate_utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must include a UTC timezone")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must be expressed in UTC")
    return value


def _normalize_unique_strings(
    values: list[str],
    field_name: str,
) -> list[str]:
    normalized = [_require_nonblank(value, field_name) for value in values]
    comparison_values = [value.casefold() for value in normalized]
    if len(comparison_values) != len(set(comparison_values)):
        raise ValueError(f"{field_name} values must be unique")
    return normalized


class DatasetRelease(ManifestModel):
    dataset_id: str
    version: str
    release_type: Literal["source_snapshot", "benchmark"]
    title: str
    description: str
    created_at_utc: datetime
    status: Literal["draft", "frozen", "retired"]
    immutable: bool
    license_or_terms: str
    intended_use: str
    limitations: list[str] = Field(min_length=1)
    languages: list[str] = Field(min_length=1)
    task_types: list[str] = Field(min_length=1)

    @field_validator("dataset_id")
    @classmethod
    def validate_dataset_id(cls, value: str) -> str:
        return _validate_identifier(value, "dataset_id")

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        normalized = _require_nonblank(value, "version")
        if not _SEMANTIC_VERSION_PATTERN.fullmatch(normalized):
            raise ValueError("version must use MAJOR.MINOR.PATCH format")
        return normalized

    @field_validator(
        "title",
        "description",
        "license_or_terms",
        "intended_use",
    )
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        return _require_nonblank(value, "release text")

    @field_validator("created_at_utc")
    @classmethod
    def validate_created_at_utc(cls, value: datetime) -> datetime:
        return _validate_utc(value, "created_at_utc")

    @field_validator("limitations", "languages", "task_types")
    @classmethod
    def validate_unique_lists(
        cls,
        values: list[str],
        info: Any,
    ) -> list[str]:
        return _normalize_unique_strings(values, info.field_name)


class ManifestSource(ManifestModel):
    source_id: str
    name: str
    source_url: str
    retrieved_at_utc: datetime
    source_modified_date: date | None
    license_or_terms: str
    snapshot_notes: str

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, value: str) -> str:
        return _validate_identifier(value, "source_id")

    @field_validator("name", "license_or_terms", "snapshot_notes")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        return _require_nonblank(value, "source text")

    @field_validator("source_url")
    @classmethod
    def validate_source_url(cls, value: str) -> str:
        normalized = _require_nonblank(value, "source_url")
        parsed = urlparse(normalized)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("source_url must be an HTTP or HTTPS URL")
        return normalized

    @field_validator("retrieved_at_utc")
    @classmethod
    def validate_retrieved_at_utc(cls, value: datetime) -> datetime:
        return _validate_utc(value, "retrieved_at_utc")

    @model_validator(mode="after")
    def validate_source_dates(self) -> "ManifestSource":
        if (
            self.source_modified_date is not None
            and self.source_modified_date > self.retrieved_at_utc.date()
        ):
            raise ValueError(
                "source_modified_date cannot be after retrieved_at_utc"
            )
        return self


class ManifestFile(ManifestModel):
    file_id: str
    source_id: str
    role: str
    relative_path: Path
    media_type: str
    language: str | None
    byte_size: int = Field(ge=0)
    sha256: str
    record_count: int | None = Field(ge=0)
    record_count_method: Literal[
        "csv_rows",
        "jsonl_records",
        "declared",
        "not_applicable",
    ]
    page_count: int | None = Field(default=None, gt=0)
    parser_schema_version: str
    validation_status: Literal["passed", "failed", "not_run"]
    validation_error_count: int = Field(ge=0)

    @field_validator("file_id", "source_id")
    @classmethod
    def validate_identifiers(cls, value: str, info: Any) -> str:
        return _validate_identifier(value, info.field_name)

    @field_validator("role", "media_type", "parser_schema_version")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        return _require_nonblank(value, "file text")

    @field_validator("language")
    @classmethod
    def validate_language(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _require_nonblank(value, "language")

    @field_validator("relative_path", mode="before")
    @classmethod
    def validate_relative_path(cls, value: Any) -> Path:
        text = str(value).strip()
        if not text:
            raise ValueError("relative_path cannot be blank")
        if "\\" in text:
            raise ValueError("relative_path must use forward slashes")

        path = PurePosixPath(text)
        windows_path = PureWindowsPath(text)
        if (
            path.is_absolute()
            or windows_path.is_absolute()
            or windows_path.drive
            or ".." in path.parts
        ):
            raise ValueError(
                "relative_path must stay within the dataset root"
            )
        if path == PurePosixPath("."):
            raise ValueError("relative_path must identify a file")
        return Path(path.as_posix())

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not _SHA256_PATTERN.fullmatch(normalized):
            raise ValueError("sha256 must contain 64 hexadecimal characters")
        return normalized

    @model_validator(mode="after")
    def validate_record_count_contract(self) -> "ManifestFile":
        if (
            self.record_count_method == "not_applicable"
            and self.record_count is not None
        ):
            raise ValueError(
                "record_count must be null when record_count_method is "
                "not_applicable"
            )
        if (
            self.record_count_method != "not_applicable"
            and self.record_count is None
        ):
            raise ValueError(
                "record_count is required unless record_count_method is "
                "not_applicable"
            )
        if self.media_type == "application/pdf" and self.page_count is None:
            raise ValueError("PDF files must declare page_count")
        return self

    @model_validator(mode="after")
    def validate_producer_validation(self) -> "ManifestFile":
        if (
            self.validation_status == "passed"
            and self.validation_error_count != 0
        ):
            raise ValueError(
                "a passed file must have validation_error_count 0"
            )
        if (
            self.validation_status == "failed"
            and self.validation_error_count == 0
        ):
            raise ValueError(
                "a failed file must have at least one validation error"
            )
        return self


class SamplingPlan(ManifestModel):
    method: Literal[
        "full_population",
        "random",
        "stratified_random",
        "purposive",
    ]
    unit: str
    population_description: str
    target_count: int = Field(gt=0)
    membership_file_id: str
    member_id_column: str
    seed: int | None
    inclusion_criteria: list[str] = Field(min_length=1)
    exclusion_criteria: list[str] = Field(min_length=1)
    strata: list[str] = Field(default_factory=list)

    @field_validator("unit", "population_description", "member_id_column")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        return _require_nonblank(value, "sampling text")

    @field_validator("membership_file_id")
    @classmethod
    def validate_membership_file_id(cls, value: str) -> str:
        return _validate_identifier(value, "membership_file_id")

    @field_validator("inclusion_criteria", "exclusion_criteria", "strata")
    @classmethod
    def validate_unique_lists(
        cls,
        values: list[str],
        info: Any,
    ) -> list[str]:
        return _normalize_unique_strings(values, info.field_name)

    @model_validator(mode="after")
    def validate_random_seed(self) -> "SamplingPlan":
        if self.method in {"random", "stratified_random"} and self.seed is None:
            raise ValueError("random sampling methods require a seed")
        return self


class DatasetSplit(ManifestModel):
    name: str
    purpose: str
    unit: str
    target_count: int = Field(gt=0)
    group_key: str
    selection_policy: str
    member_ids: list[str] = Field(min_length=1)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return _validate_identifier(value, "split name")

    @field_validator("purpose", "unit", "group_key", "selection_policy")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        return _require_nonblank(value, "split text")

    @field_validator("member_ids")
    @classmethod
    def validate_member_ids(cls, values: list[str]) -> list[str]:
        return _normalize_unique_strings(values, "member_ids")

    @model_validator(mode="after")
    def validate_member_count(self) -> "DatasetSplit":
        if len(self.member_ids) != self.target_count:
            raise ValueError(
                "member_ids must contain exactly target_count values"
            )
        return self


class ManifestLineage(ManifestModel):
    parent_dataset_version: str | None
    code_version: str
    transformations: list[str] = Field(min_length=1)

    @field_validator("parent_dataset_version")
    @classmethod
    def validate_parent_version(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = _require_nonblank(value, "parent_dataset_version")
        if not _SEMANTIC_VERSION_PATTERN.fullmatch(normalized):
            raise ValueError(
                "parent_dataset_version must use MAJOR.MINOR.PATCH format"
            )
        return normalized

    @field_validator("code_version")
    @classmethod
    def validate_code_version(cls, value: str) -> str:
        return _require_nonblank(value, "code_version")

    @field_validator("transformations")
    @classmethod
    def validate_transformations(cls, values: list[str]) -> list[str]:
        return _normalize_unique_strings(values, "transformations")


class DatasetManifest(ManifestModel):
    schema_version: Literal["1.0"]
    release: DatasetRelease
    sources: list[ManifestSource] = Field(min_length=1)
    files: list[ManifestFile] = Field(min_length=1)
    sampling: SamplingPlan | None = None
    splits: list[DatasetSplit] = Field(default_factory=list)
    lineage: ManifestLineage

    @model_validator(mode="after")
    def validate_manifest_relationships(self) -> "DatasetManifest":
        if (
            self.release.status in {"frozen", "retired"}
            and not self.release.immutable
        ):
            raise ValueError("a frozen or retired release must be immutable")

        if any(
            source.retrieved_at_utc > self.release.created_at_utc
            for source in self.sources
        ):
            raise ValueError(
                "release.created_at_utc cannot be before source retrieval"
            )

        if self.release.status == "frozen":
            unvalidated_files = sorted(
                file.file_id
                for file in self.files
                if file.validation_status != "passed"
            )
            if unvalidated_files:
                raise ValueError(
                    "every file in a frozen release must have passed "
                    "producer validation: "
                    + ", ".join(unvalidated_files)
                )

        source_ids = [source.source_id for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source_id values must be unique")

        file_ids = [file.file_id for file in self.files]
        if len(file_ids) != len(set(file_ids)):
            raise ValueError("file_id values must be unique")

        relative_paths = [
            file.relative_path.as_posix().casefold() for file in self.files
        ]
        if len(relative_paths) != len(set(relative_paths)):
            raise ValueError("file relative_path values must be unique")

        unknown_source_ids = sorted(
            {file.source_id for file in self.files} - set(source_ids)
        )
        if unknown_source_ids:
            raise ValueError(
                "files reference unknown source_id values: "
                + ", ".join(unknown_source_ids)
            )

        release_languages = {
            language.casefold() for language in self.release.languages
        }
        undeclared_languages = sorted(
            {
                file.language
                for file in self.files
                if file.language is not None
                and file.language.casefold() not in release_languages
            }
        )
        if undeclared_languages:
            raise ValueError(
                "file languages are missing from release.languages: "
                + ", ".join(undeclared_languages)
            )

        if self.release.release_type == "source_snapshot":
            if self.sampling is not None or self.splits:
                raise ValueError(
                    "a source_snapshot cannot declare sampling or splits"
                )
        else:
            if self.sampling is None:
                raise ValueError("a benchmark release requires sampling")
            if not self.splits:
                raise ValueError("a benchmark release requires splits")

            file_by_id = {file.file_id: file for file in self.files}
            membership_file = file_by_id.get(
                self.sampling.membership_file_id
            )
            if membership_file is None:
                raise ValueError(
                    "sampling.membership_file_id must reference a file"
                )
            if membership_file.record_count_method not in {
                "csv_rows",
                "jsonl_records",
            }:
                raise ValueError(
                    "the membership file must use an automatically "
                    "verified record count method"
                )
            split_names = [split.name for split in self.splits]
            if len(split_names) != len(set(split_names)):
                raise ValueError("split names must be unique")

            mismatched_units = sorted(
                {
                    split.unit
                    for split in self.splits
                    if split.unit != self.sampling.unit
                }
            )
            if mismatched_units:
                raise ValueError(
                    "every split unit must match the sampling unit"
                )

            split_total = sum(split.target_count for split in self.splits)
            if split_total != self.sampling.target_count:
                raise ValueError(
                    "split target counts must sum to sampling.target_count"
                )

            member_ids = [
                member_id
                for split in self.splits
                for member_id in split.member_ids
            ]
            comparison_member_ids = [
                member_id.casefold() for member_id in member_ids
            ]
            if len(comparison_member_ids) != len(
                set(comparison_member_ids)
            ):
                raise ValueError("split member_ids cannot overlap")

        return self


def load_dataset_manifest(path: str | Path) -> DatasetManifest:
    manifest_path = Path(path)
    try:
        with manifest_path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
    except yaml.YAMLError as error:
        raise ValueError(f"invalid dataset manifest YAML: {error}") from error

    if not isinstance(raw, dict):
        raise ValueError("dataset manifest must contain a YAML mapping")

    return DatasetManifest.model_validate(raw)


def _count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(
            1
            for row in reader
            if any(cell.strip() for cell in row)
        )


def _count_jsonl_records(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON on line {line_number}: {error.msg}"
                ) from error
            count += 1
    return count


def _read_csv_member_ids(path: Path, column: str) -> list[str]:
    member_ids: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise ValueError(f"membership column {column!r} was not found")

        for row_number, row in enumerate(reader, start=2):
            if not any(
                value is not None and value.strip()
                for value in row.values()
            ):
                continue
            value = row.get(column)
            if value is None or not value.strip():
                raise ValueError(
                    f"blank membership ID on CSV row {row_number}"
                )
            member_ids.append(value.strip())
    return member_ids


def _read_jsonl_member_ids(path: Path, column: str) -> list[str]:
    member_ids: list[str] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON on line {line_number}: {error.msg}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"JSON record on line {line_number} is not an object"
                )
            value = record.get(column)
            if value is None or not str(value).strip():
                raise ValueError(
                    f"blank membership ID on JSON line {line_number}"
                )
            member_ids.append(str(value).strip())
    return member_ids


def _verify_sampling_membership(
    manifest: DatasetManifest,
    root_path: Path,
    file_results: list[dict[str, Any]],
) -> None:
    if manifest.release.release_type != "benchmark":
        return

    sampling = manifest.sampling
    if sampling is None:
        return

    file_by_id = {file.file_id: file for file in manifest.files}
    result_by_id = {result["file_id"]: result for result in file_results}
    membership_file = file_by_id[sampling.membership_file_id]
    membership_result = result_by_id[sampling.membership_file_id]
    if not membership_result["valid"]:
        return

    membership_path = root_path / membership_file.relative_path
    try:
        if membership_file.record_count_method == "csv_rows":
            actual_member_ids = _read_csv_member_ids(
                membership_path,
                sampling.member_id_column,
            )
        else:
            actual_member_ids = _read_jsonl_member_ids(
                membership_path,
                sampling.member_id_column,
            )
    except (OSError, UnicodeError, ValueError) as error:
        membership_result["issues"].append(
            f"sampling membership could not be verified: {error}"
        )
        membership_result["sampling_membership_verified"] = False
        membership_result["valid"] = False
        return

    normalized_actual = [value.casefold() for value in actual_member_ids]
    has_duplicates = len(normalized_actual) != len(set(normalized_actual))
    if has_duplicates:
        membership_result["issues"].append(
            "sampling membership file contains duplicate member IDs"
        )

    expected_member_ids = [
        member_id
        for split in manifest.splits
        for member_id in split.member_ids
    ]
    expected_by_normalized = {
        member_id.casefold(): member_id for member_id in expected_member_ids
    }
    actual_by_normalized = {
        member_id.casefold(): member_id for member_id in actual_member_ids
    }
    missing = sorted(
        expected_by_normalized[key]
        for key in expected_by_normalized.keys() - actual_by_normalized.keys()
    )
    unexpected = sorted(
        actual_by_normalized[key]
        for key in actual_by_normalized.keys() - expected_by_normalized.keys()
    )
    if missing:
        membership_result["issues"].append(
            "split member IDs missing from membership file: "
            + ", ".join(missing)
        )
    if unexpected:
        membership_result["issues"].append(
            "membership file IDs missing from splits: "
            + ", ".join(unexpected)
        )

    membership_result["sampling_membership_verified"] = not (
        missing or unexpected or has_duplicates
    )
    membership_result["valid"] = not membership_result["issues"]


def _verify_manifest_file(
    file_entry: ManifestFile,
    root_path: Path,
) -> dict[str, Any]:
    expected_path = root_path / file_entry.relative_path
    issues: list[str] = []
    actual_byte_size: int | None = None
    actual_sha256: str | None = None
    actual_record_count: int | None = None
    record_count_verified = False

    try:
        resolved_path = expected_path.resolve()
        resolved_path.relative_to(root_path)
    except (OSError, ValueError):
        issues.append("file resolves outside the dataset root")
        resolved_path = expected_path

    if not issues and (
        not resolved_path.exists() or not resolved_path.is_file()
    ):
        issues.append("file does not exist")

    if not issues:
        actual_byte_size = resolved_path.stat().st_size
        if actual_byte_size != file_entry.byte_size:
            issues.append(
                "byte size mismatch: expected "
                f"{file_entry.byte_size}, found {actual_byte_size}"
            )

        actual_sha256 = sha256_file(resolved_path)
        if actual_sha256 != file_entry.sha256:
            issues.append(
                "SHA-256 mismatch: expected "
                f"{file_entry.sha256}, found {actual_sha256}"
            )

        try:
            if file_entry.record_count_method == "csv_rows":
                actual_record_count = _count_csv_rows(resolved_path)
                record_count_verified = True
            elif file_entry.record_count_method == "jsonl_records":
                actual_record_count = _count_jsonl_records(resolved_path)
                record_count_verified = True
        except (OSError, UnicodeError, ValueError) as error:
            issues.append(f"record count could not be verified: {error}")

        if (
            record_count_verified
            and actual_record_count != file_entry.record_count
        ):
            issues.append(
                "record count mismatch: expected "
                f"{file_entry.record_count}, found {actual_record_count}"
            )

    return {
        "file_id": file_entry.file_id,
        "relative_path": file_entry.relative_path.as_posix(),
        "valid": not issues,
        "issues": issues,
        "expected_byte_size": file_entry.byte_size,
        "actual_byte_size": actual_byte_size,
        "expected_sha256": file_entry.sha256,
        "actual_sha256": actual_sha256,
        "expected_record_count": file_entry.record_count,
        "actual_record_count": actual_record_count,
        "record_count_method": file_entry.record_count_method,
        "record_count_verified": record_count_verified,
        "sampling_membership_verified": None,
        "expected_page_count": file_entry.page_count,
        "producer_validation_status": file_entry.validation_status,
        "producer_validation_error_count": (
            file_entry.validation_error_count
        ),
    }


def verify_dataset_manifest(
    manifest: DatasetManifest,
    *,
    manifest_path: str | Path,
    root_path: str | Path,
) -> dict[str, Any]:
    requested_manifest_path = Path(manifest_path)
    requested_root_path = Path(root_path)
    manifest_file_path = requested_manifest_path.resolve()
    resolved_root_path = requested_root_path.resolve()
    file_results = [
        _verify_manifest_file(file_entry, resolved_root_path)
        for file_entry in manifest.files
    ]
    _verify_sampling_membership(
        manifest,
        resolved_root_path,
        file_results,
    )
    issue_count = sum(len(result["issues"]) for result in file_results)
    files_passed = sum(bool(result["valid"]) for result in file_results)

    return {
        "verification_schema_version": VERIFICATION_SCHEMA_VERSION,
        "manifest_schema_version": manifest.schema_version,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "evalanche_version": __version__,
        "manifest_path": str(requested_manifest_path),
        "manifest_sha256": sha256_file(manifest_file_path),
        "dataset_id": manifest.release.dataset_id,
        "dataset_version": manifest.release.version,
        "release_type": manifest.release.release_type,
        "dataset_root": str(requested_root_path),
        "valid": files_passed == len(file_results),
        "files_checked": len(file_results),
        "files_passed": files_passed,
        "issue_count": issue_count,
        "files": file_results,
    }


def verify_dataset_manifest_file(
    manifest_path: str | Path,
    *,
    root_path: str | Path,
) -> dict[str, Any]:
    manifest = load_dataset_manifest(manifest_path)
    return verify_dataset_manifest(
        manifest,
        manifest_path=manifest_path,
        root_path=root_path,
    )


def save_dataset_verification(
    verification: dict[str, Any],
    output_path: str | Path,
) -> Path:
    verification_path = Path(output_path)
    verification_path.parent.mkdir(parents=True, exist_ok=True)
    with verification_path.open("w", encoding="utf-8") as file:
        json.dump(verification, file, indent=2, ensure_ascii=True)
        file.write("\n")
    return verification_path
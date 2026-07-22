from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, BinaryIO, Callable, ContextManager
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import yaml

from evalanche import __version__
from evalanche.dataset_manifest import (
    DatasetManifest,
    sha256_file,
    verify_dataset_manifest_file,
)

DPD_SNAPSHOT_SCHEMA_VERSION = "1.0"
DPD_PARSER_SCHEMA_VERSION = "hc_dpd_utf8_extract/1.0"
DPD_VALIDATION_SCHEMA_VERSION = "hc_dpd_source_validation/1.0"

DPD_EXTRACT_PAGE_URL = (
    "https://www.canada.ca/en/health-canada/services/"
    "drugs-health-products/drug-products/drug-product-database/"
    "what-data-extract-drug-product-database.html"
)
DPD_README_URL = (
    "https://www.canada.ca/en/health-canada/services/"
    "drugs-health-products/drug-products/drug-product-database/"
    "read-file-drug-product-database-data-extract.html"
)
DPD_LICENSE_URL = (
    "https://open.canada.ca/en/open-government-licence-canada"
)
EVALANCHE_REPOSITORY_URL = (
    "https://github.com/hc-sc-ocdo-bdpd/evalanche"
)

MAX_ARCHIVE_BYTES = 100 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 250 * 1024 * 1024
DOWNLOAD_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class DpdTableSpec:
    name: str
    column_count: int


@dataclass(frozen=True)
class DpdArchiveSpec:
    cohort: str
    source_id: str
    source_url: str
    filename: str
    member_suffix: str

    @property
    def expected_members(self) -> dict[str, int]:
        return {
            f"{table.name}{self.member_suffix}.txt": table.column_count
            for table in DPD_TABLES
        }


# These counts describe the UTF-8 archive layout published on 2026-07-02.
# They intentionally fail closed if Health Canada changes the extract schema.
DPD_TABLES = (
    DpdTableSpec("biosimilar", 4),
    DpdTableSpec("comp", 18),
    DpdTableSpec("drug", 14),
    DpdTableSpec("form", 4),
    DpdTableSpec("ingred", 15),
    DpdTableSpec("package", 8),
    DpdTableSpec("pharm", 2),
    DpdTableSpec("route", 4),
    DpdTableSpec("schedule", 3),
    DpdTableSpec("status", 7),
    DpdTableSpec("ther", 4),
    DpdTableSpec("vet", 4),
)

DPD_ARCHIVES = (
    DpdArchiveSpec(
        cohort="marketed",
        source_id="hc_dpd_marketed",
        source_url=(
            "https://www.canada.ca/content/dam/hc-sc/documents/services/"
            "drug-product-database/allfiles.zip"
        ),
        filename="allfiles.zip",
        member_suffix="",
    ),
    DpdArchiveSpec(
        cohort="approved",
        source_id="hc_dpd_approved",
        source_url=(
            "https://www.canada.ca/content/dam/hc-sc/documents/services/"
            "drug-product-database/allfiles_ap.zip"
        ),
        filename="allfiles_ap.zip",
        member_suffix="_ap",
    ),
)

UrlOpener = Callable[[Request, float], ContextManager[BinaryIO]]


def _utc_text(value: datetime) -> str:
    normalized = value.astimezone(timezone.utc).replace(microsecond=0)
    return normalized.isoformat().replace("+00:00", "Z")


def _parse_source_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(
            "source_date must use YYYY-MM-DD format"
        ) from error


def _header(response: Any, name: str) -> str | None:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    value = headers.get(name)
    if value is None:
        return None
    return str(value).strip()


def _http_date(value: str, field_name: str) -> datetime:
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"invalid {field_name} HTTP header: {value!r}"
        ) from error
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _default_url_opener(
    request: Request,
    timeout: float,
) -> ContextManager[BinaryIO]:
    return urlopen(request, timeout=timeout)


def _download_archive(
    spec: DpdArchiveSpec,
    destination: Path,
    *,
    source_date: date,
    timeout: float,
    opener: UrlOpener,
) -> dict[str, Any]:
    request = Request(
        spec.source_url,
        headers={
            "Accept": "application/zip",
            "User-Agent": f"evalanche/{__version__} dpd-snapshot",
        },
    )

    with opener(request, timeout) as response:
        status = getattr(response, "status", 200)
        if status != 200:
            raise ValueError(
                f"{spec.cohort} archive returned HTTP {status}"
            )

        content_type = _header(response, "Content-Type")
        if content_type is None or "application/zip" not in content_type:
            raise ValueError(
                f"{spec.cohort} archive did not return application/zip"
            )

        last_modified_header = _header(response, "Last-Modified")
        if last_modified_header is None:
            raise ValueError(
                f"{spec.cohort} archive omitted Last-Modified"
            )
        last_modified = _http_date(
            last_modified_header,
            "Last-Modified",
        )
        if last_modified.date() != source_date:
            raise ValueError(
                f"{spec.cohort} archive source date mismatch: expected "
                f"{source_date.isoformat()}, found "
                f"{last_modified.date().isoformat()}"
            )

        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        byte_size = 0
        with destination.open("wb") as output:
            while True:
                chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                byte_size += len(chunk)
                if byte_size > MAX_ARCHIVE_BYTES:
                    raise ValueError(
                        f"{spec.cohort} archive exceeds the "
                        f"{MAX_ARCHIVE_BYTES}-byte safety limit"
                    )
                output.write(chunk)
                digest.update(chunk)

        if byte_size == 0:
            raise ValueError(f"{spec.cohort} archive was empty")

        response_date_header = _header(response, "Date")
        final_url_getter = getattr(response, "geturl", None)
        final_url = (
            str(final_url_getter())
            if callable(final_url_getter)
            else spec.source_url
        )
        parsed_final_url = urlparse(final_url)
        if (
            parsed_final_url.scheme != "https"
            or parsed_final_url.hostname != "www.canada.ca"
        ):
            raise ValueError(
                f"{spec.cohort} archive redirected outside the trusted "
                "Health Canada host"
            )

    return {
        "content_type": content_type,
        "last_modified": _utc_text(last_modified),
        "response_date": (
            _utc_text(_http_date(response_date_header, "Date"))
            if response_date_header
            else None
        ),
        "final_url": final_url,
        "byte_size": byte_size,
        "sha256": digest.hexdigest(),
    }


def _member_timestamp(info: zipfile.ZipInfo) -> str:
    # The ZIP format stores a local wall-clock value without a timezone.
    return datetime(*info.date_time).isoformat()


def _validate_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    expected_columns: int,
    source_date: date,
) -> dict[str, Any]:
    if info.flag_bits & 0x1:
        raise ValueError(f"encrypted ZIP member is not allowed: {info.filename}")
    if info.is_dir() or Path(info.filename).name != info.filename:
        raise ValueError(f"unsafe ZIP member path: {info.filename}")
    if datetime(*info.date_time).date() != source_date:
        raise ValueError(
            f"ZIP member date mismatch for {info.filename}: expected "
            f"{source_date.isoformat()}, found "
            f"{datetime(*info.date_time).date().isoformat()}"
        )

    try:
        raw = archive.read(info)
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        raise ValueError(
            f"could not read ZIP member {info.filename}: {error}"
        ) from error

    if len(raw) != info.file_size:
        raise ValueError(
            f"ZIP member size mismatch for {info.filename}"
        )

    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(
            f"ZIP member is not valid UTF-8: {info.filename}"
        ) from error
    if "\x00" in text:
        raise ValueError(f"ZIP member contains NUL bytes: {info.filename}")

    row_count = 0
    blank_row_count = 0
    try:
        rows = csv.reader(io.StringIO(text, newline=""), strict=True)
        for row_number, row in enumerate(rows, start=1):
            if not any(cell.strip() for cell in row):
                blank_row_count += 1
                continue
            if len(row) != expected_columns:
                raise ValueError(
                    f"{info.filename} row {row_number} has {len(row)} "
                    f"columns, expected {expected_columns}"
                )
            row_count += 1
    except csv.Error as error:
        raise ValueError(
            f"invalid CSV in {info.filename}: {error}"
        ) from error

    if row_count == 0:
        raise ValueError(f"ZIP member has no records: {info.filename}")
    if blank_row_count:
        raise ValueError(
            f"ZIP member contains {blank_row_count} blank rows: "
            f"{info.filename}"
        )

    return {
        "name": info.filename,
        "byte_size": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "crc32": f"{info.CRC:08x}",
        "zip_timestamp": _member_timestamp(info),
        "column_count": expected_columns,
        "record_count": row_count,
    }


def validate_dpd_archive(
    archive_path: str | Path,
    spec: DpdArchiveSpec,
    *,
    source_date: str | date,
) -> dict[str, Any]:
    path = Path(archive_path)
    expected_date = _parse_source_date(source_date)
    expected_members = spec.expected_members

    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise ValueError(
                    f"{spec.cohort} archive contains duplicate members"
                )

            actual_names = set(names)
            missing = sorted(set(expected_members) - actual_names)
            unexpected = sorted(actual_names - set(expected_members))
            if missing or unexpected:
                details: list[str] = []
                if missing:
                    details.append("missing: " + ", ".join(missing))
                if unexpected:
                    details.append(
                        "unexpected: " + ", ".join(unexpected)
                    )
                raise ValueError(
                    f"{spec.cohort} archive member set mismatch, "
                    + "; ".join(details)
                )

            total_uncompressed = sum(info.file_size for info in infos)
            if total_uncompressed > MAX_UNCOMPRESSED_BYTES:
                raise ValueError(
                    f"{spec.cohort} archive exceeds the uncompressed "
                    "safety limit"
                )

            info_by_name = {info.filename: info for info in infos}
            members = [
                _validate_member(
                    archive,
                    info_by_name[name],
                    expected_columns=expected_members[name],
                    source_date=expected_date,
                )
                for name in expected_members
            ]
    except zipfile.BadZipFile as error:
        raise ValueError(
            f"{spec.cohort} archive is not a valid ZIP file"
        ) from error

    archive_sha256 = sha256_file(path)
    if archive_sha256 is None:
        raise ValueError(f"archive does not exist: {path}")

    return {
        "cohort": spec.cohort,
        "filename": spec.filename,
        "source_url": spec.source_url,
        "byte_size": path.stat().st_size,
        "sha256": archive_sha256,
        "member_count": len(members),
        "record_count": sum(
            member["record_count"] for member in members
        ),
        "members": members,
    }


def _snapshot_paths(root_path: Path, source_date: date) -> dict[str, Path]:
    date_text = source_date.isoformat()
    snapshot_relative = (
        Path("data") / "hc" / "dpd" / "snapshots" / date_text
    )
    return {
        "snapshot_relative": snapshot_relative,
        "snapshot": root_path / snapshot_relative,
        "manifest_relative": (
            Path("configs")
            / "datasets"
            / f"hc_dpd_{date_text}_manifest.yaml"
        ),
        "manifest": (
            root_path
            / "configs"
            / "datasets"
            / f"hc_dpd_{date_text}_manifest.yaml"
        ),
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(value, file, indent=2, ensure_ascii=True)
        file.write("\n")


def _build_validation_report(
    *,
    source_date: date,
    retrieved_at: datetime,
    archive_results: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "validation_schema_version": DPD_SNAPSHOT_SCHEMA_VERSION,
        "dataset_id": "hc_dpd_source_snapshot",
        "dataset_version": (
            f"{source_date.year}.{source_date.month}.{source_date.day}"
        ),
        "source_date": source_date.isoformat(),
        "retrieved_at_utc": _utc_text(retrieved_at),
        "parser_schema_version": DPD_PARSER_SCHEMA_VERSION,
        "official_documentation": {
            "extract_page": DPD_EXTRACT_PAGE_URL,
            "readme": DPD_README_URL,
            "license": DPD_LICENSE_URL,
        },
        "status": "passed",
        "validation_error_count": 0,
        "checks": [
            "HTTP status is 200 and content type is application/zip.",
            "The final download URL remains on the trusted www.canada.ca host.",
            "HTTP Last-Modified date matches the requested source date.",
            "Archive member names match the expected cohort schema.",
            "Archive members have safe paths and matching source dates.",
            "Every member decodes as UTF-8 and parses as strict CSV.",
            "Every nonblank row has the expected number of columns.",
            "Archive, member, byte-size, row-count, and hash evidence is recorded.",
        ],
        "archives": archive_results,
    }


def _manifest_source(
    spec: DpdArchiveSpec,
    *,
    source_date: date,
    retrieved_at: datetime,
) -> dict[str, Any]:
    return {
        "source_id": spec.source_id,
        "name": f"Health Canada DPD {spec.cohort} all-files extract",
        "source_url": spec.source_url,
        "retrieved_at_utc": _utc_text(retrieved_at),
        "source_modified_date": source_date.isoformat(),
        "license_or_terms": (
            "Open Government Licence - Canada; attribute Health Canada, "
            f"Drug Product Database; {DPD_LICENSE_URL}"
        ),
        "snapshot_notes": (
            "The dynamic official URL was downloaded once and its exact "
            "bytes were frozen. HTTP and ZIP validation evidence is in "
            "validation.json."
        ),
    }


def _build_manifest(
    *,
    source_date: date,
    retrieved_at: datetime,
    snapshot_relative: Path,
    archive_results: list[dict[str, Any]],
    validation_path: Path,
) -> dict[str, Any]:
    archive_by_cohort = {
        result["cohort"]: result for result in archive_results
    }
    version = f"{source_date.year}.{source_date.month}.{source_date.day}"
    license_text = (
        "Open Government Licence - Canada; attribute Health Canada, "
        f"Drug Product Database; {DPD_LICENSE_URL}"
    )

    sources = [
        _manifest_source(
            spec,
            source_date=source_date,
            retrieved_at=retrieved_at,
        )
        for spec in DPD_ARCHIVES
    ]
    sources.append(
        {
            "source_id": "evalanche_dpd_snapshot_builder",
            "name": "Evalanche DPD snapshot builder",
            "source_url": EVALANCHE_REPOSITORY_URL,
            "retrieved_at_utc": _utc_text(retrieved_at),
            "source_modified_date": None,
            "license_or_terms": "Evalanche repository terms.",
            "snapshot_notes": (
                "Generated validation.json from the two frozen official "
                "archives without modifying their bytes."
            ),
        }
    )

    files: list[dict[str, Any]] = []
    for spec in DPD_ARCHIVES:
        result = archive_by_cohort[spec.cohort]
        files.append(
            {
                "file_id": f"dpd_{spec.cohort}_archive",
                "source_id": spec.source_id,
                "role": f"raw_dpd_{spec.cohort}_archive",
                "relative_path": (
                    snapshot_relative / "raw" / spec.filename
                ).as_posix(),
                "media_type": "application/zip",
                "language": None,
                "byte_size": result["byte_size"],
                "sha256": result["sha256"],
                "record_count": result["record_count"],
                "record_count_method": "declared",
                "parser_schema_version": DPD_PARSER_SCHEMA_VERSION,
                "validation_status": "passed",
                "validation_error_count": 0,
            }
        )

    validation_sha256 = sha256_file(validation_path)
    if validation_sha256 is None:
        raise ValueError("validation report was not created")
    files.append(
        {
            "file_id": "dpd_source_validation",
            "source_id": "evalanche_dpd_snapshot_builder",
            "role": "source_validation_report",
            "relative_path": (
                snapshot_relative / "validation.json"
            ).as_posix(),
            "media_type": "application/json",
            "language": None,
            "byte_size": validation_path.stat().st_size,
            "sha256": validation_sha256,
            "record_count": None,
            "record_count_method": "not_applicable",
            "parser_schema_version": DPD_VALIDATION_SCHEMA_VERSION,
            "validation_status": "passed",
            "validation_error_count": 0,
        }
    )

    manifest = {
        "schema_version": "1.0",
        "release": {
            "dataset_id": "hc_dpd_source_snapshot",
            "version": version,
            "release_type": "source_snapshot",
            "title": (
                "Health Canada Drug Product Database source snapshot"
            ),
            "description": (
                "Immutable official DPD marketed and approved UTF-8 "
                f"extracts published {source_date.isoformat()}."
            ),
            "created_at_utc": _utc_text(retrieved_at),
            "status": "frozen",
            "immutable": True,
            "license_or_terms": license_text,
            "intended_use": (
                "Source data for reproducible Health Canada drug-product "
                "normalization, sampling, and benchmark construction."
            ),
            "limitations": [
                "The archives contain human, veterinary, disinfectant, and radiopharmaceutical products; downstream filtering is required.",
                "The snapshot contains marketed and approved cohorts only, not cancelled or dormant products.",
                "The source archives do not include product monograph PDF files.",
                "The official download URLs are dynamic; reproducibility depends on the frozen local bytes and hashes.",
                "Record counts sum heterogeneous relational tables and are not counts of unique products.",
            ],
            "languages": ["en", "fr"],
            "task_types": ["structured_reference_data"],
        },
        "sources": sources,
        "files": files,
        "lineage": {
            "parent_dataset_version": None,
            "code_version": (
                f"evalanche {__version__}; "
                f"hc_dpd_snapshot/{DPD_SNAPSHOT_SCHEMA_VERSION}"
            ),
            "transformations": [
                "Downloaded the official marketed and approved archives without modifying their bytes.",
                "Required each HTTP Last-Modified date to match the requested source date.",
                "Validated ZIP integrity, exact member sets, member paths, source dates, UTF-8 decoding, CSV structure, and expected column counts.",
                "Recorded archive and member byte sizes, row counts, CRC32 values, and SHA-256 hashes in validation.json.",
            ],
        },
    }
    DatasetManifest.model_validate(manifest)
    return manifest


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        yaml.safe_dump(
            value,
            file,
            sort_keys=False,
            allow_unicode=False,
            width=88,
        )


def create_dpd_source_snapshot(
    *,
    root_path: str | Path,
    source_date: str | date,
    timeout: float = 120.0,
    opener: UrlOpener = _default_url_opener,
    retrieved_at: datetime | None = None,
) -> dict[str, Any]:
    if timeout <= 0:
        raise ValueError("timeout must be greater than zero")

    root = Path(root_path).resolve()
    expected_date = _parse_source_date(source_date)
    paths = _snapshot_paths(root, expected_date)
    snapshot_path = paths["snapshot"]
    manifest_path = paths["manifest"]

    if snapshot_path.exists():
        raise ValueError(
            f"snapshot path already exists and will not be overwritten: "
            f"{snapshot_path}"
        )
    if manifest_path.exists():
        raise ValueError(
            f"manifest already exists and will not be overwritten: "
            f"{manifest_path}"
        )

    retrieval_time = retrieved_at or datetime.now(timezone.utc)
    if retrieval_time.tzinfo is None or retrieval_time.utcoffset() is None:
        raise ValueError("retrieved_at must include a timezone")
    retrieval_time = retrieval_time.astimezone(timezone.utc)

    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = Path(
        tempfile.mkdtemp(
            prefix=f".{expected_date.isoformat()}-",
            dir=snapshot_path.parent,
        )
    )
    manifest_temp: Path | None = None
    published_snapshot = False
    published_manifest = False

    try:
        archive_results: list[dict[str, Any]] = []
        for spec in DPD_ARCHIVES:
            archive_path = staging_path / "raw" / spec.filename
            http_evidence = _download_archive(
                spec,
                archive_path,
                source_date=expected_date,
                timeout=timeout,
                opener=opener,
            )
            validation = validate_dpd_archive(
                archive_path,
                spec,
                source_date=expected_date,
            )
            if validation["byte_size"] != http_evidence["byte_size"]:
                raise ValueError(
                    f"{spec.cohort} archive byte size changed during "
                    "validation"
                )
            if validation["sha256"] != http_evidence["sha256"]:
                raise ValueError(
                    f"{spec.cohort} archive hash changed during validation"
                )
            validation["http"] = http_evidence
            archive_results.append(validation)

        validation_report = _build_validation_report(
            source_date=expected_date,
            retrieved_at=retrieval_time,
            archive_results=archive_results,
        )
        validation_path = staging_path / "validation.json"
        _write_json(validation_path, validation_report)

        manifest = _build_manifest(
            source_date=expected_date,
            retrieved_at=retrieval_time,
            snapshot_relative=paths["snapshot_relative"],
            archive_results=archive_results,
            validation_path=validation_path,
        )

        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, manifest_temp_text = tempfile.mkstemp(
            prefix=f".{manifest_path.name}.",
            dir=manifest_path.parent,
        )
        os.close(file_descriptor)
        manifest_temp = Path(manifest_temp_text)
        _write_yaml(manifest_temp, manifest)

        os.replace(staging_path, snapshot_path)
        published_snapshot = True
        os.replace(manifest_temp, manifest_path)
        published_manifest = True
        manifest_temp = None

        verification = verify_dataset_manifest_file(
            manifest_path,
            root_path=root,
        )
        if not verification["valid"]:
            raise ValueError(
                "materialized DPD manifest failed file verification"
            )

        return {
            "dataset_id": manifest["release"]["dataset_id"],
            "dataset_version": manifest["release"]["version"],
            "source_date": expected_date.isoformat(),
            "snapshot_path": snapshot_path,
            "manifest_path": manifest_path,
            "archives": archive_results,
            "verification": verification,
        }
    except Exception:
        if staging_path.exists():
            shutil.rmtree(staging_path)
        if manifest_temp is not None and manifest_temp.exists():
            manifest_temp.unlink()
        if published_snapshot and snapshot_path.exists():
            shutil.rmtree(snapshot_path)
        if published_manifest and manifest_path.exists():
            manifest_path.unlink()
        raise
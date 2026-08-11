from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date
import os
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlparse

import yaml


CATALOG_SCHEMA_VERSION = "1.0"
EVIDENCE_STATUSES = frozenset({"current", "stale", "superseded"})
SOURCE_KINDS = frozenset({"living", "versioned", "static", "local"})
_ENTRY_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")


class EvidenceCatalogError(ValueError):
    """Raised when curated evidence metadata is incomplete or inconsistent."""


@dataclass(frozen=True)
class EvidenceEntry:
    entry_id: str
    title: str
    evidence_type: str
    source_kind: str
    status: str
    last_reviewed: date
    review_by: date
    document: str
    section: str
    source_urls: tuple[str, ...]
    version_scope: str
    review_trigger: str
    superseded_by: str | None = None
    supersession_note: str | None = None


@dataclass(frozen=True)
class EvidenceCatalog:
    schema_version: str
    status_report_as_of: date
    entries: tuple[EvidenceEntry, ...]


@dataclass(frozen=True)
class EvidenceStatusItem:
    entry: EvidenceEntry
    effective_status: str
    overdue_days: int


@dataclass(frozen=True)
class EvidenceStatusReport:
    as_of: date
    items: tuple[EvidenceStatusItem, ...]
    warnings: tuple[str, ...]

    @property
    def counts(self) -> dict[str, int]:
        counts = Counter(item.effective_status for item in self.items)
        return {
            status: int(counts.get(status, 0))
            for status in ("current", "stale", "superseded")
        }

    @property
    def outdated_count(self) -> int:
        counts = self.counts
        return counts["stale"] + counts["superseded"]


def _require_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvidenceCatalogError(f"{field} must be a mapping")
    return value


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceCatalogError(f"{field} must be non-empty text")
    return value.strip()


def _parse_date(value: Any, field: str) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            pass
    raise EvidenceCatalogError(f"{field} must be an ISO 8601 date")


def _optional_text(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return _require_text(value, field)


def _parse_urls(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise EvidenceCatalogError(f"{field} must be a list")

    urls: list[str] = []
    for index, raw_url in enumerate(value):
        url = _require_text(raw_url, f"{field}[{index}]")
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise EvidenceCatalogError(
                f"{field}[{index}] must be an HTTP or HTTPS URL"
            )
        urls.append(url)
    return tuple(urls)


def _parse_entry(raw_entry: Any, index: int) -> EvidenceEntry:
    field = f"entries[{index}]"
    raw = _require_mapping(raw_entry, field)
    entry_id = _require_text(raw.get("id"), f"{field}.id")
    if not _ENTRY_ID_PATTERN.fullmatch(entry_id):
        raise EvidenceCatalogError(
            f"{field}.id must use lowercase letters, numbers, '.', '_', or '-'"
        )

    status = _require_text(raw.get("status"), f"{field}.status").lower()
    if status not in EVIDENCE_STATUSES:
        choices = ", ".join(sorted(EVIDENCE_STATUSES))
        raise EvidenceCatalogError(f"{field}.status must be one of: {choices}")

    source_kind = _require_text(
        raw.get("source_kind"),
        f"{field}.source_kind",
    ).lower()
    if source_kind not in SOURCE_KINDS:
        choices = ", ".join(sorted(SOURCE_KINDS))
        raise EvidenceCatalogError(
            f"{field}.source_kind must be one of: {choices}"
        )

    source_urls = _parse_urls(raw.get("source_urls"), f"{field}.source_urls")
    if not source_urls and source_kind != "local":
        raise EvidenceCatalogError(
            f"{field}.source_urls cannot be empty unless source_kind is local"
        )

    last_reviewed = _parse_date(
        raw.get("last_reviewed"),
        f"{field}.last_reviewed",
    )
    review_by = _parse_date(raw.get("review_by"), f"{field}.review_by")
    if review_by < last_reviewed:
        raise EvidenceCatalogError(
            f"{field}.review_by cannot be before last_reviewed"
        )

    superseded_by = _optional_text(
        raw.get("superseded_by"),
        f"{field}.superseded_by",
    )
    supersession_note = _optional_text(
        raw.get("supersession_note"),
        f"{field}.supersession_note",
    )
    if status == "superseded" and not (
        superseded_by or supersession_note
    ):
        raise EvidenceCatalogError(
            f"{field} is superseded and requires superseded_by or "
            "supersession_note"
        )
    if status != "superseded" and (superseded_by or supersession_note):
        raise EvidenceCatalogError(
            f"{field} can use supersession metadata only when status is "
            "superseded"
        )

    return EvidenceEntry(
        entry_id=entry_id,
        title=_require_text(raw.get("title"), f"{field}.title"),
        evidence_type=_require_text(
            raw.get("evidence_type"),
            f"{field}.evidence_type",
        ),
        source_kind=source_kind,
        status=status,
        last_reviewed=last_reviewed,
        review_by=review_by,
        document=_require_text(raw.get("document"), f"{field}.document"),
        section=_require_text(raw.get("section"), f"{field}.section"),
        source_urls=source_urls,
        version_scope=_require_text(
            raw.get("version_scope"),
            f"{field}.version_scope",
        ),
        review_trigger=_require_text(
            raw.get("review_trigger"),
            f"{field}.review_trigger",
        ),
        superseded_by=superseded_by,
        supersession_note=supersession_note,
    )


def load_evidence_catalog(path: str | Path) -> EvidenceCatalog:
    catalog_path = Path(path)
    try:
        payload = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise EvidenceCatalogError(
            f"Evidence catalog is not valid YAML: {error}"
        ) from None

    raw = _require_mapping(payload, "catalog")
    schema_version = _require_text(
        raw.get("schema_version"),
        "schema_version",
    )
    if schema_version != CATALOG_SCHEMA_VERSION:
        raise EvidenceCatalogError(
            "schema_version must be " + CATALOG_SCHEMA_VERSION
        )

    raw_entries = raw.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise EvidenceCatalogError("entries must be a non-empty list")

    entries = tuple(
        _parse_entry(raw_entry, index)
        for index, raw_entry in enumerate(raw_entries)
    )
    entry_ids = [entry.entry_id for entry in entries]
    duplicate_ids = sorted(
        entry_id
        for entry_id, count in Counter(entry_ids).items()
        if count > 1
    )
    if duplicate_ids:
        raise EvidenceCatalogError(
            "Duplicate evidence entry IDs: " + ", ".join(duplicate_ids)
        )

    known_ids = set(entry_ids)
    for entry in entries:
        if entry.superseded_by == entry.entry_id:
            raise EvidenceCatalogError(
                f"{entry.entry_id} cannot supersede itself"
            )
        if entry.superseded_by and entry.superseded_by not in known_ids:
            raise EvidenceCatalogError(
                f"{entry.entry_id}.superseded_by references unknown entry "
                f"{entry.superseded_by}"
            )

    return EvidenceCatalog(
        schema_version=schema_version,
        status_report_as_of=_parse_date(
            raw.get("status_report_as_of"),
            "status_report_as_of",
        ),
        entries=entries,
    )


def _resolve_document(root: Path, document: str) -> Path:
    relative = Path(document)
    if relative.is_absolute() or ".." in relative.parts:
        raise EvidenceCatalogError(
            f"Evidence document must be repository-relative: {document}"
        )
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        raise EvidenceCatalogError(
            f"Evidence document escapes the repository root: {document}"
        ) from None
    return resolved


def evaluate_evidence_catalog(
    catalog: EvidenceCatalog,
    *,
    root_path: str | Path,
    as_of: date | None = None,
) -> EvidenceStatusReport:
    root = Path(root_path)
    effective_date = as_of or date.today()
    warnings: list[str] = []
    items: list[EvidenceStatusItem] = []

    for entry in catalog.entries:
        document_path = _resolve_document(root, entry.document)
        if not document_path.is_file():
            raise EvidenceCatalogError(
                f"{entry.entry_id} document does not exist: {entry.document}"
            )
        document_text = document_path.read_text(encoding="utf-8")
        heading_pattern = re.compile(
            rf"^#{{2,6}} {re.escape(entry.section)}$"
        )
        if not any(
            heading_pattern.fullmatch(line)
            for line in document_text.splitlines()
        ):
            raise EvidenceCatalogError(
                f"{entry.entry_id} section is missing from {entry.document}: "
                f"{entry.section}"
            )

        overdue_days = max((effective_date - entry.review_by).days, 0)
        effective_status = entry.status
        if entry.status == "current" and overdue_days:
            effective_status = "stale"
            warnings.append(
                f"{entry.entry_id} became stale {overdue_days} day(s) ago"
            )

        items.append(
            EvidenceStatusItem(
                entry=entry,
                effective_status=effective_status,
                overdue_days=overdue_days,
            )
        )

    status_order = {"stale": 0, "superseded": 1, "current": 2}
    items.sort(
        key=lambda item: (
            status_order[item.effective_status],
            item.entry.title.casefold(),
        )
    )
    return EvidenceStatusReport(
        as_of=effective_date,
        items=tuple(items),
        warnings=tuple(warnings),
    )


def _github_anchor(heading: str) -> str:
    normalized = heading.strip().lower()
    normalized = re.sub(r"[^a-z0-9 _-]", "", normalized)
    return normalized.replace(" ", "-")


def _entry_link(entry: EvidenceEntry, report_path: str | Path) -> str:
    report_directory = Path(report_path).parent
    relative_document = os.path.relpath(
        entry.document,
        start=report_directory.as_posix(),
    ).replace(os.sep, "/")
    return f"{relative_document}#{_github_anchor(entry.section)}"


def render_evidence_status_markdown(
    report: EvidenceStatusReport,
    *,
    report_path: str | Path = "docs/evidence/status.md",
) -> str:
    counts = report.counts
    lines = [
        "# Evidence maintenance status",
        "",
        f"> Snapshot generated for {report.as_of.isoformat()}. "
        "Run the offline evidence-status command before relying on this "
        "page for a new decision.",
        "",
        "This page tracks whether each curated source has been reviewed on "
        "schedule. Status describes maintenance only. It is not a model "
        "rating or an evidence-strength grade.",
        "",
        "## Summary",
        "",
        "| Current | Stale | Superseded | Total |",
        "| ---: | ---: | ---: | ---: |",
        (
            f"| {counts['current']} | {counts['stale']} | "
            f"{counts['superseded']} | {len(report.items)} |"
        ),
        "",
    ]

    outdated = [
        item for item in report.items if item.effective_status != "current"
    ]
    lines.extend(["## Needs attention", ""])
    if not outdated:
        lines.extend(
            [
                "No entries were stale or superseded on the snapshot date.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "| Status | Evidence | Last reviewed | Review by | Action |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for item in outdated:
            entry = item.entry
            link = _entry_link(entry, report_path)
            if item.effective_status == "stale":
                action = entry.review_trigger
            elif entry.superseded_by:
                action = f"Use `{entry.superseded_by}`."
            else:
                action = entry.supersession_note or "Do not rely on this entry."
            lines.append(
                f"| {item.effective_status.title()} | "
                f"[{entry.title}]({link}) | {entry.last_reviewed} | "
                f"{entry.review_by} | {action} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Full inventory",
            "",
            "| Status | Evidence | Type | Last reviewed | Review by | "
            "Version scope |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for item in report.items:
        entry = item.entry
        link = _entry_link(entry, report_path)
        lines.append(
            f"| {item.effective_status.title()} | "
            f"[{entry.title}]({link}) | {entry.evidence_type} | "
            f"{entry.last_reviewed} | {entry.review_by} | "
            f"{entry.version_scope} |"
        )

    lines.extend(
        [
            "",
            "## Maintenance rule",
            "",
            "Review a source by its review date, and review it sooner when "
            "its recorded trigger occurs. A changed score may reflect a "
            "changed model, dataset, harness, prompt, tool setup, or provider "
            "route, so updates still require human interpretation.",
            "",
            "The machine-readable metadata is in "
            "[`catalog.yaml`](catalog.yaml).",
            "",
        ]
    )
    return "\n".join(lines)


def build_evidence_status(
    *,
    root_path: str | Path,
    catalog_path: str | Path = "docs/evidence/catalog.yaml",
    as_of: date | None = None,
) -> tuple[EvidenceCatalog, EvidenceStatusReport]:
    root = Path(root_path)
    candidate = Path(catalog_path)
    if not candidate.is_absolute():
        candidate = root / candidate
    catalog = load_evidence_catalog(candidate)
    report = evaluate_evidence_catalog(
        catalog,
        root_path=root,
        as_of=as_of,
    )
    return catalog, report

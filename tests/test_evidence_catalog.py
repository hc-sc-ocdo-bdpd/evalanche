from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import yaml

from evalanche.cli import build_parser, run_evidence_status
from evalanche.evidence_catalog import (
    EvidenceCatalogError,
    build_evidence_status,
    load_evidence_catalog,
    render_evidence_status_markdown,
)


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "docs/evidence/catalog.yaml"
STATUS_PATH = ROOT / "docs/evidence/status.md"


def _entry(
    *,
    entry_id: str = "benchmark.example",
    title: str = "Example benchmark",
    status: str = "current",
    last_reviewed: str = "2026-08-01",
    review_by: str = "2026-08-31",
    superseded_by: str | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "id": entry_id,
        "title": title,
        "evidence_type": "Living benchmark",
        "source_kind": "living",
        "status": status,
        "last_reviewed": last_reviewed,
        "review_by": review_by,
        "document": "docs/evidence/example.md",
        "section": title,
        "source_urls": ["https://example.com/benchmark"],
        "version_scope": "Current official harness and dataset release",
        "review_trigger": "Review after a harness or dataset change.",
    }
    if superseded_by:
        result["superseded_by"] = superseded_by
    return result


def _write_catalog(
    root: Path,
    entries: list[dict[str, object]],
) -> Path:
    evidence_dir = root / "docs/evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    sections = "\n".join(
        f"## {entry['section']}\n\nFixture.\n" for entry in entries
    )
    (evidence_dir / "example.md").write_text(
        "# Evidence\n\n" + sections,
        encoding="utf-8",
    )
    path = evidence_dir / "catalog.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "status_report_as_of": "2026-08-10",
                "entries": entries,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def test_repository_catalog_is_complete_and_snapshot_is_reproducible() -> None:
    catalog, report = build_evidence_status(
        root_path=ROOT,
        catalog_path=CATALOG_PATH,
        as_of=date(2026, 8, 10),
    )

    assert len(catalog.entries) == 35
    assert report.counts == {
        "current": 35,
        "stale": 0,
        "superseded": 0,
    }
    assert all(entry.version_scope for entry in catalog.entries)
    assert all(entry.review_trigger for entry in catalog.entries)
    assert STATUS_PATH.read_text(encoding="utf-8") == (
        render_evidence_status_markdown(report)
    )


def test_current_entry_becomes_stale_after_review_date(tmp_path: Path) -> None:
    path = _write_catalog(tmp_path, [_entry()])

    _, report = build_evidence_status(
        root_path=tmp_path,
        catalog_path=path,
        as_of=date(2026, 9, 5),
    )

    assert report.counts["stale"] == 1
    assert report.items[0].effective_status == "stale"
    assert report.items[0].overdue_days == 5
    assert report.warnings == (
        "benchmark.example became stale 5 day(s) ago",
    )
    rendered = render_evidence_status_markdown(report)
    assert "## Needs attention" in rendered
    assert "| Stale | [Example benchmark]" in rendered
    assert "Review after a harness or dataset change." in rendered


def test_declared_superseded_entry_names_its_replacement(
    tmp_path: Path,
) -> None:
    old = _entry(
        entry_id="benchmark.old",
        title="Old benchmark",
        status="superseded",
        superseded_by="benchmark.new",
    )
    new = _entry(
        entry_id="benchmark.new",
        title="New benchmark",
    )
    path = _write_catalog(tmp_path, [old, new])

    _, report = build_evidence_status(
        root_path=tmp_path,
        catalog_path=path,
        as_of=date(2026, 8, 10),
    )

    assert report.counts["superseded"] == 1
    assert report.items[0].entry.entry_id == "benchmark.old"
    rendered = render_evidence_status_markdown(report)
    assert "Use `benchmark.new`." in rendered


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("status", "unknown", "status must be one of"),
        ("last_reviewed", "August 1", "must be an ISO 8601 date"),
        ("review_trigger", "", "must be non-empty text"),
        ("version_scope", "", "must be non-empty text"),
        ("source_urls", [], "cannot be empty"),
    ],
)
def test_required_metadata_is_validated(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    entry = _entry()
    entry[field] = value
    path = _write_catalog(tmp_path, [entry])

    with pytest.raises(EvidenceCatalogError, match=message):
        load_evidence_catalog(path)


def test_superseded_entry_requires_a_known_replacement_or_note(
    tmp_path: Path,
) -> None:
    entry = _entry(status="superseded")
    path = _write_catalog(tmp_path, [entry])
    with pytest.raises(EvidenceCatalogError, match="requires superseded_by"):
        load_evidence_catalog(path)

    entry["superseded_by"] = "benchmark.missing"
    path = _write_catalog(tmp_path, [entry])
    with pytest.raises(EvidenceCatalogError, match="references unknown entry"):
        load_evidence_catalog(path)


def test_catalog_document_section_must_exist(tmp_path: Path) -> None:
    entry = _entry()
    path = _write_catalog(tmp_path, [entry])
    entry["section"] = "Missing section"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "status_report_as_of": "2026-08-10",
                "entries": [entry],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(EvidenceCatalogError, match="section is missing"):
        build_evidence_status(root_path=tmp_path, catalog_path=path)


def test_cli_writes_and_checks_status_without_network_calls(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _write_catalog(tmp_path, [_entry()])
    output = "docs/evidence/status.md"

    run_evidence_status(
        root_path=str(tmp_path),
        catalog_path=str(path),
        as_of_value="2026-08-10",
        output_path=output,
        write=True,
    )
    write_output = capsys.readouterr().out
    assert "Status: CURRENT" in write_output
    assert "No network or model calls were made." in write_output

    run_evidence_status(
        root_path=str(tmp_path),
        catalog_path=str(path),
        output_path=output,
        check=True,
    )
    check_output = capsys.readouterr().out
    assert "Evidence status snapshot matches" in check_output


def test_cli_can_fail_when_review_is_required(tmp_path: Path) -> None:
    path = _write_catalog(tmp_path, [_entry(status="stale")])

    with pytest.raises(SystemExit) as error:
        run_evidence_status(
            root_path=str(tmp_path),
            catalog_path=str(path),
            as_of_value="2026-08-10",
            fail_on_outdated=True,
        )

    assert error.value.code == 2


def test_evidence_status_parser_exposes_offline_controls() -> None:
    args = build_parser().parse_args(
        [
            "evidence-status",
            "--as-of",
            "2026-08-10",
            "--check",
            "--fail-on-outdated",
        ]
    )

    assert args.command == "evidence-status"
    assert args.as_of == "2026-08-10"
    assert args.check is True
    assert args.fail_on_outdated is True

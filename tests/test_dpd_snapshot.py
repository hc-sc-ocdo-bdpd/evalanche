import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request

import pytest

from evalanche.cli import build_parser
from evalanche.dataset_manifest import (
    load_dataset_manifest,
    verify_dataset_manifest_file,
)
from evalanche.dpd_snapshot import (
    DPD_ARCHIVES,
    create_dpd_source_snapshot,
    validate_dpd_archive,
)

SOURCE_DATE = "2026-07-02"
LAST_MODIFIED = "Thu, 02 Jul 2026 15:27:00 GMT"
RETRIEVED_AT = datetime(2026, 7, 22, 14, 30, tzinfo=timezone.utc)


class FakeResponse(io.BytesIO):
    status = 200

    def __init__(
        self,
        value: bytes,
        *,
        url: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(value)
        self._url = url
        self.headers = headers or {
            "Content-Type": "application/zip",
            "Last-Modified": LAST_MODIFIED,
            "Date": "Wed, 22 Jul 2026 14:30:00 GMT",
        }

    def geturl(self) -> str:
        return self._url


def make_archive(
    spec: Any,
    *,
    missing_member: str | None = None,
    wrong_columns_member: str | None = None,
) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for name, column_count in spec.expected_members.items():
            if name == missing_member:
                continue
            actual_count = (
                column_count + 1
                if name == wrong_columns_member
                else column_count
            )
            row = ",".join(f'"value_{index}"' for index in range(actual_count))
            info = zipfile.ZipInfo(
                filename=name,
                date_time=(2026, 7, 2, 6, 0, 0),
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, row + "\n")
    return output.getvalue()


def make_opener(
    archives: dict[str, bytes],
    *,
    headers: dict[str, str] | None = None,
):
    calls: list[tuple[str, float]] = []

    def opener(request: Request, timeout: float) -> FakeResponse:
        calls.append((request.full_url, timeout))
        return FakeResponse(
            archives[request.full_url],
            url=request.full_url,
            headers=headers,
        )

    opener.calls = calls
    return opener


def fake_archives() -> dict[str, bytes]:
    return {
        spec.source_url: make_archive(spec) for spec in DPD_ARCHIVES
    }


def test_create_dpd_source_snapshot_is_frozen_and_verifiable(
    tmp_path: Path,
) -> None:
    opener = make_opener(fake_archives())

    result = create_dpd_source_snapshot(
        root_path=tmp_path,
        source_date=SOURCE_DATE,
        timeout=30.0,
        opener=opener,
        retrieved_at=RETRIEVED_AT,
    )

    assert len(opener.calls) == 2
    assert result["dataset_version"] == "2026.7.2"
    assert result["verification"]["valid"] is True

    manifest = load_dataset_manifest(result["manifest_path"])
    assert manifest.release.release_type == "source_snapshot"
    assert manifest.release.status == "frozen"
    assert manifest.release.immutable is True
    assert manifest.sampling is None
    assert manifest.splits == []
    assert len(manifest.files) == 3
    assert [file.record_count for file in manifest.files[:2]] == [12, 12]

    verification = verify_dataset_manifest_file(
        result["manifest_path"],
        root_path=tmp_path,
    )
    assert verification["valid"] is True
    assert verification["files_passed"] == 3

    report_path = result["snapshot_path"] / "validation.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["validation_error_count"] == 0
    assert [archive["member_count"] for archive in report["archives"]] == [
        12,
        12,
    ]


def test_snapshot_rejects_unexpected_source_date_without_partial_release(
    tmp_path: Path,
) -> None:
    headers = {
        "Content-Type": "application/zip",
        "Last-Modified": "Fri, 03 Jul 2026 15:27:00 GMT",
        "Date": "Wed, 22 Jul 2026 14:30:00 GMT",
    }
    opener = make_opener(fake_archives(), headers=headers)

    with pytest.raises(ValueError, match="source date mismatch"):
        create_dpd_source_snapshot(
            root_path=tmp_path,
            source_date=SOURCE_DATE,
            opener=opener,
            retrieved_at=RETRIEVED_AT,
        )

    assert not (
        tmp_path / "data/hc/dpd/snapshots/2026-07-02"
    ).exists()
    assert not (
        tmp_path / "configs/datasets/hc_dpd_2026-07-02_manifest.yaml"
    ).exists()


def test_snapshot_rejects_redirect_outside_trusted_host(
    tmp_path: Path,
) -> None:
    archives = fake_archives()

    def opener(request: Request, timeout: float) -> FakeResponse:
        return FakeResponse(
            archives[request.full_url],
            url="https://example.com/allfiles.zip",
        )

    with pytest.raises(ValueError, match="trusted Health Canada host"):
        create_dpd_source_snapshot(
            root_path=tmp_path,
            source_date=SOURCE_DATE,
            opener=opener,
            retrieved_at=RETRIEVED_AT,
        )

    assert not (
        tmp_path / "data/hc/dpd/snapshots/2026-07-02"
    ).exists()


def test_snapshot_refuses_to_overwrite_an_existing_release(
    tmp_path: Path,
) -> None:
    snapshot_path = tmp_path / "data/hc/dpd/snapshots/2026-07-02"
    snapshot_path.mkdir(parents=True)
    opener = make_opener(fake_archives())

    with pytest.raises(ValueError, match="will not be overwritten"):
        create_dpd_source_snapshot(
            root_path=tmp_path,
            source_date=SOURCE_DATE,
            opener=opener,
            retrieved_at=RETRIEVED_AT,
        )

    assert opener.calls == []


def test_archive_validation_rejects_missing_member(tmp_path: Path) -> None:
    spec = DPD_ARCHIVES[0]
    missing = next(iter(spec.expected_members))
    archive_path = tmp_path / spec.filename
    archive_path.write_bytes(make_archive(spec, missing_member=missing))

    with pytest.raises(ValueError, match="member set mismatch"):
        validate_dpd_archive(
            archive_path,
            spec,
            source_date=SOURCE_DATE,
        )


def test_archive_validation_rejects_wrong_column_count(
    tmp_path: Path,
) -> None:
    spec = DPD_ARCHIVES[0]
    wrong = next(iter(spec.expected_members))
    archive_path = tmp_path / spec.filename
    archive_path.write_bytes(
        make_archive(spec, wrong_columns_member=wrong)
    )

    with pytest.raises(ValueError, match="columns, expected"):
        validate_dpd_archive(
            archive_path,
            spec,
            source_date=SOURCE_DATE,
        )


def test_cli_exposes_fail_closed_dpd_snapshot_command() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "snapshot-dpd",
            "--source-date",
            SOURCE_DATE,
            "--root",
            ".",
        ]
    )

    assert args.command == "snapshot-dpd"
    assert args.source_date == SOURCE_DATE
    assert args.timeout == 120.0


def test_materialized_dpd_snapshot_verifies() -> None:
    manifest_path = Path(
        "configs/datasets/hc_dpd_2026-07-02_manifest.yaml"
    )

    verification = verify_dataset_manifest_file(
        manifest_path,
        root_path=".",
    )

    assert verification["valid"] is True, verification
    assert verification["files_passed"] == 3


@pytest.mark.parametrize(
    ("spec", "expected_sha256", "expected_record_count"),
    [
        (
            DPD_ARCHIVES[0],
            "94c8ef8639fd07eea9c134d8ede137d77574d4ae6761955f5aa582558ec38b25",
            170236,
        ),
        (
            DPD_ARCHIVES[1],
            "410fd4da8b01ccfa42a15a4d226830619dd91b777684f505975404ec075ac440",
            40396,
        ),
    ],
)
def test_materialized_archives_pass_source_specific_validation(
    spec: Any,
    expected_sha256: str,
    expected_record_count: int,
) -> None:
    archive_path = (
        Path("data/hc/dpd/snapshots/2026-07-02/raw")
        / spec.filename
    )

    result = validate_dpd_archive(
        archive_path,
        spec,
        source_date=SOURCE_DATE,
    )

    assert result["sha256"] == expected_sha256
    assert result["record_count"] == expected_record_count
    assert result["member_count"] == 12
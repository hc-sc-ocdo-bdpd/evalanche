import copy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from evalanche.cli import build_parser, run_verify_dataset
from evalanche.dataset_manifest import (
    DatasetManifest,
    load_dataset_manifest,
    save_dataset_verification,
    sha256_file,
    verify_dataset_manifest_file,
)


def make_manifest(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    data_path = tmp_path / "cases.csv"
    data_path.write_text(
        "case_id,input\ncase_001,First\ncase_002,Second\n",
        encoding="utf-8",
    )
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "release": {
            "dataset_id": "test_dataset",
            "version": "1.2.0",
            "release_type": "benchmark",
            "title": "Test dataset",
            "description": "A test dataset manifest.",
            "created_at_utc": "2026-07-21T12:00:00Z",
            "status": "frozen",
            "immutable": True,
            "license_or_terms": "Test terms.",
            "intended_use": "Unit tests.",
            "limitations": ["Synthetic data."],
            "languages": ["en"],
            "task_types": ["classification"],
        },
        "sources": [
            {
                "source_id": "test_source",
                "name": "Test source",
                "source_url": "https://example.com/source",
                "retrieved_at_utc": "2026-07-21T12:00:00Z",
                "source_modified_date": "2026-07-20",
                "license_or_terms": "Test terms.",
                "snapshot_notes": "Frozen test fixture.",
            }
        ],
        "files": [
            {
                "file_id": "cases",
                "source_id": "test_source",
                "role": "benchmark_cases",
                "relative_path": "cases.csv",
                "media_type": "text/csv",
                "language": "en",
                "byte_size": data_path.stat().st_size,
                "sha256": sha256_file(data_path),
                "record_count": 2,
                "record_count_method": "csv_rows",
                "parser_schema_version": "cases/1.0",
                "validation_status": "passed",
                "validation_error_count": 0,
            }
        ],
        "sampling": {
            "method": "random",
            "unit": "case",
            "population_description": "Two synthetic cases.",
            "target_count": 2,
            "membership_file_id": "cases",
            "member_id_column": "case_id",
            "seed": 42,
            "inclusion_criteria": ["Complete case."],
            "exclusion_criteria": ["Incomplete case."],
            "strata": [],
        },
        "splits": [
            {
                "name": "development",
                "purpose": "Development tests.",
                "unit": "case",
                "target_count": 1,
                "group_key": "case_id",
                "selection_policy": "First selected case.",
                "member_ids": ["case_001"],
            },
            {
                "name": "heldout",
                "purpose": "Held-out tests.",
                "unit": "case",
                "target_count": 1,
                "group_key": "case_id",
                "selection_policy": "Second selected case.",
                "member_ids": ["case_002"],
            },
        ],
        "lineage": {
            "parent_dataset_version": None,
            "code_version": "test-code-version",
            "transformations": ["Created synthetic rows."],
        },
    }
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )
    return manifest_path, manifest


def test_working_example_manifest_verifies() -> None:
    manifest_path = Path(
        "configs/datasets/generic_example_manifest.yaml"
    )

    verification = verify_dataset_manifest_file(
        manifest_path,
        root_path=".",
    )

    assert verification["valid"] is True, verification
    assert verification["files_passed"] == 1
    assert verification["files"][0]["actual_record_count"] == 6
    assert verification["files"][0][
        "sampling_membership_verified"
    ] is True


def test_load_dataset_manifest_parses_valid_yaml(tmp_path: Path) -> None:
    manifest_path, _ = make_manifest(tmp_path)

    manifest = load_dataset_manifest(manifest_path)

    assert manifest.schema_version == "1.0"
    assert manifest.release.dataset_id == "test_dataset"
    assert manifest.splits[1].member_ids == ["case_002"]


def test_source_snapshot_does_not_require_sampling_or_splits(
    tmp_path: Path,
) -> None:
    _, raw = make_manifest(tmp_path)
    raw["release"]["release_type"] = "source_snapshot"
    raw.pop("sampling")
    raw.pop("splits")

    manifest = DatasetManifest.model_validate(raw)

    assert manifest.sampling is None
    assert manifest.splits == []


def test_manifest_rejects_unknown_fields(tmp_path: Path) -> None:
    _, raw = make_manifest(tmp_path)
    raw["unknown_field"] = True

    with pytest.raises(ValidationError, match="unknown_field"):
        DatasetManifest.model_validate(raw)


@pytest.mark.parametrize(
    "relative_path",
    [
        "../cases.csv",
        "/tmp/cases.csv",
        "C:/temp/cases.csv",
        "data\\cases.csv",
    ],
)
def test_manifest_rejects_unsafe_or_nonportable_paths(
    tmp_path: Path,
    relative_path: str,
) -> None:
    _, raw = make_manifest(tmp_path)
    raw["files"][0]["relative_path"] = relative_path

    with pytest.raises(ValidationError, match="relative_path"):
        DatasetManifest.model_validate(raw)


def test_random_sampling_requires_a_seed(tmp_path: Path) -> None:
    _, raw = make_manifest(tmp_path)
    raw["sampling"]["seed"] = None

    with pytest.raises(ValidationError, match="require a seed"):
        DatasetManifest.model_validate(raw)


def test_frozen_release_must_be_immutable(tmp_path: Path) -> None:
    _, raw = make_manifest(tmp_path)
    raw["release"]["immutable"] = False

    with pytest.raises(ValidationError, match="must be immutable"):
        DatasetManifest.model_validate(raw)


def test_frozen_release_requires_passed_file_validation(
    tmp_path: Path,
) -> None:
    _, raw = make_manifest(tmp_path)
    raw["files"][0]["validation_status"] = "not_run"

    with pytest.raises(ValidationError, match="producer validation"):
        DatasetManifest.model_validate(raw)


def test_files_must_reference_a_declared_source(tmp_path: Path) -> None:
    _, raw = make_manifest(tmp_path)
    raw["files"][0]["source_id"] = "missing_source"

    with pytest.raises(ValidationError, match="unknown source_id"):
        DatasetManifest.model_validate(raw)


def test_split_counts_must_match_sampling_target(tmp_path: Path) -> None:
    _, raw = make_manifest(tmp_path)
    raw["sampling"]["target_count"] = 3

    with pytest.raises(ValidationError, match="split target counts"):
        DatasetManifest.model_validate(raw)


def test_split_members_cannot_overlap(tmp_path: Path) -> None:
    _, raw = make_manifest(tmp_path)
    raw["splits"][1]["member_ids"] = ["case_001"]

    with pytest.raises(ValidationError, match="cannot overlap"):
        DatasetManifest.model_validate(raw)


def test_verifier_cross_checks_split_membership(tmp_path: Path) -> None:
    manifest_path, raw = make_manifest(tmp_path)
    changed = copy.deepcopy(raw)
    changed["splits"][1]["member_ids"] = ["case_999"]
    manifest_path.write_text(
        yaml.safe_dump(changed, sort_keys=False),
        encoding="utf-8",
    )

    verification = verify_dataset_manifest_file(
        manifest_path,
        root_path=tmp_path,
    )

    issues = verification["files"][0]["issues"]
    assert verification["valid"] is False
    assert any("case_999" in issue for issue in issues)
    assert any("case_002" in issue for issue in issues)


def test_jsonl_records_and_membership_are_verified(tmp_path: Path) -> None:
    manifest_path, raw = make_manifest(tmp_path)
    jsonl_path = tmp_path / "cases.jsonl"
    jsonl_path.write_text(
        '{"case_id":"case_001","input":"First"}\n'
        '{"case_id":"case_002","input":"Second"}\n',
        encoding="utf-8",
    )
    raw["files"][0].update(
        {
            "relative_path": "cases.jsonl",
            "media_type": "application/x-ndjson",
            "byte_size": jsonl_path.stat().st_size,
            "sha256": sha256_file(jsonl_path),
            "record_count_method": "jsonl_records",
        }
    )
    manifest_path.write_text(
        yaml.safe_dump(raw, sort_keys=False),
        encoding="utf-8",
    )

    verification = verify_dataset_manifest_file(
        manifest_path,
        root_path=tmp_path,
    )

    assert verification["valid"] is True
    assert verification["files"][0]["actual_record_count"] == 2
    assert verification["files"][0][
        "sampling_membership_verified"
    ] is True


def test_passed_file_cannot_declare_validation_errors(
    tmp_path: Path,
) -> None:
    _, raw = make_manifest(tmp_path)
    raw["files"][0]["validation_error_count"] = 1

    with pytest.raises(ValidationError, match="passed file"):
        DatasetManifest.model_validate(raw)


def test_pdf_file_requires_page_count(tmp_path: Path) -> None:
    _, raw = make_manifest(tmp_path)
    raw["files"][0].update(
        {
            "media_type": "application/pdf",
            "record_count": None,
            "record_count_method": "not_applicable",
        }
    )

    with pytest.raises(ValidationError, match="page_count"):
        DatasetManifest.model_validate(raw)


def test_verifier_detects_size_and_hash_mismatches(
    tmp_path: Path,
) -> None:
    manifest_path, _ = make_manifest(tmp_path)
    (tmp_path / "cases.csv").write_text(
        "case_id,input\ncase_001,Changed\n",
        encoding="utf-8",
    )

    verification = verify_dataset_manifest_file(
        manifest_path,
        root_path=tmp_path,
    )

    issues = verification["files"][0]["issues"]
    assert verification["valid"] is False
    assert any("byte size mismatch" in issue for issue in issues)
    assert any("SHA-256 mismatch" in issue for issue in issues)


def test_verifier_detects_record_count_mismatch(tmp_path: Path) -> None:
    manifest_path, raw = make_manifest(tmp_path)
    changed = copy.deepcopy(raw)
    changed["files"][0]["record_count"] = 3
    manifest_path.write_text(
        yaml.safe_dump(changed, sort_keys=False),
        encoding="utf-8",
    )

    verification = verify_dataset_manifest_file(
        manifest_path,
        root_path=tmp_path,
    )

    assert verification["valid"] is False
    assert "record count mismatch" in verification["files"][0][
        "issues"
    ][0]


def test_save_verification_writes_auditable_json(tmp_path: Path) -> None:
    manifest_path, _ = make_manifest(tmp_path)
    verification = verify_dataset_manifest_file(
        manifest_path,
        root_path=tmp_path,
    )
    output_path = tmp_path / "verification.json"

    saved_path = save_dataset_verification(verification, output_path)
    saved = json.loads(saved_path.read_text(encoding="utf-8"))

    assert saved_path == output_path
    assert saved["valid"] is True
    assert len(saved["manifest_sha256"]) == 64


def test_cli_parser_accepts_dataset_verification_command() -> None:
    args = build_parser().parse_args(
        [
            "verify-dataset",
            "--manifest",
            "manifest.yaml",
            "--root",
            "data",
            "--output",
            "verification.json",
        ]
    )

    assert args.command == "verify-dataset"
    assert args.manifest == "manifest.yaml"
    assert args.root == "data"
    assert args.output == "verification.json"


def test_cli_writes_verification_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest_path, _ = make_manifest(tmp_path)
    output_path = tmp_path / "verification.json"

    verification = run_verify_dataset(
        str(manifest_path),
        root_path=str(tmp_path),
        output_path=str(output_path),
    )

    output = capsys.readouterr().out
    assert verification["valid"] is True
    assert output_path.exists()
    assert "Status: VALID" in output


def test_cli_exits_nonzero_when_file_integrity_fails(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest_path, _ = make_manifest(tmp_path)
    (tmp_path / "cases.csv").write_text(
        "case_id,input\ncase_001,Changed\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as error:
        run_verify_dataset(
            str(manifest_path),
            root_path=str(tmp_path),
        )

    output = capsys.readouterr().out
    assert error.value.code == 1
    assert "Status: INVALID" in output
    assert "SHA-256 mismatch" in output
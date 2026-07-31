import json
from pathlib import Path

import pandas as pd

from evalanche.dpd_evidence_audit import (
    DPD_EVIDENCE_AUDIT_ANALYSIS_MANIFEST_PATH,
    DPD_EVIDENCE_AUDIT_OUTPUT_PATH,
    DPD_EVIDENCE_AUDIT_RELEASE_MANIFEST_PATH,
    _refresh_release_manifests,
    build_dpd_evidence_audit,
    parse_rendered_dpd_record,
)


ROOT = Path(__file__).resolve().parents[1]


def test_parse_rendered_dpd_record_preserves_compound_labels() -> None:
    source = """Extract facts.

DPD SOURCE RECORD (ENGLISH)

VARIANT 1
DRUG_CODE: 1
DIN: 01234567
BRAND_NAME: TEST DRUG
ACTIVE_INGREDIENTS:
- NAME=ACTIVE; STRENGTH=5.0; UNIT=MG
DOSAGE_FORMS: SPRAY, BAG-ON-VALVE | SOLUTION
ROUTES: ORAL
SCHEDULES: PRESCRIPTION
PRODUCT_STATUS: MARKETED
COMPANY: TEST COMPANY
"""
    parsed = parse_rendered_dpd_record(source, language="en")
    assert parsed["active_ingredients"] == [
        {"name": "ACTIVE", "strength": 5, "unit": "MG"}
    ]
    assert parsed["dosage_forms"] == [
        "SOLUTION",
        "SPRAY, BAG-ON-VALVE",
    ]


def test_published_dpd_evidence_audit_is_reproducible(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "evidence_audit.csv"
    summary_path = tmp_path / "evidence_audit_summary.json"
    result = build_dpd_evidence_audit(
        root_path=ROOT,
        output_path=output_path,
        summary_path=summary_path,
    )

    assert result["verification"] == {
        "review_cases": 105,
        "source_rebuild_matches": 105,
        "independent_expected_parses_match": 105,
        "model_outputs_rescored": 420,
        "strict_score_disagreements": 0,
    }
    assert result["outcomes"]["adjudication_counts"] == {
        "retain_result": 105
    }
    assert result["outcomes"]["leaderboard_scores_changed"] is False
    assert result["release_manifests_refreshed"] is False
    assert output_path.read_bytes() == (
        ROOT / DPD_EVIDENCE_AUDIT_OUTPUT_PATH
    ).read_bytes()

    audit = pd.read_csv(output_path, keep_default_na=False)
    assert len(audit) == 105
    assert set(audit["review_status"]) == {"complete"}
    assert set(audit["human_signoff_status"]) == {"not_claimed"}
    assert set(audit["adjudication"]) == {"retain_result"}

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "automated_evidence_audit_complete"
    assert summary["human_signoff_status"] == "not_claimed"


def test_default_audit_outputs_refresh_release_manifests(
    tmp_path: Path,
) -> None:
    analysis_directory = (
        tmp_path / DPD_EVIDENCE_AUDIT_ANALYSIS_MANIFEST_PATH.parent
    )
    analysis_directory.mkdir(parents=True)
    base_files = [
        "README.md",
        "model_overview.csv",
        "slice_performance.csv",
        "field_accuracy.csv",
        "error_taxonomy.csv",
        "pairwise_tradeoffs.csv",
        "frontier_cases.csv",
        "manual_review.csv",
    ]
    audit_files = [
        "evidence_audit.csv",
        "evidence_audit_summary.json",
        "EVIDENCE_AUDIT.md",
    ]
    for filename in (*base_files, *audit_files):
        (analysis_directory / filename).write_text(
            f"{filename}\n",
            encoding="utf-8",
        )

    analysis_manifest_path = (
        tmp_path / DPD_EVIDENCE_AUDIT_ANALYSIS_MANIFEST_PATH
    )
    analysis_manifest_path.write_text(
        json.dumps(
            {
                "analysis": {
                    "diagnostic_repairs_change_primary_score": False,
                },
                "review_set": {"status": "pending_human_review"},
                "published_files": [
                    {
                        "path": (
                            DPD_EVIDENCE_AUDIT_ANALYSIS_MANIFEST_PATH.parent
                            / filename
                        ).as_posix()
                    }
                    for filename in base_files
                ],
            }
        ),
        encoding="utf-8",
    )
    other_path = tmp_path / "reports/other.csv"
    other_path.parent.mkdir(parents=True, exist_ok=True)
    other_path.write_text("other\n", encoding="utf-8")
    release_manifest_path = (
        tmp_path / DPD_EVIDENCE_AUDIT_RELEASE_MANIFEST_PATH
    )
    release_manifest_path.write_text(
        json.dumps(
            {
                "analysis": {},
                "validation": {},
                "published_files": [
                    {
                        "path": "reports/other.csv",
                        "role": "other",
                        "size_bytes": other_path.stat().st_size,
                        "sha256": "placeholder",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    summary = {
        "status": "automated_evidence_audit_complete",
        "human_signoff_status": "not_claimed",
        "verification": {"review_cases": 105},
        "outcomes": {"leaderboard_scores_changed": False},
    }

    refreshed = _refresh_release_manifests(
        root=tmp_path,
        output_file=analysis_directory / "evidence_audit.csv",
        summary_file=(
            analysis_directory / "evidence_audit_summary.json"
        ),
        summary=summary,
    )

    assert refreshed is True
    analysis_manifest = json.loads(
        analysis_manifest_path.read_text(encoding="utf-8")
    )
    assert analysis_manifest["review_set"]["status"] == (
        "automated_evidence_audit_complete"
    )
    assert len(analysis_manifest["published_files"]) == 11
    release_manifest = json.loads(
        release_manifest_path.read_text(encoding="utf-8")
    )
    assert release_manifest["validation"][
        "automated_evidence_audit"
    ] == "complete"
    assert len(release_manifest["published_files"]) == 13

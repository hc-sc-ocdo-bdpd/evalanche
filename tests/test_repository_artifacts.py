import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_RELEASE = ROOT / "reports/hc_dpd_census/0.2.0"
ANALYSIS_RELEASE = RESULT_RELEASE / "analysis"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_published_result_release_integrity() -> None:
    manifest = json.loads(
        (RESULT_RELEASE / "manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["status"] == "published"
    assert manifest["benchmark"]["case_count"] == 14_034
    assert manifest["benchmark"]["product_family_count"] == 7_017
    assert manifest["validation"]["generation_complete"] is True
    assert manifest["validation"]["generation_failures"] == 0
    assert manifest["validation"]["automated_error_analysis"] == (
        "complete"
    )
    assert manifest["validation"][
        "grouped_product_family_analysis"
    ] == "complete"
    assert manifest["validation"]["manual_review_package"] == "ready"
    assert manifest["validation"]["automated_evidence_audit"] == (
        "complete"
    )
    assert manifest["validation"]["manual_label_review"] == (
        "not_performed"
    )
    assert manifest["validation"]["human_signoff"] == "not_claimed"
    assert manifest["validation"]["error_adjudication"] == (
        "complete_automated"
    )

    for artifact in manifest["published_files"]:
        path = ROOT / artifact["path"]
        assert path.is_file()
        assert path.stat().st_size == artifact["size_bytes"]
        assert _sha256(path) == artifact["sha256"]


def test_published_analysis_release_integrity() -> None:
    release_manifest = json.loads(
        (RESULT_RELEASE / "manifest.json").read_text(encoding="utf-8")
    )
    analysis_manifest = json.loads(
        (ANALYSIS_RELEASE / "analysis_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    retained_results = next(
        artifact
        for artifact in release_manifest["retained_raw_artifacts"]
        if artifact["role"] == "case_level_evaluation_results"
    )
    assert analysis_manifest["source"]["results_path"] == (
        retained_results["path"]
    )
    assert analysis_manifest["source"]["results_size_bytes"] == (
        retained_results["size_bytes"]
    )
    assert analysis_manifest["source"]["results_sha256"] == (
        retained_results["sha256"]
    )
    assert analysis_manifest["source"]["rows"] == 56_136
    assert analysis_manifest["review_set"] == {
        "seed": 20260730,
        "primary_model_failures": 13,
        "comparison_only_failures": 42,
        "shared_lower_tier_failure_sample": 25,
        "all_model_pass_sample": 25,
        "total_unique_cases": 105,
        "status": "automated_evidence_audit_complete",
    }
    assert analysis_manifest["analysis"][
        "diagnostic_repairs_change_primary_score"
    ] is False
    assert analysis_manifest["analysis"][
        "manual_adjudication_complete"
    ] is False
    assert analysis_manifest["analysis"][
        "automated_evidence_audit_complete"
    ] is True
    assert analysis_manifest["analysis"][
        "leaderboard_scores_changed_by_audit"
    ] is False
    assert analysis_manifest["analysis"]["human_signoff_status"] == (
        "not_claimed"
    )
    assert analysis_manifest["evidence_audit"]["verification"] == {
        "review_cases": 105,
        "source_rebuild_matches": 105,
        "independent_expected_parses_match": 105,
        "model_outputs_rescored": 420,
        "strict_score_disagreements": 0,
    }
    assert analysis_manifest["evidence_audit"]["outcomes"][
        "leaderboard_scores_changed"
    ] is False

    for artifact in analysis_manifest["published_files"]:
        path = ROOT / artifact["path"]
        assert path.is_file()
        assert path.stat().st_size == artifact["size_bytes"]
        assert _sha256(path) == artifact["sha256"]


def test_published_result_headline_values() -> None:
    with (RESULT_RELEASE / "model_summary.csv").open(
        encoding="utf-8",
        newline="",
    ) as file:
        rows = {
            row["model_name"]: row
            for row in csv.DictReader(file)
        }

    assert set(rows) == {
        "gpt_5_4_mini",
        "gpt_5_6_luna",
        "gpt_5_6_terra",
        "gpt_5_6_sol",
    }
    assert rows["gpt_5_6_sol"]["passed_cases"] == "14021"
    assert rows["gpt_5_6_terra"]["passed_cases"] == "13986"
    assert rows["gpt_5_4_mini"]["passed_cases"] == "10827"
    assert rows["gpt_5_6_luna"]["passed_cases"] == "10662"
    assert all(row["generation_errors"] == "0" for row in rows.values())


def test_published_generation_metadata_is_complete() -> None:
    metadata_paths = sorted((RESULT_RELEASE / "generation").glob("*.json"))
    assert len(metadata_paths) == 4

    for path in metadata_paths:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        assert metadata["execution"]["status"] == "completed"
        assert metadata["outputs"]["rows"] == 14_034
        assert metadata["outputs"]["success_count"] == 14_034
        assert metadata["outputs"]["error_count"] == 0


def test_published_analysis_headline_and_review_values() -> None:
    with (ANALYSIS_RELEASE / "model_overview.csv").open(
        encoding="utf-8",
        newline="",
    ) as file:
        overview = {
            row["model_name"]: row
            for row in csv.DictReader(file)
        }
    assert overview["gpt_5_6_sol"]["failed_cases"] == "13"
    assert overview["gpt_5_6_terra"]["failed_cases"] == "48"
    assert overview["gpt_5_6_sol"][
        "both_languages_passed_products"
    ] == "7007"
    assert overview["gpt_5_6_terra"][
        "both_languages_passed_products"
    ] == "6972"

    with (ANALYSIS_RELEASE / "error_taxonomy.csv").open(
        encoding="utf-8",
        newline="",
    ) as file:
        taxonomy = list(csv.DictReader(file))
    terra_compound = next(
        row
        for row in taxonomy
        if row["model_name"] == "gpt_5_6_terra"
        and row["language"] == "all"
        and row["error_type"] == "compound_label_split_only"
    )
    assert terra_compound["count"] == "26"

    with (ANALYSIS_RELEASE / "frontier_cases.csv").open(
        encoding="utf-8",
        newline="",
    ) as file:
        frontier = list(csv.DictReader(file))
    assert len(frontier) == 55
    assert Counter(row["pair_outcome"] for row in frontier) == {
        "gpt_5_6_sol_only_passed": 42,
        "gpt_5_6_terra_only_passed": 7,
        "both_failed": 6,
    }

    with (ANALYSIS_RELEASE / "manual_review.csv").open(
        encoding="utf-8",
        newline="",
    ) as file:
        review_reader = csv.DictReader(file)
        review = list(review_reader)
        columns = review_reader.fieldnames or []
    assert len(review) == 105
    assert len({row["case_id"] for row in review}) == 105
    assert Counter(row["review_group"] for row in review) == {
        "primary_model_failure": 13,
        "comparison_only_failure": 42,
        "lower_tier_shared_failure_sample": 25,
        "all_model_pass_sample": 25,
    }
    assert all(row["review_status"] == "pending" for row in review)
    for column in (
        "source_record_correct",
        "expected_answer_correct",
        "strict_score_correct",
        "error_owner",
        "operational_severity",
        "adjudication",
        "reviewer",
        "review_date",
        "review_notes",
    ):
        assert column in columns
        assert all(row[column] == "" for row in review)

    with (ANALYSIS_RELEASE / "evidence_audit.csv").open(
        encoding="utf-8",
        newline="",
    ) as file:
        audit_reader = csv.DictReader(file)
        audit = list(audit_reader)
    assert len(audit) == 105
    assert {row["review_id"] for row in audit} == {
        row["review_id"] for row in review
    }
    assert all(row["review_status"] == "complete" for row in audit)
    assert all(
        row["source_rebuild_verified"] == "yes" for row in audit
    )
    assert all(
        row["expected_answer_correct"] == "yes" for row in audit
    )
    assert all(
        row["strict_score_correct"] == "yes" for row in audit
    )
    assert all(row["adjudication"] == "retain_result" for row in audit)
    assert all(
        row["human_signoff_status"] == "not_claimed"
        for row in audit
    )

    audit_summary = json.loads(
        (
            ANALYSIS_RELEASE / "evidence_audit_summary.json"
        ).read_text(encoding="utf-8")
    )
    assert audit_summary["verification"][
        "strict_score_disagreements"
    ] == 0
    assert audit_summary["outcomes"][
        "leaderboard_scores_changed"
    ] is False


def test_analysis_notebook_is_clean_and_reproducible() -> None:
    notebook_path = ROOT / "notebooks/HC_DPD_Census_Analysis.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] == 5
    code_cells = [
        cell
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    assert len(code_cells) == 10
    assert all(cell["execution_count"] is None for cell in code_cells)
    assert all(cell["outputs"] == [] for cell in code_cells)
    source = "\n".join(cell["source"] for cell in code_cells)
    assert "AUTO_REFRESH_LOCAL_ANALYSIS = False" in source
    assert "analysis_manifest.json" in source
    assert "manual_review.csv" in source
    assert "evidence_audit.csv" in source


def test_local_markdown_links_resolve() -> None:
    markdown_paths = [
        ROOT / "README.md",
        ROOT / "CONTRIBUTING.md",
        *sorted((ROOT / "configs").glob("*.md")),
        *sorted((ROOT / "docs").rglob("*.md")),
        *sorted((ROOT / "reports").rglob("*.md")),
    ]
    missing: list[str] = []

    for markdown_path in markdown_paths:
        text = markdown_path.read_text(encoding="utf-8")
        for match in re.finditer(r"\[[^\]]*\]\(([^)]+)\)", text):
            target = match.group(1).split("#", 1)[0]
            if (
                not target
                or "://" in target
                or target.startswith("mailto:")
            ):
                continue
            resolved = markdown_path.parent / target
            if not resolved.exists():
                missing.append(
                    f"{markdown_path.relative_to(ROOT)} -> {target}"
                )

    assert missing == []

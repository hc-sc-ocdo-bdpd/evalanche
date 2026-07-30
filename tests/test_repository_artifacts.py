import csv
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_RELEASE = ROOT / "reports/hc_dpd_census/0.2.0"


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

    assert manifest["status"] == "provisional"
    assert manifest["benchmark"]["case_count"] == 14_034
    assert manifest["benchmark"]["product_family_count"] == 7_017
    assert manifest["validation"]["generation_complete"] is True
    assert manifest["validation"]["generation_failures"] == 0
    assert manifest["validation"]["manual_label_review"] == "pending"

    for artifact in manifest["published_files"]:
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

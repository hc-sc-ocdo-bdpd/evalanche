from __future__ import annotations

import json
import shutil
from hashlib import sha256
from pathlib import Path

import pandas as pd
import pytest
import yaml

import evalanche.product_monograph as pm
from evalanche import __version__
from evalanche.dataset_manifest import verify_dataset_manifest_file
from evalanche.product_monograph_review import (
    check_product_monograph_label_review,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/hc/benchmarks/product_monograph_structured_extraction/0.1.0"
MANIFEST = (
    ROOT / "configs/datasets/"
    "hc_product_monograph_structured_extraction_0.1.0_manifest.yaml"
)
NATIVE_PDF_DATA_DIR = (
    ROOT / "data/hc/benchmarks/product_monograph_native_pdf_extraction/0.1.0"
)
NATIVE_PDF_MANIFEST = (
    ROOT / "configs/datasets/"
    "hc_product_monograph_native_pdf_extraction_0.1.0_manifest.yaml"
)
NATIVE_PDF_BENCHMARK = (
    ROOT / "configs/benchmarks/hc_product_monograph_native_pdf_extraction_0.1.0.yaml"
)


def test_product_monograph_release_verifies() -> None:
    verification = verify_dataset_manifest_file(
        MANIFEST,
        root_path=ROOT,
    )
    assert verification["valid"] is True
    assert verification["files_passed"] == 6
    assert verification["files_checked"] == 6


def test_native_pdf_release_is_hash_locked_but_stays_draft() -> None:
    verification = verify_dataset_manifest_file(
        NATIVE_PDF_MANIFEST,
        root_path=ROOT,
    )
    assert verification["valid"] is True
    assert verification["files_passed"] == 6

    cases = pd.read_csv(NATIVE_PDF_DATA_DIR / "cases.csv.gz", dtype=str)
    review = pd.read_csv(NATIVE_PDF_DATA_DIR / "label_review.csv", dtype=str)
    assert len(cases) == 80
    assert len(review) == 390
    assert set(cases["input_mode"]) == {"native_pdf"}
    assert not cases["input"].str.contains("[PDF PAGE", regex=False).any()
    assert set(review["human_review_status"]) == {"pending"}

    for descriptor_list in cases["input_files"].map(json.loads):
        assert len(descriptor_list) == 1
        descriptor = descriptor_list[0]
        assert descriptor["media_type"] == "application/pdf"
        assert descriptor["path"].startswith("data/hc/product_monographs/0.1.0/raw/")
        assert len(descriptor["sha256"]) == 64
        assert "detail" not in descriptor

    manifest = yaml.safe_load(NATIVE_PDF_MANIFEST.read_text(encoding="utf-8"))
    benchmark = yaml.safe_load(NATIVE_PDF_BENCHMARK.read_text(encoding="utf-8"))
    assert manifest["release"]["status"] == "draft"
    assert manifest["release"]["immutable"] is False
    assert benchmark["status"] == "draft"
    assert benchmark["runtime"]["request_api"] == "responses"
    assert set(benchmark["required_capabilities"]) == {
        "pdf_input",
        "responses_api",
        "vision",
    }

    status = check_product_monograph_label_review(root_path=ROOT)
    assert status["valid"] is True
    assert status["promotion_ready"] is False
    assert status["approved"] == 0
    assert status["remaining"] == 390


def test_native_pdf_builder_is_deterministic_and_preserves_review(
    tmp_path: Path,
) -> None:
    parent_files = [
        pm.PM_OUTPUT_DIR / "cases.csv.gz",
        pm.PM_OUTPUT_DIR / "products.csv",
        pm.PM_OUTPUT_DIR / "field_evidence.csv.gz",
        pm.PM_SOURCES_PATH,
    ]
    for relative_path in parent_files:
        destination = tmp_path / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative_path, destination)

    first = pm.build_product_monograph_native_pdf_benchmark(
        root_path=tmp_path,
    )
    assert first["status"] == "draft"
    assert first["case_count"] == 80
    assert first["review_item_count"] == 390
    assert first["approved_review_item_count"] == 0
    assert first["verification"]["valid"] is True
    first_cases = (tmp_path / pm.PM_NATIVE_PDF_CASES_PATH).read_bytes()

    review_path = tmp_path / pm.PM_NATIVE_PDF_REVIEW_PATH
    review = pd.read_csv(review_path, dtype=str, keep_default_na=False)
    review.loc[0, "human_review_status"] = "approved"
    review.loc[0, "reviewer"] = "reviewer@example.test"
    review.loc[0, "reviewed_at_utc"] = "2026-08-04T15:00:00Z"
    review.loc[0, "notes"] = "Confirmed against the locked source page."
    review.to_csv(review_path, index=False, lineterminator="\n")

    second = pm.build_product_monograph_native_pdf_benchmark(
        root_path=tmp_path,
    )
    rebuilt_review = pd.read_csv(
        review_path,
        dtype=str,
        keep_default_na=False,
    )
    approved = rebuilt_review.loc[rebuilt_review["human_review_status"] == "approved"]
    assert len(approved) == 1
    assert approved.iloc[0]["reviewer"] == "reviewer@example.test"
    assert second["approved_review_item_count"] == 1
    assert (tmp_path / pm.PM_NATIVE_PDF_CASES_PATH).read_bytes() == first_cases

    status = check_product_monograph_label_review(root_path=tmp_path)
    assert status["valid"] is True
    assert status["reviewed"] == 1
    assert status["approved"] == 1
    assert status["remaining"] == 389

    review.loc[0, "source_page"] = "999"
    review.to_csv(review_path, index=False, lineterminator="\n")
    with pytest.raises(ValueError, match="immutable source or label"):
        pm.build_product_monograph_native_pdf_benchmark(
            root_path=tmp_path,
        )


def test_product_monograph_cohort_contract() -> None:
    products = pd.read_csv(DATA_DIR / "products.csv", dtype=str)
    cases = pd.read_csv(DATA_DIR / "cases.csv.gz", dtype=str)
    sources = pd.read_csv(DATA_DIR / "source_documents.csv", dtype=str)

    assert len(products) == 40
    assert len(cases) == 80
    assert len(sources) == 80
    assert sources["sha256"].nunique() == 80
    assert sources["language"].value_counts().to_dict() == {
        "en": 40,
        "fr": 40,
    }
    assert products["split"].value_counts().to_dict() == {
        "development": 30,
        "heldout": 10,
    }
    assert products["stratum"].value_counts().to_dict() == {
        "single_ingredient": 20,
        "multi_ingredient": 10,
        "multi_variant": 10,
    }
    assert (cases.groupby("product_id")["language"].apply(set) == {"en", "fr"}).all()
    development_groups = set(
        products.loc[
            products["split"] == "development",
            "ingredient_group_id",
        ]
    )
    heldout_groups = set(
        products.loc[
            products["split"] == "heldout",
            "ingredient_group_id",
        ]
    )
    assert development_groups.isdisjoint(heldout_groups)


def test_every_expected_item_has_page_evidence() -> None:
    cases = pd.read_csv(DATA_DIR / "cases.csv.gz", dtype=str)
    evidence = pd.read_csv(DATA_DIR / "field_evidence.csv.gz", dtype=str)
    assert len(evidence) == 390
    assert set(evidence["support_method"]) == {
        "normalized_exact",
        "normalized_name_and_strength",
        "declared_alias",
    }
    assert evidence["source_page"].astype(int).ge(1).all()
    assert set(evidence["case_id"]) == {
        metadata["dpd_case_id"] for metadata in cases["source_metadata"].map(json.loads)
    }
    assert cases["evidence_page_count"].astype(int).between(1, 3).all()

    for output in cases["expected_output"].map(json.loads):
        assert set(output) == {
            "brand_name",
            "active_ingredients",
            "dosage_forms",
            "routes",
        }
        assert output["brand_name"]
        assert output["active_ingredients"]
        assert output["dosage_forms"]
        assert output["routes"]


def test_product_monograph_build_report_is_explicit() -> None:
    report = json.loads((DATA_DIR / "build_report.json").read_text(encoding="utf-8"))
    assert report["source_documents"] == {
        "documents": 80,
        "languages": {"en": 40, "fr": 40},
        "unique_hashes": 80,
        "issues": [],
        "valid": True,
    }
    assert report["screening"] == {
        "documented_exclusions": 23,
        "by_stage": {
            "scope_alignment": 12,
            "document_availability": 10,
            "language_validation": 1,
        },
    }
    assert report["evidence"]["all_scored_items_have_source_pages"] is True
    assert report["evidence"]["automated_source_evidence_audit"] == "complete"
    assert report["evidence"]["independent_human_signoff"] == ("not_claimed")
    assert report["scoring_contract"]["unscored_alignment_fields"] == [
        "din",
        "schedule",
        "product_status",
        "company",
    ]
    assert report["software"]["evalanche_version"] == __version__


def test_product_monograph_normalization_and_overrides() -> None:
    assert pm._normalize("Pr® Crème, 5.0 %") == "creme 5 0 %"
    assert pm._base_ingredient_name("DRUG (AS SALT)") == "DRUG"
    assert pm._number_candidates(1000) == [
        "1000",
        "1,000",
        "1 000",
    ]
    assert pm._number_candidates(2.0) == ["2.0", "2", "2,0", "2 0"]
    assert pm._document_complexity(30) == "short"
    assert pm._document_complexity(31) == "medium"
    assert pm._document_complexity(61) == "long"

    expected = pm._apply_expected_overrides(
        source_expected={
            "brand_name": "SOURCE",
            "active_ingredients": [
                {"name": "DRUG (AS SALT)", "strength": 5, "unit": "MG"}
            ],
            "dosage_forms": ["SOURCE FORM"],
            "routes": ["SOURCE ROUTE"],
        },
        case_id="case_en",
        overrides={
            "cases": {
                "case_en": {
                    "brand_name": "DOCUMENT",
                    "active_ingredient_names": {"DRUG": "DRUG NAME"},
                    "dosage_forms": ["TABLET"],
                    "routes": ["ORAL"],
                }
            }
        },
    )
    assert expected == {
        "brand_name": "DOCUMENT",
        "active_ingredients": [{"name": "DRUG NAME", "strength": 5, "unit": "MG"}],
        "dosage_forms": ["TABLET"],
        "routes": ["ORAL"],
    }


def test_product_monograph_evidence_and_prompt_rendering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pages = [
        "BRAND X, TAB, ORAL",
        "DRUG A 100 mg",
    ]
    expected = {
        "brand_name": "BRAND X",
        "active_ingredients": [{"name": "DRUG A", "strength": 100, "unit": "MG"}],
        "dosage_forms": ["TABLET"],
        "routes": ["ORAL"],
    }
    overrides = {
        "normalization_aliases": {
            "dosage_forms": {"TAB": "TABLET"},
            "units": {"mg": "MG"},
        }
    }
    evidence, selected_pages = pm._build_evidence(
        case_id="case_en",
        expected=expected,
        pages=pages,
        overrides=overrides,
    )
    assert selected_pages == {1, 2}
    assert [row["field"] for row in evidence] == [
        "brand_name",
        "active_ingredients",
        "dosage_forms",
        "routes",
    ]
    assert {row["support_method"] for row in evidence} == {
        "normalized_exact",
        "normalized_name_and_strength",
        "declared_alias",
    }
    rendered = pm._render_input(
        language="en",
        pages=pages,
        selected_pages=selected_pages,
    )
    assert "PRODUCT MONOGRAPH EVIDENCE (ENGLISH)" in rendered
    assert "[PDF PAGE 1]" in rendered
    assert "[PDF PAGE 2]" in rendered
    assert (
        pm._render_input(
            language="fr",
            pages=pages,
            selected_pages={1},
        ).find("(FRENCH)")
        > 0
    )

    with pytest.raises(ValueError, match="No source evidence"):
        pm._build_evidence(
            case_id="missing_en",
            expected={**expected, "brand_name": "ABSENT"},
            pages=pages,
            overrides=overrides,
        )
    monkeypatch.setattr(pm, "PM_MAX_EVIDENCE_PAGES", 1)
    with pytest.raises(ValueError, match="above the limit"):
        pm._build_evidence(
            case_id="case_en",
            expected=expected,
            pages=pages,
            overrides=overrides,
        )


def test_pdf_page_cache_is_hash_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_path = tmp_path / "source.pdf"
    pdf_path.write_bytes(b"pdf")
    cache_path = pdf_path.with_suffix(".pages.json")
    cache_path.write_text(
        json.dumps({"pdf_sha256": "right", "pages": ["cached"]}),
        encoding="utf-8",
    )
    assert pm._extract_pdf_pages(pdf_path, "right") == ["cached"]

    cache_path.write_text("{broken", encoding="utf-8")

    class FakePage:
        def __init__(self, text: str | None) -> None:
            self.text = text

        def extract_text(self) -> str | None:
            return self.text

    class FakeReader:
        def __init__(self, path: Path) -> None:
            assert path == pdf_path
            self.pages = [FakePage("fresh"), FakePage(None)]

    monkeypatch.setattr(pm, "PdfReader", FakeReader)
    assert pm._extract_pdf_pages(pdf_path, "new") == ["fresh", ""]
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    assert cache == {"pdf_sha256": "new", "pages": ["fresh", ""]}


def test_product_monograph_acquisition_reuses_and_downloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = b"existing pdf"
    downloaded = b"downloaded pdf"
    rows = [
        {
            "product_id": "p1",
            "language": "en",
            "monograph_id": "m1",
            "source_url": "https://example.test/one.pdf",
            "sha256": sha256(existing).hexdigest(),
            "byte_size": len(existing),
            "page_count": 1,
        },
        {
            "product_id": "p2",
            "language": "fr",
            "monograph_id": "m2",
            "source_url": "https://example.test/two.pdf",
            "sha256": sha256(downloaded).hexdigest(),
            "byte_size": len(downloaded),
            "page_count": 1,
        },
    ]
    source_path = tmp_path / "sources.csv"
    pd.DataFrame(rows).to_csv(source_path, index=False)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "p1_en_m1.pdf").write_bytes(existing)

    class FakeResponse:
        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return downloaded

    def fake_urlopen(request: object, *, timeout: float) -> FakeResponse:
        assert timeout == 5
        assert "two.pdf" in str(request.full_url)
        return FakeResponse()

    monkeypatch.setattr(pm, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        pm,
        "verify_product_monograph_sources",
        lambda **kwargs: {"valid": True, "issues": []},
    )
    result = pm.acquire_product_monographs(
        root_path=tmp_path,
        sources_path="sources.csv",
        raw_dir="raw",
        timeout=5,
    )
    assert result["downloaded"] == 1
    assert result["reused"] == 1
    assert (raw_dir / "p2_fr_m2.pdf").read_bytes() == downloaded


def test_product_monograph_source_verification_reports_defects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = b"same pdf"
    digest = sha256(content).hexdigest()
    rows = [
        {
            "product_id": product,
            "language": language,
            "monograph_id": monograph,
            "source_url": f"https://example.test/{monograph}.pdf",
            "sha256": digest,
            "byte_size": len(content),
            "page_count": 1,
        }
        for product, language, monograph in (
            ("p1", "en", "m1"),
            ("p2", "fr", "m2"),
        )
    ]
    pd.DataFrame(rows).to_csv(tmp_path / "sources.csv", index=False)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for row in rows:
        (raw_dir / pm._document_filename(row)).write_bytes(content)

    class FakeReader:
        def __init__(self, path: Path) -> None:
            self.pages = [object()]

    monkeypatch.setattr(pm, "PdfReader", FakeReader)
    result = pm.verify_product_monograph_sources(
        root_path=tmp_path,
        sources_path="sources.csv",
        raw_dir="raw",
    )
    assert result["valid"] is False
    assert result["documents"] == 2
    assert result["unique_hashes"] == 1
    assert any("Expected 4 documents" in issue for issue in result["issues"])
    assert any("not unique" in issue for issue in result["issues"])
    assert any("Expected 2 English and 2 French" in issue for issue in result["issues"])
    assert any(
        "without one English and one French" in issue for issue in result["issues"]
    )

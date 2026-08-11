from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path
from urllib.error import URLError

import pandas as pd
import pytest
import yaml

import evalanche.cli as cli
import evalanche.product_monograph_expansion as expansion
from evalanche.dataset_manifest import verify_dataset_manifest_file


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/hc/benchmarks/product_monograph_structured_extraction/1.0.0"
NATIVE_DATA_DIR = (
    ROOT / "data/hc/benchmarks/product_monograph_native_pdf_extraction/1.0.0"
)


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _copy_builder_scaffold(destination: Path) -> None:
    for relative in (
        expansion.EXPANSION_CONFIG_PATH,
        Path(
            "configs/benchmarks/hc_product_monograph_structured_extraction_0.1.0.yaml"
        ),
        Path(
            "configs/benchmarks/hc_product_monograph_native_pdf_extraction_0.1.0.yaml"
        ),
        expansion.EXPANSION_AUDIT_GUIDE_PATH,
    ):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)


def _selected_from_release() -> list[expansion.ScreenedProduct]:
    products = _read(DATA_DIR / "products.csv")
    sources = _read(DATA_DIR / "source_documents.csv")
    cases = _read(DATA_DIR / "cases.csv.gz")
    evidence = _read(DATA_DIR / "field_evidence.csv.gz")
    selected: list[expansion.ScreenedProduct] = []
    for product in products.to_dict(orient="records"):
        product_id = str(product["product_id"])
        product_cases = cases.loc[cases["product_id"] == product_id]
        source_case_ids = {
            json.loads(str(row["source_metadata"]))["dpd_case_id"]
            for row in product_cases.to_dict(orient="records")
        }
        selected.append(
            expansion.ScreenedProduct(
                product=product,
                sources=sources.loc[sources["product_id"] == product_id].to_dict(
                    orient="records"
                ),
                cases=product_cases.to_dict(orient="records"),
                evidence=evidence.loc[
                    evidence["case_id"].isin(source_case_ids)
                ].to_dict(orient="records"),
            )
        )
    return selected


def test_expanded_release_contract_and_audit_gate() -> None:
    products = _read(DATA_DIR / "products.csv")
    sources = _read(DATA_DIR / "source_documents.csv")
    cases = _read(DATA_DIR / "cases.csv.gz")
    native_cases = _read(NATIVE_DATA_DIR / "cases.csv.gz")
    evidence = _read(DATA_DIR / "field_evidence.csv.gz")
    fact_audit = _read(DATA_DIR / "fact_audit.csv")
    product_audit = _read(DATA_DIR / "product_audit.csv")
    screening = _read(DATA_DIR / "screening_log.csv")

    assert len(products) == 200
    assert len(sources) == len(cases) == len(native_cases) == 400
    assert len(evidence) == len(fact_audit) == 2162
    assert len(product_audit) == 200
    assert len(screening) == 535
    assert products["product_id"].nunique() == 200
    assert products["ingredient_group_id"].nunique() == 200
    assert sources["sha256"].nunique() == 400
    assert sources["source_url"].nunique() == 400
    assert fact_audit["audit_item_id"].nunique() == 2162
    assert product_audit["audit_product_id"].nunique() == 200
    assert products["split"].value_counts().to_dict() == {
        "heldout": 160,
        "development": 40,
    }
    assert products["stratum"].value_counts().to_dict() == {
        "single_ingredient": 80,
        "multi_ingredient": 50,
        "multi_variant": 50,
        "multi_ingredient_multi_variant": 20,
    }
    assert products["route_group"].value_counts().to_dict() == {
        "oral": 91,
        "parenteral": 39,
        "ophthalmic_or_otic": 20,
        "topical": 18,
        "other": 16,
        "inhaled_or_nasal": 16,
    }
    assert screening["status"].value_counts().to_dict() == {
        "excluded": 335,
        "retained": 200,
    }
    assert (sources.groupby("product_id")["language"].agg(set) == {"en", "fr"}).all()

    pilot_ids = set(
        _read(
            ROOT / "data/hc/benchmarks/"
            "product_monograph_structured_extraction/0.1.0/products.csv"
        )["product_id"]
    )
    development_ids = set(
        products.loc[products["split"] == "development", "product_id"]
    )
    heldout_ids = set(products.loc[products["split"] == "heldout", "product_id"])
    assert development_ids == pilot_ids
    assert heldout_ids.isdisjoint(pilot_ids)

    for output in cases["expected_output"].map(json.loads):
        for field in ("active_ingredients", "dosage_forms", "routes"):
            serialized = [
                json.dumps(item, ensure_ascii=False, sort_keys=True)
                for item in output[field]
            ]
            assert len(serialized) == len(set(serialized))

    assert set(fact_audit["human_review_status"]) == {"pending"}
    for column in expansion.PRODUCT_AUDIT_STATUS_COLUMNS:
        assert set(product_audit[column]) == {"pending"}
    status = expansion.check_expanded_product_monograph_audit(root_path=ROOT)
    assert status["valid"] is True
    assert status["promotion_ready"] is False
    assert status["fact_items"] == 2162

    for manifest_path in (
        expansion.EXPANSION_MANIFEST_PATH,
        expansion.EXPANSION_NATIVE_MANIFEST_PATH,
    ):
        verification = verify_dataset_manifest_file(
            ROOT / manifest_path, root_path=ROOT
        )
        assert verification["valid"] is True
        assert verification["files_passed"] == 10
        assert verification["files_checked"] == 10

    for benchmark_path in (
        expansion.EXPANSION_BENCHMARK_PATH,
        expansion.EXPANSION_NATIVE_BENCHMARK_PATH,
    ):
        benchmark = yaml.safe_load((ROOT / benchmark_path).read_text(encoding="utf-8"))
        assert benchmark["status"] == "draft"
        assert benchmark["tiers"]["smoke"]["sampling"]["count"] == 2
        assert benchmark["tiers"]["screen"]["sampling"]["count"] == 25
        assert benchmark["tiers"]["standard"]["sampling"]["method"] == "all"


def test_release_builder_refresh_and_audit_tamper_detection(
    tmp_path: Path,
) -> None:
    _copy_builder_scaffold(tmp_path)
    screening = _read(DATA_DIR / "screening_log.csv")
    built = expansion.build_expanded_product_monograph_release(
        root_path=tmp_path,
        selected=_selected_from_release(),
        screening_records=screening.to_dict(orient="records"),
    )
    assert built["products"] == 200
    assert built["cases"] == 400
    assert built["facts"] == 2162
    assert all(item["valid"] for item in built["verifications"].values())

    fact_path = tmp_path / expansion.EXPANSION_FACT_AUDIT_PATH
    product_path = tmp_path / expansion.EXPANSION_PRODUCT_AUDIT_PATH
    fact = _read(fact_path)
    product = _read(product_path)
    fact.loc[0, "human_review_status"] = "approved"
    fact.loc[0, "reviewer"] = "reviewer@example.test"
    fact.loc[0, "reviewed_at_utc"] = "2026-08-11T18:30:00Z"
    product.loc[0, list(expansion.PRODUCT_AUDIT_STATUS_COLUMNS)] = "approved"
    product.loc[0, "reviewer"] = "reviewer@example.test"
    product.loc[0, "reviewed_at_utc"] = "2026-08-11T18:35:00+00:00"
    fact.to_csv(fact_path, index=False, lineterminator="\n")
    product.to_csv(product_path, index=False, lineterminator="\n")

    refreshed = expansion.refresh_expanded_product_monograph_release(root_path=tmp_path)
    assert refreshed["network_calls"] == 0
    assert refreshed["model_calls"] == 0
    assert refreshed["audit_summary"]["fact_status_counts"] == {
        "approved": 1,
        "pending": 2161,
    }
    checked = expansion.check_expanded_product_monograph_audit(root_path=tmp_path)
    assert checked["valid"] is True
    assert checked["promotion_ready"] is False

    fact = _read(fact_path)
    fact.loc[0, "expected_item"] = "tampered"
    fact.to_csv(fact_path, index=False, lineterminator="\n")
    checked = expansion.check_expanded_product_monograph_audit(
        root_path=tmp_path,
        verify_artifact_locks=False,
    )
    assert checked["valid"] is False
    assert any("immutable" in issue for issue in checked["issues"])
    with pytest.raises(ValueError, match="Audit files are invalid"):
        expansion.refresh_expanded_product_monograph_release(root_path=tmp_path)


def test_expansion_selection_and_text_helpers() -> None:
    assert expansion._route_group('{"routes":["ORAL"]}') == "oral"
    assert expansion._route_group('{"routes":["INTRAVENOUS"]}') == "parenteral"
    assert expansion._route_group('{"routes":["TOPICAL"]}') == "topical"
    assert expansion._route_group('{"routes":["INHALATION"]}') == "inhaled_or_nasal"
    assert expansion._route_group('{"routes":["OPHTHALMIC"]}') == "ophthalmic_or_otic"
    assert expansion._route_group('{"routes":["RECTAL"]}') == "other"

    candidates = pd.DataFrame(
        [
            {"product_id": "a", "route_group": "oral"},
            {"product_id": "b", "route_group": "topical"},
            {"product_id": "c", "route_group": "oral"},
        ]
    )
    first = expansion._coverage_order(
        candidates, seed="7", route_cycle=("oral", "topical")
    )
    second = expansion._coverage_order(
        candidates, seed="7", route_cycle=("oral", "topical")
    )
    assert first == second
    assert {item["product_id"] for item in first} == {"a", "b", "c"}

    english = expansion._validate_document_language(
        ["PRODUCT MONOGRAPH dosage and administration"], "en"
    )
    french = expansion._validate_document_language(
        ["MONOGRAPHIE DE PRODUIT posologie et administration"], "fr"
    )
    assert english["passed"] is True
    assert french["passed"] is True
    assert expansion._validate_document_language(["unknown"], "en")["passed"] is False
    excerpt = expansion._evidence_excerpt(
        "prefix " * 100 + "DRUG NAME 10 mg" + " suffix" * 100,
        '{"name":"DRUG NAME","strength":10,"unit":"MG"}',
        "active_ingredients",
    )
    assert "DRUG NAME 10 mg" in excerpt
    assert len(excerpt) < 1300
    assert expansion._valid_utc("2026-08-11T18:30:00Z") is True
    assert expansion._valid_utc("2026-08-11T14:30:00-04:00") is False
    assert expansion._valid_utc("not-a-time") is False


def test_fetch_discovery_download_and_case_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = replace(
        expansion.load_expansion_spec(root_path=ROOT),
        request_retries=3,
        retry_backoff_seconds=0,
        request_timeout_seconds=5,
    )
    attempts = 0

    def flaky(url: str, timeout: float) -> expansion.FetchedDocument:
        nonlocal attempts
        attempts += 1
        assert timeout == 5
        if attempts < 3:
            raise URLError("temporary")
        return expansion.FetchedDocument(b"ok", {})

    assert (
        expansion._fetch_with_retries(
            url="https://example.test", spec=spec, fetch=flaky
        ).content
        == b"ok"
    )
    assert attempts == 3
    with pytest.raises(OSError, match="Could not retrieve"):
        expansion._fetch_with_retries(
            url="https://example.test/fail",
            spec=spec,
            fetch=lambda url, timeout: (_ for _ in ()).throw(
                URLError("still unavailable")
            ),
        )

    html = b'<a href="https://pdf.hres.ca/dpd_pm/12345.PDF">PM</a>'
    fetched = 0

    def html_fetch(url: str, timeout: float) -> expansion.FetchedDocument:
        nonlocal fetched
        fetched += 1
        return expansion.FetchedDocument(html, {})

    discovered = expansion._discover_monograph(
        root=tmp_path,
        drug_code="42",
        language="en",
        spec=spec,
        fetch=html_fetch,
    )
    assert discovered == ("12345", "https://pdf.hres.ca/dpd_pm/12345.PDF")
    assert (
        expansion._discover_monograph(
            root=tmp_path,
            drug_code="42",
            language="en",
            spec=spec,
            fetch=html_fetch,
        )
        == discovered
    )
    assert fetched == 1
    assert (
        expansion._discover_monograph(
            root=tmp_path,
            drug_code="no-link",
            language="en",
            spec=spec,
            fetch=lambda url, timeout: expansion.FetchedDocument(b"<html/>", {}),
        )
        is None
    )
    multiple = b"https://pdf.hres.ca/dpd_pm/1.PDF https://pdf.hres.ca/dpd_pm/2.PDF"
    with pytest.raises(ValueError, match="multiple Product Monographs"):
        expansion._discover_monograph(
            root=tmp_path,
            drug_code="multiple",
            language="fr",
            spec=spec,
            fetch=lambda url, timeout: expansion.FetchedDocument(multiple, {}),
        )

    content = b"%PDF-fake-test"
    digest = hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(
        expansion,
        "_extract_pdf_pages",
        lambda path, sha256: ["PRODUCT MONOGRAPH dosage and administration"],
    )

    def pdf_fetch(url: str, timeout: float) -> expansion.FetchedDocument:
        return expansion.FetchedDocument(
            content, {"last-modified": "Tue, 11 Aug 2026 00:00:00 GMT"}
        )

    source, pages = expansion._download_and_extract(
        root=tmp_path,
        product_id="hc_dpd_test",
        language="en",
        monograph_id="12345",
        source_url="https://pdf.hres.ca/dpd_pm/12345.PDF",
        spec=spec,
        fetch=pdf_fetch,
    )
    assert source["sha256"] == digest
    assert source["page_count"] == 1
    assert pages[0].startswith("PRODUCT MONOGRAPH")

    with pytest.raises(ValueError, match="not a PDF"):
        expansion._download_and_extract(
            root=tmp_path,
            product_id="not_pdf",
            language="en",
            monograph_id="10",
            source_url="https://pdf.hres.ca/dpd_pm/10.PDF",
            spec=spec,
            fetch=lambda url, timeout: expansion.FetchedDocument(b"html", {}),
        )
    with pytest.raises(ValueError, match="hash changed"):
        expansion._download_and_extract(
            root=tmp_path,
            product_id="wrong_hash",
            language="en",
            monograph_id="11",
            source_url="https://pdf.hres.ca/dpd_pm/11.PDF",
            spec=spec,
            fetch=pdf_fetch,
            expected_sha256="0" * 64,
        )
    monkeypatch.setattr(expansion, "_extract_pdf_pages", lambda path, sha256: [])
    with pytest.raises(ValueError, match="no extractable text"):
        expansion._download_and_extract(
            root=tmp_path,
            product_id="no_text",
            language="en",
            monograph_id="12",
            source_url="https://pdf.hres.ca/dpd_pm/12.PDF",
            spec=spec,
            fetch=pdf_fetch,
        )
    monkeypatch.setattr(
        expansion,
        "_extract_pdf_pages",
        lambda path, sha256: ["MONOGRAPHIE DE PRODUIT posologie"],
    )
    with pytest.raises(ValueError, match="language could not be validated"):
        expansion._download_and_extract(
            root=tmp_path,
            product_id="wrong_language",
            language="en",
            monograph_id="13",
            source_url="https://pdf.hres.ca/dpd_pm/13.PDF",
            spec=spec,
            fetch=pdf_fetch,
        )

    monkeypatch.setattr(
        expansion,
        "_extract_pdf_pages",
        lambda path, sha256: ["PRODUCT MONOGRAPH dosage and administration"],
    )

    dpd_cases = pd.DataFrame(
        [
            {
                "product_id": "hc_dpd_test",
                "language": "en",
                "expected_output": json.dumps(
                    {
                        "brand_name": "TEST BRAND",
                        "active_ingredients": [
                            {"name": "DRUG", "strength": 10, "unit": "MG"},
                            {"name": "DRUG", "strength": 10, "unit": "MG"},
                        ],
                        "dosage_forms": ["TABLET", "TABLET"],
                        "routes": ["ORAL", "ORAL"],
                    }
                ),
            }
        ]
    )
    case, evidence = expansion._build_case(
        product={
            "product_id": "hc_dpd_test",
            "ingredient_group_id": "hc_ing_test",
            "drug_codes": '["42"]',
            "din_list": '["00000001"]',
            "split": "heldout",
            "stratum": "single_ingredient",
            "route_group": "oral",
        },
        source=source,
        pages=["TEST BRAND DRUG 10 MG TABLET ORAL PRODUCT MONOGRAPH"],
        dpd_cases=dpd_cases,
        overrides={},
    )
    output = json.loads(case["expected_output"])
    assert len(output["active_ingredients"]) == 1
    assert output["dosage_forms"] == ["TABLET"]
    assert output["routes"] == ["ORAL"]
    assert len(evidence) == 4


def test_expand_orchestrator_enforces_duplicate_gates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_spec = expansion.load_expansion_spec(root_path=ROOT)
    spec = replace(
        base_spec,
        source_dpd_products_path=Path("products.csv"),
        source_dpd_cases_path=Path("cases.csv"),
        max_workers=4,
    )
    inherited_strata = (
        ["single_ingredient"] * 20 + ["multi_ingredient"] * 10 + ["multi_variant"] * 10
    )
    inherited: list[expansion.ScreenedProduct] = []
    for index, stratum in enumerate(inherited_strata):
        product_id = f"old_{index:03d}"
        inherited.append(
            expansion.ScreenedProduct(
                product={
                    "product_id": product_id,
                    "brand_name": product_id,
                    "ingredient_group_id": f"old_group_{index:03d}",
                    "stratum": stratum,
                    "route_group": "oral",
                    "split": "development",
                    "selection_status": "inherited_development",
                },
                sources=[
                    {
                        "sha256": hashlib.sha256(
                            f"{product_id}-{language}".encode()
                        ).hexdigest()
                    }
                    for language in ("en", "fr")
                ],
                cases=[],
                evidence=[],
            )
        )

    remaining = {
        "single_ingredient": 60,
        "multi_ingredient": 40,
        "multi_variant": 40,
        "multi_ingredient_multi_variant": 20,
    }
    candidates: list[dict[str, str]] = []
    for stratum, count in remaining.items():
        for index in range(count):
            product_id = f"new_{stratum}_{index:03d}"
            candidates.append(
                {
                    "product_id": product_id,
                    "brand_name": product_id,
                    "ingredient_group_id": f"group_{stratum}_{index:03d}",
                    "stratum": stratum,
                }
            )
    candidates.insert(
        1,
        {
            "product_id": "duplicate_group",
            "brand_name": "duplicate_group",
            "ingredient_group_id": "group_single_ingredient_000",
            "stratum": "single_ingredient",
        },
    )
    candidates.insert(
        2,
        {
            "product_id": "duplicate_hash",
            "brand_name": "duplicate_hash",
            "ingredient_group_id": "group_duplicate_hash",
            "stratum": "single_ingredient",
        },
    )
    candidates.insert(
        3,
        {
            "product_id": "missing_source",
            "brand_name": "missing_source",
            "ingredient_group_id": "group_missing_source",
            "stratum": "single_ingredient",
        },
    )
    products = pd.DataFrame(candidates)
    products.to_csv(tmp_path / "products.csv", index=False)
    pd.DataFrame(
        [
            {
                "product_id": row["product_id"],
                "language": "en",
                "expected_output": '{"routes":["ORAL"]}',
            }
            for row in candidates
        ]
    ).to_csv(tmp_path / "cases.csv", index=False)

    monkeypatch.setattr(expansion, "load_expansion_spec", lambda *args, **kwargs: spec)
    monkeypatch.setattr(expansion, "_write_overrides", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        expansion,
        "_load_inherited_products",
        lambda **kwargs: inherited,
    )
    monkeypatch.setattr(
        expansion,
        "_coverage_order",
        lambda frame, **kwargs: frame.to_dict(orient="records"),
    )

    first_product = "new_single_ingredient_000"

    def fake_screen(**kwargs: object) -> expansion.ScreenedProduct:
        product = dict(kwargs["product"])
        product_id = str(product["product_id"])
        if product_id == "missing_source":
            raise ValueError("English Product Monograph missing")
        hash_product = first_product if product_id == "duplicate_hash" else product_id
        return expansion.ScreenedProduct(
            product=product,
            sources=[
                {
                    "sha256": hashlib.sha256(
                        f"{hash_product}-{language}".encode()
                    ).hexdigest()
                }
                for language in ("en", "fr")
            ],
            cases=[],
            evidence=[],
        )

    captured: dict[str, object] = {}

    def fake_build(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"selected": len(kwargs["selected"])}

    monkeypatch.setattr(expansion, "_screen_candidate", fake_screen)
    monkeypatch.setattr(
        expansion, "build_expanded_product_monograph_release", fake_build
    )
    result = expansion.expand_product_monograph_benchmark(root_path=tmp_path)
    assert result == {"selected": 200}
    selected = captured["selected"]
    screening = captured["screening_records"]
    assert len(selected) == 200
    assert len(screening) == 203
    assert {row["stage"] for row in screening if row["status"] == "excluded"} == {
        "duplicate_ingredient_group",
        "duplicate_document",
        "document_availability",
    }
    groups = [item.product["ingredient_group_id"] for item in selected]
    assert len(groups) == len(set(groups)) == 200


def test_expansion_cli_commands_and_exit_gates(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = cli.build_parser()
    expanded_args = parser.parse_args(["expand-product-monograph-benchmark"])
    assert expanded_args.config == expansion.EXPANSION_CONFIG_PATH.as_posix()
    assert expanded_args.quiet is False
    refresh_args = parser.parse_args(["refresh-product-monograph-expansion"])
    assert refresh_args.config == expansion.EXPANSION_CONFIG_PATH.as_posix()
    check_args = parser.parse_args(
        ["check-product-monograph-expansion-audit", "--require-complete"]
    )
    assert check_args.require_complete is True

    monkeypatch.setattr(
        cli,
        "expand_product_monograph_benchmark",
        lambda **kwargs: {
            "version": "1.0.0",
            "products": 200,
            "cases": 400,
            "native_pdf_cases": 400,
            "facts": 2162,
            "screened_candidates": 535,
            "screening_exclusions": 335,
            "output_dir": Path("expanded"),
        },
    )
    expanded = cli.run_expand_product_monograph_benchmark(
        root_path=".",
        config_path=expansion.EXPANSION_CONFIG_PATH.as_posix(),
        max_candidates=None,
        show_progress=False,
    )
    assert expanded["products"] == 200
    assert "DRAFT, HUMAN AUDIT REQUIRED" in capsys.readouterr().out

    monkeypatch.setattr(
        cli,
        "refresh_expanded_product_monograph_release",
        lambda **kwargs: {
            "version": "1.0.0",
            "products": 200,
            "cases": 400,
            "facts": 2162,
            "verifications": {
                "evidence_window": {"files_passed": 10, "files_checked": 10},
                "native_pdf": {"files_passed": 10, "files_checked": 10},
            },
        },
    )
    refreshed = cli.run_refresh_product_monograph_expansion(
        root_path=".",
        config_path=expansion.EXPANSION_CONFIG_PATH.as_posix(),
    )
    assert refreshed["facts"] == 2162
    assert "Network calls: 0" in capsys.readouterr().out

    incomplete = {
        "fact_items": 2162,
        "products": 200,
        "fact_status_counts": {"pending": 2162},
        "product_status_counts": {
            column: {"pending": 200}
            for column in expansion.PRODUCT_AUDIT_STATUS_COLUMNS
        },
        "fact_audit_path": Path("fact.csv"),
        "product_audit_path": Path("product.csv"),
        "issues": [],
        "promotion_ready": False,
    }
    monkeypatch.setattr(
        cli,
        "check_expanded_product_monograph_audit",
        lambda **kwargs: incomplete,
    )
    checked = cli.run_check_product_monograph_expansion_audit(
        root_path=".", require_complete=False
    )
    assert checked["promotion_ready"] is False
    assert "Remaining approvals: 2762" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="2"):
        cli.run_check_product_monograph_expansion_audit(
            root_path=".", require_complete=True
        )

    invalid = {**incomplete, "issues": ["tampered"]}
    monkeypatch.setattr(
        cli,
        "check_expanded_product_monograph_audit",
        lambda **kwargs: invalid,
    )
    with pytest.raises(SystemExit, match="1"):
        cli.run_check_product_monograph_expansion_audit(
            root_path=".", require_complete=False
        )

    monkeypatch.setattr(
        cli,
        "expand_product_monograph_benchmark",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("bad expansion")),
    )
    with pytest.raises(SystemExit, match="1"):
        cli.run_expand_product_monograph_benchmark(
            root_path=".",
            config_path="bad.yaml",
            max_candidates=None,
            show_progress=False,
        )


def test_audit_metadata_rejects_incomplete_reviews() -> None:
    fact = pd.DataFrame(
        [
            {
                "audit_item_id": "pending_metadata",
                "human_review_status": "pending",
                "reviewer": "someone",
                "reviewed_at_utc": "2026-08-11T18:00:00Z",
                "corrected_value": "unexpected",
                "notes": "",
            },
            {
                "audit_item_id": "bad_correction",
                "human_review_status": "needs_correction",
                "reviewer": "",
                "reviewed_at_utc": "2026-08-11T14:00:00-04:00",
                "corrected_value": "",
                "notes": "",
            },
            {
                "audit_item_id": "bad_approval",
                "human_review_status": "approved",
                "reviewer": "reviewer",
                "reviewed_at_utc": "2026-08-11T18:00:00Z",
                "corrected_value": "not allowed",
                "notes": "",
            },
            {
                "audit_item_id": "invalid_status",
                "human_review_status": "maybe",
                "reviewer": "reviewer",
                "reviewed_at_utc": "2026-08-11T18:00:00Z",
                "corrected_value": "",
                "notes": "",
            },
        ]
    )
    product = pd.DataFrame(
        [
            {
                "audit_product_id": "partial",
                "identity_review_status": "approved",
                "scope_review_status": "pending",
                "bilingual_review_status": "rejected",
                "reviewer": "",
                "reviewed_at_utc": "not-a-time",
                "notes": "",
            },
            {
                "audit_product_id": "pending_metadata",
                "identity_review_status": "pending",
                "scope_review_status": "pending",
                "bilingual_review_status": "pending",
                "reviewer": "reviewer",
                "reviewed_at_utc": "2026-08-11T18:00:00Z",
                "notes": "",
            },
            {
                "audit_product_id": "invalid",
                "identity_review_status": "unknown",
                "scope_review_status": "approved",
                "bilingual_review_status": "approved",
                "reviewer": "reviewer",
                "reviewed_at_utc": "2026-08-11T18:00:00Z",
                "notes": "",
            },
        ]
    )
    issues = expansion._audit_metadata_issues(fact, product)
    assert any("Invalid fact audit statuses" in issue for issue in issues)
    assert any("Invalid identity_review_status" in issue for issue in issues)
    assert any("pending fact has review metadata" in issue for issue in issues)
    assert any("pending fact has a correction" in issue for issue in issues)
    assert any("correction requires corrected_value" in issue for issue in issues)
    assert any("only a correction" in issue for issue in issues)
    assert any("finish all three" in issue for issue in issues)
    assert any("nonapproval requires notes" in issue for issue in issues)

    missing = expansion._audit_metadata_issues(
        pd.DataFrame([{"audit_item_id": "x"}]),
        pd.DataFrame([{"audit_product_id": "y"}]),
    )
    assert len(missing) == 2
    assert all("missing review columns" in issue for issue in missing)


def test_expansion_spec_fails_closed_on_contract_changes(tmp_path: Path) -> None:
    original = yaml.safe_load(
        (ROOT / expansion.EXPANSION_CONFIG_PATH).read_text(encoding="utf-8")
    )
    config_path = tmp_path / "expansion.yaml"

    def rejected(config: dict[str, object], message: str) -> None:
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        with pytest.raises(ValueError, match=message):
            expansion.load_expansion_spec(config_path, root_path=tmp_path)

    config = dict(original)
    config["schema_version"] = "2.0"
    rejected(config, "schema_version")

    config = dict(original)
    config["version"] = "1.0.1"
    rejected(config, "version must be")

    config = dict(original)
    config["selection"] = []
    rejected(config, "selection mapping")

    config = yaml.safe_load(yaml.safe_dump(original))
    config["selection"]["target_strata"].pop("multi_variant")
    rejected(config, "target_strata")

    config = yaml.safe_load(yaml.safe_dump(original))
    config["selection"]["heldout_products"] = 159
    rejected(config, "counts must equal")

    config = yaml.safe_load(yaml.safe_dump(original))
    config["selection"]["target_strata"]["single_ingredient"] = 81
    rejected(config, "Stratum quotas")

    config = yaml.safe_load(yaml.safe_dump(original))
    config["selection"]["target_products"] = 201
    config["selection"]["heldout_products"] = 161
    config["selection"]["target_strata"]["single_ingredient"] = 81
    rejected(config, "requires 200 products")

    config = yaml.safe_load(yaml.safe_dump(original))
    config["source_dpd_dataset"] = []
    rejected(config, "source and inherit mappings")

    config = yaml.safe_load(yaml.safe_dump(original))
    config["selection"]["route_cycle"] = []
    rejected(config, "route_cycle cannot be empty")


def test_candidate_screening_requires_consistent_bilingual_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = expansion.load_expansion_spec(root_path=ROOT)
    product = {
        "product_id": "hc_dpd_test",
        "brand_name": "TEST",
        "drug_codes": '["1","2"]',
        "route_group": "oral",
    }

    monkeypatch.setattr(
        expansion,
        "_discover_monograph",
        lambda **kwargs: (
            "100" if kwargs["language"] == "en" else "200",
            "https://pdf.hres.ca/dpd_pm/"
            + ("100" if kwargs["language"] == "en" else "200")
            + ".PDF",
        ),
    )

    def fake_download(**kwargs: object) -> tuple[dict[str, str], list[str]]:
        language = str(kwargs["language"])
        return (
            {
                "language": language,
                "sha256": ("a" if language == "en" else "b") * 64,
                "source_url": str(kwargs["source_url"]),
            },
            ["page"],
        )

    monkeypatch.setattr(expansion, "_download_and_extract", fake_download)
    monkeypatch.setattr(
        expansion,
        "_build_case",
        lambda **kwargs: (
            {"case_id": f"case_{kwargs['source']['language']}"},
            [{"field": "brand_name"}],
        ),
    )
    screened = expansion._screen_candidate(
        root=tmp_path,
        product=product,
        dpd_cases=pd.DataFrame(),
        overrides={},
        spec=spec,
        fetch=lambda url, timeout: expansion.FetchedDocument(b"", {}),
    )
    assert len(screened.sources) == 2
    assert len(screened.cases) == 2
    assert len(screened.evidence) == 2
    assert all(source["dpd_info_url"] for source in screened.sources)

    with pytest.raises(ValueError, match="no drug codes"):
        expansion._screen_candidate(
            root=tmp_path,
            product={**product, "drug_codes": "[]"},
            dpd_cases=pd.DataFrame(),
            overrides={},
            spec=spec,
            fetch=lambda url, timeout: expansion.FetchedDocument(b"", {}),
        )

    monkeypatch.setattr(
        expansion,
        "_discover_monograph",
        lambda **kwargs: None if kwargs["drug_code"] == "2" else ("100", "url"),
    )
    with pytest.raises(ValueError, match="missing for DPD codes"):
        expansion._screen_candidate(
            root=tmp_path,
            product=product,
            dpd_cases=pd.DataFrame(),
            overrides={},
            spec=spec,
            fetch=lambda url, timeout: expansion.FetchedDocument(b"", {}),
        )

    monkeypatch.setattr(
        expansion,
        "_discover_monograph",
        lambda **kwargs: (str(kwargs["drug_code"]), "url" + str(kwargs["drug_code"])),
    )
    with pytest.raises(ValueError, match="variants resolve"):
        expansion._screen_candidate(
            root=tmp_path,
            product=product,
            dpd_cases=pd.DataFrame(),
            overrides={},
            spec=spec,
            fetch=lambda url, timeout: expansion.FetchedDocument(b"", {}),
        )

    monkeypatch.setattr(
        expansion,
        "_discover_monograph",
        lambda **kwargs: ("100", "url"),
    )
    with pytest.raises(ValueError, match="same PDF"):
        expansion._screen_candidate(
            root=tmp_path,
            product=product,
            dpd_cases=pd.DataFrame(),
            overrides={},
            spec=spec,
            fetch=lambda url, timeout: expansion.FetchedDocument(b"", {}),
        )

    monkeypatch.setattr(
        expansion,
        "_discover_monograph",
        lambda **kwargs: (
            "100" if kwargs["language"] == "en" else "200",
            "url",
        ),
    )
    monkeypatch.setattr(
        expansion,
        "_download_and_extract",
        lambda **kwargs: (
            {
                "language": kwargs["language"],
                "sha256": "a" * 64,
                "source_url": kwargs["source_url"],
            },
            ["page"],
        ),
    )
    with pytest.raises(ValueError, match="identical hashes"):
        expansion._screen_candidate(
            root=tmp_path,
            product=product,
            dpd_cases=pd.DataFrame(),
            overrides={},
            spec=spec,
            fetch=lambda url, timeout: expansion.FetchedDocument(b"", {}),
        )

from datetime import date
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from evalanche.pricing import (
    EndpointPriceConfig,
    EndpointPricingCatalogConfig,
    build_pricing_snapshot,
    estimate_endpoint_cost_usd,
    load_endpoint_pricing_catalog,
    resolve_endpoint_price,
)


def make_endpoint(**overrides: object) -> EndpointPriceConfig:
    values = {
        "pricing_id": "azure_test_global",
        "model": "azure/test-deployment",
        "currency": "USD",
        "input_per_million_tokens": 1.0,
        "cached_input_per_million_tokens": 0.25,
        "output_per_million_tokens": 4.0,
        "effective_date": date(2026, 7, 1),
        "source": "approved internal rate card",
    }
    values.update(overrides)
    return EndpointPriceConfig.model_validate(values)


def write_catalog(path: Path) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "catalog_version": "2026-07-01",
                "endpoints": [
                    make_endpoint().model_dump(mode="json")
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def test_loads_versioned_endpoint_pricing_catalog(
    tmp_path: Path,
) -> None:
    catalog = load_endpoint_pricing_catalog(
        write_catalog(tmp_path / "pricing.yaml")
    )

    assert catalog.schema_version == "1.0"
    assert catalog.catalog_version == "2026-07-01"
    assert catalog.endpoints[0].pricing_id == "azure_test_global"
    assert catalog.endpoints[0].effective_date == date(2026, 7, 1)


def test_endpoint_pricing_ids_must_be_unique() -> None:
    with pytest.raises(ValidationError, match="pricing IDs must be unique"):
        EndpointPricingCatalogConfig(
            schema_version="1.0",
            catalog_version="test",
            endpoints=[make_endpoint(), make_endpoint()],
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pricing_id", "   "),
        ("model", "   "),
        ("source", "   "),
    ],
)
def test_endpoint_pricing_identity_fields_cannot_be_blank(
    field: str,
    value: str,
) -> None:
    with pytest.raises(ValidationError, match="cannot be blank"):
        make_endpoint(**{field: value})


def test_endpoint_pricing_rejects_non_usd_currency() -> None:
    with pytest.raises(ValidationError, match="USD"):
        make_endpoint(currency="CAD")


def test_endpoint_pricing_rejects_all_zero_billable_rates() -> None:
    with pytest.raises(ValidationError, match="token rate must be positive"):
        make_endpoint(
            input_per_million_tokens=0,
            output_per_million_tokens=0,
        )


def test_resolve_endpoint_price_requires_exact_model_match() -> None:
    catalog = EndpointPricingCatalogConfig(
        schema_version="1.0",
        catalog_version="test",
        endpoints=[make_endpoint()],
    )

    with pytest.raises(ValueError, match="configured model route"):
        resolve_endpoint_price(
            catalog=catalog,
            pricing_id="azure_test_global",
            model="azure/different-deployment",
        )


def test_resolve_endpoint_price_requires_catalog_for_reference() -> None:
    with pytest.raises(ValueError, match="endpoint_pricing_path"):
        resolve_endpoint_price(
            catalog=None,
            pricing_id="azure_test_global",
            model="azure/test-deployment",
        )


def test_estimate_endpoint_cost_uses_cached_input_rate() -> None:
    cost = estimate_endpoint_cost_usd(
        make_endpoint(),
        prompt_tokens=1_000,
        cached_prompt_tokens=400,
        completion_tokens=100,
    )

    assert cost == pytest.approx(0.0011)


def test_estimate_endpoint_cost_remains_unknown_without_split_usage() -> None:
    assert (
        estimate_endpoint_cost_usd(
            make_endpoint(),
            prompt_tokens=None,
            completion_tokens=100,
        )
        is None
    )


def test_pricing_snapshot_records_rates_hash_and_unpriced_routes(
    tmp_path: Path,
) -> None:
    pricing_path = write_catalog(tmp_path / "pricing.yaml")

    snapshot = build_pricing_snapshot(
        path=pricing_path,
        references=[
            {
                "name": "priced",
                "model": "azure/test-deployment",
                "pricing_id": "azure_test_global",
            },
            {
                "name": "fallback",
                "model": "azure/unpriced-deployment",
                "pricing_id": None,
            },
        ],
    )

    assert len(snapshot["catalog_sha256"]) == 64
    assert snapshot["catalog_version"] == "2026-07-01"
    assert snapshot["resolved_endpoints"][0]["pricing_id"] == (
        "azure_test_global"
    )
    assert snapshot["resolved_endpoints"][0][
        "output_per_million_tokens"
    ] == 4.0
    assert snapshot["unpriced_references"] == [
        {
            "name": "fallback",
            "model": "azure/unpriced-deployment",
            "pricing_id": None,
        }
    ]
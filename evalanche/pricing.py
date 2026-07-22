from __future__ import annotations

import hashlib
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from evalanche.config import resolve_env_vars


COST_CURRENCY = "USD"
CONFIGURED_COST_SOURCE = "configured_endpoint_pricing"
PROVIDER_COST_SOURCE = "litellm_response_metadata"
COST_POLICY = (
    "configured_endpoint_pricing_then_litellm_response_metadata"
)


def _nonblank(value: str, *, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} cannot be blank")
    return normalized


class EndpointPriceConfig(BaseModel):
    """One auditable token-price record for an exact model route."""

    model_config = ConfigDict(extra="forbid")

    pricing_id: str
    model: str
    currency: Literal["USD"] = "USD"
    input_per_million_tokens: float = Field(
        ge=0,
        allow_inf_nan=False,
    )
    output_per_million_tokens: float = Field(
        ge=0,
        allow_inf_nan=False,
    )
    cached_input_per_million_tokens: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
    )
    effective_date: date
    source: str
    notes: str | None = None

    @field_validator("pricing_id")
    @classmethod
    def validate_pricing_id(cls, value: str) -> str:
        return _nonblank(value, field_name="pricing_id")

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        return _nonblank(value, field_name="model")

    @field_validator("source")
    @classmethod
    def validate_source(cls, value: str) -> str:
        return _nonblank(value, field_name="source")

    @field_validator("notes")
    @classmethod
    def normalize_notes(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @model_validator(mode="after")
    def validate_positive_billable_rate(self) -> "EndpointPriceConfig":
        if (
            self.input_per_million_tokens == 0
            and self.output_per_million_tokens == 0
        ):
            raise ValueError(
                "at least one input or output token rate must be positive"
            )
        return self


class EndpointPricingCatalogConfig(BaseModel):
    """Versioned endpoint prices referenced by generation and judge configs."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"]
    catalog_version: str
    endpoints: list[EndpointPriceConfig] = Field(min_length=1)

    @field_validator("catalog_version")
    @classmethod
    def validate_catalog_version(cls, value: str) -> str:
        return _nonblank(value, field_name="catalog_version")

    @field_validator("endpoints")
    @classmethod
    def validate_unique_pricing_ids(
        cls,
        endpoints: list[EndpointPriceConfig],
    ) -> list[EndpointPriceConfig]:
        pricing_ids = [endpoint.pricing_id for endpoint in endpoints]
        if len(pricing_ids) != len(set(pricing_ids)):
            raise ValueError("endpoint pricing IDs must be unique")
        return endpoints

    def find(self, pricing_id: str) -> EndpointPriceConfig | None:
        return next(
            (
                endpoint
                for endpoint in self.endpoints
                if endpoint.pricing_id == pricing_id
            ),
            None,
        )


def load_endpoint_pricing_catalog(
    path: str | Path,
) -> EndpointPricingCatalogConfig:
    catalog, _ = load_endpoint_pricing_catalog_with_hash(path)
    return catalog


def load_endpoint_pricing_catalog_with_hash(
    path: str | Path,
) -> tuple[EndpointPricingCatalogConfig, str]:
    """Load and hash the same catalog bytes as one evidence snapshot."""
    raw_bytes = Path(path).read_bytes()
    raw = yaml.safe_load(raw_bytes.decode("utf-8"))
    resolved = resolve_env_vars(raw)
    catalog = EndpointPricingCatalogConfig.model_validate(resolved)
    return catalog, hashlib.sha256(raw_bytes).hexdigest()


def load_optional_endpoint_pricing_catalog(
    path: str | Path | None,
) -> EndpointPricingCatalogConfig | None:
    if path is None:
        return None
    return load_endpoint_pricing_catalog(path)


def load_optional_endpoint_pricing_catalog_with_hash(
    path: str | Path | None,
) -> tuple[EndpointPricingCatalogConfig | None, str | None]:
    if path is None:
        return None, None
    return load_endpoint_pricing_catalog_with_hash(path)


def resolve_endpoint_price(
    *,
    catalog: EndpointPricingCatalogConfig | None,
    pricing_id: str | None,
    model: str,
) -> EndpointPriceConfig | None:
    """Resolve an explicit price reference and reject ambiguous mismatches."""
    if pricing_id is None:
        return None

    if catalog is None:
        raise ValueError(
            f"pricing_id {pricing_id!r} was configured for {model!r}, but "
            "endpoint_pricing_path was not configured"
        )

    endpoint = catalog.find(pricing_id)
    if endpoint is None:
        raise ValueError(
            f"pricing_id {pricing_id!r} was not found in the endpoint "
            "pricing catalog"
        )

    if endpoint.model != model:
        raise ValueError(
            f"pricing_id {pricing_id!r} declares model {endpoint.model!r}, "
            f"but the configured model route is {model!r}"
        )

    return endpoint


def estimate_endpoint_cost_usd(
    endpoint: EndpointPriceConfig,
    *,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    cached_prompt_tokens: int | None = None,
) -> float | None:
    """Estimate token cost without turning missing token evidence into zero."""
    if prompt_tokens is None or completion_tokens is None:
        return None
    if prompt_tokens < 0 or completion_tokens < 0:
        return None

    cached_tokens = cached_prompt_tokens or 0
    if cached_tokens < 0 or cached_tokens > prompt_tokens:
        return None

    uncached_tokens = prompt_tokens - cached_tokens
    cached_rate = (
        endpoint.cached_input_per_million_tokens
        if endpoint.cached_input_per_million_tokens is not None
        else endpoint.input_per_million_tokens
    )

    cost = (
        Decimal(uncached_tokens)
        * Decimal(str(endpoint.input_per_million_tokens))
        + Decimal(cached_tokens) * Decimal(str(cached_rate))
        + Decimal(completion_tokens)
        * Decimal(str(endpoint.output_per_million_tokens))
    ) / Decimal(1_000_000)
    return float(cost)


def endpoint_price_fields(
    endpoint: EndpointPriceConfig | None,
) -> dict[str, Any]:
    if endpoint is None:
        return {
            "pricing_id": None,
            "pricing_model": None,
            "pricing_currency": None,
            "pricing_input_per_million_tokens": None,
            "pricing_cached_input_per_million_tokens": None,
            "pricing_output_per_million_tokens": None,
            "pricing_effective_date": None,
            "pricing_source": None,
        }

    return {
        "pricing_id": endpoint.pricing_id,
        "pricing_model": endpoint.model,
        "pricing_currency": endpoint.currency,
        "pricing_input_per_million_tokens": (
            endpoint.input_per_million_tokens
        ),
        "pricing_cached_input_per_million_tokens": (
            endpoint.cached_input_per_million_tokens
        ),
        "pricing_output_per_million_tokens": (
            endpoint.output_per_million_tokens
        ),
        "pricing_effective_date": endpoint.effective_date.isoformat(),
        "pricing_source": endpoint.source,
    }


def build_pricing_snapshot(
    *,
    path: str | Path | None,
    references: list[dict[str, Any]],
    catalog: EndpointPricingCatalogConfig | None = None,
    catalog_sha256: str | None = None,
) -> dict[str, Any]:
    """Snapshot exact rates and unpriced routes used by a run."""
    if catalog is None:
        catalog, loaded_sha256 = (
            load_optional_endpoint_pricing_catalog_with_hash(path)
        )
        catalog_sha256 = loaded_sha256
    resolved: list[dict[str, Any]] = []
    unpriced: list[dict[str, Any]] = []
    included_ids: set[str] = set()

    for reference in references:
        model = str(reference["model"])
        pricing_id = reference.get("pricing_id")
        endpoint = resolve_endpoint_price(
            catalog=catalog,
            pricing_id=pricing_id,
            model=model,
        )
        if endpoint is None:
            unpriced.append(dict(reference))
            continue
        if endpoint.pricing_id in included_ids:
            continue
        included_ids.add(endpoint.pricing_id)
        resolved.append(endpoint.model_dump(mode="json"))

    return {
        "cost_policy": COST_POLICY,
        "catalog_path": str(path) if path is not None else None,
        "catalog_sha256": catalog_sha256,
        "catalog_schema_version": (
            catalog.schema_version if catalog is not None else None
        ),
        "catalog_version": (
            catalog.catalog_version if catalog is not None else None
        ),
        "resolved_endpoints": resolved,
        "unpriced_references": unpriced,
    }
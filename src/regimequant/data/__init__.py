"""RegimeQuant M1 market-data foundation public API."""

from .canonicalize import canonicalize_market_data
from .errors import (
	MarketDataDomainError,
	MarketDataError,
	MarketDataProviderError,
	MarketDataValidationError,
)
from .models import (
	ALLOWED_AVAILABILITY_POLICY_IDS,
	AVAILABILITY_POLICY_EXTERNAL_VERIFIED,
	AVAILABILITY_POLICY_FIXTURE_DECLARED,
	AVAILABILITY_POLICY_UNKNOWN_UNVERIFIED,
	AdjustmentState,
	AvailabilityStatus,
	MarketDataProvenance,
	PriceType,
	ReturnType,
	UniverseType,
)
from .providers import FixtureMarketDataProvider, MarketDataProvider
from .returns import compute_log_returns, compute_simple_returns

__all__ = [
	"AdjustmentState",
	"ALLOWED_AVAILABILITY_POLICY_IDS",
	"AVAILABILITY_POLICY_EXTERNAL_VERIFIED",
	"AVAILABILITY_POLICY_FIXTURE_DECLARED",
	"AVAILABILITY_POLICY_UNKNOWN_UNVERIFIED",
	"AvailabilityStatus",
	"FixtureMarketDataProvider",
	"MarketDataDomainError",
	"MarketDataError",
	"MarketDataProvider",
	"MarketDataProviderError",
	"MarketDataProvenance",
	"MarketDataValidationError",
	"PriceType",
	"ReturnType",
	"UniverseType",
	"canonicalize_market_data",
	"compute_log_returns",
	"compute_simple_returns",
]

"""Domain-specific errors for RegimeQuant market data foundations."""


class MarketDataError(Exception):
    """Base error for market-data boundary failures."""


class MarketDataValidationError(MarketDataError):
    """Raised when the input schema or duplicate semantics are invalid."""


class MarketDataDomainError(MarketDataError):
    """Raised when values violate the canonical market-data domain."""


class MarketDataProviderError(MarketDataError):
    """Raised when a provider adapter cannot produce normalized raw data."""
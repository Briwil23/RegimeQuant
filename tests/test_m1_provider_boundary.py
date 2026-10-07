from __future__ import annotations

import pandas as pd
import pytest

from regimequant.data import AdjustmentState, AvailabilityStatus, FixtureMarketDataProvider, PriceType
from regimequant.data.errors import MarketDataProviderError


def test_fixture_provider_normalizes_provider_specific_columns() -> None:
    source = pd.DataFrame(
        [
            {
                "provider_symbol": "AAPL",
                "provider_timestamp": pd.Timestamp("2024-01-02 16:00:00"),
                "provider_available_at": pd.Timestamp("2024-01-02 16:00:00"),
                "provider_availability_status": AvailabilityStatus.VERIFIED_TIMESTAMP.value,
                "provider_price": 100.0,
                "provider_price_type": PriceType.CLOSE.value,
                "provider_adjustment_state": AdjustmentState.RAW.value,
                "provider_currency": "USD",
                "provider_source_observation_id": "a1",
            }
        ]
    )
    provider = FixtureMarketDataProvider(
        source_table=source,
        source_timezone="America/New_York",
        column_map={
            "provider_symbol": "instrument",
            "provider_timestamp": "timestamp",
            "provider_available_at": "available_at",
            "provider_availability_status": "availability_status",
            "provider_price": "price",
            "provider_price_type": "price_type",
            "provider_adjustment_state": "adjustment_state",
            "provider_currency": "currency",
            "provider_source_observation_id": "source_observation_id",
        },
    )
    raw, provenance = provider.fetch_raw(instruments=("AAPL",))
    assert list(raw.columns) == [
        "instrument",
        "timestamp",
        "available_at",
        "availability_status",
        "price",
        "price_type",
        "adjustment_state",
        "currency",
        "source_observation_id",
    ]
    assert raw.iloc[0]["timestamp"] == pd.Timestamp("2024-01-02 21:00:00+00:00")
    assert raw.iloc[0]["available_at"] == pd.Timestamp("2024-01-02 21:00:00+00:00")
    assert provenance.price_type == PriceType.CLOSE
    assert provenance.adjustment_state == AdjustmentState.RAW


def test_fixture_provider_requires_explicit_timezone_for_naive_source_timestamps() -> None:
    provider = FixtureMarketDataProvider(
        source_table=pd.DataFrame(
            [
                {
                    "instrument": "AAPL",
                    "timestamp": pd.Timestamp("2024-01-02 16:00:00"),
                    "available_at": pd.Timestamp("2024-01-02 16:00:00"),
                    "availability_status": AvailabilityStatus.VERIFIED_TIMESTAMP.value,
                    "price": 100.0,
                    "price_type": PriceType.CLOSE.value,
                    "adjustment_state": AdjustmentState.RAW.value,
                    "currency": "USD",
                }
            ]
        ),
        source_timezone=None,
    )

    with pytest.raises(MarketDataProviderError):
        provider.fetch_raw()
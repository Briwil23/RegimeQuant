from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from regimequant.data import (
    AVAILABILITY_POLICY_FIXTURE_DECLARED,
    AdjustmentState,
    AvailabilityStatus,
    FixtureMarketDataProvider,
    MarketDataProvenance,
    PriceType,
    UniverseType,
    canonicalize_market_data,
    compute_log_returns,
    compute_simple_returns,
)
from regimequant.data.errors import MarketDataProviderError


def _provenance(adjustment_state: AdjustmentState = AdjustmentState.RAW) -> MarketDataProvenance:
    return MarketDataProvenance(
        provider_id="fixture",
        requested_instruments=("AAPL", "MSFT"),
        requested_start_timestamp=pd.Timestamp("2024-01-01T00:00:00Z"),
        requested_end_timestamp=pd.Timestamp("2024-01-06T00:00:00Z"),
        requested_price_type=PriceType.CLOSE,
        requested_adjustment_state=adjustment_state,
        retrieval_timestamp_utc=pd.Timestamp("2024-01-06T12:00:00Z"),
        retrieval_environment="fixture",
        price_type=PriceType.CLOSE,
        adjustment_state=adjustment_state,
        timestamp_timezone_policy="utc",
        availability_policy_id=AVAILABILITY_POLICY_FIXTURE_DECLARED,
        source_availability_status_distribution=(),
        availability_status_distribution=(),
        universe_type=UniverseType.STATIC,
        universe_instruments=("AAPL", "MSFT"),
        universe_definition="static fixture universe",
        survivorship_limitation_flag=True,
        survivorship_limitation_code="static-universe",
        canonical_schema_version="1",
        returns_schema_version="1",
        regimequant_version="0.0.0",
        adjusted_vintage_unverified_flag=adjustment_state == AdjustmentState.ADJUSTED,
        pit_membership_not_supported_flag=True,
        limitations=("fixture-provider",),
    )


def _canonical_prices() -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "instrument": "AAPL",
                "timestamp": pd.Timestamp("2024-01-02 16:00:00-05:00"),
                "available_at": pd.Timestamp("2024-01-02 16:00:00-05:00"),
                "availability_status": AvailabilityStatus.VERIFIED_TIMESTAMP.value,
                "price": 100.0,
                "price_type": PriceType.CLOSE.value,
                "adjustment_state": AdjustmentState.RAW.value,
                "currency": "USD",
                "source_observation_id": "a1",
            },
            {
                "instrument": "AAPL",
                "timestamp": pd.Timestamp("2024-01-03 16:00:00-05:00"),
                "available_at": pd.Timestamp("2024-01-03 16:00:00-05:00"),
                "availability_status": AvailabilityStatus.VERIFIED_TIMESTAMP.value,
                "price": 110.0,
                "price_type": PriceType.CLOSE.value,
                "adjustment_state": AdjustmentState.RAW.value,
                "currency": "USD",
                "source_observation_id": "a2",
            },
            {
                "instrument": "MSFT",
                "timestamp": pd.Timestamp("2024-01-03 16:00:00-05:00"),
                "available_at": pd.Timestamp("2024-01-03 16:00:00-05:00"),
                "availability_status": AvailabilityStatus.VERIFIED_TIMESTAMP.value,
                "price": 200.0,
                "price_type": PriceType.CLOSE.value,
                "adjustment_state": AdjustmentState.RAW.value,
                "currency": "USD",
                "source_observation_id": "m1",
            },
        ]
    )
    canonical, _ = canonicalize_market_data(frame, _provenance())
    return canonical


def test_fixture_provider_normalizes_provider_specific_columns_and_timezone() -> None:
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
        universe_instruments=("AAPL",),
        survivorship_limitation_flag=True,
        survivorship_limitation_code="static-universe",
    )
    raw, provenance = provider.fetch_raw(instruments=("AAPL",))
    assert raw.iloc[0]["timestamp"] == pd.Timestamp("2024-01-02 21:00:00+00:00")
    assert raw.iloc[0]["available_at"] == pd.Timestamp("2024-01-02 21:00:00+00:00")
    assert provenance.universe_type == UniverseType.STATIC
    assert provenance.survivorship_limitation_flag is True
    assert provenance.source_availability_status_distribution == ((AvailabilityStatus.VERIFIED_TIMESTAMP, 1),)
    assert provenance.availability_status_distribution == ()


def test_canonicalization_distinguishes_source_and_canonical_distributions() -> None:
    source = pd.DataFrame(
        [
            {
                "instrument": "AAPL",
                "timestamp": pd.Timestamp("2024-01-02 16:00:00-05:00"),
                "available_at": pd.Timestamp("2024-01-02 16:00:00-05:00"),
                "availability_status": AvailabilityStatus.VERIFIED_TIMESTAMP.value,
                "price": 100.0,
                "price_type": PriceType.CLOSE.value,
                "adjustment_state": AdjustmentState.RAW.value,
                "currency": "USD",
                "source_observation_id": "b",
            },
            {
                "instrument": "AAPL",
                "timestamp": pd.Timestamp("2024-01-02 16:00:00-05:00"),
                "available_at": pd.Timestamp("2024-01-02 16:00:00-05:00"),
                "availability_status": AvailabilityStatus.VERIFIED_TIMESTAMP.value,
                "price": 100.0,
                "price_type": PriceType.CLOSE.value,
                "adjustment_state": AdjustmentState.RAW.value,
                "currency": "USD",
                "source_observation_id": "a",
            },
        ]
    )
    provider = FixtureMarketDataProvider(source_table=source, source_timezone="America/New_York")
    raw, provenance = provider.fetch_raw(instruments=("AAPL",))
    assert provenance.source_availability_status_distribution == ((AvailabilityStatus.VERIFIED_TIMESTAMP, 2),)
    assert provenance.availability_status_distribution == ()

    canonical, diagnostics = canonicalize_market_data(raw, provenance)
    assert diagnostics["exact_duplicate_collapse_count"] == 1
    canonical_provenance = canonical.attrs["provenance"]
    assert canonical_provenance.source_availability_status_distribution == ((AvailabilityStatus.VERIFIED_TIMESTAMP, 2),)
    assert canonical_provenance.availability_status_distribution == (
        (AvailabilityStatus.VERIFIED_TIMESTAMP, 1),
        (AvailabilityStatus.UNKNOWN_UNVERIFIED, 0),
    )


def test_provider_missing_timezone_policy_fails_on_naive_timestamps() -> None:
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


def test_future_append_leaves_historical_returns_unchanged() -> None:
    base = _canonical_prices()
    base_simple, _ = compute_simple_returns(base)
    base_log, _ = compute_log_returns(base)

    extended = pd.concat(
        [
            base,
            pd.DataFrame(
                [
                    {
                        "instrument": "AAPL",
                        "timestamp": pd.Timestamp("2024-01-10 16:00:00-05:00"),
                        "available_at": pd.Timestamp("2024-01-10 16:00:00-05:00"),
                        "availability_status": AvailabilityStatus.VERIFIED_TIMESTAMP.value,
                        "price": 150.0,
                        "price_type": PriceType.CLOSE.value,
                        "adjustment_state": AdjustmentState.RAW.value,
                        "currency": "USD",
                        "source_observation_id": "a3",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    extended_canonical, _ = canonicalize_market_data(extended, _provenance())
    extended_simple, _ = compute_simple_returns(extended_canonical)
    extended_log, _ = compute_log_returns(extended_canonical)

    cutoff = pd.Timestamp("2024-01-05 16:00:00-05:00").tz_convert("UTC")
    base_historical_simple = base_simple[base_simple["timestamp"] <= cutoff].sort_values(["instrument", "timestamp"]).reset_index(drop=True)
    extended_historical_simple = extended_simple[extended_simple["timestamp"] <= cutoff].sort_values(["instrument", "timestamp"]).reset_index(drop=True)
    base_historical_log = base_log[base_log["timestamp"] <= cutoff].sort_values(["instrument", "timestamp"]).reset_index(drop=True)
    extended_historical_log = extended_log[extended_log["timestamp"] <= cutoff].sort_values(["instrument", "timestamp"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(base_historical_simple, extended_historical_simple)
    pd.testing.assert_frame_equal(base_historical_log, extended_historical_log)


def test_information_availability_causality_uses_later_required_input() -> None:
    canonical = _canonical_prices()
    canonical.loc[canonical.index[0], "available_at"] = pd.Timestamp("2024-01-04T20:00:00Z")
    returns, _ = compute_simple_returns(canonical)
    aapl = returns[returns["instrument"] == "AAPL"].reset_index(drop=True)
    assert aapl.loc[1, "available_at"] == pd.Timestamp("2024-01-04T20:00:00Z")


def test_provenance_instrument_sets_are_deterministic_under_permutations() -> None:
    base_table = pd.DataFrame(
        [
            {
                "instrument": "AAPL",
                "timestamp": pd.Timestamp("2024-01-02 16:00:00-05:00"),
                "available_at": pd.Timestamp("2024-01-02 16:00:00-05:00"),
                "availability_status": AvailabilityStatus.VERIFIED_TIMESTAMP.value,
                "price": 100.0,
                "price_type": PriceType.CLOSE.value,
                "adjustment_state": AdjustmentState.RAW.value,
                "currency": "USD",
            }
        ]
    )
    provider_a = FixtureMarketDataProvider(
        source_table=base_table,
        source_timezone="America/New_York",
        universe_instruments=("MSFT", "AAPL"),
    )
    provider_b = FixtureMarketDataProvider(
        source_table=base_table,
        source_timezone="America/New_York",
        universe_instruments=("AAPL", "MSFT"),
    )

    _, provenance_a = provider_a.fetch_raw(instruments=("MSFT", "AAPL"))
    _, provenance_b = provider_b.fetch_raw(instruments=("AAPL", "MSFT"))

    assert provenance_a.requested_instruments == ("AAPL", "MSFT")
    assert provenance_b.requested_instruments == ("AAPL", "MSFT")
    assert provenance_a.universe_instruments == ("AAPL", "MSFT")
    assert provenance_b.universe_instruments == ("AAPL", "MSFT")


def test_attrs_provenance_is_not_durable_across_csv_roundtrip(tmp_path: Path) -> None:
    canonical = _canonical_prices()
    csv_path = tmp_path / "canonical.csv"
    canonical.to_csv(csv_path, index=False)
    reloaded = pd.read_csv(csv_path)
    assert "provenance" not in reloaded.attrs
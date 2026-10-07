from __future__ import annotations

import math

import pandas as pd
import pytest

from regimequant.data import (
    AVAILABILITY_POLICY_FIXTURE_DECLARED,
    AdjustmentState,
    AvailabilityStatus,
    MarketDataProvenance,
    PriceType,
    UniverseType,
    canonicalize_market_data,
    compute_log_returns,
    compute_simple_returns,
)
from regimequant.data.errors import MarketDataDomainError, MarketDataValidationError


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


def _canonical_prices(adjustment_state: AdjustmentState = AdjustmentState.RAW) -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "instrument": "AAPL",
                "timestamp": pd.Timestamp("2024-01-02 16:00:00-05:00"),
                "available_at": pd.Timestamp("2024-01-02 16:00:00-05:00"),
                "availability_status": AvailabilityStatus.VERIFIED_TIMESTAMP.value,
                "price": 100.0,
                "price_type": PriceType.CLOSE.value,
                "adjustment_state": adjustment_state.value,
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
                "adjustment_state": adjustment_state.value,
                "currency": "USD",
                "source_observation_id": "a2",
            },
            {
                "instrument": "AAPL",
                "timestamp": pd.Timestamp("2024-01-05 16:00:00-05:00"),
                "available_at": pd.Timestamp("2024-01-05 16:00:00-05:00"),
                "availability_status": AvailabilityStatus.VERIFIED_TIMESTAMP.value,
                "price": 121.0,
                "price_type": PriceType.CLOSE.value,
                "adjustment_state": adjustment_state.value,
                "currency": "USD",
                "source_observation_id": "a3",
            },
            {
                "instrument": "MSFT",
                "timestamp": pd.Timestamp("2024-01-03 16:00:00-05:00"),
                "available_at": pd.Timestamp("2024-01-03 16:00:00-05:00"),
                "availability_status": AvailabilityStatus.VERIFIED_TIMESTAMP.value,
                "price": 200.0,
                "price_type": PriceType.CLOSE.value,
                "adjustment_state": adjustment_state.value,
                "currency": "USD",
                "source_observation_id": "m1",
            },
        ]
    )
    canonical, _ = canonicalize_market_data(frame, _provenance(adjustment_state))
    return canonical


def test_simple_return_benchmark() -> None:
    canonical = _canonical_prices()
    returns, diagnostics = compute_simple_returns(canonical)
    assert diagnostics == {"input_row_count": 4, "output_row_count": 4}
    aapl = returns[returns["instrument"] == "AAPL"].reset_index(drop=True)
    assert math.isnan(aapl.loc[0, "return_value"])
    assert aapl.loc[1, "return_value"] == pytest.approx(0.1)
    assert aapl.loc[2, "return_value"] == pytest.approx(0.1)


def test_log_return_benchmark() -> None:
    canonical = _canonical_prices()
    returns, _ = compute_log_returns(canonical)
    aapl = returns[returns["instrument"] == "AAPL"].reset_index(drop=True)
    assert math.isnan(aapl.loc[0, "return_value"])
    assert aapl.loc[1, "return_value"] == pytest.approx(math.log(1.1))


def test_first_return_is_nan_and_first_gap_metadata_missing() -> None:
    canonical = _canonical_prices()
    returns, _ = compute_simple_returns(canonical)
    first = returns[returns["instrument"] == "AAPL"].iloc[0]
    assert math.isnan(first["return_value"])
    assert pd.isna(first["previous_timestamp"])
    assert pd.isna(first["interval_elapsed_seconds"])
    assert first["availability_status"] == AvailabilityStatus.UNDEFINED.value
    assert pd.isna(first["available_at"])


def test_constant_positive_prices_yield_zero_returns_after_first() -> None:
    frame = _canonical_prices().iloc[[0, 1]].copy()
    frame.loc[:, "price"] = 50.0
    returns, _ = compute_simple_returns(frame)
    assert returns.iloc[1]["return_value"] == pytest.approx(0.0)


def test_cross_instrument_isolation_and_non_monotonic_input() -> None:
    canonical = _canonical_prices()
    shuffled = canonical.sample(frac=1.0, random_state=7).reset_index(drop=True)
    returns, _ = compute_simple_returns(shuffled)
    aapl = returns[returns["instrument"] == "AAPL"].reset_index(drop=True)
    msft = returns[returns["instrument"] == "MSFT"].reset_index(drop=True)
    assert aapl.iloc[1]["return_value"] == pytest.approx(0.1)
    assert math.isnan(msft.iloc[0]["return_value"])


def test_missing_date_gap_uses_previous_observation_and_exposes_interval() -> None:
    canonical = _canonical_prices()
    returns, _ = compute_simple_returns(canonical)
    aapl = returns[returns["instrument"] == "AAPL"].reset_index(drop=True)
    assert aapl.loc[2, "return_value"] == pytest.approx(0.1)
    assert aapl.loc[2, "interval_elapsed_seconds"] == pytest.approx(2 * 24 * 60 * 60)
    assert aapl.loc[2, "previous_timestamp"] == pd.Timestamp("2024-01-05 21:00:00+00:00") - pd.Timedelta(days=2)


def test_return_availability_uses_all_required_inputs() -> None:
    canonical = _canonical_prices()
    canonical.loc[canonical.index[1], "available_at"] = pd.NaT
    canonical.loc[canonical.index[1], "availability_status"] = AvailabilityStatus.UNKNOWN_UNVERIFIED.value
    returns, _ = compute_simple_returns(canonical)
    aapl = returns[returns["instrument"] == "AAPL"].reset_index(drop=True)
    assert aapl.loc[1, "availability_status"] == AvailabilityStatus.UNKNOWN_UNVERIFIED.value
    assert pd.isna(aapl.loc[1, "available_at"])


def test_verified_inputs_yield_max_available_at_for_returns() -> None:
    canonical = _canonical_prices()
    canonical.loc[canonical.index[1], "available_at"] = pd.Timestamp("2024-01-04T20:00:00Z")
    returns, _ = compute_simple_returns(canonical)
    aapl = returns[returns["instrument"] == "AAPL"].reset_index(drop=True)
    assert aapl.loc[1, "available_at"] == pd.Timestamp("2024-01-04T20:00:00Z")


def test_returns_are_separate_from_prices_and_inputs_unchanged() -> None:
    canonical = _canonical_prices()
    original = canonical.copy(deep=True)
    returns, _ = compute_simple_returns(canonical)
    assert "return_value" not in canonical.columns
    pd.testing.assert_frame_equal(canonical, original)
    assert list(returns.columns) == [
        "instrument",
        "timestamp",
        "available_at",
        "availability_status",
        "previous_timestamp",
        "previous_available_at",
        "previous_availability_status",
        "return_type",
        "return_value",
        "price_type",
        "adjustment_state",
        "interval_elapsed_seconds",
    ]


def test_first_return_undefined_does_not_contaminate_canonical_distribution() -> None:
    canonical = _canonical_prices()
    returns, _ = compute_simple_returns(canonical)
    assert returns.iloc[0]["availability_status"] == AvailabilityStatus.UNDEFINED.value
    assert canonical.attrs["provenance"].availability_status_distribution == (
        (AvailabilityStatus.VERIFIED_TIMESTAMP, 4),
        (AvailabilityStatus.UNKNOWN_UNVERIFIED, 0),
    )


def test_invalid_canonical_input_rejected() -> None:
    canonical = _canonical_prices()
    canonical.loc[:, "price"] = -1.0
    with pytest.raises(MarketDataDomainError):
        compute_simple_returns(canonical)


def test_mixed_adjustment_state_rejected() -> None:
    canonical = _canonical_prices()
    canonical.loc[canonical.index[1], "adjustment_state"] = AdjustmentState.ADJUSTED.value
    with pytest.raises(MarketDataValidationError):
        compute_simple_returns(canonical)
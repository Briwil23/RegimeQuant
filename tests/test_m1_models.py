from __future__ import annotations

from datetime import timezone

import pandas as pd
import pytest

from regimequant.data import (
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


def test_enum_values_are_string_backed() -> None:
    assert PriceType.CLOSE.value == "close"
    assert AdjustmentState.RAW.value == "raw"
    assert AdjustmentState.ADJUSTED.value == "adjusted"
    assert AvailabilityStatus.VERIFIED_TIMESTAMP.value == "verified_timestamp"
    assert AvailabilityStatus.UNKNOWN_UNVERIFIED.value == "unknown_unverified"
    assert AvailabilityStatus.UNDEFINED.value == "undefined"
    assert UniverseType.STATIC.value == "static"
    assert ReturnType.SIMPLE.value == "simple"
    assert ReturnType.LOG.value == "log"


def test_market_data_provenance_is_frozen_and_validated() -> None:
    provenance = MarketDataProvenance(
        provider_id="fixture",
        requested_instruments=("AAPL", "MSFT"),
        requested_start_timestamp=pd.Timestamp("2024-01-01T00:00:00Z"),
        requested_end_timestamp=pd.Timestamp("2024-01-03T00:00:00Z"),
        requested_price_type=PriceType.CLOSE,
        requested_adjustment_state=AdjustmentState.RAW,
        retrieval_timestamp_utc=pd.Timestamp("2024-01-05T00:00:00Z"),
        retrieval_environment="fixture",
        price_type=PriceType.CLOSE,
        adjustment_state=AdjustmentState.RAW,
        timestamp_timezone_policy="utc",
        availability_policy_id=AVAILABILITY_POLICY_FIXTURE_DECLARED,
        source_availability_status_distribution=((AvailabilityStatus.VERIFIED_TIMESTAMP, 3),),
        availability_status_distribution=((AvailabilityStatus.VERIFIED_TIMESTAMP, 2),),
        universe_type=UniverseType.STATIC,
        universe_instruments=("AAPL", "MSFT"),
        universe_definition="static fixture universe",
        survivorship_limitation_flag=True,
        survivorship_limitation_code="static-universe",
        canonical_schema_version="1",
        returns_schema_version="1",
        regimequant_version="0.0.0",
        adjusted_vintage_unverified_flag=False,
        pit_membership_not_supported_flag=True,
        limitations=("fixture-provider",),
    )

    assert provenance.retrieval_timestamp_utc.tz == timezone.utc
    assert provenance.requested_instruments == ("AAPL", "MSFT")
    assert provenance.universe_instruments == ("AAPL", "MSFT")
    assert provenance.source_availability_status_distribution == ((AvailabilityStatus.VERIFIED_TIMESTAMP, 3),)
    with pytest.raises(Exception):
        provenance.provider_id = "other"  # type: ignore[misc]


def test_distribution_counts_reject_negative_and_noninteger_values() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        MarketDataProvenance(
            provider_id="fixture",
            requested_instruments=("AAPL",),
            retrieval_timestamp_utc=pd.Timestamp("2024-01-05T00:00:00Z"),
            requested_price_type=PriceType.CLOSE,
            requested_adjustment_state=AdjustmentState.RAW,
            price_type=PriceType.CLOSE,
            adjustment_state=AdjustmentState.RAW,
            universe_type=UniverseType.STATIC,
            universe_instruments=("AAPL",),
            availability_policy_id=AVAILABILITY_POLICY_FIXTURE_DECLARED,
            availability_status_distribution=((AvailabilityStatus.VERIFIED_TIMESTAMP, -1),),
            pit_membership_not_supported_flag=True,
        )

    with pytest.raises(ValueError, match="integers"):
        MarketDataProvenance(
            provider_id="fixture",
            requested_instruments=("AAPL",),
            retrieval_timestamp_utc=pd.Timestamp("2024-01-05T00:00:00Z"),
            requested_price_type=PriceType.CLOSE,
            requested_adjustment_state=AdjustmentState.RAW,
            price_type=PriceType.CLOSE,
            adjustment_state=AdjustmentState.RAW,
            universe_type=UniverseType.STATIC,
            universe_instruments=("AAPL",),
            availability_policy_id=AVAILABILITY_POLICY_FIXTURE_DECLARED,
            source_availability_status_distribution=((AvailabilityStatus.VERIFIED_TIMESTAMP, 1.5),),
            pit_membership_not_supported_flag=True,
        )


def test_requested_and_universe_instruments_are_normalized_deterministically() -> None:
    provenance = MarketDataProvenance(
        provider_id="fixture",
        requested_instruments=("MSFT", "AAPL", "MSFT"),
        retrieval_timestamp_utc=pd.Timestamp("2024-01-05T00:00:00Z"),
        requested_price_type=PriceType.CLOSE,
        requested_adjustment_state=AdjustmentState.RAW,
        price_type=PriceType.CLOSE,
        adjustment_state=AdjustmentState.RAW,
        universe_type=UniverseType.STATIC,
        universe_instruments=("MSFT", "AAPL", "MSFT"),
        availability_policy_id=AVAILABILITY_POLICY_FIXTURE_DECLARED,
        pit_membership_not_supported_flag=True,
    )
    assert provenance.requested_instruments == ("AAPL", "MSFT")
    assert provenance.universe_instruments == ("AAPL", "MSFT")


def test_availability_policy_id_is_controlled() -> None:
    with pytest.raises(ValueError, match="availability_policy_id"):
        MarketDataProvenance(
            provider_id="fixture",
            requested_instruments=("AAPL",),
            retrieval_timestamp_utc=pd.Timestamp("2024-01-05T00:00:00Z"),
            requested_price_type=PriceType.CLOSE,
            requested_adjustment_state=AdjustmentState.RAW,
            price_type=PriceType.CLOSE,
            adjustment_state=AdjustmentState.RAW,
            universe_type=UniverseType.STATIC,
            universe_instruments=("AAPL",),
            availability_policy_id="free_text_policy",
            pit_membership_not_supported_flag=True,
        )

    assert AVAILABILITY_POLICY_FIXTURE_DECLARED in ALLOWED_AVAILABILITY_POLICY_IDS
    assert AVAILABILITY_POLICY_EXTERNAL_VERIFIED in ALLOWED_AVAILABILITY_POLICY_IDS
    assert AVAILABILITY_POLICY_UNKNOWN_UNVERIFIED in ALLOWED_AVAILABILITY_POLICY_IDS


def test_unknown_unverified_policy_cannot_report_verified_distribution() -> None:
    with pytest.raises(ValueError, match="cannot claim verified availability"):
        MarketDataProvenance(
            provider_id="fixture",
            requested_instruments=("AAPL",),
            retrieval_timestamp_utc=pd.Timestamp("2024-01-05T00:00:00Z"),
            requested_price_type=PriceType.CLOSE,
            requested_adjustment_state=AdjustmentState.RAW,
            price_type=PriceType.CLOSE,
            adjustment_state=AdjustmentState.RAW,
            universe_type=UniverseType.STATIC,
            universe_instruments=("AAPL",),
            availability_policy_id=AVAILABILITY_POLICY_UNKNOWN_UNVERIFIED,
            availability_status_distribution=((AvailabilityStatus.VERIFIED_TIMESTAMP, 1),),
            pit_membership_not_supported_flag=True,
        )


def test_adjusted_vintage_limitation_required_for_adjusted_data() -> None:
    with pytest.raises(ValueError, match="adjusted historical data"):
        MarketDataProvenance(
            provider_id="fixture",
            requested_instruments=("AAPL",),
            retrieval_timestamp_utc=pd.Timestamp("2024-01-05T00:00:00Z"),
            requested_price_type=PriceType.CLOSE,
            requested_adjustment_state=AdjustmentState.ADJUSTED,
            price_type=PriceType.CLOSE,
            adjustment_state=AdjustmentState.ADJUSTED,
            universe_type=UniverseType.STATIC,
            universe_instruments=("AAPL",),
            adjusted_vintage_unverified_flag=False,
            pit_membership_not_supported_flag=True,
        )


def test_static_universe_requires_machine_readable_metadata() -> None:
    provenance = MarketDataProvenance(
        provider_id="fixture",
        requested_instruments=("AAPL",),
        retrieval_timestamp_utc=pd.Timestamp("2024-01-05T00:00:00Z"),
        requested_price_type=PriceType.CLOSE,
        requested_adjustment_state=AdjustmentState.RAW,
        price_type=PriceType.CLOSE,
        adjustment_state=AdjustmentState.RAW,
        universe_type=UniverseType.STATIC,
        universe_instruments=("AAPL",),
        survivorship_limitation_flag=True,
        survivorship_limitation_code="static-universe",
        adjusted_vintage_unverified_flag=False,
        pit_membership_not_supported_flag=True,
    )
    assert provenance.universe_type == UniverseType.STATIC
    assert provenance.survivorship_limitation_flag is True
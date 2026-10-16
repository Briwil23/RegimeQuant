from __future__ import annotations

import pandas as pd

from regimequant.data import (
    AVAILABILITY_POLICY_FIXTURE_DECLARED,
    AdjustmentState,
    AvailabilityStatus,
    MarketDataProvenance,
    PriceType,
    ReturnType,
    UniverseType,
)
from regimequant.features import FeatureSpec, compute_features_from_returns
from regimequant.features.definitions import FEATURE_ROLLING_MEAN


def _source_provenance() -> MarketDataProvenance:
    return MarketDataProvenance(
        provider_id="fixture",
        requested_instruments=("AAPL",),
        retrieval_timestamp_utc=pd.Timestamp("2024-01-06T12:00:00Z"),
        requested_price_type=PriceType.CLOSE,
        requested_adjustment_state=AdjustmentState.RAW,
        price_type=PriceType.CLOSE,
        adjustment_state=AdjustmentState.RAW,
        universe_type=UniverseType.STATIC,
        universe_instruments=("AAPL",),
        availability_policy_id=AVAILABILITY_POLICY_FIXTURE_DECLARED,
        pit_membership_not_supported_flag=True,
    )


def _returns(order: list[int]) -> pd.DataFrame:
    base_rows = [
        {
            "instrument": "AAPL",
            "timestamp": pd.Timestamp("2024-01-01T10:05:00Z"),
            "available_at": pd.Timestamp("2024-01-01T10:05:00Z"),
            "availability_status": AvailabilityStatus.VERIFIED_TIMESTAMP.value,
            "previous_timestamp": pd.NaT,
            "previous_available_at": pd.NaT,
            "previous_availability_status": None,
            "return_type": ReturnType.SIMPLE.value,
            "return_value": 0.10,
            "price_type": PriceType.CLOSE.value,
            "adjustment_state": AdjustmentState.RAW.value,
            "interval_elapsed_seconds": None,
        },
        {
            "instrument": "AAPL",
            "timestamp": pd.Timestamp("2024-01-01T10:10:00Z"),
            "available_at": pd.Timestamp("2024-01-01T10:10:00Z"),
            "availability_status": AvailabilityStatus.VERIFIED_TIMESTAMP.value,
            "previous_timestamp": pd.Timestamp("2024-01-01T10:05:00Z"),
            "previous_available_at": pd.Timestamp("2024-01-01T10:05:00Z"),
            "previous_availability_status": AvailabilityStatus.VERIFIED_TIMESTAMP.value,
            "return_type": ReturnType.SIMPLE.value,
            "return_value": -0.05,
            "price_type": PriceType.CLOSE.value,
            "adjustment_state": AdjustmentState.RAW.value,
            "interval_elapsed_seconds": 300.0,
        },
        {
            "instrument": "AAPL",
            "timestamp": pd.Timestamp("2024-01-01T10:15:00Z"),
            "available_at": pd.Timestamp("2024-01-01T10:15:00Z"),
            "availability_status": AvailabilityStatus.VERIFIED_TIMESTAMP.value,
            "previous_timestamp": pd.Timestamp("2024-01-01T10:10:00Z"),
            "previous_available_at": pd.Timestamp("2024-01-01T10:10:00Z"),
            "previous_availability_status": AvailabilityStatus.VERIFIED_TIMESTAMP.value,
            "return_type": ReturnType.SIMPLE.value,
            "return_value": 0.02,
            "price_type": PriceType.CLOSE.value,
            "adjustment_state": AdjustmentState.RAW.value,
            "interval_elapsed_seconds": 300.0,
        },
    ]
    frame = pd.DataFrame([base_rows[index] for index in order])
    frame.attrs["provenance"] = {
        "source_canonical_provenance": _source_provenance(),
        "return_policy_version": "1",
        "return_type": ReturnType.SIMPLE.value,
        "generated_at_utc": pd.Timestamp("2024-01-06T12:00:00Z"),
    }
    return frame


def test_permutation_invariance_of_returns_input_order() -> None:
    spec = FeatureSpec(feature_name=FEATURE_ROLLING_MEAN, window_size=2, min_observations=2)
    ordered, _ = compute_features_from_returns(_returns([0, 1, 2]), [spec])
    permuted, _ = compute_features_from_returns(_returns([2, 0, 1]), [spec])
    assert ordered.reset_index(drop=True).equals(permuted.reset_index(drop=True))

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
from regimequant.features.definitions import FEATURE_CUMULATIVE_RETURN, FEATURE_ROLLING_MEAN
from regimequant.features.models import MathStatus, UndefinedReason


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


def _returns_with_unknown_status() -> pd.DataFrame:
    rows = [
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
            "available_at": None,
            "availability_status": AvailabilityStatus.UNKNOWN_UNVERIFIED.value,
            "previous_timestamp": pd.Timestamp("2024-01-01T10:05:00Z"),
            "previous_available_at": pd.Timestamp("2024-01-01T10:05:00Z"),
            "previous_availability_status": AvailabilityStatus.VERIFIED_TIMESTAMP.value,
            "return_type": ReturnType.SIMPLE.value,
            "return_value": 0.05,
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
            "previous_available_at": None,
            "previous_availability_status": AvailabilityStatus.UNKNOWN_UNVERIFIED.value,
            "return_type": ReturnType.SIMPLE.value,
            "return_value": 0.02,
            "price_type": PriceType.CLOSE.value,
            "adjustment_state": AdjustmentState.RAW.value,
            "interval_elapsed_seconds": 300.0,
        },
    ]
    frame = pd.DataFrame(rows)
    frame.attrs["provenance"] = {
        "source_canonical_provenance": _source_provenance(),
        "return_policy_version": "1",
        "return_type": ReturnType.SIMPLE.value,
        "generated_at_utc": pd.Timestamp("2024-01-06T12:00:00Z"),
    }
    return frame


def test_defined_feature_with_unknown_input_status_has_unknown_availability_and_null_available_at() -> None:
    frame = _returns_with_unknown_status()
    spec = FeatureSpec(feature_name=FEATURE_ROLLING_MEAN, window_size=2, min_observations=2)
    features, _ = compute_features_from_returns(frame, [spec])

    latest = features.sort_values("timestamp").iloc[-1]
    assert latest["math_status"] == MathStatus.DEFINED.value
    assert latest["input_availability_status"] == AvailabilityStatus.UNKNOWN_UNVERIFIED.value
    assert pd.isna(latest["available_at"])


def test_complete_window_feature_is_undefined_when_window_includes_undefined_input_return() -> None:
    frame = _returns_with_unknown_status().copy()
    frame.loc[1, "return_value"] = float("nan")
    frame.loc[1, "availability_status"] = AvailabilityStatus.UNDEFINED.value

    spec = FeatureSpec(
        feature_name=FEATURE_CUMULATIVE_RETURN,
        window_size=2,
        min_observations=2,
        require_complete_window=True,
        allow_partial_valid=False,
    )
    features, _ = compute_features_from_returns(frame, [spec])

    latest = features.sort_values("timestamp").iloc[-1]
    assert latest["math_status"] == MathStatus.UNDEFINED.value
    assert latest["undefined_reason"] == UndefinedReason.INPUT_UNDEFINED_RETURN.value
    assert pd.isna(latest["available_at"])

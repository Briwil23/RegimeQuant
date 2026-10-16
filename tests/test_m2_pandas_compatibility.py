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
from regimequant.features import compute_features_from_returns, list_m2_v1_feature_specs


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


def _returns_with_nullable_fields() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "instrument": pd.Series(["AAPL", "AAPL"], dtype="string"),
            "timestamp": [pd.Timestamp("2024-01-01T10:05:00Z"), pd.Timestamp("2024-01-01T10:10:00Z")],
            "available_at": [pd.Timestamp("2024-01-01T10:05:00Z"), pd.Timestamp("2024-01-01T10:10:00Z")],
            "availability_status": pd.Series(
                [
                    AvailabilityStatus.VERIFIED_TIMESTAMP.value,
                    AvailabilityStatus.VERIFIED_TIMESTAMP.value,
                ],
                dtype="string",
            ),
            "previous_timestamp": [pd.NaT, pd.Timestamp("2024-01-01T10:05:00Z")],
            "previous_available_at": [pd.NaT, pd.Timestamp("2024-01-01T10:05:00Z")],
            "previous_availability_status": pd.Series(
                [None, AvailabilityStatus.VERIFIED_TIMESTAMP.value],
                dtype="string",
            ),
            "return_type": pd.Series([ReturnType.SIMPLE.value, ReturnType.SIMPLE.value], dtype="string"),
            "return_value": pd.Series([0.1, 0.2], dtype="Float64"),
            "price_type": pd.Series([PriceType.CLOSE.value, PriceType.CLOSE.value], dtype="string"),
            "adjustment_state": pd.Series(
                [AdjustmentState.RAW.value, AdjustmentState.RAW.value],
                dtype="string",
            ),
            "interval_elapsed_seconds": pd.Series([None, 300.0], dtype="Float64"),
        }
    )
    frame.attrs["provenance"] = {
        "source_canonical_provenance": _source_provenance(),
        "return_policy_version": "1",
        "return_type": ReturnType.SIMPLE.value,
        "generated_at_utc": pd.Timestamp("2024-01-06T12:00:00Z"),
    }
    return frame


def test_nullable_and_extension_dtypes_are_accepted() -> None:
    features, diagnostics = compute_features_from_returns(
        _returns_with_nullable_fields(),
        list_m2_v1_feature_specs(),
    )
    assert diagnostics["input_row_count"] == 2
    assert diagnostics["output_row_count"] == len(features)

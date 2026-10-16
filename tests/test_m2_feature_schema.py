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
from regimequant.features.models import FEATURE_COLUMNS, PRIMARY_KEY_COLUMNS, ScopeType


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


def _returns_table() -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "instrument": "AAPL",
                "timestamp": pd.Timestamp("2024-01-01T10:00:00Z"),
                "available_at": None,
                "availability_status": AvailabilityStatus.UNDEFINED.value,
                "previous_timestamp": pd.NaT,
                "previous_available_at": pd.NaT,
                "previous_availability_status": None,
                "return_type": ReturnType.SIMPLE.value,
                "return_value": float("nan"),
                "price_type": PriceType.CLOSE.value,
                "adjustment_state": AdjustmentState.RAW.value,
                "interval_elapsed_seconds": None,
            },
            {
                "instrument": "AAPL",
                "timestamp": pd.Timestamp("2024-01-01T10:05:00Z"),
                "available_at": pd.Timestamp("2024-01-01T10:05:00Z"),
                "availability_status": AvailabilityStatus.VERIFIED_TIMESTAMP.value,
                "previous_timestamp": pd.Timestamp("2024-01-01T10:00:00Z"),
                "previous_available_at": pd.NaT,
                "previous_availability_status": AvailabilityStatus.UNDEFINED.value,
                "return_type": ReturnType.SIMPLE.value,
                "return_value": 0.10,
                "price_type": PriceType.CLOSE.value,
                "adjustment_state": AdjustmentState.RAW.value,
                "interval_elapsed_seconds": 300.0,
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
    )
    frame.attrs["provenance"] = {
        "source_canonical_provenance": _source_provenance(),
        "return_policy_version": "1",
        "return_type": ReturnType.SIMPLE.value,
        "generated_at_utc": pd.Timestamp("2024-01-06T12:00:00Z"),
    }
    return frame


def test_output_schema_and_scope_contract() -> None:
    features, diagnostics = compute_features_from_returns(_returns_table(), list_m2_v1_feature_specs())
    assert list(features.columns) == FEATURE_COLUMNS
    assert diagnostics["input_row_count"] == 4
    assert diagnostics["output_row_count"] == len(features)
    assert set(features["scope_type"].astype(str).tolist()) == {ScopeType.INSTRUMENT.value}
    assert not features.duplicated(subset=PRIMARY_KEY_COLUMNS).any()


def test_empty_spec_list_returns_typed_empty_output() -> None:
    features, diagnostics = compute_features_from_returns(_returns_table(), [])
    assert list(features.columns) == FEATURE_COLUMNS
    assert features.empty
    assert diagnostics["output_row_count"] == 0

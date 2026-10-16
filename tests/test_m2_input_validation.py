from __future__ import annotations

import pandas as pd
import pytest

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
from regimequant.features.models import FeatureValidationError


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


def _base_returns() -> pd.DataFrame:
    frame = pd.DataFrame(
        [
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
            }
        ]
    )
    frame.attrs["provenance"] = {
        "source_canonical_provenance": _source_provenance(),
        "return_policy_version": "1",
        "return_type": ReturnType.SIMPLE.value,
        "generated_at_utc": pd.Timestamp("2024-01-06T12:00:00Z"),
    }
    return frame


def test_rejects_missing_required_return_columns() -> None:
    frame = _base_returns().drop(columns=["return_value"])
    spec = FeatureSpec(feature_name=FEATURE_ROLLING_MEAN, window_size=1, min_observations=1)
    with pytest.raises(FeatureValidationError):
        compute_features_from_returns(frame, [spec])


def test_rejects_duplicate_return_keys() -> None:
    frame = pd.concat([_base_returns(), _base_returns()], ignore_index=True)
    spec = FeatureSpec(feature_name=FEATURE_ROLLING_MEAN, window_size=1, min_observations=1)
    with pytest.raises(FeatureValidationError):
        compute_features_from_returns(frame, [spec])


def test_rejects_mixed_return_types() -> None:
    frame = pd.concat([_base_returns(), _base_returns()], ignore_index=True)
    frame.loc[1, "timestamp"] = pd.Timestamp("2024-01-01T10:10:00Z")
    frame.loc[1, "return_type"] = ReturnType.LOG.value
    spec = FeatureSpec(feature_name=FEATURE_ROLLING_MEAN, window_size=1, min_observations=1)
    with pytest.raises(FeatureValidationError):
        compute_features_from_returns(frame, [spec])


def test_rejects_missing_provenance_metadata() -> None:
    frame = _base_returns()
    frame.attrs.pop("provenance")
    spec = FeatureSpec(feature_name=FEATURE_ROLLING_MEAN, window_size=1, min_observations=1)
    with pytest.raises(FeatureValidationError):
        compute_features_from_returns(frame, [spec])

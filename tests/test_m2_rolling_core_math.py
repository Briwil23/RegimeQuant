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
    ReturnType,
    UniverseType,
)
from regimequant.features import FeatureSpec, compute_features_from_returns
from regimequant.features.definitions import (
    FEATURE_CUMULATIVE_RETURN,
    FEATURE_DOWNSIDE_DEVIATION,
    FEATURE_DOWNSIDE_FREQUENCY,
    FEATURE_LAGGED_MOMENTUM,
    FEATURE_REALIZED_VARIANCE,
    FEATURE_ROLLING_MEAN,
    FEATURE_ROLLING_STD,
    FEATURE_VOLATILITY_RATIO,
)


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


def _returns_table(return_type: ReturnType, values: list[float]) -> pd.DataFrame:
    ts = [
        pd.Timestamp("2024-01-01T10:05:00Z"),
        pd.Timestamp("2024-01-01T10:10:00Z"),
        pd.Timestamp("2024-01-01T10:15:00Z"),
    ]
    rows = []
    for idx, value in enumerate(values):
        rows.append(
            {
                "instrument": "AAPL",
                "timestamp": ts[idx],
                "available_at": ts[idx],
                "availability_status": AvailabilityStatus.VERIFIED_TIMESTAMP.value,
                "previous_timestamp": pd.NaT if idx == 0 else ts[idx - 1],
                "previous_available_at": pd.NaT if idx == 0 else ts[idx - 1],
                "previous_availability_status": None if idx == 0 else AvailabilityStatus.VERIFIED_TIMESTAMP.value,
                "return_type": return_type.value,
                "return_value": value,
                "price_type": PriceType.CLOSE.value,
                "adjustment_state": AdjustmentState.RAW.value,
                "interval_elapsed_seconds": None if idx == 0 else 300.0,
            }
        )

    frame = pd.DataFrame(rows)
    frame.attrs["provenance"] = {
        "source_canonical_provenance": _source_provenance(),
        "return_policy_version": "1",
        "return_type": return_type.value,
        "generated_at_utc": pd.Timestamp("2024-01-06T12:00:00Z"),
    }
    return frame


def _feature_at_latest(features: pd.DataFrame, feature_name: str) -> float:
    row = features[features["feature_name"] == feature_name].sort_values("timestamp").iloc[-1]
    return float(row["feature_value"])


def test_core_simple_return_feature_math_matches_reference_values() -> None:
    frame = _returns_table(ReturnType.SIMPLE, [0.10, -0.05, 0.02])
    specs = [
        FeatureSpec(feature_name=FEATURE_ROLLING_MEAN, window_size=3, min_observations=3),
        FeatureSpec(
            feature_name=FEATURE_CUMULATIVE_RETURN,
            output_transformation_mode="simple_equivalent",
            window_size=3,
            min_observations=3,
            require_complete_window=True,
            allow_partial_valid=False,
        ),
        FeatureSpec(
            feature_name=FEATURE_LAGGED_MOMENTUM,
            output_transformation_mode="simple_equivalent",
            window_size=2,
            min_observations=2,
            include_current=False,
            require_complete_window=True,
            allow_partial_valid=False,
        ),
        FeatureSpec(feature_name=FEATURE_ROLLING_STD, window_size=3, min_observations=3, ddof=1),
        FeatureSpec(feature_name=FEATURE_DOWNSIDE_DEVIATION, window_size=3, min_observations=3),
        FeatureSpec(feature_name=FEATURE_REALIZED_VARIANCE, window_size=3, min_observations=3),
        FeatureSpec(
            feature_name=FEATURE_VOLATILITY_RATIO,
            window_size=3,
            short_window_size=2,
            long_window_size=3,
            min_observations=2,
            ddof=1,
        ),
        FeatureSpec(feature_name=FEATURE_DOWNSIDE_FREQUENCY, window_size=3, min_observations=3),
    ]

    features, _ = compute_features_from_returns(frame, specs)
    assert _feature_at_latest(features, FEATURE_ROLLING_MEAN) == pytest.approx(0.023333333333333334)
    assert _feature_at_latest(features, FEATURE_CUMULATIVE_RETURN) == pytest.approx(0.06589999999999985)
    assert _feature_at_latest(features, FEATURE_LAGGED_MOMENTUM) == pytest.approx(0.04499999999999993)
    assert _feature_at_latest(features, FEATURE_ROLLING_STD) == pytest.approx(0.07505553499465135)
    assert _feature_at_latest(features, FEATURE_DOWNSIDE_DEVIATION) == pytest.approx(0.02886751345948129)
    assert _feature_at_latest(features, FEATURE_REALIZED_VARIANCE) == pytest.approx(0.012900000000000002)
    assert _feature_at_latest(features, FEATURE_VOLATILITY_RATIO) == pytest.approx(0.6594780076723942)
    assert _feature_at_latest(features, FEATURE_DOWNSIDE_FREQUENCY) == pytest.approx(1.0 / 3.0)


def test_log_input_cumulative_simple_equivalent_matches_simple_cumulative() -> None:
    simple = [0.10, -0.05, 0.02]
    log_values = [math.log(1.0 + value) for value in simple]

    frame = _returns_table(ReturnType.LOG, log_values)
    spec = FeatureSpec(
        feature_name=FEATURE_CUMULATIVE_RETURN,
        input_return_type=ReturnType.LOG,
        output_transformation_mode="simple_equivalent",
        window_size=3,
        min_observations=3,
        require_complete_window=True,
        allow_partial_valid=False,
    )
    features, _ = compute_features_from_returns(frame, [spec])
    latest = _feature_at_latest(features, FEATURE_CUMULATIVE_RETURN)
    assert latest == pytest.approx(0.06590000000000007)

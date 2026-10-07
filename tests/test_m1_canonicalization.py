from __future__ import annotations

import pandas as pd
import pytest

from regimequant.data import (
    AVAILABILITY_POLICY_EXTERNAL_VERIFIED,
    AVAILABILITY_POLICY_FIXTURE_DECLARED,
    AVAILABILITY_POLICY_UNKNOWN_UNVERIFIED,
    AdjustmentState,
    AvailabilityStatus,
    MarketDataProvenance,
    PriceType,
    UniverseType,
    canonicalize_market_data,
)
from regimequant.data.errors import MarketDataDomainError, MarketDataValidationError


def _base_provenance(adjustment_state: AdjustmentState = AdjustmentState.RAW) -> MarketDataProvenance:
    return MarketDataProvenance(
        provider_id="fixture",
        requested_instruments=("AAPL", "MSFT"),
        requested_start_timestamp=pd.Timestamp("2024-01-01T00:00:00Z"),
        requested_end_timestamp=pd.Timestamp("2024-01-05T00:00:00Z"),
        requested_price_type=PriceType.CLOSE,
        requested_adjustment_state=adjustment_state,
        retrieval_timestamp_utc=pd.Timestamp("2024-01-05T12:00:00Z"),
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


def _provenance_with_declared_distribution(
    distribution: tuple[tuple[AvailabilityStatus, int], ...],
    adjustment_state: AdjustmentState = AdjustmentState.RAW,
) -> MarketDataProvenance:
    return MarketDataProvenance(
        provider_id="fixture",
        requested_instruments=("AAPL", "MSFT"),
        requested_start_timestamp=pd.Timestamp("2024-01-01T00:00:00Z"),
        requested_end_timestamp=pd.Timestamp("2024-01-05T00:00:00Z"),
        requested_price_type=PriceType.CLOSE,
        requested_adjustment_state=adjustment_state,
        retrieval_timestamp_utc=pd.Timestamp("2024-01-05T12:00:00Z"),
        retrieval_environment="fixture",
        price_type=PriceType.CLOSE,
        adjustment_state=adjustment_state,
        timestamp_timezone_policy="utc",
        availability_policy_id=AVAILABILITY_POLICY_FIXTURE_DECLARED,
        source_availability_status_distribution=(),
        availability_status_distribution=distribution,
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


def _canonical_input() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "instrument": "MSFT",
                "timestamp": pd.Timestamp("2024-01-03 16:00:00-05:00"),
                "available_at": pd.Timestamp("2024-01-03 16:00:00-05:00"),
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
                "instrument": "AAPL",
                "timestamp": pd.Timestamp("2024-01-03 16:00:00-05:00"),
                "available_at": pd.Timestamp("2024-01-03 16:00:00-05:00"),
                "availability_status": AvailabilityStatus.VERIFIED_TIMESTAMP.value,
                "price": 110.0,
                "price_type": PriceType.CLOSE.value,
                "adjustment_state": AdjustmentState.RAW.value,
                "currency": "USD",
                "source_observation_id": "a2-dup",
            },
        ]
    )


def test_canonicalization_valid_and_column_order() -> None:
    canonical, diagnostics = canonicalize_market_data(_canonical_input(), _base_provenance())
    assert list(canonical.columns) == [
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
    assert diagnostics == {
        "input_row_count": 4,
        "output_row_count": 3,
        "exact_duplicate_collapse_count": 1,
    }
    # Duplicate rows with identical market semantics retain the lexicographically smallest source id.
    assert canonical.iloc[1]["source_observation_id"] == "a2"
    assert canonical["instrument"].tolist() == ["AAPL", "AAPL", "MSFT"]
    assert canonical["timestamp"].dt.tz is not None
    provenance = canonical.attrs["provenance"]
    assert provenance.availability_status_distribution == (
        (AvailabilityStatus.VERIFIED_TIMESTAMP, 3),
        (AvailabilityStatus.UNKNOWN_UNVERIFIED, 0),
    )


@pytest.mark.parametrize(
    "price",
    [float("nan"), float("inf"), float("-inf"), 0.0, -1.0],
)
def test_canonical_price_domain_rejects_invalid_values(price: float) -> None:
    frame = _canonical_input().iloc[[0]].copy()
    frame.loc[:, "price"] = price
    with pytest.raises(MarketDataDomainError):
        canonicalize_market_data(frame, _base_provenance())


def test_missing_observation_is_absent_row_not_nan_price() -> None:
    frame = _canonical_input().iloc[[0]].copy()
    frame.loc[:, "price"] = 100.0
    canonical, _ = canonicalize_market_data(frame, _base_provenance())
    assert canonical.shape[0] == 1


def test_naive_timestamp_rejected() -> None:
    row = _canonical_input().iloc[0].to_dict()
    row["timestamp"] = pd.Timestamp("2024-01-03 16:00:00")
    frame = pd.DataFrame([row])
    with pytest.raises(MarketDataDomainError):
        canonicalize_market_data(frame, _base_provenance())


def test_unknown_availability_requires_null_available_at() -> None:
    frame = _canonical_input().iloc[[0]].copy()
    frame.loc[:, "availability_status"] = AvailabilityStatus.UNKNOWN_UNVERIFIED.value
    frame.loc[:, "available_at"] = None
    canonical, _ = canonicalize_market_data(frame, _base_provenance())
    assert canonical.iloc[0]["availability_status"] == AvailabilityStatus.UNKNOWN_UNVERIFIED.value
    assert pd.isna(canonical.iloc[0]["available_at"])


def test_conflicting_duplicate_rejected() -> None:
    frame = _canonical_input().iloc[[2, 3]].copy()
    frame.iloc[1, frame.columns.get_loc("price")] = 111.0
    with pytest.raises(MarketDataValidationError):
        canonicalize_market_data(frame, _base_provenance())


def test_duplicate_source_id_retention_is_deterministic_under_permutations() -> None:
    base_row = {
        "instrument": "AAPL",
        "timestamp": pd.Timestamp("2024-01-03 16:00:00-05:00"),
        "available_at": pd.Timestamp("2024-01-03 16:00:00-05:00"),
        "availability_status": AvailabilityStatus.VERIFIED_TIMESTAMP.value,
        "price": 110.0,
        "price_type": PriceType.CLOSE.value,
        "adjustment_state": AdjustmentState.RAW.value,
        "currency": "USD",
    }
    rows_a = [
        {**base_row, "source_observation_id": "z-record"},
        {**base_row, "source_observation_id": "a-record"},
    ]
    rows_b = list(reversed(rows_a))

    canonical_a, _ = canonicalize_market_data(pd.DataFrame(rows_a), _base_provenance())
    canonical_b, _ = canonicalize_market_data(pd.DataFrame(rows_b), _base_provenance())

    assert canonical_a.iloc[0]["source_observation_id"] == "a-record"
    assert canonical_b.iloc[0]["source_observation_id"] == "a-record"


@pytest.mark.parametrize(
    "bad_instrument",
    [123, None, "", "   "],
)
def test_instrument_must_be_nonempty_string(bad_instrument: object) -> None:
    row = _canonical_input().iloc[0].to_dict()
    row["instrument"] = bad_instrument
    frame = pd.DataFrame([row])
    with pytest.raises(MarketDataValidationError, match="instrument must be a non-empty string"):
        canonicalize_market_data(frame, _base_provenance())


def test_canonical_observations_reject_undefined_availability_status() -> None:
    frame = _canonical_input().iloc[[0]].copy()
    frame.loc[:, "availability_status"] = AvailabilityStatus.UNDEFINED.value
    with pytest.raises(MarketDataValidationError, match="reserved for derived return observations"):
        canonicalize_market_data(frame, _base_provenance())


def test_unknown_unverified_policy_cannot_claim_verified_observations() -> None:
    frame = _canonical_input().iloc[[0]].copy()
    provenance = MarketDataProvenance(
        provider_id="fixture",
        requested_instruments=("AAPL", "MSFT"),
        requested_start_timestamp=pd.Timestamp("2024-01-01T00:00:00Z"),
        requested_end_timestamp=pd.Timestamp("2024-01-05T00:00:00Z"),
        requested_price_type=PriceType.CLOSE,
        requested_adjustment_state=AdjustmentState.RAW,
        retrieval_timestamp_utc=pd.Timestamp("2024-01-05T12:00:00Z"),
        retrieval_environment="fixture",
        price_type=PriceType.CLOSE,
        adjustment_state=AdjustmentState.RAW,
        timestamp_timezone_policy="utc",
        availability_policy_id=AVAILABILITY_POLICY_UNKNOWN_UNVERIFIED,
        availability_status_distribution=((AvailabilityStatus.UNKNOWN_UNVERIFIED, 1),),
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
    with pytest.raises(
        MarketDataValidationError,
        match="verified availability observations require a fixture-declared or external-verified availability policy",
    ):
        canonicalize_market_data(frame, provenance)


def test_external_verified_policy_is_valid_for_verified_observations() -> None:
    frame = _canonical_input().iloc[[0]].copy()
    provenance = MarketDataProvenance(
        provider_id="fixture",
        requested_instruments=("AAPL", "MSFT"),
        requested_start_timestamp=pd.Timestamp("2024-01-01T00:00:00Z"),
        requested_end_timestamp=pd.Timestamp("2024-01-05T00:00:00Z"),
        requested_price_type=PriceType.CLOSE,
        requested_adjustment_state=AdjustmentState.RAW,
        retrieval_timestamp_utc=pd.Timestamp("2024-01-05T12:00:00Z"),
        retrieval_environment="fixture",
        price_type=PriceType.CLOSE,
        adjustment_state=AdjustmentState.RAW,
        timestamp_timezone_policy="utc",
        availability_policy_id=AVAILABILITY_POLICY_EXTERNAL_VERIFIED,
        availability_status_distribution=((AvailabilityStatus.VERIFIED_TIMESTAMP, 1),),
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
    canonical, _ = canonicalize_market_data(frame, provenance)
    assert canonical.iloc[0]["availability_status"] == AvailabilityStatus.VERIFIED_TIMESTAMP.value


def test_canonicalization_does_not_mutate_input() -> None:
    frame = _canonical_input().iloc[[0]].copy()
    original = frame.copy(deep=True)
    canonicalize_market_data(frame, _base_provenance())
    pd.testing.assert_frame_equal(frame, original)


def test_adjusted_history_provenance_requires_vintage_limitation() -> None:
    provenance = _base_provenance(AdjustmentState.ADJUSTED)
    frame = _canonical_input().iloc[[0]].copy()
    frame.loc[:, "adjustment_state"] = AdjustmentState.ADJUSTED.value
    canonical, _ = canonicalize_market_data(frame, provenance)
    assert canonical.iloc[0]["adjustment_state"] == AdjustmentState.ADJUSTED.value


def test_canonical_distribution_all_verified_matches_actual_rows() -> None:
    frame = _canonical_input().iloc[[0, 1]].copy()
    provenance = _provenance_with_declared_distribution(
        (
            (AvailabilityStatus.VERIFIED_TIMESTAMP, 2),
            (AvailabilityStatus.UNKNOWN_UNVERIFIED, 0),
        )
    )
    canonical, _ = canonicalize_market_data(frame, provenance)
    assert canonical.attrs["provenance"].availability_status_distribution == (
        (AvailabilityStatus.VERIFIED_TIMESTAMP, 2),
        (AvailabilityStatus.UNKNOWN_UNVERIFIED, 0),
    )


def test_canonical_distribution_all_unknown_matches_actual_rows() -> None:
    frame = _canonical_input().iloc[[0, 1]].copy()
    frame.loc[:, "availability_status"] = AvailabilityStatus.UNKNOWN_UNVERIFIED.value
    frame.loc[:, "available_at"] = None
    provenance = _base_provenance()
    canonical, _ = canonicalize_market_data(frame, provenance)
    assert canonical.attrs["provenance"].availability_status_distribution == (
        (AvailabilityStatus.VERIFIED_TIMESTAMP, 0),
        (AvailabilityStatus.UNKNOWN_UNVERIFIED, 2),
    )


def test_canonical_distribution_mixed_verified_and_unknown_matches_actual_rows() -> None:
    frame = _canonical_input().iloc[[0, 1, 2]].copy()
    frame.loc[frame.index[0], "availability_status"] = AvailabilityStatus.UNKNOWN_UNVERIFIED.value
    frame.loc[frame.index[0], "available_at"] = None
    provenance = _base_provenance()
    canonical, _ = canonicalize_market_data(frame, provenance)
    assert canonical.attrs["provenance"].availability_status_distribution == (
        (AvailabilityStatus.VERIFIED_TIMESTAMP, 2),
        (AvailabilityStatus.UNKNOWN_UNVERIFIED, 1),
    )


def test_declared_verified_only_distribution_conflicts_with_unknown_row() -> None:
    frame = _canonical_input().iloc[[0]].copy()
    frame.loc[:, "availability_status"] = AvailabilityStatus.UNKNOWN_UNVERIFIED.value
    frame.loc[:, "available_at"] = None
    provenance = _provenance_with_declared_distribution(
        (
            (AvailabilityStatus.VERIFIED_TIMESTAMP, 1),
            (AvailabilityStatus.UNKNOWN_UNVERIFIED, 0),
        )
    )
    with pytest.raises(MarketDataValidationError, match="declared canonical availability_status_distribution"):
        canonicalize_market_data(frame, provenance)


def test_duplicate_collapse_distribution_reflects_canonical_rows() -> None:
    frame = _canonical_input().iloc[[2, 3]].copy()
    provenance = _base_provenance()
    canonical, diagnostics = canonicalize_market_data(frame, provenance)
    assert diagnostics["exact_duplicate_collapse_count"] == 1
    assert canonical.shape[0] == 1
    assert canonical.attrs["provenance"].availability_status_distribution == (
        (AvailabilityStatus.VERIFIED_TIMESTAMP, 1),
        (AvailabilityStatus.UNKNOWN_UNVERIFIED, 0),
    )


def test_empty_canonical_dataset_emits_zero_count_distribution() -> None:
    frame = pd.DataFrame(columns=[
        "instrument",
        "timestamp",
        "available_at",
        "availability_status",
        "price",
        "price_type",
        "adjustment_state",
        "currency",
        "source_observation_id",
    ])
    canonical, diagnostics = canonicalize_market_data(frame, _base_provenance())
    assert diagnostics["output_row_count"] == 0
    assert canonical.attrs["provenance"].availability_status_distribution == (
        (AvailabilityStatus.VERIFIED_TIMESTAMP, 0),
        (AvailabilityStatus.UNKNOWN_UNVERIFIED, 0),
    )


def test_distribution_is_deterministic_under_row_permutations() -> None:
    frame_a = _canonical_input().iloc[[0, 1, 2]].copy()
    frame_a.loc[frame_a.index[0], "availability_status"] = AvailabilityStatus.UNKNOWN_UNVERIFIED.value
    frame_a.loc[frame_a.index[0], "available_at"] = None
    frame_b = frame_a.sample(frac=1.0, random_state=11).reset_index(drop=True)
    canonical_a, _ = canonicalize_market_data(frame_a, _base_provenance())
    canonical_b, _ = canonicalize_market_data(frame_b, _base_provenance())
    assert canonical_a.attrs["provenance"].availability_status_distribution == canonical_b.attrs["provenance"].availability_status_distribution
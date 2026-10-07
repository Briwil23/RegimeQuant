"""Canonical market-data normalization for RegimeQuant M1."""

from __future__ import annotations

from dataclasses import replace
from math import isfinite

import pandas as pd

from .errors import MarketDataDomainError, MarketDataValidationError
from .models import (
    AVAILABILITY_POLICY_EXTERNAL_VERIFIED,
    AVAILABILITY_POLICY_FIXTURE_DECLARED,
    AdjustmentState,
    AvailabilityStatus,
    MarketDataProvenance,
    PriceType,
    UniverseType,
)


CANONICAL_COLUMNS = [
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


def _coerce_utc_timestamp(value: object, field_name: str, *, allow_none: bool = False) -> pd.Timestamp | None:
    if value is None or value is pd.NaT or pd.isna(value):
        if allow_none:
            return None
        raise MarketDataDomainError(f"{field_name} is required")

    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None or timestamp.tz is None:
        raise MarketDataDomainError(f"{field_name} must be timezone-aware")
    return timestamp.tz_convert("UTC")


def _coerce_enum(enum_cls, value: object, field_name: str):
    try:
        return enum_cls(value)
    except Exception as exc:  # pragma: no cover - defensive
        raise MarketDataValidationError(f"invalid {field_name}: {value!r}") from exc


def _normalize_instrument(value: object) -> str:
    if not isinstance(value, str):
        raise MarketDataValidationError("instrument must be a non-empty string")
    normalized = value.strip()
    if not normalized:
        raise MarketDataValidationError("instrument must be a non-empty string")
    return normalized


def _normalize_source_observation_id(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    normalized = str(value).strip()
    return normalized or None


def _retained_source_observation_id(values: pd.Series) -> str | None:
    normalized_ids = sorted(
        {
            normalized
            for normalized in (_normalize_source_observation_id(value) for value in values)
            if normalized is not None
        }
    )
    if not normalized_ids:
        return None
    return normalized_ids[0]


def _canonical_availability_distribution(frame: pd.DataFrame) -> tuple[tuple[AvailabilityStatus, int], ...]:
    verified_count = int((frame["availability_status"] == AvailabilityStatus.VERIFIED_TIMESTAMP.value).sum())
    unknown_count = int((frame["availability_status"] == AvailabilityStatus.UNKNOWN_UNVERIFIED.value).sum())
    return (
        (AvailabilityStatus.VERIFIED_TIMESTAMP, verified_count),
        (AvailabilityStatus.UNKNOWN_UNVERIFIED, unknown_count),
    )


def _expanded_canonical_distribution(
    distribution: tuple[tuple[AvailabilityStatus, int], ...],
) -> tuple[tuple[AvailabilityStatus, int], ...]:
    counts = {status: count for status, count in distribution}
    return (
        (AvailabilityStatus.VERIFIED_TIMESTAMP, int(counts.get(AvailabilityStatus.VERIFIED_TIMESTAMP, 0))),
        (AvailabilityStatus.UNKNOWN_UNVERIFIED, int(counts.get(AvailabilityStatus.UNKNOWN_UNVERIFIED, 0))),
    )


def canonicalize_market_data(
    raw_table: pd.DataFrame,
    provenance: MarketDataProvenance,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if provenance.universe_type != UniverseType.STATIC:
        raise MarketDataValidationError("only static universes are supported in M1")

    data = raw_table.copy(deep=True)
    required_columns = set(CANONICAL_COLUMNS)
    missing = required_columns.difference(data.columns)
    if missing:
        raise MarketDataValidationError(f"missing required columns: {sorted(missing)!r}")

    extras = set(data.columns).difference(required_columns)
    if extras:
        raise MarketDataValidationError(f"unexpected columns in canonicalization input: {sorted(extras)!r}")

    if data.empty:
        canonical = data.loc[:, CANONICAL_COLUMNS].copy().iloc[0:0].copy()
        diagnostics = {"input_row_count": 0, "output_row_count": 0, "exact_duplicate_collapse_count": 0}
        canonical_distribution = _canonical_availability_distribution(canonical)
        if (
            provenance.availability_status_distribution
            and _expanded_canonical_distribution(provenance.availability_status_distribution) != canonical_distribution
        ):
            raise MarketDataValidationError(
                "declared canonical availability_status_distribution does not match canonical observations"
            )
        canonical.attrs["provenance"] = replace(
            provenance,
            availability_status_distribution=canonical_distribution,
        )
        canonical.attrs["diagnostics"] = diagnostics
        return canonical, diagnostics

    data["instrument"] = [_normalize_instrument(value) for value in data["instrument"]]

    raw_instruments = tuple(sorted(set(data["instrument"].tolist())))
    if provenance.universe_instruments and not set(raw_instruments).issubset(set(provenance.universe_instruments)):
        raise MarketDataValidationError("canonical data instruments must be contained within the declared static universe")

    normalized_rows = []
    for row_index, row in data.reset_index(drop=True).iterrows():
        instrument = _normalize_instrument(row["instrument"])

        timestamp = _coerce_utc_timestamp(row["timestamp"], "timestamp")
        availability_status = _coerce_enum(AvailabilityStatus, row["availability_status"], "availability_status")
        if availability_status == AvailabilityStatus.VERIFIED_TIMESTAMP:
            if provenance.availability_policy_id not in {
                AVAILABILITY_POLICY_FIXTURE_DECLARED,
                AVAILABILITY_POLICY_EXTERNAL_VERIFIED,
            }:
                raise MarketDataValidationError(
                    "verified availability observations require a fixture-declared or external-verified availability policy"
                )
            available_at = _coerce_utc_timestamp(row["available_at"], "available_at")
        elif availability_status == AvailabilityStatus.UNDEFINED:
            raise MarketDataValidationError("undefined availability status is reserved for derived return observations")
        else:
            if row["available_at"] is not None and not pd.isna(row["available_at"]):
                raise MarketDataDomainError("unknown availability observations must not carry available_at")
            available_at = None

        price_type = _coerce_enum(PriceType, row["price_type"], "price_type")
        adjustment_state = _coerce_enum(AdjustmentState, row["adjustment_state"], "adjustment_state")

        if price_type != provenance.price_type:
            raise MarketDataValidationError("price_type must match provenance")
        if adjustment_state != provenance.adjustment_state:
            raise MarketDataValidationError("adjustment_state must match provenance")

        price = row["price"]
        if price is None or pd.isna(price):
            raise MarketDataDomainError("canonical price must not be NaN")
        try:
            price_value = float(price)
        except Exception as exc:  # pragma: no cover - defensive
            raise MarketDataDomainError(f"invalid price value: {price!r}") from exc
        if not isfinite(price_value) or price_value <= 0.0:
            raise MarketDataDomainError("canonical price must be finite and strictly greater than zero")

        currency = None if row["currency"] is None or pd.isna(row["currency"]) else str(row["currency"])
        source_observation_id = _normalize_source_observation_id(row["source_observation_id"])

        normalized_rows.append(
            {
                "instrument": instrument,
                "timestamp": timestamp,
                "available_at": available_at,
                "availability_status": availability_status.value,
                "price": price_value,
                "price_type": price_type.value,
                "adjustment_state": adjustment_state.value,
                "currency": currency,
                "source_observation_id": source_observation_id,
                "_row_index": row_index,
            }
        )

    normalized = pd.DataFrame(normalized_rows)
    normalized = normalized.sort_values(
        ["instrument", "timestamp", "price_type", "adjustment_state", "source_observation_id", "_row_index"],
        kind="mergesort",
    ).reset_index(drop=True)

    deduped_rows: list[dict[str, object]] = []
    exact_duplicate_collapse_count = 0
    for _, group in normalized.groupby(["instrument", "timestamp", "price_type", "adjustment_state"], sort=False, dropna=False):
        if len(group) == 1:
            deduped_rows.append(group.iloc[0].drop(labels=["_row_index"]).to_dict())
            continue

        semantic_columns = [
            "instrument",
            "timestamp",
            "available_at",
            "availability_status",
            "price",
            "price_type",
            "adjustment_state",
            "currency",
        ]
        semantic_frame = group.loc[:, semantic_columns].copy()
        if semantic_frame.drop_duplicates().shape[0] != 1:
            raise MarketDataValidationError("conflicting duplicate observations share the same canonical key")

        retained_source_observation_id = _retained_source_observation_id(group["source_observation_id"])
        exact_duplicate_collapse_count += len(group) - 1
        retained_row = group.iloc[0].drop(labels=["_row_index"]).to_dict()
        retained_row["source_observation_id"] = retained_source_observation_id
        deduped_rows.append(retained_row)

    canonical = pd.DataFrame(deduped_rows).sort_values(
        ["instrument", "timestamp", "price_type", "adjustment_state"],
        kind="mergesort",
    ).reset_index(drop=True)
    canonical = canonical.loc[:, CANONICAL_COLUMNS]
    canonical_distribution = _canonical_availability_distribution(canonical)
    if (
        provenance.availability_status_distribution
        and _expanded_canonical_distribution(provenance.availability_status_distribution) != canonical_distribution
    ):
        raise MarketDataValidationError(
            "declared canonical availability_status_distribution does not match canonical observations"
        )

    diagnostics = {
        "input_row_count": int(len(data)),
        "output_row_count": int(len(canonical)),
        "exact_duplicate_collapse_count": int(exact_duplicate_collapse_count),
    }
    canonical.attrs["provenance"] = replace(
        provenance,
        availability_status_distribution=canonical_distribution,
    )
    canonical.attrs["diagnostics"] = diagnostics
    return canonical, diagnostics
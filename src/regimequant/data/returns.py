"""Return construction for RegimeQuant M1."""

from __future__ import annotations

from datetime import datetime, timezone
from math import isfinite, log

import pandas as pd

from .errors import MarketDataDomainError, MarketDataValidationError
from .models import AvailabilityStatus, MarketDataProvenance, ReturnType


RETURN_COLUMNS = [
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

_REQUIRED_INPUT_COLUMNS = {
    "instrument",
    "timestamp",
    "available_at",
    "availability_status",
    "price",
    "price_type",
    "adjustment_state",
    "currency",
    "source_observation_id",
}


def _coerce_provenance(frame: pd.DataFrame, provenance: MarketDataProvenance | None) -> MarketDataProvenance:
    if provenance is not None:
        return provenance
    inferred = frame.attrs.get("provenance")
    if isinstance(inferred, MarketDataProvenance):
        return inferred
    raise MarketDataValidationError("canonical price table provenance is required")


def _validate_canonical_input(frame: pd.DataFrame) -> None:
    missing = _REQUIRED_INPUT_COLUMNS.difference(frame.columns)
    if missing:
        raise MarketDataValidationError(f"missing required canonical columns: {sorted(missing)!r}")
    extras = set(frame.columns).difference(_REQUIRED_INPUT_COLUMNS)
    if extras:
        raise MarketDataValidationError(f"unexpected columns in canonical price input: {sorted(extras)!r}")

    for price in frame["price"]:
        if price is None or pd.isna(price):
            raise MarketDataDomainError("canonical price must not be NaN")
        try:
            price_value = float(price)
        except Exception as exc:  # pragma: no cover - defensive
            raise MarketDataDomainError(f"invalid price value: {price!r}") from exc
        if not isfinite(price_value) or price_value <= 0.0:
            raise MarketDataDomainError("canonical price must be finite and strictly greater than zero")


def _compute_returns(
    canonical_prices: pd.DataFrame,
    *,
    return_type: ReturnType,
    provenance: MarketDataProvenance | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    data = canonical_prices.copy(deep=True)
    provenance = _coerce_provenance(data, provenance)
    _validate_canonical_input(data)

    price_types = tuple(sorted({str(value) for value in data["price_type"].astype(str).tolist()}))
    adjustment_states = tuple(sorted({str(value) for value in data["adjustment_state"].astype(str).tolist()}))
    if len(price_types) != 1 or len(adjustment_states) != 1:
        raise MarketDataValidationError("return construction requires one price_type and one adjustment_state")
    if price_types[0] != provenance.price_type.value:
        raise MarketDataValidationError("canonical price type does not match provenance")
    if adjustment_states[0] != provenance.adjustment_state.value:
        raise MarketDataValidationError("canonical adjustment state does not match provenance")

    ordered = data.sort_values(["instrument", "timestamp"], kind="mergesort").reset_index(drop=True)
    grouped = ordered.groupby("instrument", sort=False, dropna=False)

    previous_timestamp = grouped["timestamp"].shift(1)
    previous_available_at = grouped["available_at"].shift(1)
    previous_availability_status = grouped["availability_status"].shift(1)
    previous_price = grouped["price"].shift(1)

    return_values: list[float] = []
    return_available_at: list[pd.Timestamp | None] = []
    return_availability_status: list[str] = []
    interval_elapsed_seconds: list[float | None] = []

    for index, row in ordered.iterrows():
        previous_price_value = previous_price.iloc[index]
        current_price_value = float(row["price"])
        current_status = AvailabilityStatus(row["availability_status"])
        previous_status_value = previous_availability_status.iloc[index]
        previous_status = None if pd.isna(previous_status_value) else AvailabilityStatus(previous_status_value)
        current_available_at = None if pd.isna(row["available_at"]) else pd.Timestamp(row["available_at"]).tz_convert("UTC")
        previous_available_at_value = previous_available_at.iloc[index]
        previous_available = None if pd.isna(previous_available_at_value) else pd.Timestamp(previous_available_at_value).tz_convert("UTC")

        if pd.isna(previous_price_value):
            return_values.append(float("nan"))
            return_available_at.append(None)
            return_availability_status.append(AvailabilityStatus.UNDEFINED.value)
            interval_elapsed_seconds.append(None)
            continue

        if current_status == AvailabilityStatus.VERIFIED_TIMESTAMP and current_available_at is None:
            raise MarketDataDomainError("verified observations must carry available_at")
        if previous_status == AvailabilityStatus.VERIFIED_TIMESTAMP and previous_available is None:
            raise MarketDataDomainError("verified observations must carry previous available_at")

        if current_status == AvailabilityStatus.VERIFIED_TIMESTAMP and previous_status == AvailabilityStatus.VERIFIED_TIMESTAMP:
            derived_available_at = max(current_available_at, previous_available)
            derived_status = AvailabilityStatus.VERIFIED_TIMESTAMP.value
        else:
            derived_available_at = None
            derived_status = AvailabilityStatus.UNKNOWN_UNVERIFIED.value

        if return_type == ReturnType.SIMPLE:
            return_value = current_price_value / float(previous_price_value) - 1.0
        elif return_type == ReturnType.LOG:
            return_value = log(current_price_value / float(previous_price_value))
        else:  # pragma: no cover - defensive
            raise MarketDataValidationError(f"unsupported return type: {return_type!r}")

        return_values.append(return_value)
        return_available_at.append(derived_available_at)
        return_availability_status.append(derived_status)
        interval_elapsed_seconds.append((row["timestamp"] - previous_timestamp.iloc[index]).total_seconds())

    returns = pd.DataFrame(
        {
            "instrument": ordered["instrument"].astype(str),
            "timestamp": ordered["timestamp"],
            "available_at": return_available_at,
            "availability_status": return_availability_status,
            "previous_timestamp": previous_timestamp,
            "previous_available_at": previous_available_at,
            "previous_availability_status": previous_availability_status,
            "return_type": return_type.value,
            "return_value": return_values,
            "price_type": ordered["price_type"],
            "adjustment_state": ordered["adjustment_state"],
            "interval_elapsed_seconds": interval_elapsed_seconds,
        }
    )

    returns = returns.loc[:, RETURN_COLUMNS]
    diagnostics = {"input_row_count": int(len(data)), "output_row_count": int(len(returns))}
    returns.attrs["provenance"] = {
        "source_canonical_provenance": provenance,
        "return_policy_version": "1",
        "return_type": return_type.value,
        "generated_at_utc": pd.Timestamp(datetime.now(timezone.utc)),
    }
    returns.attrs["diagnostics"] = diagnostics
    return returns, diagnostics


def compute_simple_returns(
    canonical_prices: pd.DataFrame,
    provenance: MarketDataProvenance | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    return _compute_returns(canonical_prices, return_type=ReturnType.SIMPLE, provenance=provenance)


def compute_log_returns(
    canonical_prices: pd.DataFrame,
    provenance: MarketDataProvenance | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    return _compute_returns(canonical_prices, return_type=ReturnType.LOG, provenance=provenance)
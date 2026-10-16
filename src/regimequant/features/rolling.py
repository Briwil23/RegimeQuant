"""Instrument-level rolling feature computations for RegimeQuant M2."""

from __future__ import annotations

import math
import statistics
from typing import Any

import pandas as pd

from regimequant.data.models import AvailabilityStatus, ReturnType

from .definitions import (
    FEATURE_CUMULATIVE_RETURN,
    FEATURE_DOWNSIDE_DEVIATION,
    FEATURE_DOWNSIDE_FREQUENCY,
    FEATURE_LAGGED_MOMENTUM,
    FEATURE_REALIZED_VARIANCE,
    FEATURE_ROLLING_MEAN,
    FEATURE_ROLLING_STD,
    FEATURE_VOLATILITY_RATIO,
    FeatureSpec,
    feature_instance_id,
)
from .models import MathStatus, ScopeType, UndefinedReason


def _window_indices(end_idx: int, window_size: int, include_current: bool) -> list[int]:
    if include_current:
        start = max(0, end_idx - window_size + 1)
        return list(range(start, end_idx + 1))
    start = max(0, end_idx - window_size)
    return list(range(start, end_idx))


def _is_valid_numeric_return(row: pd.Series) -> bool:
    status = str(row["availability_status"])
    if status == AvailabilityStatus.UNDEFINED.value:
        return False
    value = row["return_value"]
    return value is not None and not pd.isna(value) and math.isfinite(float(value))


def _derive_input_availability_status(
    required_rows: pd.DataFrame,
    fallback_rows: pd.DataFrame,
) -> str:
    rows = required_rows if not required_rows.empty else fallback_rows
    statuses = rows["availability_status"].astype(str).tolist()
    if any(status == AvailabilityStatus.UNKNOWN_UNVERIFIED.value for status in statuses):
        return AvailabilityStatus.UNKNOWN_UNVERIFIED.value
    return AvailabilityStatus.VERIFIED_TIMESTAMP.value


def _defined_available_at(required_rows: pd.DataFrame, input_availability_status: str) -> pd.Timestamp | None:
    if input_availability_status != AvailabilityStatus.VERIFIED_TIMESTAMP.value:
        return None
    if required_rows.empty:
        return None
    timestamps = [
        pd.Timestamp(value).tz_convert("UTC")
        for value in required_rows["available_at"].tolist()
        if value is not None and not pd.isna(value)
    ]
    if not timestamps:
        return None
    return max(timestamps)


def _finalize_record(
    *,
    spec: FeatureSpec,
    instrument: str,
    timestamp: pd.Timestamp,
    feature_value: float,
    math_status: MathStatus,
    undefined_reason: UndefinedReason | None,
    input_availability_status: str,
    available_at: pd.Timestamp | None,
    observation_count_used: int,
    price_type: str,
    adjustment_state: str,
) -> dict[str, Any]:
    return {
        "scope_type": ScopeType.INSTRUMENT.value,
        "scope_id": instrument,
        "timestamp": timestamp,
        "feature_name": spec.feature_name,
        "feature_instance_id": feature_instance_id(spec),
        "feature_value": feature_value,
        "math_status": math_status.value,
        "undefined_reason": None if undefined_reason is None else undefined_reason.value,
        "input_availability_status": input_availability_status,
        "available_at": available_at,
        "input_return_type": spec.input_return_type.value,
        "price_type": price_type,
        "adjustment_state": adjustment_state,
        "window_kind": spec.window_kind,
        "window_size": spec.window_size,
        "min_observations": spec.min_observations,
        "observation_count_used": observation_count_used,
        "feature_definition_version": spec.feature_definition_version,
    }


def _simple_equivalent_compound(values: list[float], return_type: ReturnType) -> float:
    if return_type == ReturnType.SIMPLE:
        return math.prod(1.0 + value for value in values) - 1.0
    return math.exp(sum(values)) - 1.0


def _compute_volatility_ratio(
    *,
    frame: pd.DataFrame,
    end_idx: int,
    spec: FeatureSpec,
) -> tuple[float, MathStatus, UndefinedReason | None, pd.DataFrame, int]:
    short_indices = _window_indices(end_idx, int(spec.short_window_size), True)
    long_indices = _window_indices(end_idx, int(spec.long_window_size), True)

    short_slots = frame.iloc[short_indices].copy()
    long_slots = frame.iloc[long_indices].copy()

    short_valid_mask = short_slots.apply(_is_valid_numeric_return, axis=1)
    long_valid_mask = long_slots.apply(_is_valid_numeric_return, axis=1)
    short_contrib = short_slots.loc[short_valid_mask].copy()
    long_contrib = long_slots.loc[long_valid_mask].copy()

    short_n = len(short_contrib)
    long_n = len(long_contrib)

    required_rows = pd.concat([short_contrib, long_contrib], ignore_index=True)

    if short_n < spec.min_observations or long_n < spec.min_observations:
        return (
            float("nan"),
            MathStatus.UNDEFINED,
            UndefinedReason.INSUFFICIENT_OBSERVATIONS,
            required_rows,
            min(short_n, long_n),
        )

    if spec.ddof is None:
        ddof = 1
    else:
        ddof = spec.ddof

    if short_n <= ddof or long_n <= ddof:
        return (
            float("nan"),
            MathStatus.UNDEFINED,
            UndefinedReason.INSUFFICIENT_OBSERVATIONS,
            required_rows,
            min(short_n, long_n),
        )

    short_values = [float(value) for value in short_contrib["return_value"].tolist()]
    long_values = [float(value) for value in long_contrib["return_value"].tolist()]

    short_std = statistics.stdev(short_values)
    long_std = statistics.stdev(long_values)

    if long_std == 0.0:
        return (
            float("nan"),
            MathStatus.UNDEFINED,
            UndefinedReason.ZERO_DENOMINATOR,
            required_rows,
            min(short_n, long_n),
        )

    return (
        short_std / long_std,
        MathStatus.DEFINED,
        None,
        required_rows,
        min(short_n, long_n),
    )


def compute_feature_rows_for_instrument(
    frame: pd.DataFrame,
    spec: FeatureSpec,
    *,
    price_type: str,
    adjustment_state: str,
) -> list[dict[str, Any]]:
    """Compute a feature's rows for one instrument time series."""

    instrument = str(frame.iloc[0]["instrument"]) if not frame.empty else ""
    rows: list[dict[str, Any]] = []

    for end_idx in range(len(frame)):
        row = frame.iloc[end_idx]
        timestamp = pd.Timestamp(row["timestamp"]).tz_convert("UTC")

        indices = _window_indices(end_idx, spec.window_size, spec.include_current)
        slot_frame = frame.iloc[indices].copy()
        valid_mask = slot_frame.apply(_is_valid_numeric_return, axis=1)
        contributors = slot_frame.loc[valid_mask].copy()
        n_contrib = len(contributors)

        if spec.feature_name == FEATURE_VOLATILITY_RATIO:
            (
                feature_value,
                math_status,
                undefined_reason,
                required_rows,
                observation_count_used,
            ) = _compute_volatility_ratio(frame=frame, end_idx=end_idx, spec=spec)
            input_availability_status = _derive_input_availability_status(required_rows, slot_frame)
            available_at = (
                _defined_available_at(required_rows, input_availability_status)
                if math_status == MathStatus.DEFINED
                else None
            )
            rows.append(
                _finalize_record(
                    spec=spec,
                    instrument=instrument,
                    timestamp=timestamp,
                    feature_value=feature_value,
                    math_status=math_status,
                    undefined_reason=undefined_reason,
                    input_availability_status=input_availability_status,
                    available_at=available_at,
                    observation_count_used=observation_count_used,
                    price_type=price_type,
                    adjustment_state=adjustment_state,
                )
            )
            continue

        math_status = MathStatus.DEFINED
        undefined_reason: UndefinedReason | None = None
        feature_value = float("nan")

        if spec.require_complete_window:
            if len(slot_frame) < spec.window_size or n_contrib != len(slot_frame):
                math_status = MathStatus.UNDEFINED
                undefined_reason = (
                    UndefinedReason.INSUFFICIENT_OBSERVATIONS
                    if len(slot_frame) < spec.window_size
                    else UndefinedReason.INPUT_UNDEFINED_RETURN
                )
            elif n_contrib < spec.min_observations:
                math_status = MathStatus.UNDEFINED
                undefined_reason = UndefinedReason.INSUFFICIENT_OBSERVATIONS
        else:
            if n_contrib < spec.min_observations:
                math_status = MathStatus.UNDEFINED
                undefined_reason = UndefinedReason.INSUFFICIENT_OBSERVATIONS

        if math_status == MathStatus.DEFINED:
            values = [float(value) for value in contributors["return_value"].tolist()]
            if spec.feature_name == FEATURE_ROLLING_MEAN:
                feature_value = sum(values) / len(values)
            elif spec.feature_name == FEATURE_CUMULATIVE_RETURN:
                feature_value = _simple_equivalent_compound(values, spec.input_return_type)
            elif spec.feature_name == FEATURE_LAGGED_MOMENTUM:
                feature_value = _simple_equivalent_compound(values, spec.input_return_type)
            elif spec.feature_name == FEATURE_ROLLING_STD:
                ddof = 1 if spec.ddof is None else spec.ddof
                if len(values) <= ddof:
                    math_status = MathStatus.UNDEFINED
                    undefined_reason = UndefinedReason.INSUFFICIENT_OBSERVATIONS
                else:
                    feature_value = statistics.stdev(values)
            elif spec.feature_name == FEATURE_DOWNSIDE_DEVIATION:
                threshold = spec.downside_threshold
                downside_sq = [min(value - threshold, 0.0) ** 2 for value in values]
                feature_value = math.sqrt(sum(downside_sq) / len(values))
            elif spec.feature_name == FEATURE_REALIZED_VARIANCE:
                feature_value = sum(value * value for value in values)
            elif spec.feature_name == FEATURE_DOWNSIDE_FREQUENCY:
                threshold = spec.downside_threshold
                downside_count = sum(1 for value in values if value < threshold)
                feature_value = downside_count / len(values)
            else:
                raise ValueError(f"unsupported feature_name: {spec.feature_name!r}")

            if math_status == MathStatus.DEFINED and not math.isfinite(feature_value):
                math_status = MathStatus.UNDEFINED
                undefined_reason = UndefinedReason.NONFINITE_INPUT
                feature_value = float("nan")

        required_rows = slot_frame if spec.require_complete_window else contributors
        input_availability_status = _derive_input_availability_status(required_rows, slot_frame)
        available_at = (
            _defined_available_at(required_rows, input_availability_status)
            if math_status == MathStatus.DEFINED
            else None
        )

        rows.append(
            _finalize_record(
                spec=spec,
                instrument=instrument,
                timestamp=timestamp,
                feature_value=feature_value,
                math_status=math_status,
                undefined_reason=undefined_reason,
                input_availability_status=input_availability_status,
                available_at=available_at,
                observation_count_used=n_contrib,
                price_type=price_type,
                adjustment_state=adjustment_state,
            )
        )

    return rows

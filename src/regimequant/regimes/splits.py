from __future__ import annotations

from datetime import timedelta

import pandas as pd

from .contracts import M3ValidationError


def _validate_interval_pair(name: str, interval: tuple[pd.Timestamp, pd.Timestamp] | None) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    if interval is None:
        return None
    start, end = interval
    if pd.Timestamp(start) >= pd.Timestamp(end):
        raise M3ValidationError(f"{name} interval must be strictly ordered")
    return (pd.Timestamp(start), pd.Timestamp(end))


def build_m3_split_plan(
    observation_frame: pd.DataFrame,
    *,
    interval_start: pd.Timestamp | None = None,
    interval_end: pd.Timestamp | None = None,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
) -> tuple[tuple[pd.Timestamp, pd.Timestamp], tuple[pd.Timestamp, pd.Timestamp] | None, tuple[pd.Timestamp, pd.Timestamp] | None]:
    """Build deterministic chronological train/validation/test intervals.

    Intervals are half-open and strictly non-overlapping:
    [train_start, train_end), [train_end, validation_end), [validation_end, end).
    """

    if observation_frame.empty:
        raise M3ValidationError("empty observation frame cannot define a training split")

    timestamps = pd.to_datetime(observation_frame["timestamp"], errors="coerce")
    valid = timestamps.dropna()
    if valid.empty:
        raise M3ValidationError("observation timestamps are required for split planning")

    start = pd.Timestamp(interval_start) if interval_start is not None else valid.min()
    end = pd.Timestamp(interval_end) if interval_end is not None else valid.max()
    if start >= end:
        raise M3ValidationError("training interval must be strictly increasing")

    total_seconds = (end - start).total_seconds()
    if total_seconds <= 0:
        raise M3ValidationError("split plan requires a positive interval length")

    total = train_fraction + validation_fraction + test_fraction
    if abs(total - 1.0) > 1e-9:
        raise M3ValidationError("train/validation/test fractions must sum to 1.0")

    train_end = start + timedelta(seconds=total_seconds * train_fraction)
    validation_end = train_end + timedelta(seconds=total_seconds * validation_fraction)
    train_interval = (start, train_end)
    validation_interval = (train_end, validation_end)
    test_interval = (validation_end, end)

    _validate_interval_pair("train", train_interval)
    _validate_interval_pair("validation", validation_interval)
    _validate_interval_pair("test", test_interval)
    if train_interval[1] > validation_interval[0]:
        raise M3ValidationError("train and validation intervals overlap")
    if validation_interval[1] > test_interval[0]:
        raise M3ValidationError("validation and test intervals overlap")
    if validation_interval[0] != train_interval[1]:
        raise M3ValidationError("train/validation boundaries must be exactly equal to avoid ambiguous inclusion")
    if test_interval[0] != validation_interval[1]:
        raise M3ValidationError("validation/test boundaries must be exactly equal to avoid ambiguous inclusion")
    return train_interval, validation_interval, test_interval

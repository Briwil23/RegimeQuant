from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from regimequant.cointegration.integration_screening import _canonical_pair_order
from regimequant.cointegration.price_policy import DailyPricePolicy, PricePolicyError
from regimequant.signals.m8_numerics import M8NumericsError, NumericalDiagnostics, validate_numerics


class M8FormationError(ValueError):
    """Raised when formation data cannot produce a valid deterministic model."""


@dataclass(frozen=True, slots=True)
class FormationFit:
    alpha: float
    beta: float
    spread_mean: float
    spread_std: float
    diagnostics: NumericalDiagnostics
    aligned_index: pd.DatetimeIndex
    x_series: pd.Series
    y_series: pd.Series


def _as_utc(ts: object) -> pd.Timestamp:
    value = pd.Timestamp(ts)
    if value.tz is None:
        return value.tz_localize("UTC")
    return value.tz_convert("UTC")


def _validate_no_duplicates(series: pd.Series) -> None:
    if series.index.has_duplicates:
        raise M8FormationError("duplicate/conflicting observations are not permitted")


def fit_formation_model(
    x: pd.Series,
    y: pd.Series,
    *,
    formation_start: object,
    formation_end: object,
    eval_observation_time: object,
) -> FormationFit:
    formation_start_utc = _as_utc(formation_start)
    formation_end_utc = _as_utc(formation_end)
    eval_time_utc = _as_utc(eval_observation_time)

    if formation_start_utc > formation_end_utc:
        raise M8FormationError("formation_start must be <= formation_end")
    if formation_end_utc >= eval_time_utc:
        raise M8FormationError("formation observations must strictly precede evaluation session")

    ordered_x, ordered_y = _canonical_pair_order(x, y)
    _validate_no_duplicates(ordered_x)
    _validate_no_duplicates(ordered_y)

    try:
        validated_x = DailyPricePolicy.validate_series(ordered_x, ticker="x").price_series
        validated_y = DailyPricePolicy.validate_series(ordered_y, ticker="y").price_series
    except PricePolicyError as exc:
        raise M8FormationError(str(exc)) from exc

    x_window = validated_x[(validated_x.index >= formation_start_utc) & (validated_x.index <= formation_end_utc)]
    y_window = validated_y[(validated_y.index >= formation_start_utc) & (validated_y.index <= formation_end_utc)]
    aligned = pd.concat([x_window, y_window], axis=1, join="inner").dropna()
    if len(aligned) < 253:
        raise M8FormationError("at least 253 aligned formation levels are required")

    x_values = aligned.iloc[:, 0].to_numpy(dtype=float)
    y_values = aligned.iloc[:, 1].to_numpy(dtype=float)

    X = np.column_stack([np.ones(len(x_values), dtype=float), x_values])
    coefficients, *_ = np.linalg.lstsq(X, y_values, rcond=None)
    alpha = float(coefficients[0])
    beta = float(coefficients[1])
    if not np.isfinite(alpha) or not np.isfinite(beta):
        raise M8FormationError("OLS coefficients must be finite")

    spread = y_values - alpha - beta * x_values
    try:
        diagnostics = validate_numerics(x_values, y_values, spread)
    except M8NumericsError as exc:
        raise M8FormationError(str(exc)) from exc

    spread_mean = float(np.mean(spread))
    spread_std = float(np.std(spread, ddof=1))

    return FormationFit(
        alpha=alpha,
        beta=beta,
        spread_mean=spread_mean,
        spread_std=spread_std,
        diagnostics=diagnostics,
        aligned_index=pd.DatetimeIndex(aligned.index),
        x_series=aligned.iloc[:, 0].copy(),
        y_series=aligned.iloc[:, 1].copy(),
    )

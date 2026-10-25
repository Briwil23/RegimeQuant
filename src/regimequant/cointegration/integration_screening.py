from __future__ import annotations

import math

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from regimequant.cointegration.price_policy import DailyPricePolicy, PricePolicyError


class IntegrationScreeningError(ValueError):
    """Raised when a pair cannot satisfy the M7 I(1)-compatibility gate."""


def _canonical_pair_order(x: pd.Series, y: pd.Series) -> tuple[pd.Series, pd.Series]:
    if not isinstance(x, pd.Series) or not isinstance(y, pd.Series):
        raise IntegrationScreeningError("x and y must both be pandas Series")

    left_name = str(x.name if x.name is not None else "left").strip()
    right_name = str(y.name if y.name is not None else "right").strip()

    if not left_name or not right_name or left_name == right_name:
        raise IntegrationScreeningError("pair requires distinct canonical security IDs")

    ordered_left, ordered_right = sorted((left_name, right_name))
    if (left_name, right_name) != (ordered_left, ordered_right):
        return y.copy(), x.copy()
    return x.copy(), y.copy()


def screen_pair_for_i1(x: pd.Series, y: pd.Series) -> dict[str, object]:
    x, y = _canonical_pair_order(x, y)

    try:
        left = DailyPricePolicy.validate_series(x, ticker="left")
        right = DailyPricePolicy.validate_series(y, ticker="right")
    except PricePolicyError as exc:
        raise IntegrationScreeningError(str(exc)) from exc

    left_series = left.price_series.sort_index()
    right_series = right.price_series.sort_index()
    joined = pd.concat([left_series, right_series], axis=1, join="inner")
    if joined.empty or len(joined) < 253:
        raise IntegrationScreeningError("insufficient aligned prices for pair screening")

    left_values = joined.iloc[:, 0].astype(float)
    right_values = joined.iloc[:, 1].astype(float)
    if np.isnan(left_values.to_numpy()).any() or np.isnan(right_values.to_numpy()).any():
        raise IntegrationScreeningError("aligned pair contains NaN values")

    level_x = left_values
    level_y = right_values
    x_diff = level_x.diff().dropna()
    y_diff = level_y.diff().dropna()

    if x_diff.empty or y_diff.empty:
        raise IntegrationScreeningError("insufficient first differences")
    if np.isclose(x_diff.var(ddof=1), 0.0) or np.isclose(y_diff.var(ddof=1), 0.0):
        raise IntegrationScreeningError("pair contains a degenerate differenced series")

    try:
        x_level_p = adfuller(level_x, maxlag=12, regression="c", autolag="AIC")[1]
        y_level_p = adfuller(level_y, maxlag=12, regression="c", autolag="AIC")[1]
        x_diff_p = adfuller(x_diff, maxlag=12, regression="c", autolag="AIC")[1]
        y_diff_p = adfuller(y_diff, maxlag=12, regression="c", autolag="AIC")[1]
    except Exception as exc:
        raise IntegrationScreeningError("ADF screening failed on provided input") from exc

    acceptable = (
        math.isfinite(x_level_p)
        and math.isfinite(y_level_p)
        and math.isfinite(x_diff_p)
        and math.isfinite(y_diff_p)
        and x_level_p > 0.05
        and y_level_p > 0.05
        and x_diff_p <= 0.05
        and y_diff_p <= 0.05
    )

    if not acceptable:
        raise IntegrationScreeningError("pair does not satisfy I(1)-compatibility gate")

    return {
        "acceptable": True,
        "x_level_p_value": float(x_level_p),
        "y_level_p_value": float(y_level_p),
        "x_diff_p_value": float(x_diff_p),
        "y_diff_p_value": float(y_diff_p),
    }

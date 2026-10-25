from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regimequant.cointegration.price_policy import DailyPricePolicy, PricePolicyError


def test_price_policy_accepts_valid_daily_close_series() -> None:
    dates = pd.date_range("2020-01-02", periods=300, freq="B")
    prices = pd.Series(range(1, 301), index=dates, dtype=float)
    result = DailyPricePolicy.validate_series(prices, ticker="AAPL")
    assert result.valid is True
    assert len(result.aligned_index) >= 253


def test_price_policy_requires_253_levels_and_rejects_252_levels() -> None:
    short_dates = pd.date_range("2020-01-02", periods=252, freq="B")
    short_prices = pd.Series(np.linspace(100.0, 200.0, 252), index=short_dates, dtype=float)
    with pytest.raises(PricePolicyError):
        DailyPricePolicy.validate_series(short_prices, ticker="AAPL")

    valid_dates = pd.date_range("2020-01-02", periods=253, freq="B")
    valid_prices = pd.Series(np.linspace(100.0, 200.0, 253), index=valid_dates, dtype=float)
    result = DailyPricePolicy.validate_series(valid_prices, ticker="AAPL")
    assert result.valid is True
    assert result.first_difference_count >= 252


def test_price_policy_rejects_duplicate_rows_and_rescaling_preserves_admissibility() -> None:
    duplicate_index = pd.to_datetime(["2020-01-02", "2020-01-02", "2020-01-03"], utc=True)
    duplicate_prices = pd.Series([100.0, 101.0, 102.0], index=duplicate_index, dtype=float)
    with pytest.raises(PricePolicyError):
        DailyPricePolicy.validate_series(duplicate_prices, ticker="AAPL")

    valid_dates = pd.date_range("2020-01-02", periods=253, freq="B")
    valid_prices = pd.Series(np.linspace(100.0, 200.0, 253), index=valid_dates, dtype=float)
    rescaled = valid_prices * 2.5
    result = DailyPricePolicy.validate_series(rescaled, ticker="AAPL")
    assert result.valid is True
    assert result.first_difference_count >= 252


def test_price_policy_rejects_missing_required_sessions_and_non_positive_prices() -> None:
    dates = pd.date_range("2020-01-02", periods=300, freq="B")
    prices = pd.Series([0.0] + [100.0] * (len(dates) - 1), index=dates)
    try:
        DailyPricePolicy.validate_series(prices, ticker="AAPL")
    except PricePolicyError:
        return
    raise AssertionError("non-positive prices should be rejected")

from __future__ import annotations

import numpy as np
import pandas as pd

import regimequant.cointegration.engle_granger as engle_granger_module
from regimequant.cointegration.engle_granger import EngleGrangerResult, canonical_engle_granger_test


def test_canonical_engle_granger_test_returns_valid_result() -> None:
    index = pd.date_range("2020-01-02", periods=400, freq="B")
    t = np.arange(400, dtype=float)
    x = pd.Series(100.0 + t + 0.25 * np.sin(t / 7.0), index=index, dtype=float)
    y = pd.Series(90.0 + 1.1 * x.to_numpy() + 0.35 * np.cos(t / 5.0), index=index, dtype=float)
    result = canonical_engle_granger_test(x, y)
    assert isinstance(result, EngleGrangerResult)
    assert result.p_value >= 0.0
    assert result.p_value <= 1.0


def test_canonical_engle_granger_test_uses_frozen_canonical_contract(monkeypatch) -> None:
    index = pd.date_range("2020-01-02", periods=400, freq="B")
    t = np.arange(400, dtype=float)
    x = pd.Series(100.0 + t + 0.25 * np.sin(t / 7.0), index=index, name="MSFT", dtype=float)
    y = pd.Series(90.0 + 1.1 * x.to_numpy() + 0.35 * np.cos(t / 5.0), index=index, name="AAPL", dtype=float)
    observed = {}
    original = engle_granger_module.coint

    def spy(left, right, *args, **kwargs):
        observed["left_name"] = left.name
        observed["right_name"] = right.name
        observed["kwargs"] = kwargs
        return original(left, right, *args, **kwargs)

    monkeypatch.setattr(engle_granger_module, "coint", spy)
    result = canonical_engle_granger_test(y, x)
    assert isinstance(result, EngleGrangerResult)
    assert observed["left_name"] == "AAPL"
    assert observed["right_name"] == "MSFT"
    assert observed["kwargs"]["trend"] == "c"
    assert observed["kwargs"]["maxlag"] == 12
    assert observed["kwargs"]["autolag"] == "AIC"

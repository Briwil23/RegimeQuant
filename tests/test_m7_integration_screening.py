from __future__ import annotations

import numpy as np
import pandas as pd

import regimequant.cointegration.integration_screening as integration_screening_module
from regimequant.cointegration.integration_screening import IntegrationScreeningError, screen_pair_for_i1


def test_screen_pair_for_i1_accepts_i1_compatible_pair() -> None:
    index = pd.date_range("2020-01-02", periods=400, freq="B")
    t = np.arange(400, dtype=float)
    x = pd.Series(100.0 + t + 0.25 * np.sin(t / 7.0), index=index, dtype=float)
    y = pd.Series(90.0 + 1.1 * x.to_numpy() + 0.35 * np.cos(t / 5.0), index=index, dtype=float)
    result = screen_pair_for_i1(x, y)
    assert result["acceptable"] is True


def test_screen_pair_for_i1_uses_frozen_adf_contract(monkeypatch) -> None:
    index = pd.date_range("2020-01-02", periods=400, freq="B")
    t = np.arange(400, dtype=float)
    x = pd.Series(100.0 + t + 0.25 * np.sin(t / 7.0), index=index, name="MSFT", dtype=float)
    y = pd.Series(90.0 + 1.1 * x.to_numpy() + 0.35 * np.cos(t / 5.0), index=index, name="AAPL", dtype=float)
    observed = {}
    original = integration_screening_module.adfuller

    def spy(series, *args, **kwargs):
        observed["kwargs"] = kwargs
        return original(series, *args, **kwargs)

    monkeypatch.setattr(integration_screening_module, "adfuller", spy)
    result = screen_pair_for_i1(x, y)
    assert result["acceptable"] is True
    assert observed["kwargs"]["maxlag"] == 12
    assert observed["kwargs"]["regression"] == "c"
    assert observed["kwargs"]["autolag"] == "AIC"


def test_screen_pair_for_i1_canonicalizes_reversed_arguments() -> None:
    index = pd.date_range("2020-01-02", periods=400, freq="B")
    t = np.arange(400, dtype=float)
    x = pd.Series(100.0 + t + 0.25 * np.sin(t / 7.0), index=index, name="MSFT", dtype=float)
    y = pd.Series(90.0 + 1.1 * x.to_numpy() + 0.35 * np.cos(t / 5.0), index=index, name="AAPL", dtype=float)

    ordered_x, ordered_y = integration_screening_module._canonical_pair_order(x, y)
    assert ordered_x.name == "AAPL"
    assert ordered_y.name == "MSFT"

    direct = screen_pair_for_i1(x, y)
    reversed_pair = screen_pair_for_i1(y, x)
    assert direct["acceptable"] is True
    assert reversed_pair["acceptable"] is True
    assert direct["x_level_p_value"] == reversed_pair["x_level_p_value"]
    assert direct["y_level_p_value"] == reversed_pair["y_level_p_value"]


def test_screen_pair_for_i1_rejects_i0_or_ambiguous_pair() -> None:
    index = pd.date_range("2020-01-02", periods=400, freq="B")
    x = pd.Series([1.0] * len(index), index=index, dtype=float)
    y = pd.Series([2.0] * len(index), index=index, dtype=float)
    try:
        screen_pair_for_i1(x, y)
    except IntegrationScreeningError:
        return
    raise AssertionError("constant series should be rejected")

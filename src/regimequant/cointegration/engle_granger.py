from __future__ import annotations

import math

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from regimequant.cointegration.contracts import EngleGrangerResult
from regimequant.cointegration.integration_screening import IntegrationScreeningError, _canonical_pair_order, screen_pair_for_i1


class EngleGrangerError(ValueError):
    """Raised when the canonical Engle–Granger test cannot be computed."""


def _canonical_ordered_pair(x: pd.Series, y: pd.Series) -> tuple[pd.Series, pd.Series]:
    ordered_x, ordered_y = _canonical_pair_order(x, y)
    return ordered_x.sort_index(), ordered_y.sort_index()


def canonical_engle_granger_test(x: pd.Series, y: pd.Series, *, pair_id: str = "pair") -> EngleGrangerResult:
    try:
        screen_pair_for_i1(x, y)
    except IntegrationScreeningError as exc:
        raise EngleGrangerError(str(exc)) from exc

    ordered_x, ordered_y = _canonical_ordered_pair(x, y)
    aligned = pd.concat([ordered_x.astype(float), ordered_y.astype(float)], axis=1, join="inner")
    if aligned.empty:
        raise EngleGrangerError("unable to form aligned pair")

    try:
        stat, p_value, *_ = coint(aligned.iloc[:, 0], aligned.iloc[:, 1], trend="c", maxlag=12, autolag="AIC")
    except Exception as exc:  # pragma: no cover - defensive path
        raise EngleGrangerError("cointegration test failed numerically") from exc

    if not math.isfinite(float(stat)) or not math.isfinite(float(p_value)):
        raise EngleGrangerError("cointegration test produced non-finite values")

    return EngleGrangerResult(
        pair_id=str(pair_id),
        statistic=float(stat),
        p_value=float(p_value),
        valid=True,
        warnings=(),
    )

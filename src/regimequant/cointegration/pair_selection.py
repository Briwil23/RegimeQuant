from __future__ import annotations

from itertools import combinations
from typing import Iterable

import pandas as pd

from regimequant.cointegration.contracts import CointegrationCandidate
from regimequant.cointegration.engle_granger import canonical_engle_granger_test
from regimequant.cointegration.integration_screening import screen_pair_for_i1
from regimequant.cointegration.price_policy import DailyPricePolicy


def deterministic_pair_selection(securities: Iterable[str], price_table: dict[str, pd.Series], *, threshold: float = 0.05) -> list[CointegrationCandidate]:
    instruments = sorted({str(item).strip() for item in securities if str(item).strip()})
    if len(instruments) < 2:
        return []

    allowed: list[CointegrationCandidate] = []
    for left, right in combinations(instruments, 2):
        left_series = price_table[left].copy()
        right_series = price_table[right].copy()
        DailyPricePolicy.validate_series(left_series, left)
        DailyPricePolicy.validate_series(right_series, right)
        screening = screen_pair_for_i1(left_series, right_series)
        if not screening["acceptable"]:
            continue
        result = canonical_engle_granger_test(left_series, right_series)
        if not result.valid:
            continue
        candidate = CointegrationCandidate(
            instrument_a=left,
            instrument_b=right,
            pair_id=f"pair_{left}_{right}",
            status="candidate",
            p_value=result.p_value,
            adjusted_p_value=result.p_value,
            score=float(result.p_value),
            source_mode="production",
        )
        if result.p_value <= threshold:
            allowed.append(candidate)

    allowed.sort(key=lambda item: (item.p_value if item.p_value is not None else 1.0, item.pair_id))
    return allowed


def select_canonical_pair(securities: Iterable[str], price_table: dict[str, pd.Series], *, threshold: float = 0.05) -> CointegrationCandidate | None:
    candidates = deterministic_pair_selection(securities, price_table, threshold=threshold)
    return candidates[0] if candidates else None

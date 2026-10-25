from __future__ import annotations

import math

import numpy as np


class BYAdjustmentError(ValueError):
    """Raised when the BY family-wise error gate cannot be applied."""


def by_fdr_adjustment(p_values: list[float] | tuple[float, ...] | np.ndarray) -> list[float]:
    values = np.asarray(list(p_values), dtype=float)
    if values.size == 0:
        return []
    if not np.all(np.isfinite(values)):
        raise BYAdjustmentError("p-values must be finite")
    if np.any((values < 0.0) | (values > 1.0)):
        raise BYAdjustmentError("p-values must lie in [0, 1]")

    m = values.size
    harmonic = float(sum(1.0 / float(i) for i in range(1, m + 1)))
    order = np.argsort(values)
    adjusted_sorted = np.empty(m, dtype=float)
    cumulative = 1.0
    for rank_index in range(m - 1, -1, -1):
        original_rank = order[rank_index]
        rank = rank_index + 1
        candidate = values[original_rank] * m * harmonic / rank
        cumulative = min(cumulative, candidate)
        adjusted_sorted[original_rank] = min(1.0, cumulative)

    return adjusted_sorted.tolist()

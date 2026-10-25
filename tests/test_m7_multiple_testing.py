from __future__ import annotations

from regimequant.cointegration.multiple_testing import by_fdr_adjustment


def test_by_fdr_adjustment_uses_harmonic_correction() -> None:
    p_values = [0.01, 0.04, 0.06]
    adjusted = by_fdr_adjustment(p_values)
    assert len(adjusted) == 3
    assert all(value >= 0.0 for value in adjusted)
    assert all(value <= 1.0 for value in adjusted)

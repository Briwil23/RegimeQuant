from __future__ import annotations

import numpy as np

from regimequant.signals.m8_numerics import validate_numerics


def test_numerical_diagnostics_are_scale_invariant() -> None:
    x = np.linspace(100.0, 200.0, 253)
    y = 3.0 + 1.5 * x + 0.1 * np.sin(np.arange(253))
    spread = y - (3.0 + 1.5 * x)

    base = validate_numerics(x, y, spread)
    scaled = validate_numerics(x * 100.0, y * 100.0, spread * 100.0)

    assert base.x_variance_ratio > 0.0
    assert scaled.x_variance_ratio > 0.0
    assert abs(base.x_variance_ratio - scaled.x_variance_ratio) < 1e-12
    assert abs(base.spread_scale_ratio - scaled.spread_scale_ratio) < 1e-12

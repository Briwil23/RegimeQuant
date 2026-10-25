from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class M8NumericsError(ValueError):
    """Raised when formation data fails deterministic numerical requirements."""


@dataclass(frozen=True, slots=True)
class NumericalDiagnostics:
    x_scale: float
    y_scale: float
    x_variance_ratio: float
    condition_number: float
    spread_scale_ratio: float


def _safe_scale(values: np.ndarray) -> float:
    scale = float(np.median(np.abs(values)))
    return max(scale, 1e-12)


def _scaled_design_condition_number(x: np.ndarray) -> float:
    column = x.reshape(-1, 1)
    rms = float(np.sqrt(np.mean(np.square(column))))
    if rms <= 0.0:
        rms = 1.0
    scaled = column / rms
    design = np.column_stack([np.ones_like(scaled), scaled])
    return float(np.linalg.cond(design))


def validate_numerics(x: np.ndarray, y: np.ndarray, spread: np.ndarray) -> NumericalDiagnostics:
    if x.size < 253 or y.size < 253:
        raise M8NumericsError("at least 253 aligned formation observations are required")
    if x.size != y.size or x.size != spread.size:
        raise M8NumericsError("x, y, and spread must have identical lengths")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(spread).all():
        raise M8NumericsError("formation arrays must be finite")

    x_scale = _safe_scale(x)
    y_scale = _safe_scale(y)
    x_var = float(np.var(x, ddof=1))
    x_var_ratio = x_var / max(x_scale * x_scale, 1e-12)
    if x_var_ratio <= 1e-10:
        raise M8NumericsError("x is near-constant after scale normalization")

    condition_number = _scaled_design_condition_number(x)
    if condition_number > 1e12:
        raise M8NumericsError("formation regression is ill-conditioned")

    spread_std = float(np.std(spread, ddof=1))
    spread_ratio = spread_std / max(max(x_scale, y_scale), 1e-12)
    if spread_ratio <= 1e-8:
        raise M8NumericsError("residual spread variance is too small relative to formation scale")

    return NumericalDiagnostics(
        x_scale=x_scale,
        y_scale=y_scale,
        x_variance_ratio=x_var_ratio,
        condition_number=condition_number,
        spread_scale_ratio=spread_ratio,
    )

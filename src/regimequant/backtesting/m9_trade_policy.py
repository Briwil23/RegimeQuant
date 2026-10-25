from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN

from regimequant.backtesting.m9_contracts import BETA_MIN_ABS, RejectReason, SignalType


DECIMAL_0 = Decimal("0")


@dataclass(frozen=True, slots=True)
class SizingResult:
    accepted: bool
    reason: RejectReason | None
    q_y: Decimal
    q_x: Decimal
    gross_notional: Decimal
    gross_residual: Decimal
    hedge_error: Decimal


def _floor_to_step(value: Decimal, step: Decimal) -> Decimal:
    if step <= DECIMAL_0:
        raise ValueError("step must be positive")
    sign = Decimal("1") if value >= DECIMAL_0 else Decimal("-1")
    absolute = value.copy_abs()
    units = (absolute / step).to_integral_value(rounding=ROUND_DOWN)
    return sign * units * step


def _direction_from_signal(signal: SignalType) -> int:
    if signal == SignalType.LONG_SPREAD:
        return 1
    if signal == SignalType.SHORT_SPREAD:
        return -1
    raise ValueError("signal must be LONG_SPREAD or SHORT_SPREAD")


def compute_share_ratio_sizing(
    *,
    signal: SignalType,
    beta: Decimal,
    gross_budget: Decimal,
    price_y: Decimal,
    price_x: Decimal,
    lot_step_y: Decimal = Decimal("0.0001"),
    lot_step_x: Decimal = Decimal("0.0001"),
    hedge_tolerance_fraction: Decimal = Decimal("0.001"),
) -> SizingResult:
    if gross_budget <= DECIMAL_0:
        return SizingResult(False, RejectReason.INVALID_BUDGET, DECIMAL_0, DECIMAL_0, DECIMAL_0, DECIMAL_0, DECIMAL_0)
    if price_y <= DECIMAL_0 or price_x <= DECIMAL_0:
        return SizingResult(False, RejectReason.INVALID_PRICE, DECIMAL_0, DECIMAL_0, DECIMAL_0, DECIMAL_0, DECIMAL_0)
    if beta.copy_abs() < BETA_MIN_ABS:
        return SizingResult(False, RejectReason.DEGENERATE_HEDGE, DECIMAL_0, DECIMAL_0, DECIMAL_0, DECIMAL_0, DECIMAL_0)

    direction = _direction_from_signal(signal)
    denominator = price_y + beta.copy_abs() * price_x
    if denominator <= DECIMAL_0:
        return SizingResult(False, RejectReason.INVALID_PRICE, DECIMAL_0, DECIMAL_0, DECIMAL_0, DECIMAL_0, DECIMAL_0)

    raw_q_y = Decimal(direction) * gross_budget / denominator
    raw_q_x = -beta * raw_q_y

    q_y = _floor_to_step(raw_q_y, lot_step_y)
    q_x = _floor_to_step(raw_q_x, lot_step_x)

    gross_notional = q_y.copy_abs() * price_y + q_x.copy_abs() * price_x
    gross_residual = gross_budget - gross_notional
    if gross_residual < DECIMAL_0:
        return SizingResult(False, RejectReason.RISK_PRECHECK_FAILED, DECIMAL_0, DECIMAL_0, DECIMAL_0, DECIMAL_0, DECIMAL_0)

    hedge_error = (q_x + beta * q_y).copy_abs()
    tolerance = max(Decimal("0.000001"), hedge_tolerance_fraction * raw_q_x.copy_abs())
    if hedge_error > tolerance:
        return SizingResult(False, RejectReason.RISK_PRECHECK_FAILED, DECIMAL_0, DECIMAL_0, DECIMAL_0, DECIMAL_0, hedge_error)

    return SizingResult(True, None, q_y, q_x, gross_notional, gross_residual, hedge_error)

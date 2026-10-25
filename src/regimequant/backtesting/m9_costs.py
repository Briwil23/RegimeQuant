from __future__ import annotations

from decimal import Decimal


DAYS_PER_YEAR = Decimal("252")


def commission_cost(notional: Decimal, bps: Decimal = Decimal("1")) -> Decimal:
    return notional.copy_abs() * (bps / Decimal("10000"))


def slippage_cost(notional: Decimal, bps: Decimal = Decimal("2")) -> Decimal:
    return notional.copy_abs() * (bps / Decimal("10000"))


def borrow_accrual(short_market_value: Decimal, annual_rate: Decimal = Decimal("0.03"), days: int = 1) -> Decimal:
    if days < 0:
        raise ValueError("days must be non-negative")
    return short_market_value.copy_abs() * annual_rate * Decimal(days) / DAYS_PER_YEAR


def financing_accrual(free_cash: Decimal, annual_rate: Decimal = Decimal("0.04"), days: int = 1) -> Decimal:
    if days < 0:
        raise ValueError("days must be non-negative")
    if free_cash >= Decimal("0"):
        return Decimal("0")
    return free_cash.copy_abs() * annual_rate * Decimal(days) / DAYS_PER_YEAR

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from regimequant.backtesting.m9_contracts import RiskCaps


DECIMAL_0 = Decimal("0")


@dataclass(frozen=True, slots=True)
class RiskCheckResult:
    accepted: bool
    reason: str | None


def pair_budget_from_initial_equity(initial_equity: Decimal, caps: RiskCaps) -> Decimal:
    if initial_equity <= DECIMAL_0:
        raise ValueError("initial_equity must be positive")
    return initial_equity * caps.pair_budget_fraction


def aggregate_security_exposure(current: dict[str, Decimal], delta: dict[str, Decimal]) -> dict[str, Decimal]:
    merged = dict(current)
    for security_id, value in delta.items():
        merged[security_id] = merged.get(security_id, DECIMAL_0) + value
    return merged


def gross_notional(security_exposure: dict[str, Decimal]) -> Decimal:
    return sum((value.copy_abs() for value in security_exposure.values()), DECIMAL_0)


def net_notional(security_exposure: dict[str, Decimal]) -> Decimal:
    return sum(security_exposure.values(), DECIMAL_0)


def validate_caps(*, security_exposure: dict[str, Decimal], equity: Decimal, caps: RiskCaps) -> RiskCheckResult:
    if equity <= DECIMAL_0:
        return RiskCheckResult(False, "NON_POSITIVE_EQUITY")

    gross = gross_notional(security_exposure)
    net = net_notional(security_exposure)

    if gross > caps.gross_cap_fraction * equity:
        return RiskCheckResult(False, "GROSS_CAP_EXCEEDED")
    if net.copy_abs() > caps.net_cap_fraction * equity:
        return RiskCheckResult(False, "NET_CAP_EXCEEDED")

    sec_cap = caps.security_cap_fraction * equity
    for value in security_exposure.values():
        if value.copy_abs() > sec_cap:
            return RiskCheckResult(False, "SECURITY_CAP_EXCEEDED")

    return RiskCheckResult(True, None)


def reserve_capital(*, free_cash: Decimal, current_reserved: Decimal, amount: Decimal) -> tuple[bool, Decimal]:
    if amount < DECIMAL_0:
        raise ValueError("amount must be non-negative")
    available = free_cash - current_reserved
    if available < amount:
        return False, current_reserved
    return True, current_reserved + amount

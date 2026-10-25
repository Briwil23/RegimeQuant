from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal, ROUND_HALF_EVEN

from regimequant.backtesting.m9_contracts import FillLeg


CENT = Decimal("0.01")
DECIMAL_0 = Decimal("0")


def money(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_HALF_EVEN)


@dataclass(frozen=True, slots=True)
class LedgerState:
    free_cash: Decimal
    restricted_cash: Decimal
    positions: dict[str, Decimal]
    restricted_by_security: dict[str, Decimal]
    basis_by_security: dict[str, Decimal]
    realized_pnl: Decimal
    fee_expense: Decimal
    borrow_expense: Decimal
    financing_expense: Decimal
    rounding_expense: Decimal
    unrealized_pnl: Decimal = DECIMAL_0


def create_ledger(initial_cash: Decimal) -> LedgerState:
    return LedgerState(
        free_cash=money(initial_cash),
        restricted_cash=DECIMAL_0,
        positions={},
        restricted_by_security={},
        basis_by_security={},
        realized_pnl=DECIMAL_0,
        fee_expense=DECIMAL_0,
        borrow_expense=DECIMAL_0,
        financing_expense=DECIMAL_0,
        rounding_expense=DECIMAL_0,
    )


def _transfer_free_to_restricted(ledger: LedgerState, amount: Decimal, security_id: str) -> LedgerState:
    amount = money(amount)
    return replace(
        ledger,
        free_cash=money(ledger.free_cash - amount),
        restricted_cash=money(ledger.restricted_cash + amount),
        restricted_by_security={
            **ledger.restricted_by_security,
            security_id: money(ledger.restricted_by_security.get(security_id, DECIMAL_0) + amount),
        },
    )


def _release_restricted_to_free(ledger: LedgerState, amount: Decimal, security_id: str) -> LedgerState:
    amount = money(amount)
    current = ledger.restricted_by_security.get(security_id, DECIMAL_0)
    released = amount if amount <= current else current
    return replace(
        ledger,
        free_cash=money(ledger.free_cash + released),
        restricted_cash=money(ledger.restricted_cash - released),
        restricted_by_security={
            **ledger.restricted_by_security,
            security_id: money(current - released),
        },
    )


def post_atomic_fill(
    ledger: LedgerState,
    legs: tuple[FillLeg, FillLeg],
    *,
    commissions: tuple[Decimal, Decimal],
    slippages: tuple[Decimal, Decimal],
) -> LedgerState:
    updated = ledger
    positions = dict(updated.positions)
    basis = dict(updated.basis_by_security)

    for idx, leg in enumerate(legs):
        security = leg.security_id
        qty = leg.quantity
        price = leg.close_price
        notional = money(qty.copy_abs() * price)

        prior_qty = positions.get(security, DECIMAL_0)
        prior_basis = basis.get(security, DECIMAL_0)
        new_qty = prior_qty + qty
        realized_delta = DECIMAL_0

        if prior_qty > DECIMAL_0 and qty < DECIMAL_0:
            close_qty = min(prior_qty, qty.copy_abs())
            realized_delta = money(close_qty * money(price - prior_basis))
        elif prior_qty < DECIMAL_0 and qty > DECIMAL_0:
            close_qty = min(prior_qty.copy_abs(), qty)
            realized_delta = money(close_qty * money(prior_basis - price))

        if new_qty > DECIMAL_0:
            if prior_qty >= DECIMAL_0 and qty > DECIMAL_0:
                prior_cost = money(prior_qty * prior_basis)
                add_cost = money(qty * price)
                basis[security] = money((prior_cost + add_cost) / new_qty)
            elif prior_qty < DECIMAL_0:
                basis[security] = money(price)
        elif new_qty < DECIMAL_0:
            if prior_qty <= DECIMAL_0 and qty < DECIMAL_0:
                prior_cost = money(prior_qty.copy_abs() * prior_basis)
                add_cost = money(qty.copy_abs() * price)
                basis[security] = money((prior_cost + add_cost) / new_qty.copy_abs())
            elif prior_qty > DECIMAL_0:
                basis[security] = money(price)
        else:
            basis[security] = DECIMAL_0

        if qty > DECIMAL_0:
            updated = replace(updated, free_cash=money(updated.free_cash - notional))
            if prior_qty < DECIMAL_0:
                cover_qty = min(qty, prior_qty.copy_abs())
                if prior_qty.copy_abs() > DECIMAL_0:
                    prior_restricted = updated.restricted_by_security.get(security, DECIMAL_0)
                    release = money(prior_restricted * (cover_qty / prior_qty.copy_abs()))
                    updated = _release_restricted_to_free(updated, release, security)
        elif qty < DECIMAL_0:
            updated = replace(updated, free_cash=money(updated.free_cash + notional))
            incremental_short = max(new_qty.copy_abs() - prior_qty.copy_abs(), DECIMAL_0) if new_qty < DECIMAL_0 else DECIMAL_0
            if incremental_short > DECIMAL_0:
                transfer = money(incremental_short * price)
                updated = _transfer_free_to_restricted(updated, transfer, security)

        positions[security] = new_qty

        commission = money(commissions[idx])
        slippage = money(slippages[idx])
        fee_total = money(commission + slippage)
        updated = replace(
            updated,
            free_cash=money(updated.free_cash - fee_total),
            fee_expense=money(updated.fee_expense + fee_total),
            realized_pnl=money(updated.realized_pnl + realized_delta),
        )

    return replace(updated, positions=positions, basis_by_security=basis)


def post_borrow_and_financing(ledger: LedgerState, *, borrow_cost: Decimal, financing_cost: Decimal) -> LedgerState:
    borrow_cost = money(borrow_cost)
    financing_cost = money(financing_cost)
    total = money(borrow_cost + financing_cost)
    return replace(
        ledger,
        free_cash=money(ledger.free_cash - total),
        borrow_expense=money(ledger.borrow_expense + borrow_cost),
        financing_expense=money(ledger.financing_expense + financing_cost),
    )


def post_cash_dividend(
    ledger: LedgerState,
    *,
    security_id: str,
    dividend_per_share: Decimal,
    evidence_complete: bool,
) -> tuple[LedgerState, bool]:
    if not evidence_complete:
        return ledger, False
    qty = ledger.positions.get(security_id, DECIMAL_0)
    if qty == DECIMAL_0:
        return ledger, True
    cash_flow = money(qty * dividend_per_share)
    return replace(ledger, free_cash=money(ledger.free_cash + cash_flow)), True


def post_split(
    ledger: LedgerState,
    *,
    security_id: str,
    split_ratio: Decimal,
    evidence_complete: bool,
) -> tuple[LedgerState, bool]:
    if not evidence_complete:
        return ledger, False
    if split_ratio <= DECIMAL_0:
        return ledger, False
    qty = ledger.positions.get(security_id, DECIMAL_0)
    if qty == DECIMAL_0:
        return ledger, True
    positions = dict(ledger.positions)
    basis = dict(ledger.basis_by_security)
    positions[security_id] = qty * split_ratio
    prior_basis = basis.get(security_id, DECIMAL_0)
    if prior_basis != DECIMAL_0:
        basis[security_id] = money(prior_basis / split_ratio)
    return replace(ledger, positions=positions, basis_by_security=basis), True


def equity(ledger: LedgerState, prices: dict[str, Decimal]) -> Decimal:
    market_value = DECIMAL_0
    for security, qty in ledger.positions.items():
        market_value = money(market_value + money(qty * prices[security]))
    return money(ledger.free_cash + ledger.restricted_cash + market_value)


def cash_conserved_before_after_transfer(before: LedgerState, after: LedgerState) -> bool:
    return money(before.free_cash + before.restricted_cash) == money(after.free_cash + after.restricted_cash)


def reconciliation_residual(
    *,
    equity_prev: Decimal,
    equity_now: Decimal,
    realized: Decimal,
    delta_unrealized: Decimal,
    costs: Decimal,
    corp_action_cash: Decimal,
    rounding_adjustment: Decimal,
) -> Decimal:
    lhs = money(equity_now - equity_prev)
    rhs = money(realized + delta_unrealized - costs + corp_action_cash + rounding_adjustment)
    return money(lhs - rhs)


def close_pair_with_realized_pnl(
    ledger: LedgerState,
    *,
    y_security_id: str,
    x_security_id: str,
    close_price_y: Decimal,
    close_price_x: Decimal,
) -> tuple[LedgerState, Decimal]:
    positions = dict(ledger.positions)
    qy = positions.get(y_security_id, DECIMAL_0)
    qx = positions.get(x_security_id, DECIMAL_0)
    if qy == DECIMAL_0 and qx == DECIMAL_0:
        return ledger, DECIMAL_0

    updated = ledger
    realized = DECIMAL_0

    if qy != DECIMAL_0:
        notional_y = money(qy.copy_abs() * close_price_y)
        if qy > DECIMAL_0:
            updated = replace(updated, free_cash=money(updated.free_cash + notional_y))
            realized = money(realized + notional_y)
        else:
            updated = replace(updated, free_cash=money(updated.free_cash - notional_y))
            release = updated.restricted_by_security.get(y_security_id, DECIMAL_0)
            updated = _release_restricted_to_free(updated, release, y_security_id)
            realized = money(realized - notional_y)
        positions[y_security_id] = DECIMAL_0

    if qx != DECIMAL_0:
        notional_x = money(qx.copy_abs() * close_price_x)
        if qx > DECIMAL_0:
            updated = replace(updated, free_cash=money(updated.free_cash + notional_x))
            realized = money(realized + notional_x)
        else:
            updated = replace(updated, free_cash=money(updated.free_cash - notional_x))
            release = updated.restricted_by_security.get(x_security_id, DECIMAL_0)
            updated = _release_restricted_to_free(updated, release, x_security_id)
            realized = money(realized - notional_x)
        positions[x_security_id] = DECIMAL_0

    updated = replace(
        updated,
        positions=positions,
        realized_pnl=money(updated.realized_pnl + realized),
    )
    return updated, realized


def validate_ledger_residual(residual: Decimal) -> None:
    if money(residual).copy_abs() > CENT:
        raise ValueError("INVALID_LEDGER")


def mark_unrealized_pnl(
    ledger: LedgerState,
    *,
    prices: dict[str, Decimal],
    prior_prices: dict[str, Decimal],
) -> tuple[LedgerState, Decimal]:
    prior_unrealized = ledger.unrealized_pnl
    current_unrealized = DECIMAL_0
    for security_id, qty in ledger.positions.items():
        if security_id in prices and qty != DECIMAL_0:
            basis = ledger.basis_by_security.get(security_id, DECIMAL_0)
            current_unrealized = money(current_unrealized + money(qty * (prices[security_id] - basis)))
    delta = money(current_unrealized - prior_unrealized)
    return replace(ledger, unrealized_pnl=current_unrealized), delta

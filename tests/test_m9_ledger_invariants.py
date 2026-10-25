from __future__ import annotations

from decimal import Decimal

from regimequant.backtesting.m9_accounting import (
    create_ledger,
    equity,
    mark_unrealized_pnl,
    money,
    post_atomic_fill,
    post_borrow_and_financing,
    post_cash_dividend,
    post_split,
    reconciliation_residual,
)
from regimequant.backtesting.m9_contracts import FillLeg


def test_restricted_short_proceeds_in_equity_and_not_free_cash_spendable() -> None:
    ledger = create_ledger(Decimal("100000"))
    legs = (
        FillLeg("A", Decimal("-10"), Decimal("100")),
        FillLeg("B", Decimal("0"), Decimal("50")),
    )
    ledger = post_atomic_fill(ledger, legs, commissions=(Decimal("0"), Decimal("0")), slippages=(Decimal("0"), Decimal("0")))

    assert ledger.restricted_cash == Decimal("1000.00")
    assert ledger.free_cash == Decimal("100000.00")

    eq = equity(ledger, {"A": Decimal("100"), "B": Decimal("50")})
    assert eq == Decimal("100000.00")


def test_costs_reduce_equity_once_and_reconcile() -> None:
    ledger = create_ledger(Decimal("10000"))
    legs = (
        FillLeg("A", Decimal("1"), Decimal("100")),
        FillLeg("B", Decimal("-1"), Decimal("100")),
    )
    prev_equity = equity(ledger, {"A": Decimal("100"), "B": Decimal("100")})
    ledger = post_atomic_fill(ledger, legs, commissions=(Decimal("1"), Decimal("1")), slippages=(Decimal("2"), Decimal("2")))
    ledger = post_borrow_and_financing(ledger, borrow_cost=Decimal("1"), financing_cost=Decimal("1"))
    now_equity = equity(ledger, {"A": Decimal("100"), "B": Decimal("100")})

    residual = reconciliation_residual(
        equity_prev=prev_equity,
        equity_now=now_equity,
        realized=Decimal("0"),
        delta_unrealized=Decimal("0"),
        costs=Decimal("8"),
        corp_action_cash=Decimal("0"),
        rounding_adjustment=Decimal("0"),
    )
    assert residual == Decimal("0.00")


def test_mandatory_numerical_accounting_case() -> None:
    ledger = create_ledger(Decimal("100000"))
    entry_legs = (
        FillLeg("A", Decimal("10"), Decimal("100")),
        FillLeg("B", Decimal("-10"), Decimal("100")),
    )
    ledger = post_atomic_fill(
        ledger,
        entry_legs,
        commissions=(Decimal("1"), Decimal("1")),
        slippages=(Decimal("2"), Decimal("2")),
    )

    assert ledger.free_cash == Decimal("98994.00")
    assert ledger.restricted_cash == Decimal("1000.00")
    assert ledger.positions["A"] == Decimal("10")
    assert ledger.positions["B"] == Decimal("-10")

    marks = {"A": Decimal("100"), "B": Decimal("100")}
    assert equity(ledger, marks) == Decimal("99994.00")

    marks_up = {"A": Decimal("110"), "B": Decimal("95")}
    ledger, period_delta = mark_unrealized_pnl(ledger, prices=marks_up, prior_prices=marks)
    assert period_delta == Decimal("150.00")
    assert equity(ledger, marks_up) == Decimal("100144.00")

    exit_legs = (
        FillLeg("A", Decimal("-10"), Decimal("110")),
        FillLeg("B", Decimal("10"), Decimal("95")),
    )
    ledger = post_atomic_fill(
        ledger,
        exit_legs,
        commissions=(Decimal("1"), Decimal("1")),
        slippages=(Decimal("2"), Decimal("2")),
    )

    assert ledger.restricted_cash == Decimal("0.00")
    assert ledger.positions["A"] == Decimal("0")
    assert ledger.positions["B"] == Decimal("0")
    assert ledger.realized_pnl == Decimal("150.00")
    assert equity(ledger, marks_up) == Decimal("100138.00")


def test_unrealized_period_delta_not_cumulative() -> None:
    ledger = create_ledger(Decimal("1000"))
    ledger = post_atomic_fill(
        ledger,
        (FillLeg("A", Decimal("1"), Decimal("100")), FillLeg("B", Decimal("0"), Decimal("100"))),
        commissions=(Decimal("0"), Decimal("0")),
        slippages=(Decimal("0"), Decimal("0")),
    )
    ledger, d1 = mark_unrealized_pnl(ledger, prices={"A": Decimal("110"), "B": Decimal("100")}, prior_prices={"A": Decimal("100"), "B": Decimal("100")})
    ledger, d2 = mark_unrealized_pnl(ledger, prices={"A": Decimal("105"), "B": Decimal("100")}, prior_prices={"A": Decimal("110"), "B": Decimal("100")})

    assert d1 == Decimal("10.00")
    assert d2 == Decimal("-5.00")
    assert ledger.unrealized_pnl == Decimal("5.00")


def test_no_equity_creation_from_short_proceeds() -> None:
    ledger = create_ledger(Decimal("100000"))
    ledger = post_atomic_fill(
        ledger,
        (FillLeg("B", Decimal("-10"), Decimal("100")), FillLeg("A", Decimal("0"), Decimal("100"))),
        commissions=(Decimal("0"), Decimal("0")),
        slippages=(Decimal("0"), Decimal("0")),
    )
    assert ledger.free_cash == Decimal("100000.00")
    assert ledger.restricted_cash == Decimal("1000.00")
    assert equity(ledger, {"B": Decimal("100"), "A": Decimal("100")}) == Decimal("100000.00")


def test_split_adjusts_quantity_and_basis_inversely() -> None:
    ledger = create_ledger(Decimal("1000"))
    ledger = post_atomic_fill(
        ledger,
        (FillLeg("A", Decimal("10"), Decimal("100")), FillLeg("B", Decimal("0"), Decimal("100"))),
        commissions=(Decimal("0"), Decimal("0")),
        slippages=(Decimal("0"), Decimal("0")),
    )
    ledger, ok = post_split(ledger, security_id="A", split_ratio=Decimal("2"), evidence_complete=True)
    assert ok
    assert ledger.positions["A"] == Decimal("20")
    assert ledger.basis_by_security["A"] == Decimal("50.00")


def test_dividend_posts_once() -> None:
    ledger = create_ledger(Decimal("1000"))
    ledger = post_atomic_fill(
        ledger,
        (FillLeg("A", Decimal("10"), Decimal("10")), FillLeg("B", Decimal("0"), Decimal("10"))),
        commissions=(Decimal("0"), Decimal("0")),
        slippages=(Decimal("0"), Decimal("0")),
    )
    before = ledger.free_cash
    ledger, ok = post_cash_dividend(ledger, security_id="A", dividend_per_share=Decimal("1"), evidence_complete=True)
    assert ok
    assert ledger.free_cash == money(before + Decimal("10"))

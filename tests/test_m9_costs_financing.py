from __future__ import annotations

from decimal import Decimal

from regimequant.backtesting.m9_costs import borrow_accrual, commission_cost, financing_accrual, slippage_cost


def test_cost_formulas_are_separate_from_execution_price() -> None:
    notional = Decimal("1000000")
    assert commission_cost(notional) == Decimal("100")
    assert slippage_cost(notional) == Decimal("200")


def test_borrow_and_financing_accrual() -> None:
    assert borrow_accrual(Decimal("1000000"), annual_rate=Decimal("0.03"), days=1) > Decimal("0")
    assert financing_accrual(Decimal("-1000000"), annual_rate=Decimal("0.04"), days=1) > Decimal("0")
    assert financing_accrual(Decimal("1000000"), annual_rate=Decimal("0.04"), days=1) == Decimal("0")

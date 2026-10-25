from __future__ import annotations

from decimal import Decimal

from regimequant.backtesting.m9_contracts import SignalType
from regimequant.backtesting.m9_trade_policy import compute_share_ratio_sizing
from regimequant.backtesting.m9_accounting import money


def test_money_rounding_is_deterministic_minor_unit() -> None:
    assert money(Decimal("1.005")) == Decimal("1.00")
    assert money(Decimal("1.015")) == Decimal("1.02")


def test_rounding_never_exceeds_budget() -> None:
    gross = Decimal("12345.67")
    result = compute_share_ratio_sizing(
        signal=SignalType.LONG_SPREAD,
        beta=Decimal("1.2"),
        gross_budget=gross,
        price_y=Decimal("101.23"),
        price_x=Decimal("49.87"),
        lot_step_y=Decimal("0.01"),
        lot_step_x=Decimal("0.01"),
    )
    assert result.accepted
    assert result.gross_notional <= gross
    assert result.gross_residual >= Decimal("0")

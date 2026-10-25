from __future__ import annotations

from decimal import Decimal

from regimequant.backtesting.m9_contracts import RejectReason, SignalType
from regimequant.backtesting.m9_trade_policy import compute_share_ratio_sizing


def test_share_ratio_sizing_preserves_gross_and_ratio_with_fine_lots() -> None:
    gross = Decimal("2000000")
    beta = Decimal("1.25")
    price_y = Decimal("100")
    price_x = Decimal("80")
    result = compute_share_ratio_sizing(
        signal=SignalType.LONG_SPREAD,
        beta=beta,
        gross_budget=gross,
        price_y=price_y,
        price_x=price_x,
        lot_step_y=Decimal("0.0000001"),
        lot_step_x=Decimal("0.0000001"),
    )
    assert result.accepted
    assert result.gross_notional <= gross
    ratio = result.q_x / result.q_y
    assert abs(ratio + beta) < Decimal("0.000001")


def test_degenerate_beta_is_rejected() -> None:
    result = compute_share_ratio_sizing(
        signal=SignalType.LONG_SPREAD,
        beta=Decimal("0.0000009"),
        gross_budget=Decimal("1000"),
        price_y=Decimal("10"),
        price_x=Decimal("10"),
    )
    assert not result.accepted
    assert result.reason == RejectReason.DEGENERATE_HEDGE


def test_negative_beta_flips_x_leg_side() -> None:
    result = compute_share_ratio_sizing(
        signal=SignalType.SHORT_SPREAD,
        beta=Decimal("-0.8"),
        gross_budget=Decimal("10000"),
        price_y=Decimal("50"),
        price_x=Decimal("20"),
    )
    assert result.accepted
    assert result.q_y < 0
    assert result.q_x < 0

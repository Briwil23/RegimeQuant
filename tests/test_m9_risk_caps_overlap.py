from __future__ import annotations

from decimal import Decimal

from regimequant.backtesting.m9_contracts import RiskCaps
from regimequant.backtesting.m9_risk import aggregate_security_exposure, reserve_capital, validate_caps


def test_overlap_exposure_aggregation_and_caps() -> None:
    current = {"A": Decimal("5000000"), "B": Decimal("-2000000")}
    delta = {"A": Decimal("3000000"), "C": Decimal("1000000")}
    merged = aggregate_security_exposure(current, delta)
    assert merged["A"] == Decimal("8000000")

    check = validate_caps(security_exposure=merged, equity=Decimal("100000000"), caps=RiskCaps())
    assert check.accepted


def test_cap_reject_and_capital_reservation() -> None:
    exposure = {"A": Decimal("12000000")}
    check = validate_caps(security_exposure=exposure, equity=Decimal("100000000"), caps=RiskCaps())
    assert not check.accepted

    ok, reserved = reserve_capital(free_cash=Decimal("100"), current_reserved=Decimal("90"), amount=Decimal("20"))
    assert not ok
    assert reserved == Decimal("90")

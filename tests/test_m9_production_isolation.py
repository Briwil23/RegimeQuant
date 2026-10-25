from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from regimequant.backtesting.m9_contracts import AuthorizationInput, OrderStatus, RejectReason, RiskCaps, SignalType, SourceMode
from regimequant.backtesting.m9_execution import Stage1Context, authorize_order


def test_production_mode_is_rejected() -> None:
    auth_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    decision = authorize_order(
        AuthorizationInput(
            pair_id="pair_A_B",
            y_security_id="A",
            x_security_id="B",
            signal=SignalType.LONG_SPREAD,
            source_mode=SourceMode.PRODUCTION,
            beta=Decimal("1.2"),
            authorization_time=auth_time,
            decision_session_index=20,
            evidence_available=True,
            provenance_complete=True,
            initial_equity=Decimal("100000000"),
            current_equity=Decimal("100000000"),
            free_cash=Decimal("10000000"),
        ),
        Stage1Context(
            evidence_ready_at=auth_time,
            preliminary_risk_ok=True,
            current_reserved_capital=Decimal("0"),
        ),
        RiskCaps(),
    )
    assert decision.status == OrderStatus.REJECTED
    assert decision.reason == RejectReason.PRODUCTION_DISABLED
    assert decision.policy is None

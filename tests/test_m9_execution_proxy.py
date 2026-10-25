from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from regimequant.backtesting.m9_contracts import AuthorizationInput, OrderStatus, RiskCaps, SignalType, SourceMode
from regimequant.backtesting.m9_execution import Stage1Context, Stage2RiskContext, authorize_order, execute_atomic_at_selected_close


def _auth_input(mode: SourceMode = SourceMode.RESEARCH) -> AuthorizationInput:
    auth_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    return AuthorizationInput(
        pair_id="pair_A_B",
        y_security_id="A",
        x_security_id="B",
        signal=SignalType.LONG_SPREAD,
        source_mode=mode,
        beta=Decimal("1.1"),
        authorization_time=auth_time,
        decision_session_index=12,
        evidence_available=True,
        provenance_complete=True,
        initial_equity=Decimal("100000000"),
        current_equity=Decimal("100000000"),
        free_cash=Decimal("50000000"),
    )


def test_authorize_then_atomic_fill() -> None:
    input_data = _auth_input()
    decision = authorize_order(
        input_data,
        Stage1Context(evidence_ready_at=input_data.authorization_time, preliminary_risk_ok=True, current_reserved_capital=Decimal("0")),
        RiskCaps(),
    )
    assert decision.status == OrderStatus.AUTHORIZED
    assert decision.policy is not None

    fill = execute_atomic_at_selected_close(
        policy=decision.policy,
        selected_session_index=13,
        close_price_y=Decimal("100"),
        close_price_x=Decimal("80"),
        stage2_risk=Stage2RiskContext(
            current_security_exposure={},
            free_cash_before_fill=Decimal("50000000"),
            reservation_released=decision.policy.gross_budget,
        ),
        caps=RiskCaps(),
        current_equity=Decimal("100000000"),
    )
    assert fill.status == OrderStatus.FILLED
    assert len(fill.legs) == 2


def test_missing_selected_close_cancels_atomic_order() -> None:
    input_data = _auth_input()
    decision = authorize_order(
        input_data,
        Stage1Context(evidence_ready_at=input_data.authorization_time, preliminary_risk_ok=True, current_reserved_capital=Decimal("0")),
        RiskCaps(),
    )
    assert decision.policy is not None
    fill = execute_atomic_at_selected_close(
        policy=decision.policy,
        selected_session_index=13,
        close_price_y=None,
        close_price_x=Decimal("80"),
        stage2_risk=Stage2RiskContext(
            current_security_exposure={},
            free_cash_before_fill=Decimal("50000000"),
            reservation_released=decision.policy.gross_budget,
        ),
        caps=RiskCaps(),
        current_equity=Decimal("100000000"),
    )
    assert fill.status == OrderStatus.CANCELLED
    assert len(fill.legs) == 0

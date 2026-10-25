from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from regimequant.backtesting.m9_contracts import (
    AtomicFill,
    AuthorizationDecision,
    AuthorizationInput,
    FillLeg,
    FrozenSizingPolicy,
    OrderStatus,
    RejectReason,
    RiskCaps,
    SignalType,
    SourceMode,
)
from regimequant.backtesting.m9_risk import pair_budget_from_initial_equity, reserve_capital
from regimequant.backtesting.m9_risk import aggregate_security_exposure, validate_caps
from regimequant.backtesting.m9_temporal import forbid_same_session_fill
from regimequant.backtesting.m9_trade_policy import compute_share_ratio_sizing


@dataclass(frozen=True, slots=True)
class Stage1Context:
    evidence_ready_at: datetime
    preliminary_risk_ok: bool
    current_reserved_capital: Decimal
    lot_step_y: Decimal = Decimal("0.0001")
    lot_step_x: Decimal = Decimal("0.0001")


@dataclass(frozen=True, slots=True)
class Stage2RiskContext:
    current_security_exposure: dict[str, Decimal]
    free_cash_before_fill: Decimal
    reservation_released: Decimal
    current_pair_gross: Decimal = Decimal("0")


def authorize_order(input_data: AuthorizationInput, stage1: Stage1Context, caps: RiskCaps) -> AuthorizationDecision:
    if input_data.source_mode == SourceMode.PRODUCTION:
        return AuthorizationDecision(OrderStatus.REJECTED, RejectReason.PRODUCTION_DISABLED, Decimal("0"), None)
    if input_data.signal not in {SignalType.LONG_SPREAD, SignalType.SHORT_SPREAD}:
        return AuthorizationDecision(OrderStatus.REJECTED, RejectReason.INVALID_SIGNAL, Decimal("0"), None)
    if not input_data.evidence_available or stage1.evidence_ready_at > input_data.authorization_time:
        return AuthorizationDecision(OrderStatus.REJECTED, RejectReason.MISSING_PROVENANCE, Decimal("0"), None)
    if not input_data.provenance_complete:
        return AuthorizationDecision(OrderStatus.REJECTED, RejectReason.MISSING_PROVENANCE, Decimal("0"), None)
    if not stage1.preliminary_risk_ok:
        return AuthorizationDecision(OrderStatus.REJECTED, RejectReason.RISK_PRECHECK_FAILED, Decimal("0"), None)

    gross_budget = pair_budget_from_initial_equity(input_data.initial_equity, caps)
    direction = 1 if input_data.signal == SignalType.LONG_SPREAD else -1
    policy = FrozenSizingPolicy(
        pair_id=input_data.pair_id,
        y_security_id=input_data.y_security_id,
        x_security_id=input_data.x_security_id,
        signal=input_data.signal,
        direction=direction,
        beta=input_data.beta,
        gross_budget=gross_budget,
        lot_step_y=stage1.lot_step_y,
        lot_step_x=stage1.lot_step_x,
        created_at=input_data.authorization_time,
        decision_session_index=input_data.decision_session_index,
    )

    ok, updated_reserved = reserve_capital(
        free_cash=input_data.free_cash,
        current_reserved=stage1.current_reserved_capital,
        amount=gross_budget,
    )
    if not ok:
        return AuthorizationDecision(OrderStatus.REJECTED, RejectReason.INSUFFICIENT_CAPITAL, Decimal("0"), None)

    return AuthorizationDecision(OrderStatus.AUTHORIZED, None, updated_reserved, policy)


def execute_atomic_at_selected_close(
    *,
    policy: FrozenSizingPolicy,
    selected_session_index: int,
    close_price_y: Decimal | None,
    close_price_x: Decimal | None,
    stage2_risk: Stage2RiskContext,
    caps: RiskCaps,
    current_equity: Decimal,
) -> AtomicFill:
    if not forbid_same_session_fill(
        decision_session_index=policy.decision_session_index,
        selected_session_index=selected_session_index,
    ):
        return AtomicFill(policy.pair_id, selected_session_index, tuple(), OrderStatus.REJECTED, RejectReason.CHRONOLOGY_INVALID)

    if close_price_y is None or close_price_x is None:
        return AtomicFill(policy.pair_id, selected_session_index, tuple(), OrderStatus.CANCELLED, RejectReason.MISSING_SELECTED_CLOSE)

    sizing = compute_share_ratio_sizing(
        signal=policy.signal,
        beta=policy.beta,
        gross_budget=policy.gross_budget,
        price_y=close_price_y,
        price_x=close_price_x,
        lot_step_y=policy.lot_step_y,
        lot_step_x=policy.lot_step_x,
    )
    if not sizing.accepted:
        return AtomicFill(policy.pair_id, selected_session_index, tuple(), OrderStatus.REJECTED, sizing.reason)

    delta_exposure = {
        policy.y_security_id: sizing.q_y * close_price_y,
        policy.x_security_id: sizing.q_x * close_price_x,
    }
    projected_exposure = aggregate_security_exposure(stage2_risk.current_security_exposure, delta_exposure)
    cap_check = validate_caps(security_exposure=projected_exposure, equity=current_equity, caps=caps)
    if not cap_check.accepted:
        return AtomicFill(policy.pair_id, selected_session_index, tuple(), OrderStatus.REJECTED, RejectReason.RISK_STAGE2_FAILED)
    projected_pair_gross = stage2_risk.current_pair_gross + sizing.gross_notional
    if projected_pair_gross > caps.gross_cap_fraction * current_equity:
        return AtomicFill(policy.pair_id, selected_session_index, tuple(), OrderStatus.REJECTED, RejectReason.RISK_STAGE2_FAILED)

    entry_cash_outflow = Decimal("0")
    if sizing.q_y > 0:
        entry_cash_outflow += sizing.q_y.copy_abs() * close_price_y
    if sizing.q_x > 0:
        entry_cash_outflow += sizing.q_x.copy_abs() * close_price_x
    if stage2_risk.free_cash_before_fill + stage2_risk.reservation_released < entry_cash_outflow:
        return AtomicFill(policy.pair_id, selected_session_index, tuple(), OrderStatus.REJECTED, RejectReason.INSUFFICIENT_CAPITAL)

    legs = (
        FillLeg(security_id=policy.y_security_id, quantity=sizing.q_y, close_price=close_price_y),
        FillLeg(security_id=policy.x_security_id, quantity=sizing.q_x, close_price=close_price_x),
    )
    return AtomicFill(policy.pair_id, selected_session_index, legs, OrderStatus.FILLED, None)

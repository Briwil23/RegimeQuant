from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum


DECIMAL_0 = Decimal("0")
DECIMAL_1 = Decimal("1")
BETA_MIN_ABS = Decimal("0.000001")


class SourceMode(str, Enum):
    RESEARCH = "replay"
    PRODUCTION = "production"


class SignalType(str, Enum):
    LONG_SPREAD = "LONG_SPREAD_CANDIDATE"
    SHORT_SPREAD = "SHORT_SPREAD_CANDIDATE"
    NEUTRAL = "NEUTRAL"
    ABSTAIN = "ABSTAIN"


class OrderIntent(str, Enum):
    FORCED_EXIT = "FORCED_EXIT"
    SCHEDULED_EXIT = "SCHEDULED_EXIT"
    OPPOSING_CLOSE = "OPPOSING_CLOSE"
    NEW_ENTRY = "NEW_ENTRY"


class OrderStatus(str, Enum):
    AUTHORIZED = "AUTHORIZED"
    REJECTED = "REJECTED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"


class RejectReason(str, Enum):
    PRODUCTION_DISABLED = "PRODUCTION_DISABLED"
    INVALID_SIGNAL = "INVALID_SIGNAL"
    MISSING_PROVENANCE = "MISSING_PROVENANCE"
    DEGENERATE_HEDGE = "DEGENERATE_HEDGE"
    INVALID_PRICE = "INVALID_PRICE"
    INVALID_BUDGET = "INVALID_BUDGET"
    RISK_PRECHECK_FAILED = "RISK_PRECHECK_FAILED"
    RISK_STAGE2_FAILED = "RISK_STAGE2_FAILED"
    INSUFFICIENT_CAPITAL = "INSUFFICIENT_CAPITAL"
    MISSING_SELECTED_CLOSE = "MISSING_SELECTED_CLOSE"
    CHRONOLOGY_INVALID = "CHRONOLOGY_INVALID"
    SIGNAL_EXPIRED = "SIGNAL_EXPIRED"
    EVIDENCE_UNVERIFIED = "EVIDENCE_UNVERIFIED"


@dataclass(frozen=True, slots=True)
class AuthorizationInput:
    pair_id: str
    y_security_id: str
    x_security_id: str
    signal: SignalType
    source_mode: SourceMode
    beta: Decimal
    authorization_time: datetime
    decision_session_index: int
    evidence_available: bool
    provenance_complete: bool
    initial_equity: Decimal
    current_equity: Decimal
    free_cash: Decimal


@dataclass(frozen=True, slots=True)
class FrozenSizingPolicy:
    pair_id: str
    y_security_id: str
    x_security_id: str
    signal: SignalType
    direction: int
    beta: Decimal
    gross_budget: Decimal
    lot_step_y: Decimal
    lot_step_x: Decimal
    created_at: datetime
    decision_session_index: int


@dataclass(frozen=True, slots=True)
class AuthorizationDecision:
    status: OrderStatus
    reason: RejectReason | None
    reserved_capital: Decimal
    policy: FrozenSizingPolicy | None


@dataclass(frozen=True, slots=True)
class FillLeg:
    security_id: str
    quantity: Decimal
    close_price: Decimal


@dataclass(frozen=True, slots=True)
class AtomicFill:
    pair_id: str
    selected_session_index: int
    legs: tuple[FillLeg, FillLeg]
    status: OrderStatus
    reason: RejectReason | None


@dataclass(frozen=True, slots=True)
class Stage2Context:
    selected_session_index: int
    close_price_y: Decimal | None
    close_price_x: Decimal | None
    free_cash: Decimal
    current_equity: Decimal


@dataclass(frozen=True, slots=True)
class RiskCaps:
    pair_budget_fraction: Decimal = Decimal("0.02")
    gross_cap_fraction: Decimal = Decimal("1.50")
    net_cap_fraction: Decimal = Decimal("0.30")
    security_cap_fraction: Decimal = Decimal("0.10")


@dataclass(frozen=True, slots=True)
class RiskState:
    security_notional: dict[str, Decimal] = field(default_factory=dict)
    gross_notional: Decimal = DECIMAL_0
    net_notional: Decimal = DECIMAL_0

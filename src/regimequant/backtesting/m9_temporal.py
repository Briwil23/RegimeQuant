from __future__ import annotations

from datetime import datetime
from typing import Sequence

from regimequant.backtesting.m9_contracts import OrderIntent


def require_evidence_before_authorization(*, evidence_ready_at: datetime, authorization_time: datetime) -> bool:
    return evidence_ready_at <= authorization_time


def select_next_eligible_session_close(*, decision_session_index: int, eligible_session_indices: Sequence[int]) -> int:
    for session_idx in eligible_session_indices:
        if session_idx > decision_session_index:
            return session_idx
    raise ValueError("no eligible session after decision session")


def validate_submission_before_close(*, submitted_at: datetime, selected_close_event_time: datetime) -> bool:
    return submitted_at < selected_close_event_time


def forbid_same_session_fill(*, decision_session_index: int, selected_session_index: int) -> bool:
    return selected_session_index > decision_session_index


def cancellation_on_missing_selected_close(*, has_close_y: bool, has_close_x: bool) -> bool:
    return not (has_close_y and has_close_x)


def holding_exit_session(*, entry_session_index: int, completed_holding_sessions: int = 5) -> int:
    if completed_holding_sessions <= 0:
        raise ValueError("completed_holding_sessions must be positive")
    return entry_session_index + completed_holding_sessions + 1


def deterministic_order_priority(intent: OrderIntent) -> int:
    ordering = {
        OrderIntent.FORCED_EXIT: 0,
        OrderIntent.SCHEDULED_EXIT: 1,
        OrderIntent.OPPOSING_CLOSE: 2,
        OrderIntent.NEW_ENTRY: 3,
    }
    return ordering[intent]


def deterministic_tiebreak_key(*, pair_id: str, authorization_timestamp: datetime, signal_identity: str) -> tuple[int, str, str]:
    return (int(authorization_timestamp.timestamp()), pair_id, signal_identity)

from __future__ import annotations

from regimequant.backtesting.m9_temporal import holding_exit_session, select_next_eligible_session_close


def test_holding_exit_on_sixth_subsequent_session() -> None:
    assert holding_exit_session(entry_session_index=100, completed_holding_sessions=5) == 106


def test_holiday_gaps_are_skipped_by_eligible_sessions() -> None:
    decision_session = 10
    eligible = [11, 14, 15]
    assert select_next_eligible_session_close(decision_session_index=decision_session, eligible_session_indices=eligible) == 11

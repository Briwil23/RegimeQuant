from __future__ import annotations

from datetime import datetime, timedelta, timezone

from regimequant.backtesting.m9_temporal import (
    cancellation_on_missing_selected_close,
    forbid_same_session_fill,
    require_evidence_before_authorization,
    select_next_eligible_session_close,
    validate_submission_before_close,
)


def test_temporal_ordering_guards() -> None:
    now = datetime.now(tz=timezone.utc)
    assert require_evidence_before_authorization(evidence_ready_at=now, authorization_time=now + timedelta(seconds=1))
    assert not require_evidence_before_authorization(evidence_ready_at=now + timedelta(seconds=2), authorization_time=now)


def test_no_same_session_fill_and_next_eligible_selection() -> None:
    selected = select_next_eligible_session_close(decision_session_index=10, eligible_session_indices=[8, 10, 11, 14])
    assert selected == 11
    assert forbid_same_session_fill(decision_session_index=10, selected_session_index=11)
    assert not forbid_same_session_fill(decision_session_index=10, selected_session_index=10)


def test_submission_and_missing_close_rules() -> None:
    now = datetime.now(tz=timezone.utc)
    assert validate_submission_before_close(submitted_at=now, selected_close_event_time=now + timedelta(minutes=1))
    assert cancellation_on_missing_selected_close(has_close_y=False, has_close_x=True)
    assert not cancellation_on_missing_selected_close(has_close_y=True, has_close_x=True)

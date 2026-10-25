from __future__ import annotations

import pandas as pd
import pytest

from regimequant.signals.m8_calendar import M8CalendarError, TradingCalendarEvidence


def test_calendar_requires_explicit_ordered_unique_sessions() -> None:
    with pytest.raises(M8CalendarError):
        TradingCalendarEvidence(
            calendar_id="XNYS",
            calendar_version="1",
            sessions=(pd.Timestamp("2024-01-03", tz="UTC"), pd.Timestamp("2024-01-02", tz="UTC")),
            session_sequence_digest="d",
        )


def test_calendar_session_count_uses_declared_sequence_not_weekdays() -> None:
    sessions = (
        pd.Timestamp("2024-01-02", tz="UTC"),
        pd.Timestamp("2024-01-03", tz="UTC"),
        pd.Timestamp("2024-01-05", tz="UTC"),
    )
    calendar = TradingCalendarEvidence(
        calendar_id="XNYS",
        calendar_version="1",
        sessions=sessions,
        session_sequence_digest="d",
    )
    assert calendar.completed_sessions_after(sessions[0], sessions[2]) == 2

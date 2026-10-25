from __future__ import annotations

import pandas as pd

from regimequant.signals.m8_calendar import TradingCalendarEvidence, discovery_is_expired


def test_discovery_expiry_boundary_20_vs_21_sessions() -> None:
    sessions = tuple(pd.date_range("2024-01-02", periods=30, freq="B", tz="UTC"))
    calendar = TradingCalendarEvidence(
        calendar_id="XNYS",
        calendar_version="1",
        sessions=sessions,
        session_sequence_digest="d",
    )
    discovery = sessions[0]
    at_twenty = sessions[20]
    at_twenty_one = sessions[21]
    assert discovery_is_expired(
        calendar=calendar,
        discovery_observation_session=discovery,
        decision_observation_session=at_twenty,
        max_completed_sessions=20,
    ) is False
    assert discovery_is_expired(
        calendar=calendar,
        discovery_observation_session=discovery,
        decision_observation_session=at_twenty_one,
        max_completed_sessions=20,
    ) is True

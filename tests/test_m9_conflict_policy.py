from __future__ import annotations

from datetime import datetime, timezone

from regimequant.backtesting.m9_contracts import OrderIntent
from regimequant.backtesting.m9_temporal import deterministic_order_priority, deterministic_tiebreak_key


def test_event_priority_ordering() -> None:
    intents = [
        OrderIntent.NEW_ENTRY,
        OrderIntent.FORCED_EXIT,
        OrderIntent.OPPOSING_CLOSE,
        OrderIntent.SCHEDULED_EXIT,
    ]
    ordered = sorted(intents, key=deterministic_order_priority)
    assert ordered == [
        OrderIntent.FORCED_EXIT,
        OrderIntent.SCHEDULED_EXIT,
        OrderIntent.OPPOSING_CLOSE,
        OrderIntent.NEW_ENTRY,
    ]


def test_tiebreaker_is_deterministic() -> None:
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    key1 = deterministic_tiebreak_key(pair_id="pair_A_B", authorization_timestamp=ts, signal_identity="abc")
    key2 = deterministic_tiebreak_key(pair_id="pair_A_B", authorization_timestamp=ts, signal_identity="abc")
    assert key1 == key2

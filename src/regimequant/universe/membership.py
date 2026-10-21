from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import pandas as pd

from .contracts import AuthorityConflictError, MembershipEvent


def _coerce_event(event: MembershipEvent | dict[str, Any]) -> MembershipEvent:
    if isinstance(event, MembershipEvent):
        return event
    if isinstance(event, dict):
        return MembershipEvent(**event)
    raise TypeError(f"membership events must be MembershipEvent or dict, got {type(event)!r}")


def _normalize_decision_timestamp(value: Any) -> pd.Timestamp:
    if value is None:
        raise ValueError("decision_timestamp is required")
    ts = pd.Timestamp(value)
    if ts.tzinfo is None or ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _event_lineage(events: list[MembershipEvent]) -> list[MembershipEvent]:
    by_id = {event.event_id: event for event in events}
    if not events:
        return []
    ordered = sorted(events, key=lambda event: (event.effective_at, event.verified_at, event.event_id))
    lineage: list[MembershipEvent] = []
    seen_ids: set[str] = set()
    for event in ordered:
        if event.event_id in seen_ids:
            continue
        lineage.append(event)
        seen_ids.add(event.event_id)
        for superseded_id in event.supersedes:
            if superseded_id in by_id and superseded_id not in seen_ids:
                lineage.append(by_id[superseded_id])
                seen_ids.add(superseded_id)
    return sorted(lineage, key=lambda event: (event.effective_at, event.verified_at, event.event_id))


def build_membership_history(
    events: Iterable[MembershipEvent | dict[str, Any]],
    *,
    decision_timestamp: Any,
    allow_inferred: bool = False,
) -> dict[str, dict[str, Any]]:
    """Resolve the admissible membership history for a point-in-time decision.

    The ledger is preserved as immutable event history. Resolution at T only inspects the
    admissible subset where verified_at <= T and effective_at <= T. Explicit supersession
    chains are respected, but later independent events remain in the ledger and do not erase
    earlier as-of-T decisions.
    """

    decision_ts = _normalize_decision_timestamp(decision_timestamp)
    admissible: list[MembershipEvent] = []
    for raw_event in events:
        event = _coerce_event(raw_event)
        if event.effective_at > decision_ts:
            continue
        if event.verified_at > decision_ts:
            continue
        if event.is_inferred and not allow_inferred:
            continue
        admissible.append(event)

    by_ticker: dict[str, list[MembershipEvent]] = defaultdict(list)
    for event in admissible:
        by_ticker[event.ticker].append(event)

    result: dict[str, dict[str, Any]] = {}
    for ticker, grouped in by_ticker.items():
        ordered = sorted(grouped, key=lambda item: (item.effective_at, item.verified_at, item.event_id))
        by_id = {event.event_id: event for event in ordered}

        supersession_sources: dict[str, str] = {}
        supersession_targets: set[str] = set()
        for event in ordered:
            for superseded_id in event.supersedes:
                if superseded_id in by_id:
                    supersession_targets.add(superseded_id)
                    supersession_sources[superseded_id] = event.event_id

        if supersession_targets:
            final_event_id = max(
                (supersession_sources.get(event_id, event_id) for event_id in supersession_targets),
                key=lambda item: (by_id[item].effective_at, by_id[item].verified_at, item),
            )
            chosen = by_id[final_event_id]
        else:
            unique_security_ids = {event.security_id for event in ordered}
            if len(unique_security_ids) > 1:
                same_timestamp = [event for event in ordered if event.effective_at == ordered[-1].effective_at]
                if len({event.security_id for event in same_timestamp}) > 1:
                    raise AuthorityConflictError(
                        f"conflicting authority for ticker {ticker!r} at decision {decision_ts.isoformat()}"
                    )
                chosen = ordered[0]
            else:
                chosen = ordered[-1]

        result[ticker] = {
            "ticker": chosen.ticker,
            "security_id": chosen.security_id,
            "status": chosen.status,
            "effective_at": chosen.effective_at,
            "verified_at": chosen.verified_at,
            "source": chosen.source,
            "authority": chosen.authority,
            "event_id": chosen.event_id,
            "is_inferred": chosen.is_inferred,
            "evidence_digest": chosen.evidence_digest,
            "ledger": [
                {
                    "event_id": event.event_id,
                    "security_id": event.security_id,
                    "ticker": event.ticker,
                    "effective_at": event.effective_at.isoformat(),
                    "verified_at": event.verified_at.isoformat(),
                    "status": event.status,
                    "source": event.source,
                    "authority": event.authority,
                    "supersedes": event.supersedes,
                }
                for event in _event_lineage(ordered)
            ],
        }

    return result


def resolve_membership_events(
    events: Iterable[MembershipEvent | dict[str, Any]],
    *,
    decision_timestamp: Any,
    allow_inferred: bool = False,
) -> dict[str, dict[str, Any]]:
    return build_membership_history(
        events,
        decision_timestamp=decision_timestamp,
        allow_inferred=allow_inferred,
    )


__all__ = ["build_membership_history", "resolve_membership_events"]

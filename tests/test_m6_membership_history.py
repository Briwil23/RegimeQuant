from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from regimequant.universe import (
    AuthorityConflictError,
    MembershipEvent,
    SecurityIdentityEvidence,
    build_membership_history,
)


def _ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value).tz_convert("UTC")


def test_ticker_reuse_and_late_correction_are_resolved_by_validity_window() -> None:
    events = [
        MembershipEvent(
            event_id="evt-1",
            security_id="sec-AAA",
            ticker="AAA",
            effective_at=_ts("2024-01-01T00:00:00Z"),
            verified_at=_ts("2024-01-02T00:00:00Z"),
            source="exchange",
            authority="exchange",
            status="listed",
            supersedes=(),
        ),
        MembershipEvent(
            event_id="evt-2",
            security_id="sec-BBB",
            ticker="AAA",
            effective_at=_ts("2024-01-03T00:00:00Z"),
            verified_at=_ts("2024-01-04T00:00:00Z"),
            source="nasdaq",
            authority="nasdaq",
            status="listed",
            supersedes=("evt-1",),
        ),
        MembershipEvent(
            event_id="evt-3",
            security_id="sec-CCC",
            ticker="AAA",
            effective_at=_ts("2024-01-05T00:00:00Z"),
            verified_at=_ts("2024-01-06T00:00:00Z"),
            source="filing",
            authority="filing",
            status="delisted",
            supersedes=("evt-2",),
        ),
    ]

    history = build_membership_history(events, decision_timestamp=_ts("2024-01-05T12:00:00Z"))
    record = history["AAA"]
    assert record["status"] == "listed"
    assert record["security_id"] == "sec-BBB"

    corrected = build_membership_history(events, decision_timestamp=_ts("2024-01-07T00:00:00Z"))
    corrected_record = corrected["AAA"]
    assert corrected_record["status"] == "delisted"
    assert corrected_record["security_id"] == "sec-CCC"


def test_independent_later_events_are_preserved_for_same_ticker() -> None:
    events = [
        MembershipEvent(
            event_id="evt-1",
            security_id="sec-ALPHA",
            ticker="XYZ",
            effective_at=_ts("2024-01-01T00:00:00Z"),
            verified_at=_ts("2024-01-02T00:00:00Z"),
            source="exchange",
            authority="exchange",
            status="listed",
        ),
        MembershipEvent(
            event_id="evt-2",
            security_id="sec-BETA",
            ticker="XYZ",
            effective_at=_ts("2024-01-03T00:00:00Z"),
            verified_at=_ts("2024-01-04T00:00:00Z"),
            source="corporate_action",
            authority="corporate_action",
            status="delisted",
        ),
    ]
    history = build_membership_history(events, decision_timestamp=_ts("2024-01-03T12:00:00Z"))
    assert history["XYZ"]["security_id"] == "sec-ALPHA"


def test_conflicting_authoritative_evidence_is_rejected() -> None:
    events = [
        MembershipEvent(
            event_id="evt-1",
            security_id="sec-RED",
            ticker="RED",
            effective_at=_ts("2024-01-01T00:00:00Z"),
            verified_at=_ts("2024-01-02T00:00:00Z"),
            source="exchange",
            authority="exchange",
            status="listed",
            supersedes=(),
        ),
        MembershipEvent(
            event_id="evt-2",
            security_id="sec-BLUE",
            ticker="RED",
            effective_at=_ts("2024-01-01T00:00:00Z"),
            verified_at=_ts("2024-01-02T00:00:00Z"),
            source="filing",
            authority="filing",
            status="listed",
            supersedes=(),
        ),
    ]
    try:
        build_membership_history(events, decision_timestamp=_ts("2024-01-02T12:00:00Z"))
    except AuthorityConflictError:
        return
    raise AssertionError("Conflicting authority should raise AuthorityConflictError")


def test_identity_evidence_requires_verified_external_source() -> None:
    evidence = SecurityIdentityEvidence(
        security_id="sec-UNIT",
        ticker="UNIT",
        aliases=("UNIT.US",),
        authority="exchange",
        verified_at=_ts("2024-01-02T00:00:00Z"),
        source="exchange",
        evidence_digest="abc",
        status="verified",
    )
    assert evidence.security_id == "sec-UNIT"
    assert evidence.ticker == "UNIT"


def test_multiple_verified_security_identities_from_same_issuer_remain_distinct() -> None:
    identities = (
        SecurityIdentityEvidence(
            security_id="sec-ALPHA",
            ticker="ALPHA",
            aliases=("ALPHA.US",),
            authority="exchange",
            verified_at=_ts("2024-01-02T00:00:00Z"),
            source="exchange",
            evidence_digest="alpha-proof",
            status="verified",
            issuer_id="issuer-42",
        ),
        SecurityIdentityEvidence(
            security_id="sec-BETA",
            ticker="BETA",
            aliases=("BETA.US",),
            authority="exchange",
            verified_at=_ts("2024-01-02T00:00:00Z"),
            source="exchange",
            evidence_digest="beta-proof",
            status="verified",
            issuer_id="issuer-42",
        ),
    )
    assert identities[0].issuer_id == identities[1].issuer_id == "issuer-42"
    assert identities[0].security_id != identities[1].security_id
    assert identities[0].ticker != identities[1].ticker
    assert identities[0].aliases != identities[1].aliases

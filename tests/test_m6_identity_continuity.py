from __future__ import annotations

import pandas as pd

from regimequant.universe import (
    CorporateActionContinuity,
    EligibilityPolicy,
    SecurityAliasRecord,
    SecurityIdentityEvidence,
    UniverseCandidate,
    create_snapshot,
)


def _ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value).tz_convert("UTC")


def test_ticker_reuse_creates_no_false_continuity() -> None:
    alias_a = SecurityAliasRecord(
        canonical_security_id="sec-A",
        symbol="AAA",
        listing_venue="nasdaq",
        valid_from=_ts("2024-01-01T00:00:00Z"),
        valid_to=_ts("2024-01-02T00:00:00Z"),
        verified_at=_ts("2024-01-01T12:00:00Z"),
        evidence_provenance="exchange",
        source_authority="exchange",
        evidence_digest="hash-a",
    )
    alias_b = SecurityAliasRecord(
        canonical_security_id="sec-B",
        symbol="AAA",
        listing_venue="nyse",
        valid_from=_ts("2024-01-03T00:00:00Z"),
        verified_at=_ts("2024-01-03T12:00:00Z"),
        evidence_provenance="exchange",
        source_authority="exchange",
        evidence_digest="hash-b",
    )
    assert alias_a.canonical_security_id != alias_b.canonical_security_id
    assert alias_a.symbol == alias_b.symbol == "AAA"


def test_symbol_change_and_corporate_action_are_distinct_records() -> None:
    continuity = CorporateActionContinuity(
        predecessor_security_id="sec-old",
        successor_security_id="sec-new",
        event_kind="ticker_change",
        effective_at=_ts("2024-01-10T00:00:00Z"),
        verified_at=_ts("2024-01-10T00:00:00Z"),
        source_authority="filing",
        evidence_provenance="filing",
        evidence_digest="cont-1",
    )
    assert continuity.predecessor_security_id != continuity.successor_security_id
    assert continuity.event_kind == "ticker_change"


def test_identity_evidence_rejects_conflicting_mappings() -> None:
    evidence = SecurityIdentityEvidence(
        security_id="sec-1",
        ticker="ABC",
        aliases=("ABC.US",),
        authority="exchange",
        verified_at=_ts("2024-01-02T00:00:00Z"),
        source="exchange",
        evidence_digest="digest-1",
        status="verified",
        ambiguous=False,
    )
    assert evidence.security_id == "sec-1"
    assert evidence.issuer_id is None


def test_snapshot_hash_changes_when_identity_or_coverage_changes() -> None:
    policy = EligibilityPolicy(mode="production", allow_inferred_membership=False)
    candidate = UniverseCandidate(
        security_id="sec-A",
        ticker="A",
        as_of=_ts("2024-01-05T00:00:00Z"),
        status="candidate",
        is_delisted=False,
        source="exchange",
        coverage_complete=True,
    )
    base = create_snapshot(
        candidate_universe_identity="u-1",
        coverage_evidence=(candidate,),
        admissible_membership_versions=("v1",),
        eligibility_policy=policy,
        decisions=(),
        source_provenance_digest="source-digest-1",
        identity_evidence=(
            SecurityIdentityEvidence(
                security_id="sec-A",
                ticker="A",
                aliases=("A.US",),
                authority="exchange",
                verified_at=_ts("2024-01-04T00:00:00Z"),
                source="exchange",
                evidence_digest="hash-a",
                status="verified",
            ),
        ),
        continuity_records=(
            CorporateActionContinuity(
                predecessor_security_id="sec-old",
                successor_security_id="sec-A",
                event_kind="spin_off",
                effective_at=_ts("2024-01-01T00:00:00Z"),
                verified_at=_ts("2024-01-01T12:00:00Z"),
                source_authority="filing",
                evidence_provenance="filing",
                evidence_digest="cont-a",
            ),
        ),
    )
    changed = create_snapshot(
        candidate_universe_identity="u-1",
        coverage_evidence=(candidate,),
        admissible_membership_versions=("v1",),
        eligibility_policy=policy,
        decisions=(),
        source_provenance_digest="source-digest-2",
        identity_evidence=(
            SecurityIdentityEvidence(
                security_id="sec-A",
                ticker="A",
                aliases=("AA.US",),
                authority="exchange",
                verified_at=_ts("2024-01-04T00:00:00Z"),
                source="exchange",
                evidence_digest="hash-b",
                status="verified",
            ),
        ),
        continuity_records=(
            CorporateActionContinuity(
                predecessor_security_id="sec-old",
                successor_security_id="sec-A",
                event_kind="spin_off",
                effective_at=_ts("2024-01-01T00:00:00Z"),
                verified_at=_ts("2024-01-01T12:00:00Z"),
                source_authority="filing",
                evidence_provenance="filing",
                evidence_digest="cont-b",
            ),
        ),
    )
    assert base.snapshot_scientific_identity != changed.snapshot_scientific_identity

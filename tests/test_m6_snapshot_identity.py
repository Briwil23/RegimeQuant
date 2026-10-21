from __future__ import annotations

import pandas as pd

from regimequant.universe import (
    EligibilityDecision,
    EligibilityPolicy,
    UniverseCandidate,
    UniverseSnapshot,
    compute_snapshot_identity,
    create_snapshot,
)


def _ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value).tz_convert("UTC")


def test_snapshot_identity_is_deterministic_and_order_invariant() -> None:
    policy = EligibilityPolicy(mode="production", allow_inferred_membership=False)
    candidate_a = UniverseCandidate(
        security_id="sec-A",
        ticker="A",
        as_of=_ts("2024-01-05T00:00:00Z"),
        status="candidate",
        is_delisted=False,
        source="exchange",
        coverage_complete=True,
    )
    candidate_b = UniverseCandidate(
        security_id="sec-B",
        ticker="B",
        as_of=_ts("2024-01-05T00:00:00Z"),
        status="candidate",
        is_delisted=False,
        source="exchange",
        coverage_complete=True,
    )
    decision_a = EligibilityDecision(
        security_id="sec-A",
        decision_timestamp=_ts("2024-01-05T12:00:00Z"),
        eligibility_status="eligible",
        evidence_available=True,
        snapshot_identity="snap-a",
        provenance_digest="hash-a",
        coverage_digest="cov-a",
        exclusion_reasons=(),
        mode="production",
    )
    decision_b = EligibilityDecision(
        security_id="sec-B",
        decision_timestamp=_ts("2024-01-05T12:00:00Z"),
        eligibility_status="eligible",
        evidence_available=True,
        snapshot_identity="snap-b",
        provenance_digest="hash-b",
        coverage_digest="cov-b",
        exclusion_reasons=(),
        mode="production",
    )
    snapshot_one = create_snapshot(
        candidate_universe_identity="u-1",
        coverage_evidence=(candidate_a, candidate_b),
        admissible_membership_versions=("v1", "v2"),
        eligibility_policy=policy,
        decisions=(decision_a, decision_b),
        source_provenance_digest="source-digest",
    )
    snapshot_two = create_snapshot(
        candidate_universe_identity="u-1",
        coverage_evidence=(candidate_b, candidate_a),
        admissible_membership_versions=("v2", "v1"),
        eligibility_policy=policy,
        decisions=(decision_b, decision_a),
        source_provenance_digest="source-digest",
    )
    assert compute_snapshot_identity(snapshot_one) == compute_snapshot_identity(snapshot_two)
    assert snapshot_one.snapshot_scientific_identity == snapshot_two.snapshot_scientific_identity


def test_snapshot_excludes_runtime_metadata() -> None:
    policy = EligibilityPolicy(mode="production", allow_inferred_membership=False)
    snapshot = create_snapshot(
        candidate_universe_identity="u-1",
        coverage_evidence=(),
        admissible_membership_versions=("v1",),
        eligibility_policy=policy,
        decisions=(),
        source_provenance_digest="source-digest",
    )
    payload = snapshot.scientific_payload
    assert "generated_at" not in payload
    assert "runtime_metadata" not in payload
    assert "input_order" not in payload

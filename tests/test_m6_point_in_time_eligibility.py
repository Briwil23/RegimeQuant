from __future__ import annotations

import pandas as pd

from regimequant.universe import (
    EligibilityPolicy,
    ExclusionReason,
    MembershipEvent,
    UniverseCandidate,
    create_snapshot,
    evaluate_eligibility,
)


def _ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value).tz_convert("UTC")


def test_production_mode_requires_observed_verified_membership() -> None:
    candidates = [
        UniverseCandidate(
            security_id="sec-A",
            ticker="A",
            as_of=_ts("2024-01-05T00:00:00Z"),
            status="candidate",
            is_delisted=False,
            source="exchange",
            coverage_complete=True,
        )
    ]
    membership = [
        MembershipEvent(
            event_id="m-1",
            security_id="sec-A",
            ticker="A",
            effective_at=_ts("2024-01-03T00:00:00Z"),
            verified_at=_ts("2024-01-04T00:00:00Z"),
            source="exchange",
            authority="exchange",
            status="listed",
        )
    ]
    decision = evaluate_eligibility(
        candidates,
        membership,
        policy=EligibilityPolicy(mode="production", allow_inferred_membership=False),
        decision_timestamp=_ts("2024-01-05T12:00:00Z"),
    )[0]
    assert decision.eligibility_status == "eligible"
    assert decision.exclusion_reasons == ()


def test_research_mode_allows_explicitly_labeled_inference_only() -> None:
    candidates = [
        UniverseCandidate(
            security_id="sec-I",
            ticker="I",
            as_of=_ts("2024-01-05T00:00:00Z"),
            status="candidate",
            is_delisted=False,
            source="research",
            coverage_complete=True,
        )
    ]
    policy = EligibilityPolicy(mode="research", allow_inferred_membership=True)
    decision = evaluate_eligibility(
        candidates,
        [],
        policy=policy,
        decision_timestamp=_ts("2024-01-05T12:00:00Z"),
    )[0]
    assert decision.eligibility_status == "inferred"
    assert "research_inferred" in decision.exclusion_reasons or decision.exclusion_reasons == ()


def test_partial_coverage_is_flagged_and_excluded_in_production_with_explicit_research_override() -> None:
    candidate = UniverseCandidate(
        security_id="sec-P",
        ticker="P",
        as_of=_ts("2024-01-05T00:00:00Z"),
        status="candidate",
        is_delisted=True,
        source="exchange",
        coverage_complete=False,
    )

    production_decision = evaluate_eligibility(
        [candidate],
        [],
        policy=EligibilityPolicy(mode="production", allow_inferred_membership=False),
        decision_timestamp=_ts("2024-01-05T12:00:00Z"),
    )[0]
    assert production_decision.coverage_status == "partial"
    assert production_decision.eligibility_status == "ineligible"
    assert ExclusionReason.INCOMPLETE_CANDIDATE_COVERAGE in production_decision.exclusion_reasons
    assert ExclusionReason.INSUFFICIENT_VERIFIED_COVERAGE in production_decision.exclusion_reasons
    assert production_decision.research_override is False

    research_decision = evaluate_eligibility(
        [candidate],
        [],
        policy=EligibilityPolicy(mode="research", allow_inferred_membership=True),
        decision_timestamp=_ts("2024-01-05T12:00:00Z"),
    )[0]
    assert research_decision.coverage_status == "partial"
    assert research_decision.research_override is True
    assert research_decision.eligibility_status == "ineligible"

    snapshot = create_snapshot(
        candidate_universe_identity="u-1",
        coverage_evidence=(candidate,),
        admissible_membership_versions=("v1",),
        eligibility_policy=EligibilityPolicy(mode="production", allow_inferred_membership=False),
        decisions=(production_decision,),
        source_provenance_digest="source-digest",
    )
    assert snapshot.scientific_payload["decisions"][0]["coverage_status"] == "partial"
    assert snapshot.scientific_payload["decisions"][0]["coverage_status"] != "complete"

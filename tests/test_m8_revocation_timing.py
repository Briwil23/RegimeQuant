from __future__ import annotations

import pandas as pd

from regimequant.cointegration.contracts import CointegrationCandidate, DecisionEnvelope
from regimequant.cointegration.identity import compute_canonical_pair_id
from regimequant.signals.m8_calendar import TradingCalendarEvidence
from regimequant.signals.m8_contracts import M8AbstainReason, M8DiscoveryFamilyEvidence, M8FamilyMember, M8RevocationRecord
from regimequant.signals.m8_eligibility import evaluate_eligibility
from regimequant.signals.m8_identity import compute_family_digest


def _family(mode: str = "replay") -> M8DiscoveryFamilyEvidence:
    canonical = compute_canonical_pair_id("AAPL", "MSFT")
    member = M8FamilyMember("h1", canonical, "AAPL", "MSFT", True, raw_p_value=0.001)
    discovery_window_start = pd.Timestamp("2024-01-01", tz="UTC")
    discovery_window_end = pd.Timestamp("2024-01-10", tz="UTC")
    discovery_observation_session = pd.Timestamp("2024-01-05", tz="UTC")
    discovery_decision_cutoff = pd.Timestamp("2024-01-06", tz="UTC")
    payload = {
        "family_definition_version": "v1",
        "family_id": "fam-1",
        "universe_snapshot_identity": "u1",
        "discovery_window_start": str(discovery_window_start),
        "discovery_window_end": str(discovery_window_end),
        "discovery_observation_session": str(discovery_observation_session),
        "discovery_decision_cutoff": str(discovery_decision_cutoff),
        "dataset_identity": "d1",
        "statistical_config_digest": "s1",
        "exchange_calendar_id": "XNYS",
        "exchange_calendar_version": "1",
        "exchange_session_sequence_digest": "c1",
        "provenance_mode": mode,
        "members": [
            (
                member.hypothesis_id,
                member.canonical_pair_id,
                member.valid_for_discovery,
                member.raw_p_value,
                member.invalid_reason,
            )
        ],
    }
    return M8DiscoveryFamilyEvidence(
        family_definition_version="v1",
        family_id="fam-1",
        universe_snapshot_identity="u1",
        discovery_window_start=discovery_window_start,
        discovery_window_end=discovery_window_end,
        discovery_observation_session=discovery_observation_session,
        discovery_available_at=pd.Timestamp("2024-01-06", tz="UTC"),
        discovery_decision_cutoff=discovery_decision_cutoff,
        dataset_identity="d1",
        statistical_config_digest="s1",
        declared_membership_ordered=(member,),
        supplied_by_adjusted_values=(0.001,),
        predeclared_committed_at=pd.Timestamp("2024-01-04", tz="UTC"),
        outcomes_available_at=pd.Timestamp("2024-01-06", tz="UTC"),
        exchange_calendar_id="XNYS",
        exchange_calendar_version="1",
        exchange_session_sequence_digest="c1",
        provenance_mode=mode,
        availability_verified=True,
        identity_verified=True,
        continuity_verified=True,
        evidence_provenance_digest="p1",
        family_digest=compute_family_digest(payload),
    )


def test_known_and_effective_revocation_rejects_eligibility() -> None:
    canonical = compute_canonical_pair_id("AAPL", "MSFT")
    candidate = CointegrationCandidate("AAPL", "MSFT", "pair_AAPL_MSFT", source_mode="replay")
    cutoff = pd.Timestamp("2024-01-20", tz="UTC")
    envelope = DecisionEnvelope("replay", cutoff, True, "replay_src", "e", "d", replay_reference="r1")
    calendar = TradingCalendarEvidence(
        "XNYS",
        "1",
        tuple(pd.date_range("2024-01-02", periods=30, freq="B", tz="UTC")),
        "cd",
    )
    revocation = M8RevocationRecord(
        canonical_pair_id=canonical,
        authority_id="auth",
        effective_at_session=pd.Timestamp("2024-01-15", tz="UTC"),
        known_at=pd.Timestamp("2024-01-18", tz="UTC"),
        coverage_complete=True,
        reason_code="issuer-change",
        revocation_digest="r1",
    )
    decision = evaluate_eligibility(
        candidate=candidate,
        family_evidence=_family(),
        decision_envelope=envelope,
        calendar=calendar,
        decision_observation_session=pd.Timestamp("2024-01-19", tz="UTC"),
        revocations=(revocation,),
    )
    assert decision.eligible is False
    assert decision.reason == M8AbstainReason.REVOKED_EFFECTIVE


def test_production_mode_abstains_without_independent_revocation_authority() -> None:
    candidate = CointegrationCandidate("AAPL", "MSFT", "pair_AAPL_MSFT", source_mode="production")
    cutoff = pd.Timestamp("2024-01-20", tz="UTC")
    envelope = DecisionEnvelope("production", cutoff, True, "prod", "e", "d")
    calendar = TradingCalendarEvidence(
        "XNYS",
        "1",
        tuple(pd.date_range("2024-01-02", periods=30, freq="B", tz="UTC")),
        "cd",
    )
    decision = evaluate_eligibility(
        candidate=candidate,
        family_evidence=_family("production"),
        decision_envelope=envelope,
        calendar=calendar,
        decision_observation_session=pd.Timestamp("2024-01-19", tz="UTC"),
        revocations=(),
    )
    assert decision.eligible is False
    assert decision.reason == M8AbstainReason.PREDECLARATION_UNPROVEN

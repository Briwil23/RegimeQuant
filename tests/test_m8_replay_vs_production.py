from __future__ import annotations

import pandas as pd

from regimequant.cointegration.contracts import CointegrationCandidate, DecisionEnvelope
from regimequant.cointegration.identity import compute_canonical_pair_id
from regimequant.signals.m8_calendar import TradingCalendarEvidence
from regimequant.signals.m8_contracts import M8AbstainReason, M8DiscoveryFamilyEvidence, M8FamilyMember
from regimequant.signals.m8_eligibility import evaluate_eligibility
from regimequant.signals.m8_family_evidence import recompute_by_adjusted_values
from regimequant.signals.m8_identity import compute_family_digest


def _family(mode: str) -> M8DiscoveryFamilyEvidence:
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


def test_mode_mismatch_rejected() -> None:
    candidate = CointegrationCandidate("AAPL", "MSFT", "pair_AAPL_MSFT", source_mode="replay")
    envelope = DecisionEnvelope("production", pd.Timestamp("2024-01-20", tz="UTC"), True, "prod", "e", "d")
    calendar = TradingCalendarEvidence("XNYS", "1", tuple(pd.date_range("2024-01-02", periods=30, freq="B", tz="UTC")), "cd")
    decision = evaluate_eligibility(
        candidate=candidate,
        family_evidence=_family("production"),
        decision_envelope=envelope,
        calendar=calendar,
        decision_observation_session=pd.Timestamp("2024-01-19", tz="UTC"),
    )
    assert decision.eligible is False
    assert decision.reason == M8AbstainReason.MODE_MISMATCH


def test_raw_significant_but_by_not_significant_is_rejected() -> None:
    canonical = compute_canonical_pair_id("AAPL", "MSFT")
    members = tuple(
        M8FamilyMember(f"h{i:02d}", f"pair_{i:02d}", f"A{i:02d}", f"B{i:02d}", True, raw_p_value=0.02)
        for i in range(1, 50)
    ) + (M8FamilyMember("h_target", canonical, "AAPL", "MSFT", True, raw_p_value=0.01),)
    members = tuple(sorted(members, key=lambda m: m.canonical_pair_id))
    discovery_window_start = pd.Timestamp("2024-01-01", tz="UTC")
    discovery_window_end = pd.Timestamp("2024-01-10", tz="UTC")
    discovery_observation_session = pd.Timestamp("2024-01-05", tz="UTC")
    discovery_decision_cutoff = pd.Timestamp("2024-01-06", tz="UTC")
    payload = {
        "family_definition_version": "v1",
        "family_id": "fam-big",
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
        "provenance_mode": "replay",
        "members": [
            (
                m.hypothesis_id,
                m.canonical_pair_id,
                m.valid_for_discovery,
                m.raw_p_value,
                m.invalid_reason,
            )
            for m in members
        ],
    }
    evidence_template = M8DiscoveryFamilyEvidence(
        family_definition_version="v1",
        family_id="fam-big",
        universe_snapshot_identity="u1",
        discovery_window_start=discovery_window_start,
        discovery_window_end=discovery_window_end,
        discovery_observation_session=discovery_observation_session,
        discovery_available_at=pd.Timestamp("2024-01-06", tz="UTC"),
        discovery_decision_cutoff=discovery_decision_cutoff,
        dataset_identity="d1",
        statistical_config_digest="s1",
        declared_membership_ordered=members,
        supplied_by_adjusted_values=tuple(1.0 for _ in members),
        predeclared_committed_at=pd.Timestamp("2024-01-04", tz="UTC"),
        outcomes_available_at=pd.Timestamp("2024-01-06", tz="UTC"),
        exchange_calendar_id="XNYS",
        exchange_calendar_version="1",
        exchange_session_sequence_digest="c1",
        provenance_mode="replay",
        availability_verified=True,
        identity_verified=True,
        continuity_verified=True,
        evidence_provenance_digest="p1",
        family_digest=compute_family_digest(payload),
    )
    evidence = M8DiscoveryFamilyEvidence(
        family_definition_version=evidence_template.family_definition_version,
        family_id=evidence_template.family_id,
        universe_snapshot_identity=evidence_template.universe_snapshot_identity,
        discovery_window_start=evidence_template.discovery_window_start,
        discovery_window_end=evidence_template.discovery_window_end,
        discovery_observation_session=evidence_template.discovery_observation_session,
        discovery_available_at=evidence_template.discovery_available_at,
        discovery_decision_cutoff=evidence_template.discovery_decision_cutoff,
        dataset_identity=evidence_template.dataset_identity,
        statistical_config_digest=evidence_template.statistical_config_digest,
        declared_membership_ordered=evidence_template.declared_membership_ordered,
        supplied_by_adjusted_values=recompute_by_adjusted_values(evidence_template),
        predeclared_committed_at=evidence_template.predeclared_committed_at,
        outcomes_available_at=evidence_template.outcomes_available_at,
        exchange_calendar_id=evidence_template.exchange_calendar_id,
        exchange_calendar_version=evidence_template.exchange_calendar_version,
        exchange_session_sequence_digest=evidence_template.exchange_session_sequence_digest,
        provenance_mode=evidence_template.provenance_mode,
        availability_verified=evidence_template.availability_verified,
        identity_verified=evidence_template.identity_verified,
        continuity_verified=evidence_template.continuity_verified,
        evidence_provenance_digest=evidence_template.evidence_provenance_digest,
        family_digest=evidence_template.family_digest,
    )
    candidate = CointegrationCandidate("AAPL", "MSFT", "pair_AAPL_MSFT", p_value=0.01, adjusted_p_value=0.01, source_mode="replay")
    envelope = DecisionEnvelope("replay", pd.Timestamp("2024-01-20", tz="UTC"), True, "replay_src", "e", "d", replay_reference="r1")
    calendar = TradingCalendarEvidence("XNYS", "1", tuple(pd.date_range("2024-01-02", periods=60, freq="B", tz="UTC")), "cd")
    decision = evaluate_eligibility(
        candidate=candidate,
        family_evidence=evidence,
        decision_envelope=envelope,
        calendar=calendar,
        decision_observation_session=pd.Timestamp("2024-01-19", tz="UTC"),
    )
    assert decision.eligible is False
    assert decision.reason == M8AbstainReason.BY_NOT_SIGNIFICANT


def test_claimed_adjusted_values_must_match_recomputed_by() -> None:
    candidate = CointegrationCandidate("AAPL", "MSFT", "pair_AAPL_MSFT", source_mode="replay")
    envelope = DecisionEnvelope("replay", pd.Timestamp("2024-01-20", tz="UTC"), True, "replay_src", "e", "d", replay_reference="r1")
    calendar = TradingCalendarEvidence("XNYS", "1", tuple(pd.date_range("2024-01-02", periods=30, freq="B", tz="UTC")), "cd")
    bad = _family("replay")
    mutated = M8DiscoveryFamilyEvidence(
        family_definition_version=bad.family_definition_version,
        family_id=bad.family_id,
        universe_snapshot_identity=bad.universe_snapshot_identity,
        discovery_window_start=bad.discovery_window_start,
        discovery_window_end=bad.discovery_window_end,
        discovery_observation_session=bad.discovery_observation_session,
        discovery_available_at=bad.discovery_available_at,
        discovery_decision_cutoff=bad.discovery_decision_cutoff,
        dataset_identity=bad.dataset_identity,
        statistical_config_digest=bad.statistical_config_digest,
        declared_membership_ordered=bad.declared_membership_ordered,
        supplied_by_adjusted_values=(0.9,),
        predeclared_committed_at=bad.predeclared_committed_at,
        outcomes_available_at=bad.outcomes_available_at,
        exchange_calendar_id=bad.exchange_calendar_id,
        exchange_calendar_version=bad.exchange_calendar_version,
        exchange_session_sequence_digest=bad.exchange_session_sequence_digest,
        provenance_mode=bad.provenance_mode,
        availability_verified=bad.availability_verified,
        identity_verified=bad.identity_verified,
        continuity_verified=bad.continuity_verified,
        evidence_provenance_digest=bad.evidence_provenance_digest,
        family_digest=bad.family_digest,
    )
    decision = evaluate_eligibility(
        candidate=candidate,
        family_evidence=mutated,
        decision_envelope=envelope,
        calendar=calendar,
        decision_observation_session=pd.Timestamp("2024-01-19", tz="UTC"),
    )
    assert decision.eligible is False
    assert decision.reason == M8AbstainReason.BY_INCONSISTENT_EVIDENCE


def test_production_fail_closed_without_independent_provenance_authority() -> None:
    candidate = CointegrationCandidate("AAPL", "MSFT", "pair_AAPL_MSFT", source_mode="production")
    envelope = DecisionEnvelope("production", pd.Timestamp("2024-01-20", tz="UTC"), True, "prod", "e", "d")
    calendar = TradingCalendarEvidence("XNYS", "1", tuple(pd.date_range("2024-01-02", periods=30, freq="B", tz="UTC")), "cd")
    decision = evaluate_eligibility(
        candidate=candidate,
        family_evidence=_family("production"),
        decision_envelope=envelope,
        calendar=calendar,
        decision_observation_session=pd.Timestamp("2024-01-19", tz="UTC"),
    )
    assert decision.eligible is False
    assert decision.reason == M8AbstainReason.PREDECLARATION_UNPROVEN

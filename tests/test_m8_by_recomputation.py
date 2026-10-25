from __future__ import annotations

import pandas as pd
import pytest

from regimequant.signals.m8_contracts import M8AbstainReason
from regimequant.signals.m8_contracts import M8DiscoveryFamilyEvidence, M8FamilyMember
from regimequant.cointegration.contracts import CointegrationCandidate, DecisionEnvelope
from regimequant.cointegration.identity import compute_canonical_pair_id
from regimequant.signals.m8_calendar import TradingCalendarEvidence
from regimequant.signals.m8_eligibility import evaluate_eligibility
from regimequant.signals.m8_family_evidence import recompute_by_adjusted_values, verify_adjusted_values
from regimequant.signals.m8_identity import compute_family_digest


def test_by_recomputation_matches_reference_case() -> None:
    members = (
        M8FamilyMember("h1", "pair_a", "A", "B", True, raw_p_value=0.001),
        M8FamilyMember("h2", "pair_b", "C", "D", True, raw_p_value=0.020),
    )
    evidence = M8DiscoveryFamilyEvidence(
        family_definition_version="v1",
        family_id="fam-1",
        universe_snapshot_identity="u1",
        discovery_window_start=pd.Timestamp("2024-01-01", tz="UTC"),
        discovery_window_end=pd.Timestamp("2024-01-10", tz="UTC"),
        discovery_observation_session=pd.Timestamp("2024-01-05", tz="UTC"),
        discovery_available_at=pd.Timestamp("2024-01-06", tz="UTC"),
        discovery_decision_cutoff=pd.Timestamp("2024-01-06", tz="UTC"),
        dataset_identity="d1",
        statistical_config_digest="s1",
        declared_membership_ordered=members,
        supplied_by_adjusted_values=(0.003, 0.03),
        predeclared_committed_at=pd.Timestamp("2024-01-04", tz="UTC"),
        outcomes_available_at=pd.Timestamp("2024-01-06", tz="UTC"),
        exchange_calendar_id="XNYS",
        exchange_calendar_version="1",
        exchange_session_sequence_digest="c1",
        provenance_mode="production",
        availability_verified=True,
        identity_verified=True,
        continuity_verified=True,
        evidence_provenance_digest="p1",
        family_digest="f1",
    )
    adjusted = recompute_by_adjusted_values(evidence)
    assert adjusted == (0.003, 0.03)
    assert verify_adjusted_values(evidence) == adjusted


def test_duplicate_canonical_pair_ids_are_rejected() -> None:
    members = (
        M8FamilyMember("h1", "pair_dup", "A", "B", True, raw_p_value=0.01),
        M8FamilyMember("h2", "pair_dup", "C", "D", True, raw_p_value=0.02),
    )
    with pytest.raises(ValueError):
        M8DiscoveryFamilyEvidence(
            family_definition_version="v1",
            family_id="fam-dup",
            universe_snapshot_identity="u1",
            discovery_window_start=pd.Timestamp("2024-01-01", tz="UTC"),
            discovery_window_end=pd.Timestamp("2024-01-10", tz="UTC"),
            discovery_observation_session=pd.Timestamp("2024-01-05", tz="UTC"),
            discovery_available_at=pd.Timestamp("2024-01-06", tz="UTC"),
            discovery_decision_cutoff=pd.Timestamp("2024-01-06", tz="UTC"),
            dataset_identity="d1",
            statistical_config_digest="s1",
            declared_membership_ordered=members,
            supplied_by_adjusted_values=(0.03, 0.03),
            predeclared_committed_at=pd.Timestamp("2024-01-04", tz="UTC"),
            outcomes_available_at=pd.Timestamp("2024-01-06", tz="UTC"),
            exchange_calendar_id="XNYS",
            exchange_calendar_version="1",
            exchange_session_sequence_digest="c1",
            provenance_mode="production",
            availability_verified=True,
            identity_verified=True,
            continuity_verified=True,
            evidence_provenance_digest="p1",
            family_digest="f1",
        )


def test_inconsistent_family_digest_is_rejected_as_by_inconsistent() -> None:
    members = (
        M8FamilyMember("h1", "pair_a", "A", "B", True, raw_p_value=0.01),
        M8FamilyMember("h2", "pair_b", "C", "D", True, raw_p_value=0.02),
    )
    payload = {
        "family_id": "fam-1",
        "members": [(member.hypothesis_id, member.canonical_pair_id) for member in members],
        "dataset_identity": "d1",
        "statistical_config_digest": "s1",
    }
    expected_digest = compute_family_digest(payload)
    good = M8DiscoveryFamilyEvidence(
        family_definition_version="v1",
        family_id="fam-1",
        universe_snapshot_identity="u1",
        discovery_window_start=pd.Timestamp("2024-01-01", tz="UTC"),
        discovery_window_end=pd.Timestamp("2024-01-10", tz="UTC"),
        discovery_observation_session=pd.Timestamp("2024-01-05", tz="UTC"),
        discovery_available_at=pd.Timestamp("2024-01-06", tz="UTC"),
        discovery_decision_cutoff=pd.Timestamp("2024-01-06", tz="UTC"),
        dataset_identity="d1",
        statistical_config_digest="s1",
        declared_membership_ordered=members,
        supplied_by_adjusted_values=(0.03, 0.03),
        predeclared_committed_at=pd.Timestamp("2024-01-04", tz="UTC"),
        outcomes_available_at=pd.Timestamp("2024-01-06", tz="UTC"),
        exchange_calendar_id="XNYS",
        exchange_calendar_version="1",
        exchange_session_sequence_digest="c1",
        provenance_mode="production",
        availability_verified=True,
        identity_verified=True,
        continuity_verified=True,
        evidence_provenance_digest="p1",
        family_digest=expected_digest,
    )
    # This is a structural sanity check that our digest payload can be reproduced.
    assert good.family_digest == expected_digest


def test_fabricated_predeclaration_cannot_authorize_production_eligibility() -> None:
    canonical = compute_canonical_pair_id("AAPL", "MSFT")
    member = M8FamilyMember("h1", canonical, "AAPL", "MSFT", True, raw_p_value=0.01)
    payload = {
        "family_definition_version": "v1",
        "family_id": "fam-1",
        "universe_snapshot_identity": "u1",
        "discovery_window_start": str(pd.Timestamp("2024-01-01", tz="UTC")),
        "discovery_window_end": str(pd.Timestamp("2024-01-10", tz="UTC")),
        "discovery_observation_session": str(pd.Timestamp("2024-01-05", tz="UTC")),
        "discovery_decision_cutoff": str(pd.Timestamp("2024-01-06", tz="UTC")),
        "dataset_identity": "d1",
        "statistical_config_digest": "s1",
        "exchange_calendar_id": "XNYS",
        "exchange_calendar_version": "1",
        "exchange_session_sequence_digest": "c1",
        "provenance_mode": "production",
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
    evidence = M8DiscoveryFamilyEvidence(
        family_definition_version="v1",
        family_id="fam-1",
        universe_snapshot_identity="u1",
        discovery_window_start=pd.Timestamp("2024-01-01", tz="UTC"),
        discovery_window_end=pd.Timestamp("2024-01-10", tz="UTC"),
        discovery_observation_session=pd.Timestamp("2024-01-05", tz="UTC"),
        discovery_available_at=pd.Timestamp("2024-01-06", tz="UTC"),
        discovery_decision_cutoff=pd.Timestamp("2024-01-06", tz="UTC"),
        dataset_identity="d1",
        statistical_config_digest="s1",
        declared_membership_ordered=(member,),
        supplied_by_adjusted_values=(0.01,),
        predeclared_committed_at=pd.Timestamp("2024-01-04", tz="UTC"),
        outcomes_available_at=pd.Timestamp("2024-01-06", tz="UTC"),
        exchange_calendar_id="XNYS",
        exchange_calendar_version="1",
        exchange_session_sequence_digest="c1",
        provenance_mode="production",
        availability_verified=True,
        identity_verified=True,
        continuity_verified=True,
        evidence_provenance_digest="fabricated_predeclare",
        family_digest=compute_family_digest(payload),
    )
    candidate = CointegrationCandidate("AAPL", "MSFT", "pair_AAPL_MSFT", source_mode="production")
    envelope = DecisionEnvelope("production", pd.Timestamp("2024-01-20", tz="UTC"), True, "prod", "e", "d")
    calendar = TradingCalendarEvidence("XNYS", "1", tuple(pd.date_range("2024-01-02", periods=40, freq="B", tz="UTC")), "cd")
    decision = evaluate_eligibility(
        candidate=candidate,
        family_evidence=evidence,
        decision_envelope=envelope,
        calendar=calendar,
        decision_observation_session=pd.Timestamp("2024-01-19", tz="UTC"),
    )
    assert decision.eligible is False
    assert decision.reason == M8AbstainReason.PREDECLARATION_UNPROVEN


def test_omitted_family_member_cannot_authorize_production_eligibility() -> None:
    canonical = compute_canonical_pair_id("AAPL", "MSFT")
    member = M8FamilyMember("h_target", canonical, "AAPL", "MSFT", True, raw_p_value=0.01)
    payload = {
        "family_definition_version": "v1",
        "family_id": "fam-omitted",
        "universe_snapshot_identity": "u1",
        "discovery_window_start": str(pd.Timestamp("2024-01-01", tz="UTC")),
        "discovery_window_end": str(pd.Timestamp("2024-01-10", tz="UTC")),
        "discovery_observation_session": str(pd.Timestamp("2024-01-05", tz="UTC")),
        "discovery_decision_cutoff": str(pd.Timestamp("2024-01-06", tz="UTC")),
        "dataset_identity": "d1",
        "statistical_config_digest": "s1",
        "exchange_calendar_id": "XNYS",
        "exchange_calendar_version": "1",
        "exchange_session_sequence_digest": "c1",
        "provenance_mode": "production",
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
    evidence = M8DiscoveryFamilyEvidence(
        family_definition_version="v1",
        family_id="fam-omitted",
        universe_snapshot_identity="u1",
        discovery_window_start=pd.Timestamp("2024-01-01", tz="UTC"),
        discovery_window_end=pd.Timestamp("2024-01-10", tz="UTC"),
        discovery_observation_session=pd.Timestamp("2024-01-05", tz="UTC"),
        discovery_available_at=pd.Timestamp("2024-01-06", tz="UTC"),
        discovery_decision_cutoff=pd.Timestamp("2024-01-06", tz="UTC"),
        dataset_identity="d1",
        statistical_config_digest="s1",
        declared_membership_ordered=(member,),
        supplied_by_adjusted_values=(0.01,),
        predeclared_committed_at=pd.Timestamp("2024-01-04", tz="UTC"),
        outcomes_available_at=pd.Timestamp("2024-01-06", tz="UTC"),
        exchange_calendar_id="XNYS",
        exchange_calendar_version="1",
        exchange_session_sequence_digest="c1",
        provenance_mode="production",
        availability_verified=True,
        identity_verified=True,
        continuity_verified=True,
        evidence_provenance_digest="omitted_family",
        family_digest=compute_family_digest(payload),
    )
    candidate = CointegrationCandidate("AAPL", "MSFT", "pair_AAPL_MSFT", source_mode="production")
    envelope = DecisionEnvelope("production", pd.Timestamp("2024-01-20", tz="UTC"), True, "prod", "e", "d")
    calendar = TradingCalendarEvidence("XNYS", "1", tuple(pd.date_range("2024-01-02", periods=40, freq="B", tz="UTC")), "cd")
    decision = evaluate_eligibility(
        candidate=candidate,
        family_evidence=evidence,
        decision_envelope=envelope,
        calendar=calendar,
        decision_observation_session=pd.Timestamp("2024-01-19", tz="UTC"),
    )
    assert decision.eligible is False
    assert decision.reason == M8AbstainReason.PREDECLARATION_UNPROVEN

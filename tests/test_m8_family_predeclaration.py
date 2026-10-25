from __future__ import annotations

import pandas as pd
import pytest

from regimequant.signals.m8_contracts import M8DiscoveryFamilyEvidence, M8FamilyMember
from regimequant.signals.m8_family_evidence import M8FamilyEvidenceError, validate_predeclaration


def test_family_predeclaration_must_precede_outcomes() -> None:
    member = M8FamilyMember(
        hypothesis_id="h1",
        canonical_pair_id="pair_a",
        canonical_security_id_left="A",
        canonical_security_id_right="B",
        valid_for_discovery=True,
        raw_p_value=0.01,
    )
    with pytest.raises(ValueError):
        M8DiscoveryFamilyEvidence(
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
            predeclared_committed_at=pd.Timestamp("2024-01-07", tz="UTC"),
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


def test_validate_predeclaration_rejects_late_commitment() -> None:
    member = M8FamilyMember(
        hypothesis_id="h1",
        canonical_pair_id="pair_a",
        canonical_security_id_left="A",
        canonical_security_id_right="B",
        valid_for_discovery=True,
        raw_p_value=0.01,
    )
    evidence = M8DiscoveryFamilyEvidence(
        family_definition_version="v1",
        family_id="fam-1",
        universe_snapshot_identity="u1",
        discovery_window_start=pd.Timestamp("2024-01-01", tz="UTC"),
        discovery_window_end=pd.Timestamp("2024-01-10", tz="UTC"),
        discovery_observation_session=pd.Timestamp("2024-01-05", tz="UTC"),
        discovery_available_at=pd.Timestamp("2024-01-06", tz="UTC"),
        discovery_decision_cutoff=pd.Timestamp("2024-01-07", tz="UTC"),
        dataset_identity="d1",
        statistical_config_digest="s1",
        declared_membership_ordered=(member,),
        supplied_by_adjusted_values=(0.01,),
        predeclared_committed_at=pd.Timestamp("2024-01-06", tz="UTC"),
        outcomes_available_at=pd.Timestamp("2024-01-07", tz="UTC"),
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
    validate_predeclaration(evidence)
    with pytest.raises(ValueError):
        M8DiscoveryFamilyEvidence(
            family_definition_version=evidence.family_definition_version,
            family_id=evidence.family_id,
            universe_snapshot_identity=evidence.universe_snapshot_identity,
            discovery_window_start=evidence.discovery_window_start,
            discovery_window_end=evidence.discovery_window_end,
            discovery_observation_session=evidence.discovery_observation_session,
            discovery_available_at=evidence.discovery_available_at,
            discovery_decision_cutoff=evidence.discovery_decision_cutoff,
            dataset_identity=evidence.dataset_identity,
            statistical_config_digest=evidence.statistical_config_digest,
            declared_membership_ordered=evidence.declared_membership_ordered,
            supplied_by_adjusted_values=evidence.supplied_by_adjusted_values,
            predeclared_committed_at=pd.Timestamp("2024-01-08", tz="UTC"),
            outcomes_available_at=evidence.outcomes_available_at,
            exchange_calendar_id=evidence.exchange_calendar_id,
            exchange_calendar_version=evidence.exchange_calendar_version,
            exchange_session_sequence_digest=evidence.exchange_session_sequence_digest,
            provenance_mode=evidence.provenance_mode,
            availability_verified=evidence.availability_verified,
            identity_verified=evidence.identity_verified,
            continuity_verified=evidence.continuity_verified,
            evidence_provenance_digest=evidence.evidence_provenance_digest,
            family_digest=evidence.family_digest,
        )

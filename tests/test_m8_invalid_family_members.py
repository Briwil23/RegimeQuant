from __future__ import annotations

import pandas as pd

from regimequant.signals.m8_contracts import M8DiscoveryFamilyEvidence, M8FamilyMember
from regimequant.signals.m8_family_evidence import recompute_by_adjusted_values


def test_invalid_hypothesis_remains_in_family_size_with_p_one() -> None:
    members = (
        M8FamilyMember("h1", "pair_a", "A", "B", True, raw_p_value=0.001),
        M8FamilyMember("h2", "pair_b", "C", "D", False, invalid_reason="missing-data"),
        M8FamilyMember("h3", "pair_c", "E", "F", True, raw_p_value=0.020),
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
        supplied_by_adjusted_values=(0.0055, 1.0, 0.055),
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
    assert len(adjusted) == 3
    assert adjusted[1] == 1.0

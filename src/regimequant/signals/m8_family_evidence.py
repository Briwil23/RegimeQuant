from __future__ import annotations

import math

from regimequant.signals.m8_contracts import M8DiscoveryFamilyEvidence, M8FamilyMember
from regimequant.signals.m8_identity import compute_family_digest


class M8FamilyEvidenceError(ValueError):
    """Raised when discovery family evidence is incomplete or inconsistent."""


TOLERANCE = 1e-12


def _member_computational_p(member: M8FamilyMember) -> float:
    if member.valid_for_discovery:
        if member.raw_p_value is None:
            raise M8FamilyEvidenceError("valid discovery members require raw_p_value")
        p = float(member.raw_p_value)
        if not math.isfinite(p) or not 0.0 <= p <= 1.0:
            raise M8FamilyEvidenceError("raw p-values must be finite and lie in [0, 1]")
        return p
    return 1.0


def recompute_by_adjusted_values(evidence: M8DiscoveryFamilyEvidence) -> tuple[float, ...]:
    members = evidence.declared_membership_ordered
    if len(members) == 0:
        raise M8FamilyEvidenceError("family evidence must include at least one member")

    p_values = [_member_computational_p(member) for member in members]
    m = len(p_values)
    harmonic = sum(1.0 / float(i) for i in range(1, m + 1))
    order = sorted(range(m), key=lambda idx: p_values[idx])

    adjusted = [1.0] * m
    cumulative = 1.0
    for rank_index in range(m - 1, -1, -1):
        original_idx = order[rank_index]
        rank = float(rank_index + 1)
        candidate = p_values[original_idx] * m * harmonic / rank
        cumulative = min(cumulative, candidate)
        adjusted[original_idx] = min(1.0, cumulative)

    return tuple(float(value) for value in adjusted)


def verify_adjusted_values(evidence: M8DiscoveryFamilyEvidence) -> tuple[float, ...]:
    expected = recompute_by_adjusted_values(evidence)
    supplied = evidence.supplied_by_adjusted_values
    if len(expected) != len(supplied):
        raise M8FamilyEvidenceError("supplied adjusted values length does not match declared family size")
    for idx, (lhs, rhs) in enumerate(zip(expected, supplied)):
        if abs(float(lhs) - float(rhs)) > TOLERANCE:
            raise M8FamilyEvidenceError(f"BY adjusted values mismatch at index {idx}")
    return expected


def verify_family_digest(evidence: M8DiscoveryFamilyEvidence) -> str:
    payload = {
        "family_definition_version": evidence.family_definition_version,
        "family_id": evidence.family_id,
        "universe_snapshot_identity": evidence.universe_snapshot_identity,
        "discovery_window_start": str(evidence.discovery_window_start),
        "discovery_window_end": str(evidence.discovery_window_end),
        "discovery_observation_session": str(evidence.discovery_observation_session),
        "discovery_decision_cutoff": str(evidence.discovery_decision_cutoff),
        "dataset_identity": evidence.dataset_identity,
        "statistical_config_digest": evidence.statistical_config_digest,
        "exchange_calendar_id": evidence.exchange_calendar_id,
        "exchange_calendar_version": evidence.exchange_calendar_version,
        "exchange_session_sequence_digest": evidence.exchange_session_sequence_digest,
        "provenance_mode": evidence.provenance_mode,
        "members": [
            (
                member.hypothesis_id,
                member.canonical_pair_id,
                member.valid_for_discovery,
                None if member.raw_p_value is None else float(member.raw_p_value),
                member.invalid_reason,
            )
            for member in evidence.declared_membership_ordered
        ],
    }
    expected = compute_family_digest(payload)
    if expected != evidence.family_digest:
        raise M8FamilyEvidenceError("declared family_digest does not match material family payload")
    return expected


def validate_predeclaration(evidence: M8DiscoveryFamilyEvidence) -> None:
    if evidence.predeclared_committed_at > evidence.outcomes_available_at:
        raise M8FamilyEvidenceError("family predeclaration cannot occur after outcomes become available")
    if evidence.predeclared_committed_at > evidence.discovery_decision_cutoff:
        raise M8FamilyEvidenceError("family predeclaration cannot occur after discovery cutoff")


def member_index_by_pair_id(evidence: M8DiscoveryFamilyEvidence, canonical_pair_id: str) -> int:
    for idx, member in enumerate(evidence.declared_membership_ordered):
        if member.canonical_pair_id == canonical_pair_id:
            return idx
    raise M8FamilyEvidenceError("canonical_pair_id is not in the declared discovery family")

from __future__ import annotations

from typing import Any

from .contracts import AuthorityConflictError, EligibilityDecision, MembershipEvent, SecurityIdentityEvidence, UniverseCandidate, UniverseSnapshot


def validate_identity_evidence(evidence: SecurityIdentityEvidence) -> None:
    if evidence.ambiguous:
        raise AuthorityConflictError("ambiguous identity evidence")
    if not evidence.security_id or not evidence.ticker:
        raise ValueError("security_id and ticker are required")
    if evidence.status == "verified" and not evidence.evidence_digest:
        raise ValueError("verified evidence requires a digest")


def validate_membership_event(event: MembershipEvent) -> None:
    if not event.event_id:
        raise ValueError("event_id required")
    if not event.security_id or not event.ticker:
        raise ValueError("security_id and ticker required")
    if event.verified_at > event.effective_at:
        raise ValueError("verified_at must not be after effective_at")


def validate_candidate(candidate: UniverseCandidate) -> None:
    if not candidate.security_id or not candidate.ticker:
        raise ValueError("candidate security_id and ticker are required")
    if candidate.liquidity_metadata and not isinstance(candidate.liquidity_metadata, dict):
        raise TypeError("liquidity_metadata must be a dictionary")


def validate_snapshot(snapshot: UniverseSnapshot) -> None:
    if not snapshot.candidate_universe_identity:
        raise ValueError("candidate_universe_identity is required")
    if "generated_at" in snapshot.scientific_payload:
        raise ValueError("scientific payload must not include generation timestamps")
    if "runtime_metadata" in snapshot.scientific_payload:
        raise ValueError("scientific payload must not include runtime metadata")
    if "input_order" in snapshot.scientific_payload:
        raise ValueError("scientific payload must not include incidental ordering")
    for decision in snapshot.decisions:
        if not isinstance(decision, EligibilityDecision):
            raise TypeError("snapshot decisions must be EligibilityDecision objects")


__all__ = [
    "validate_identity_evidence",
    "validate_membership_event",
    "validate_candidate",
    "validate_snapshot",
]

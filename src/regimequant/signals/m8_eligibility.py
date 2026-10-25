from __future__ import annotations

from dataclasses import dataclass

from regimequant.cointegration.contracts import CointegrationCandidate, DecisionEnvelope
from regimequant.cointegration.identity import compute_canonical_pair_id
from regimequant.signals.m8_calendar import TradingCalendarEvidence, discovery_is_expired
from regimequant.signals.m8_contracts import M8AbstainReason, M8DiscoveryFamilyEvidence, M8RevocationRecord
from regimequant.signals.m8_family_evidence import (
    M8FamilyEvidenceError,
    member_index_by_pair_id,
    validate_predeclaration,
    verify_adjusted_values,
    verify_family_digest,
)
from regimequant.signals.m8_identity import compute_eligibility_digest
from regimequant.signals.m8_validation import M8ValidationError, require_available_by_cutoff, validate_m8_decision_envelope


@dataclass(frozen=True, slots=True)
class M8EligibilityProof:
    canonical_pair_id: str
    family_id: str
    family_digest: str
    by_adjusted_p_value: float
    by_threshold: float
    discovery_observation_session: object
    discovery_available_at: object
    discovery_decision_cutoff: object
    revoked_effective: bool
    revocation_reason: str | None
    source_mode: str
    proof_digest: str


@dataclass(frozen=True, slots=True)
class M8EligibilityDecision:
    eligible: bool
    proof: M8EligibilityProof | None
    reason: M8AbstainReason | None


def _canonical_pair_from_candidate(candidate: CointegrationCandidate) -> str:
    return compute_canonical_pair_id(candidate.instrument_a, candidate.instrument_b)


def _revocation_effective(
    *,
    revocations: tuple[M8RevocationRecord, ...],
    canonical_pair_id: str,
    decision_cutoff,
    decision_session,
    source_mode: str,
) -> tuple[bool, str | None, M8AbstainReason | None]:
    relevant = [record for record in revocations if record.canonical_pair_id == canonical_pair_id]
    if not relevant:
        return False, None, None

    for record in relevant:
        if not record.coverage_complete:
            return False, None, M8AbstainReason.REVOCATION_COVERAGE_UNKNOWN

    known_by_cutoff = [record for record in relevant if record.known_at <= decision_cutoff]
    if source_mode == "production" and any(record.known_at > decision_cutoff for record in relevant):
        return False, None, M8AbstainReason.REVOCATION_COVERAGE_UNKNOWN

    for record in known_by_cutoff:
        if record.effective_at_session <= decision_session:
            return True, record.reason_code, None

    return False, None, None


def evaluate_eligibility(
    *,
    candidate: CointegrationCandidate,
    family_evidence: M8DiscoveryFamilyEvidence,
    decision_envelope: DecisionEnvelope,
    calendar: TradingCalendarEvidence,
    decision_observation_session,
    revocations: tuple[M8RevocationRecord, ...] = (),
    max_completed_sessions: int = 20,
) -> M8EligibilityDecision:
    try:
        validate_m8_decision_envelope(decision_envelope)
    except M8ValidationError:
        return M8EligibilityDecision(eligible=False, proof=None, reason=M8AbstainReason.MODE_MISMATCH)

    if decision_envelope.mode not in {"production", "replay"}:
        return M8EligibilityDecision(eligible=False, proof=None, reason=M8AbstainReason.MODE_MISMATCH)
    if candidate.source_mode != decision_envelope.mode:
        return M8EligibilityDecision(eligible=False, proof=None, reason=M8AbstainReason.MODE_MISMATCH)

    # Fail closed in production: with the frozen M0-M7 interfaces available here, M8 cannot
    # independently substantiate full discovery-family authority, exchange-calendar authority,
    # or revocation-coverage authority end-to-end from authoritative records.
    if decision_envelope.mode == "production":
        return M8EligibilityDecision(eligible=False, proof=None, reason=M8AbstainReason.PREDECLARATION_UNPROVEN)

    try:
        validate_predeclaration(family_evidence)
    except M8FamilyEvidenceError:
        return M8EligibilityDecision(eligible=False, proof=None, reason=M8AbstainReason.PREDECLARATION_UNPROVEN)

    try:
        verify_family_digest(family_evidence)
    except M8FamilyEvidenceError:
        return M8EligibilityDecision(eligible=False, proof=None, reason=M8AbstainReason.BY_INCONSISTENT_EVIDENCE)

    canonical_pair_id = _canonical_pair_from_candidate(candidate)
    if family_evidence.provenance_mode != decision_envelope.mode:
        return M8EligibilityDecision(eligible=False, proof=None, reason=M8AbstainReason.MODE_MISMATCH)

    try:
        require_available_by_cutoff(
            available_at=family_evidence.discovery_available_at,
            decision_cutoff=decision_envelope.cutoff,
            field_name="discovery evidence",
        )
    except M8ValidationError:
        return M8EligibilityDecision(eligible=False, proof=None, reason=M8AbstainReason.EVIDENCE_LATE)

    try:
        member_idx = member_index_by_pair_id(family_evidence, canonical_pair_id)
    except M8FamilyEvidenceError:
        return M8EligibilityDecision(eligible=False, proof=None, reason=M8AbstainReason.FAMILY_INCOMPLETE)

    member = family_evidence.declared_membership_ordered[member_idx]
    if not member.valid_for_discovery:
        return M8EligibilityDecision(eligible=False, proof=None, reason=M8AbstainReason.MEMBER_INVALID)

    try:
        by_values = verify_adjusted_values(family_evidence)
    except M8FamilyEvidenceError:
        return M8EligibilityDecision(eligible=False, proof=None, reason=M8AbstainReason.BY_INCONSISTENT_EVIDENCE)

    adjusted = float(by_values[member_idx])
    if adjusted > 0.05:
        return M8EligibilityDecision(eligible=False, proof=None, reason=M8AbstainReason.BY_NOT_SIGNIFICANT)

    expired = discovery_is_expired(
        calendar=calendar,
        discovery_observation_session=family_evidence.discovery_observation_session,
        decision_observation_session=decision_observation_session,
        max_completed_sessions=max_completed_sessions,
    )
    if expired:
        return M8EligibilityDecision(eligible=False, proof=None, reason=M8AbstainReason.DISCOVERY_STALE)

    revoked_effective, revoked_reason, revocation_abstain_reason = _revocation_effective(
        revocations=revocations,
        canonical_pair_id=canonical_pair_id,
        decision_cutoff=decision_envelope.cutoff,
        decision_session=decision_observation_session,
        source_mode=decision_envelope.mode,
    )
    if revocation_abstain_reason is not None:
        return M8EligibilityDecision(eligible=False, proof=None, reason=revocation_abstain_reason)
    if revoked_effective:
        return M8EligibilityDecision(eligible=False, proof=None, reason=M8AbstainReason.REVOKED_EFFECTIVE)

    digest = compute_eligibility_digest(
        {
            "canonical_pair_id": canonical_pair_id,
            "family_id": family_evidence.family_id,
            "family_digest": family_evidence.family_digest,
            "by_adjusted_p_value": adjusted,
            "by_threshold": 0.05,
            "discovery_observation_session": str(family_evidence.discovery_observation_session),
            "discovery_available_at": str(family_evidence.discovery_available_at),
            "discovery_decision_cutoff": str(family_evidence.discovery_decision_cutoff),
            "source_mode": decision_envelope.mode,
            "revoked_effective": False,
        }
    )

    proof = M8EligibilityProof(
        canonical_pair_id=canonical_pair_id,
        family_id=family_evidence.family_id,
        family_digest=family_evidence.family_digest,
        by_adjusted_p_value=adjusted,
        by_threshold=0.05,
        discovery_observation_session=family_evidence.discovery_observation_session,
        discovery_available_at=family_evidence.discovery_available_at,
        discovery_decision_cutoff=family_evidence.discovery_decision_cutoff,
        revoked_effective=False,
        revocation_reason=revoked_reason,
        source_mode=decision_envelope.mode,
        proof_digest=digest,
    )
    return M8EligibilityDecision(eligible=True, proof=proof, reason=None)

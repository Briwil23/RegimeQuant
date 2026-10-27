from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from regimequant.cointegration.contracts import DecisionEnvelope
from regimequant.signals.m8_contracts import M8AbstainReason
from regimequant.signals.m8_eligibility import M8EligibilityDecision
from regimequant.signals.m8_validation import M8ValidationError, validate_m8_decision_envelope
from regimequant.walkforward.contracts import MetricStatus, ScientificStatus, ValidationResult


class EvidenceAuthorityState(str, Enum):
    LOCALLY_VERIFIED_CAUSAL_TIMING = "LOCALLY_VERIFIED_CAUSAL_TIMING"
    LOCALLY_VERIFIED_IDENTITY = "LOCALLY_VERIFIED_IDENTITY"
    MISSING_EVIDENCE = "MISSING_EVIDENCE"
    LATE_EVIDENCE = "LATE_EVIDENCE"
    UNVERIFIED_FAMILY_AUTHORITY = "UNVERIFIED_FAMILY_AUTHORITY"
    INVALID_IDENTITY = "INVALID_IDENTITY"
    INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"


@dataclass(frozen=True, slots=True)
class EvidenceAuthorityDecision:
    state: EvidenceAuthorityState
    reason_code: str
    detail: str | None
    scientific_status: ScientificStatus
    metric_status: MetricStatus


def classify_evidence_authority(
    *,
    has_evidence: bool,
    causal_timing_verified: bool,
    identity_verified: bool,
    evidence_late: bool,
    family_authority_independently_verified: bool,
    insufficient_history: bool,
    caller_family_completeness_claim: bool,
) -> EvidenceAuthorityDecision:
    if not has_evidence:
        return EvidenceAuthorityDecision(
            state=EvidenceAuthorityState.MISSING_EVIDENCE,
            reason_code="MISSING_EVIDENCE",
            detail="Required evidence artifacts are missing",
            scientific_status="MISSING_CAUSAL_EVIDENCE",
            metric_status="METRICS_MISSING",
        )
    if evidence_late:
        return EvidenceAuthorityDecision(
            state=EvidenceAuthorityState.LATE_EVIDENCE,
            reason_code="LATE_EVIDENCE",
            detail="Evidence became available after stage cutoff",
            scientific_status="MISSING_CAUSAL_EVIDENCE",
            metric_status="METRICS_MISSING",
        )
    if not identity_verified:
        return EvidenceAuthorityDecision(
            state=EvidenceAuthorityState.INVALID_IDENTITY,
            reason_code="INVALID_IDENTITY",
            detail="Identity verification failed",
            scientific_status="INVALID_SCIENTIFIC_IDENTITY",
            metric_status="METRICS_INVALID",
        )
    if insufficient_history:
        return EvidenceAuthorityDecision(
            state=EvidenceAuthorityState.INSUFFICIENT_HISTORY,
            reason_code="INSUFFICIENT_HISTORY",
            detail="Causally eligible sample floor not satisfied",
            scientific_status="INSUFFICIENT_HISTORY",
            metric_status="METRICS_INSUFFICIENT",
        )
    if not causal_timing_verified:
        return EvidenceAuthorityDecision(
            state=EvidenceAuthorityState.MISSING_EVIDENCE,
            reason_code="TIMING_UNVERIFIED",
            detail="Causal timing cannot be locally verified",
            scientific_status="MISSING_CAUSAL_EVIDENCE",
            metric_status="METRICS_MISSING",
        )

    # Caller declarations never become independent family authority.
    if caller_family_completeness_claim and not family_authority_independently_verified:
        return EvidenceAuthorityDecision(
            state=EvidenceAuthorityState.UNVERIFIED_FAMILY_AUTHORITY,
            reason_code="FAMILY_AUTHORITY_UNVERIFIED",
            detail="Caller family-completeness declaration is non-authoritative",
            scientific_status="MISSING_CAUSAL_EVIDENCE",
            metric_status="METRICS_MISSING",
        )
    if family_authority_independently_verified:
        return EvidenceAuthorityDecision(
            state=EvidenceAuthorityState.UNVERIFIED_FAMILY_AUTHORITY,
            reason_code="FAMILY_AUTHORITY_UNVERIFIED",
            detail="Caller-supplied family authority cannot establish independent provenance",
            scientific_status="MISSING_CAUSAL_EVIDENCE",
            metric_status="METRICS_MISSING",
        )
    if not family_authority_independently_verified:
        return EvidenceAuthorityDecision(
            state=EvidenceAuthorityState.UNVERIFIED_FAMILY_AUTHORITY,
            reason_code="FAMILY_AUTHORITY_UNVERIFIED",
            detail="Family authority is unverified under frozen interfaces",
            scientific_status="MISSING_CAUSAL_EVIDENCE",
            metric_status="METRICS_MISSING",
        )


def validate_decision_envelope_timing(envelope: DecisionEnvelope) -> bool:
    try:
        validate_m8_decision_envelope(envelope)
    except M8ValidationError:
        return False
    return True


def classify_from_m8_eligibility(
    *,
    envelope: DecisionEnvelope,
    eligibility: M8EligibilityDecision,
) -> EvidenceAuthorityDecision:
    if not validate_decision_envelope_timing(envelope):
        return EvidenceAuthorityDecision(
            state=EvidenceAuthorityState.LATE_EVIDENCE,
            reason_code="LATE_DECISION_EVIDENCE",
            detail="Decision envelope failed frozen timing checks",
            scientific_status="MISSING_CAUSAL_EVIDENCE",
            metric_status="METRICS_MISSING",
        )

    if eligibility.eligible:
        return EvidenceAuthorityDecision(
            state=EvidenceAuthorityState.LOCALLY_VERIFIED_CAUSAL_TIMING,
            reason_code="LOCAL_TIMING_OK",
            detail="Eligibility timing checks passed",
            scientific_status="VALID_RESEARCH_EXECUTION",
            metric_status="METRICS_MISSING",
        )

    reason = eligibility.reason
    if reason == M8AbstainReason.EVIDENCE_LATE:
        return EvidenceAuthorityDecision(
            state=EvidenceAuthorityState.LATE_EVIDENCE,
            reason_code="EVIDENCE_LATE",
            detail="M8 evidence is late",
            scientific_status="MISSING_CAUSAL_EVIDENCE",
            metric_status="METRICS_MISSING",
        )
    if reason in {M8AbstainReason.BY_INCONSISTENT_EVIDENCE, M8AbstainReason.IDENTITY_MISMATCH}:
        return EvidenceAuthorityDecision(
            state=EvidenceAuthorityState.INVALID_IDENTITY,
            reason_code="INVALID_FAMILY_IDENTITY",
            detail="Family identity validation failed",
            scientific_status="INVALID_SCIENTIFIC_IDENTITY",
            metric_status="METRICS_INVALID",
        )
    if reason in {M8AbstainReason.PREDECLARATION_UNPROVEN, M8AbstainReason.FAMILY_INCOMPLETE}:
        return EvidenceAuthorityDecision(
            state=EvidenceAuthorityState.UNVERIFIED_FAMILY_AUTHORITY,
            reason_code="FAMILY_AUTHORITY_UNVERIFIED",
            detail="Family completeness/authority remains unverified",
            scientific_status="MISSING_CAUSAL_EVIDENCE",
            metric_status="METRICS_MISSING",
        )

    return EvidenceAuthorityDecision(
        state=EvidenceAuthorityState.MISSING_EVIDENCE,
        reason_code="MISSING_EVIDENCE",
        detail="M8 eligibility abstained under fail-closed semantics",
        scientific_status="MISSING_CAUSAL_EVIDENCE",
        metric_status="METRICS_MISSING",
    )


def to_validation_result(decision: EvidenceAuthorityDecision) -> ValidationResult:
    return ValidationResult(
        valid=decision.state == EvidenceAuthorityState.LOCALLY_VERIFIED_CAUSAL_TIMING,
        scientific_status=decision.scientific_status,
        metric_status=decision.metric_status,
        reason_code=decision.reason_code,
        details=decision.detail,
        authoritative=False,
        status_role="STRUCTURAL_ONLY",
    )

from __future__ import annotations

import math

from regimequant.cointegration.contracts import CointegrationCandidate, DecisionEnvelope
from regimequant.signals.m8_calendar import TradingCalendarEvidence
from regimequant.signals.m8_contracts import M8AbstainReason, M8SignalEnum, M8SignalObservation
from regimequant.signals.m8_contracts import M8DiscoveryFamilyEvidence, M8RevocationRecord
from regimequant.signals.m8_eligibility import evaluate_eligibility
from regimequant.signals.m8_formation import M8FormationError, fit_formation_model
from regimequant.signals.m8_identity import compute_signal_identity
from regimequant.signals.m8_validation import M8ValidationError, require_available_by_cutoff


class M8SignalError(ValueError):
    """Raised when an M8 signal cannot be emitted due to invalid inputs."""


def _abstain(
    *,
    canonical_pair_id: str,
    eval_observation_time,
    eval_available_at,
    m8_decision_cutoff,
    source_mode: str,
    scientific_identity: str,
    evidence_digest: str,
    decision_digest: str,
    reason: M8AbstainReason,
) -> M8SignalObservation:
    return M8SignalObservation(
        canonical_pair_id=canonical_pair_id,
        eval_observation_time=eval_observation_time,
        eval_available_at=eval_available_at,
        m8_decision_cutoff=m8_decision_cutoff,
        signal=M8SignalEnum.ABSTAIN,
        z_score=None,
        abstain_reason=reason,
        source_mode=source_mode,
        scientific_identity=scientific_identity,
        evidence_digest=evidence_digest,
        decision_digest=decision_digest,
    )


def evaluate_stateless_signal(
    *,
    canonical_pair_id: str,
    eval_observation_time,
    eval_available_at,
    m8_decision_cutoff,
    spread_eval: float,
    spread_mean: float,
    spread_std: float,
    source_mode: str,
    scientific_identity: str,
    evidence_digest: str,
    decision_digest: str,
    z_entry: float = 2.0,
    invalid_reason: M8AbstainReason | None = None,
) -> M8SignalObservation:
    if invalid_reason is not None:
        return M8SignalObservation(
            canonical_pair_id=canonical_pair_id,
            eval_observation_time=eval_observation_time,
            eval_available_at=eval_available_at,
            m8_decision_cutoff=m8_decision_cutoff,
            signal=M8SignalEnum.ABSTAIN,
            z_score=None,
            abstain_reason=invalid_reason,
            source_mode=source_mode,
            scientific_identity=scientific_identity,
            evidence_digest=evidence_digest,
            decision_digest=decision_digest,
        )

    if spread_std <= 0.0 or not math.isfinite(spread_std):
        return M8SignalObservation(
            canonical_pair_id=canonical_pair_id,
            eval_observation_time=eval_observation_time,
            eval_available_at=eval_available_at,
            m8_decision_cutoff=m8_decision_cutoff,
            signal=M8SignalEnum.ABSTAIN,
            z_score=None,
            abstain_reason=M8AbstainReason.SPREAD_VAR_TOO_SMALL,
            source_mode=source_mode,
            scientific_identity=scientific_identity,
            evidence_digest=evidence_digest,
            decision_digest=decision_digest,
        )

    z = (float(spread_eval) - float(spread_mean)) / float(spread_std)
    if not math.isfinite(z):
        return M8SignalObservation(
            canonical_pair_id=canonical_pair_id,
            eval_observation_time=eval_observation_time,
            eval_available_at=eval_available_at,
            m8_decision_cutoff=m8_decision_cutoff,
            signal=M8SignalEnum.ABSTAIN,
            z_score=None,
            abstain_reason=M8AbstainReason.Z_NONFINITE,
            source_mode=source_mode,
            scientific_identity=scientific_identity,
            evidence_digest=evidence_digest,
            decision_digest=decision_digest,
        )

    if z >= z_entry:
        signal = M8SignalEnum.SHORT_SPREAD_CANDIDATE
    elif z <= -z_entry:
        signal = M8SignalEnum.LONG_SPREAD_CANDIDATE
    else:
        signal = M8SignalEnum.NEUTRAL

    return M8SignalObservation(
        canonical_pair_id=canonical_pair_id,
        eval_observation_time=eval_observation_time,
        eval_available_at=eval_available_at,
        m8_decision_cutoff=m8_decision_cutoff,
        signal=signal,
        z_score=float(z),
        abstain_reason=None,
        source_mode=source_mode,
        scientific_identity=scientific_identity,
        evidence_digest=evidence_digest,
        decision_digest=decision_digest,
    )


def generate_stateless_signal(
    *,
    candidate: CointegrationCandidate,
    family_evidence: M8DiscoveryFamilyEvidence,
    decision_envelope: DecisionEnvelope,
    calendar: TradingCalendarEvidence,
    revocations: tuple[M8RevocationRecord, ...],
    x_series,
    y_series,
    formation_start,
    formation_end,
    formation_available_at,
    eval_observation_time,
    eval_available_at,
    eval_x: float,
    eval_y: float,
    model_version: str = "m8v1",
) -> M8SignalObservation:
    default_identity = compute_signal_identity(
        {
            "pair_id": candidate.pair_id,
            "family_digest": family_evidence.family_digest,
            "model_version": model_version,
            "source_mode": decision_envelope.mode,
        }
    )

    eligibility = evaluate_eligibility(
        candidate=candidate,
        family_evidence=family_evidence,
        decision_envelope=decision_envelope,
        calendar=calendar,
        decision_observation_session=eval_observation_time,
        revocations=revocations,
    )
    if not eligibility.eligible:
        return _abstain(
            canonical_pair_id=eligibility.proof.canonical_pair_id if eligibility.proof is not None else candidate.pair_id,
            eval_observation_time=eval_observation_time,
            eval_available_at=eval_available_at,
            m8_decision_cutoff=decision_envelope.cutoff,
            source_mode=decision_envelope.mode,
            scientific_identity=default_identity,
            evidence_digest=family_evidence.evidence_provenance_digest,
            decision_digest=decision_envelope.decision_digest,
            reason=eligibility.reason or M8AbstainReason.MODE_MISMATCH,
        )

    canonical_pair_id = eligibility.proof.canonical_pair_id
    try:
        require_available_by_cutoff(
            available_at=eval_available_at,
            decision_cutoff=decision_envelope.cutoff,
            field_name="evaluation evidence",
        )
        require_available_by_cutoff(
            available_at=formation_available_at,
            decision_cutoff=formation_end,
            field_name="formation evidence",
        )
    except M8ValidationError:
        return _abstain(
            canonical_pair_id=canonical_pair_id,
            eval_observation_time=eval_observation_time,
            eval_available_at=eval_available_at,
            m8_decision_cutoff=decision_envelope.cutoff,
            source_mode=decision_envelope.mode,
            scientific_identity=default_identity,
            evidence_digest=family_evidence.evidence_provenance_digest,
            decision_digest=decision_envelope.decision_digest,
            reason=M8AbstainReason.EVIDENCE_LATE,
        )

    try:
        fit = fit_formation_model(
            x_series,
            y_series,
            formation_start=formation_start,
            formation_end=formation_end,
            eval_observation_time=eval_observation_time,
        )
    except M8FormationError as exc:
        message = str(exc).lower()
        if "strictly precede" in message:
            reason = M8AbstainReason.LEAKAGE_FORMATION_EVAL_OVERLAP
        elif "duplicate" in message:
            reason = M8AbstainReason.DUPLICATE_OR_CONFLICTING_OBS
        elif "missing required sessions" in message:
            reason = M8AbstainReason.MISSING_REQUIRED_SESSION
        elif "253 aligned formation" in message:
            reason = M8AbstainReason.FORMATION_TOO_SHORT
        elif "near-constant" in message:
            reason = M8AbstainReason.X_NEAR_CONSTANT
        elif "ill-conditioned" in message:
            reason = M8AbstainReason.ILL_CONDITIONED
        elif "spread variance" in message:
            reason = M8AbstainReason.SPREAD_VAR_TOO_SMALL
        else:
            reason = M8AbstainReason.FORMATION_TOO_SHORT
        return _abstain(
            canonical_pair_id=canonical_pair_id,
            eval_observation_time=eval_observation_time,
            eval_available_at=eval_available_at,
            m8_decision_cutoff=decision_envelope.cutoff,
            source_mode=decision_envelope.mode,
            scientific_identity=default_identity,
            evidence_digest=family_evidence.evidence_provenance_digest,
            decision_digest=decision_envelope.decision_digest,
            reason=reason,
        )

    spread_eval = float(eval_y) - fit.alpha - fit.beta * float(eval_x)
    signal_identity = compute_signal_identity(
        {
            "pair_id": canonical_pair_id,
            "family_digest": family_evidence.family_digest,
            "eligibility_digest": eligibility.proof.proof_digest,
            "formation_sessions": [str(ts) for ts in fit.aligned_index],
            "formation_values_x": [float(v) for v in fit.x_series.to_numpy()],
            "formation_values_y": [float(v) for v in fit.y_series.to_numpy()],
            "alpha": float(fit.alpha),
            "beta": float(fit.beta),
            "spread_mean": float(fit.spread_mean),
            "spread_std": float(fit.spread_std),
            "eval_observation_time": str(eval_observation_time),
            "eval_available_at": str(eval_available_at),
            "eval_x": float(eval_x),
            "eval_y": float(eval_y),
            "decision_cutoff": str(decision_envelope.cutoff),
            "threshold": 2.0,
            "model_version": model_version,
            "source_mode": decision_envelope.mode,
            "calendar_version": calendar.calendar_version,
            "revocations": [record.revocation_digest for record in revocations],
        }
    )

    return evaluate_stateless_signal(
        canonical_pair_id=canonical_pair_id,
        eval_observation_time=eval_observation_time,
        eval_available_at=eval_available_at,
        m8_decision_cutoff=decision_envelope.cutoff,
        spread_eval=spread_eval,
        spread_mean=fit.spread_mean,
        spread_std=fit.spread_std,
        source_mode=decision_envelope.mode,
        scientific_identity=signal_identity,
        evidence_digest=eligibility.proof.proof_digest,
        decision_digest=decision_envelope.decision_digest,
    )

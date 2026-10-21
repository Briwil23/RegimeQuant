from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd

from .contracts import CoverageStatus, EligibilityDecision, EligibilityStatus, ExclusionReason, MembershipEvent, UniverseCandidate
from .hash import compute_digest
from .membership import build_membership_history


@dataclass(frozen=True, slots=True)
class EligibilityPolicy:
    mode: str = "production"
    allow_inferred_membership: bool = False
    require_verified_membership: bool = True
    require_coverage_complete: bool = True
    unsupported_liquidity_metadata_allowed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", str(self.mode).strip().lower())
        if self.mode not in {"production", "research"}:
            raise ValueError("mode must be 'production' or 'research'")


def determine_exclusion_reasons(
    *,
    observed_member: bool,
    verified: bool,
    unsupported_liquidity_metadata: bool = False,
    future_effective_event: bool = False,
    coverage_complete: bool = True,
    mode: str = "production",
    allow_inferred_membership: bool = False,
    coverage_status: str | None = None,
) -> tuple[str, ...]:
    reasons: list[str] = []

    if not observed_member:
        if mode == "research" and allow_inferred_membership:
            reasons.append(ExclusionReason.RESEARCH_INFERRED.value)
        else:
            reasons.append(ExclusionReason.MISSING_VERIFIED_MEMBERSHIP.value)
    elif not verified:
        reasons.append(ExclusionReason.UNKNOWN_UNVERIFIED_EVIDENCE.value)

    if unsupported_liquidity_metadata:
        reasons.append(ExclusionReason.UNSUPPORTED_LIQUIDITY_METADATA.value)
    if future_effective_event:
        reasons.append(ExclusionReason.FUTURE_EFFECTIVE_EVENT.value)
    if coverage_status in {CoverageStatus.PARTIAL.value, CoverageStatus.UNAVAILABLE.value}:
        reasons.append(ExclusionReason.INCOMPLETE_CANDIDATE_COVERAGE.value)
    if not coverage_complete:
        reasons.append(ExclusionReason.INCOMPLETE_CANDIDATE_COVERAGE.value)

    # Strict production rejects incomplete verified coverage; research mode retains the diagnostic.
    if mode == "production" and (coverage_status in {CoverageStatus.PARTIAL.value, CoverageStatus.UNAVAILABLE.value} or not coverage_complete):
        reasons.append(ExclusionReason.INSUFFICIENT_VERIFIED_COVERAGE.value)

    ordered: list[str] = []
    for reason in reasons:
        if reason not in ordered:
            ordered.append(reason)
    return tuple(ordered)


def _coverage_status(candidate: UniverseCandidate) -> str:
    if candidate.coverage_complete:
        return CoverageStatus.COMPLETE.value
    if candidate.is_delisted:
        return CoverageStatus.PARTIAL.value
    return CoverageStatus.UNAVAILABLE.value


def evaluate_eligibility(
    candidates: Iterable[UniverseCandidate],
    membership_events: Iterable[MembershipEvent | dict[str, Any]],
    *,
    policy: EligibilityPolicy | None = None,
    decision_timestamp: Any,
) -> list[EligibilityDecision]:
    policy = policy or EligibilityPolicy()
    decision_ts = pd.Timestamp(decision_timestamp)
    if decision_ts.tzinfo is None or decision_ts.tz is None:
        decision_ts = decision_ts.tz_localize("UTC")
    else:
        decision_ts = decision_ts.tz_convert("UTC")

    membership_by_ticker = build_membership_history(
        membership_events,
        decision_timestamp=decision_ts,
        allow_inferred=policy.allow_inferred_membership,
    )

    results: list[EligibilityDecision] = []
    for candidate in candidates:
        record = membership_by_ticker.get(candidate.ticker)
        observed_member = record is not None and record["security_id"] == candidate.security_id
        verified = bool(record and record["status"] in {"listed", "delisted"})
        unsupported = bool(candidate.liquidity_metadata) and not policy.unsupported_liquidity_metadata_allowed
        future_effective_event = False
        if record is not None and record["effective_at"] > decision_ts:
            future_effective_event = True
        coverage_status = _coverage_status(candidate)

        reasons = determine_exclusion_reasons(
            observed_member=observed_member,
            verified=verified,
            unsupported_liquidity_metadata=unsupported,
            future_effective_event=future_effective_event,
            coverage_complete=candidate.coverage_complete,
            mode=policy.mode,
            allow_inferred_membership=policy.allow_inferred_membership,
            coverage_status=coverage_status,
        )

        if policy.mode == "production":
            if observed_member and verified and not reasons:
                status = EligibilityStatus.ELIGIBLE.value
            else:
                status = EligibilityStatus.INELIGIBLE.value
        else:
            if not observed_member and policy.allow_inferred_membership and not unsupported and candidate.coverage_complete:
                status = EligibilityStatus.INFERRED.value
            elif observed_member and verified and not reasons:
                status = EligibilityStatus.ELIGIBLE.value
            else:
                status = EligibilityStatus.INELIGIBLE.value

        if policy.mode == "production" and not observed_member:
            evidence_available = False
        else:
            evidence_available = bool(record or (policy.mode == "research" and policy.allow_inferred_membership))

        decision = EligibilityDecision(
            security_id=candidate.security_id,
            decision_timestamp=decision_ts,
            eligibility_status=status,
            evidence_available=evidence_available,
            snapshot_identity=f"{candidate.security_id}:{decision_ts.isoformat()}:{policy.mode}",
            provenance_digest=compute_digest(
                {
                    "security_id": candidate.security_id,
                    "ticker": candidate.ticker,
                    "decision_timestamp": decision_ts.isoformat(),
                    "mode": policy.mode,
                }
            ),
            coverage_digest=compute_digest(
                {
                    "security_id": candidate.security_id,
                    "as_of": candidate.as_of.isoformat(),
                    "coverage_complete": candidate.coverage_complete,
                    "is_delisted": candidate.is_delisted,
                    "coverage_status": coverage_status,
                }
            ),
            exclusion_reasons=reasons,
            mode=policy.mode,
            coverage_status=coverage_status,
            research_override=(policy.mode == "research" and policy.allow_inferred_membership and not observed_member),
        )
        results.append(decision)

    return results


__all__ = ["EligibilityPolicy", "determine_exclusion_reasons", "evaluate_eligibility"]

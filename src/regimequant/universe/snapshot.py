from __future__ import annotations

from typing import Any, Iterable

from .contracts import EligibilityDecision, UniverseSnapshot
from .eligibility import EligibilityPolicy
from .hash import compute_digest


def _decision_payload(decision: EligibilityDecision) -> dict[str, Any]:
    return {
        "security_id": decision.security_id,
        "decision_timestamp": decision.decision_timestamp.isoformat(),
        "eligibility_status": decision.eligibility_status,
        "evidence_available": decision.evidence_available,
        "snapshot_identity": decision.snapshot_identity,
        "provenance_digest": decision.provenance_digest,
        "coverage_digest": decision.coverage_digest,
        "exclusion_reasons": list(decision.exclusion_reasons),
        "mode": decision.mode,
        "coverage_status": decision.coverage_status,
        "research_override": decision.research_override,
    }


def create_snapshot(
    *,
    candidate_universe_identity: str,
    coverage_evidence: Iterable[Any],
    admissible_membership_versions: Iterable[str],
    eligibility_policy: EligibilityPolicy,
    decisions: Iterable[EligibilityDecision],
    source_provenance_digest: str,
    identity_evidence: Iterable[Any] | None = None,
    continuity_records: Iterable[Any] | None = None,
) -> UniverseSnapshot:
    coverage_sequence = tuple(sorted(coverage_evidence, key=lambda item: str(getattr(item, "security_id", str(item)))))
    flattened_decisions = tuple(sorted(decisions, key=lambda item: item.security_id))
    identity_sequence = tuple(sorted(identity_evidence or (), key=lambda item: str(getattr(item, "security_id", str(item)))))
    continuity_sequence = tuple(sorted(continuity_records or (), key=lambda item: f"{getattr(item, 'predecessor_security_id', '')}:{getattr(item, 'successor_security_id', '')}:{getattr(item, 'effective_at', '')}"))
    payload = {
        "candidate_universe_identity": candidate_universe_identity,
        "coverage_evidence": [
            {
                "security_id": getattr(item, "security_id", str(item)),
                "ticker": getattr(item, "ticker", ""),
                "as_of": getattr(item, "as_of", None).isoformat() if getattr(item, "as_of", None) is not None else None,
                "status": getattr(item, "status", ""),
                "is_delisted": bool(getattr(item, "is_delisted", False)),
                "source": getattr(item, "source", ""),
                "coverage_complete": bool(getattr(item, "coverage_complete", True)),
            }
            for item in coverage_sequence
        ],
        "identity_evidence": [
            {
                "security_id": getattr(item, "security_id", str(item)),
                "ticker": getattr(item, "ticker", ""),
                "aliases": list(getattr(item, "aliases", ())),
                "authority": getattr(item, "authority", ""),
                "verified_at": getattr(item, "verified_at", None).isoformat() if getattr(item, "verified_at", None) is not None else None,
                "source": getattr(item, "source", ""),
                "evidence_digest": getattr(item, "evidence_digest", ""),
                "status": getattr(item, "status", ""),
                "issuer_id": getattr(item, "issuer_id", None),
            }
            for item in identity_sequence
        ],
        "continuity_records": [
            {
                "predecessor_security_id": getattr(item, "predecessor_security_id", ""),
                "successor_security_id": getattr(item, "successor_security_id", ""),
                "event_kind": getattr(item, "event_kind", ""),
                "effective_at": getattr(item, "effective_at", None).isoformat() if getattr(item, "effective_at", None) is not None else None,
                "verified_at": getattr(item, "verified_at", None).isoformat() if getattr(item, "verified_at", None) is not None else None,
                "source_authority": getattr(item, "source_authority", ""),
                "evidence_digest": getattr(item, "evidence_digest", ""),
            }
            for item in continuity_sequence
        ],
        "admissible_membership_versions": sorted(set(str(item) for item in admissible_membership_versions)),
        "eligibility_policy": {
            "mode": eligibility_policy.mode,
            "allow_inferred_membership": eligibility_policy.allow_inferred_membership,
            "require_verified_membership": eligibility_policy.require_verified_membership,
            "require_coverage_complete": eligibility_policy.require_coverage_complete,
            "unsupported_liquidity_metadata_allowed": eligibility_policy.unsupported_liquidity_metadata_allowed,
        },
        "decisions": [_decision_payload(decision) for decision in flattened_decisions],
        "reason_codes": sorted({reason for decision in flattened_decisions for reason in decision.exclusion_reasons}),
        "source_provenance_digest": source_provenance_digest,
    }
    identity = f"m6v1_sha256_{compute_digest(payload)}"
    snapshot = UniverseSnapshot(
        candidate_universe_identity=candidate_universe_identity,
        coverage_evidence=coverage_sequence,
        admissible_membership_versions=tuple(sorted(set(str(item) for item in admissible_membership_versions))),
        eligibility_policy=payload["eligibility_policy"],
        decisions=flattened_decisions,
        source_provenance_digest=source_provenance_digest,
        snapshot_scientific_identity=identity,
        scientific_payload=payload,
    )
    return snapshot


def compute_snapshot_identity(snapshot: UniverseSnapshot) -> str:
    return snapshot.snapshot_scientific_identity


__all__ = ["create_snapshot", "compute_snapshot_identity"]

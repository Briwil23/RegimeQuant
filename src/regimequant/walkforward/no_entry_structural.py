from __future__ import annotations

from dataclasses import dataclass

from regimequant.walkforward.contracts import NoEntryArtifact
from regimequant.walkforward.identity import build_no_entry_digest


@dataclass(frozen=True, slots=True)
class StructuralNoEntryResult:
    artifact: NoEntryArtifact
    evidence_dependency_identities: tuple[str, ...]


def build_structural_no_entry(
    *,
    experiment_id: str,
    fold_id: str,
    fold_input_identity: str,
    decision_session: int,
    boundary_b: int,
    reason_code: str,
    evidence_dependency_identities: tuple[str, ...],
) -> StructuralNoEntryResult:
    if not isinstance(evidence_dependency_identities, tuple) or not evidence_dependency_identities:
        raise ValueError("evidence_dependency_identities must be a non-empty tuple")

    digest = build_no_entry_digest(
        experiment_id=experiment_id,
        fold_id=fold_id,
        fold_input_identity=fold_input_identity,
        decision_session=decision_session,
        boundary_b=boundary_b,
        reason_code=reason_code,
        evidence_dependency_identities=evidence_dependency_identities,
        invoked_m9=False,
    )
    artifact = NoEntryArtifact(
        schema="regimequant.walkforward.no_entry",
        version="m10.v1",
        experiment_id=experiment_id,
        fold_id=fold_id,
        fold_input_identity=fold_input_identity,
        decision_session=decision_session,
        boundary_b=boundary_b,
        reason_code=reason_code,
        evidence_dependency_identities=evidence_dependency_identities,
        invoked_m9=False,
        no_entry_digest=digest,
        structural_only=True,
    )
    return StructuralNoEntryResult(artifact=artifact, evidence_dependency_identities=evidence_dependency_identities)

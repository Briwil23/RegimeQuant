from __future__ import annotations

from dataclasses import dataclass

from regimequant.walkforward.causal_projection import CausalEligibilityProjection
from regimequant.walkforward.contracts import ExperimentPlan, FoldEvidenceArtifact, FoldInput, FoldInterval
from regimequant.walkforward.identity import build_fold_input_digest


@dataclass(frozen=True, slots=True)
class FoldBindingResult:
    fold_input: FoldInput
    evidence_dependency_identities: tuple[str, ...]


def build_fold_input_binding(
    *,
    plan: ExperimentPlan,
    fold_id: str,
    interval: FoldInterval,
    projection: CausalEligibilityProjection,
    evidence: FoldEvidenceArtifact,
    evidence_dependency_identities: tuple[str, ...],
) -> FoldBindingResult:
    if projection.fold_id != fold_id:
        raise ValueError("projection fold_id must match requested fold_id")
    if projection.fold_interval != interval:
        raise ValueError("projection interval must match fold interval")
    if projection.source_dataset_identity != plan.dataset_identity:
        raise ValueError("projection dataset identity must match experiment plan")
    if projection.session_calendar_id != plan.calendar_id:
        raise ValueError("projection calendar identity must match experiment plan")
    if projection.session_calendar_version != plan.calendar_version:
        raise ValueError("projection calendar version must match experiment plan")
    if not isinstance(projection.session_sequence_digest, str) or not projection.session_sequence_digest.strip():
        raise ValueError("projection session sequence digest must be a non-empty string")
    if projection.m3_policy_identity != plan.model_policy_identity:
        raise ValueError("projection m3 policy identity must match plan model policy identity")
    if projection.m5_policy_identity != plan.formation_policy_identity:
        raise ValueError("projection m5 policy identity must match plan formation policy identity")
    if not isinstance(evidence_dependency_identities, tuple) or not evidence_dependency_identities:
        raise ValueError("evidence_dependency_identities must be a non-empty tuple")
    if any(not isinstance(dep, str) or not dep.strip() for dep in evidence_dependency_identities):
        raise ValueError("evidence_dependency_identities entries must be non-empty strings")
    dependency_set = set(evidence_dependency_identities)
    required_dependencies = {
        evidence.family_provenance_digest,
        evidence.calendar_provenance_digest,
        evidence.model_provenance_digest,
        evidence.validation_digest,
        evidence.evidence_digest,
        projection.session_sequence_digest,
    }
    missing_dependencies = sorted(dep for dep in required_dependencies if dep not in dependency_set)
    if missing_dependencies:
        raise ValueError("evidence_dependency_identities missing required bound lineage identities")

    digest = build_fold_input_digest(
        plan_id=plan.experiment_id,
        fold_id=fold_id,
        interval=interval,
        dataset_identity=plan.dataset_identity,
        feature_spec_set_hash=plan.feature_spec_set_hash,
        calendar_identity=plan.calendar_id,
        evidence=evidence,
        model_policy_identity=plan.model_policy_identity,
        formation_policy_identity=plan.formation_policy_identity,
        family_policy_identity=plan.family_policy_identity,
    )

    fold_input = FoldInput(
        schema="regimequant.walkforward.fold_input",
        version="m10.v1",
        plan_id=plan.experiment_id,
        fold_id=fold_id,
        interval=interval,
        dataset_identity=plan.dataset_identity,
        feature_spec_set_hash=plan.feature_spec_set_hash,
        calendar_identity=plan.calendar_id,
        evidence=evidence,
        model_policy_identity=plan.model_policy_identity,
        formation_policy_identity=plan.formation_policy_identity,
        family_policy_identity=plan.family_policy_identity,
        input_digest=digest,
    )
    return FoldBindingResult(
        fold_input=fold_input,
        evidence_dependency_identities=tuple(sorted(dependency_set)),
    )

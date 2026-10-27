from __future__ import annotations

from datetime import datetime, timezone

from .contracts import ExperimentPlan, FoldInput, M10Policy

M10_RATIFIED_POLICY = M10Policy(
    initial_training_sessions=120,
    validation_sessions=20,
    outer_test_sessions=20,
    stride_sessions=20,
    final_holdout_sessions=20,
    formation_min_aligned_levels=253,
    discovery_freshness_limit_completed_sessions=20,
    discovery_refresh_sessions=20,
    by_q_threshold=0.05,
    empty_signal_policy="valid_abstention_only",
    research_mode="research-only",
)


def build_experiment_plan(
    *,
    experiment_id: str,
    experiment_version: str,
    regime_primary: str,
    regime_comparator: str | None,
    calendar_id: str,
    calendar_version: str,
    dataset_identity: str,
    feature_spec_set_hash: str,
    model_policy_identity: str,
    formation_policy_identity: str,
    family_policy_identity: str,
    final_holdout_sessions: int,
    created_at: datetime | None = None,
    folds: tuple[FoldInput, ...] = (),
) -> ExperimentPlan:
    if experiment_version != "m10.v1":
        raise ValueError("experiment_version must be 'm10.v1'")
    if regime_primary != "M5":
        raise ValueError("regime_primary must be 'M5'")
    if regime_comparator not in (None, "M3"):
        raise ValueError("regime_comparator must be None or 'M3'")
    if final_holdout_sessions != M10_RATIFIED_POLICY.final_holdout_sessions:
        raise ValueError("final_holdout_sessions must equal 20")
    if not experiment_id.strip():
        raise ValueError("experiment_id is required")
    if not calendar_id.strip():
        raise ValueError("calendar_id is required")
    if not dataset_identity.strip():
        raise ValueError("dataset_identity is required")
    if not feature_spec_set_hash.strip():
        raise ValueError("feature_spec_set_hash is required")
    if not model_policy_identity.strip():
        raise ValueError("model_policy_identity is required")
    if not formation_policy_identity.strip():
        raise ValueError("formation_policy_identity is required")
    if not family_policy_identity.strip():
        raise ValueError("family_policy_identity is required")
    created = created_at or datetime.now(timezone.utc)
    if created.tzinfo is None:
        raise ValueError("created_at must be timezone-aware")
    plan = ExperimentPlan(
        schema="regimequant.walkforward.experiment_plan",
        experiment_id=experiment_id,
        experiment_version=experiment_version,
        regime_primary=regime_primary,
        regime_comparator=regime_comparator,
        calendar_id=calendar_id,
        calendar_version=calendar_version,
        dataset_identity=dataset_identity,
        feature_spec_set_hash=feature_spec_set_hash,
        model_policy_identity=model_policy_identity,
        formation_policy_identity=formation_policy_identity,
        family_policy_identity=family_policy_identity,
        final_holdout_sessions=final_holdout_sessions,
        created_at=created,
        folds=tuple(folds),
        structural_only=True,
        scientific_certification_status="STRUCTURAL_ONLY",
    )
    validate_experiment_plan(plan)
    return plan


def validate_experiment_plan(plan: ExperimentPlan) -> ExperimentPlan:
    if plan.schema != "regimequant.walkforward.experiment_plan":
        raise ValueError("Unsupported schema")
    if plan.experiment_version != "m10.v1":
        raise ValueError("Unsupported schema version")
    if plan.regime_primary != "M5":
        raise ValueError("Only M5 is supported as primary model")
    if plan.regime_comparator not in (None, "M3"):
        raise ValueError("Comparator must be M3 or None")
    if plan.final_holdout_sessions != M10_RATIFIED_POLICY.final_holdout_sessions:
        raise ValueError("Invalid final holdout policy")
    if plan.created_at is None or plan.created_at.tzinfo is None:
        raise ValueError("Malformed timestamp: created_at must be timezone-aware")
    if plan.structural_only is not True:
        raise ValueError("Manifest must remain structural-only in Phase 1")
    if plan.scientific_certification_status != "STRUCTURAL_ONLY":
        raise ValueError("Manifest cannot claim scientific certification in Phase 1")
    seen: set[str] = set()
    for fold in plan.folds:
        if fold.fold_id in seen:
            raise ValueError("Duplicate fold identifiers are not allowed")
        seen.add(fold.fold_id)
        if fold.plan_id != plan.experiment_id:
            raise ValueError("Fold plan_id must match experiment_id")
    return plan


def _missing_required_identity(value: str | None, label: str) -> None:
    if not value or not str(value).strip():
        raise ValueError(f"Missing required identity field: {label}")

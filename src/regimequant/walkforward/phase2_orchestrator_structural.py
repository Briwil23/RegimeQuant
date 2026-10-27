from __future__ import annotations

from dataclasses import dataclass

from regimequant.walkforward.causal_projection import (
    CausalEligibilityProjection,
    ProjectionStage,
    required_m3_heldout_minimum,
    required_m3_training_minimum,
    required_m5_training_minimum,
)
from regimequant.walkforward.contracts import ExperimentPlan, FoldEvidenceArtifact, M10Policy
from regimequant.walkforward.fold_contract_binding import FoldBindingResult, build_fold_input_binding
from regimequant.walkforward.fold_geometry import DeterministicFold, build_expanding_fold_intervals
from regimequant.walkforward.no_entry_structural import StructuralNoEntryResult, build_structural_no_entry


@dataclass(frozen=True, slots=True)
class StructuralAssemblyResult:
    folds: tuple[DeterministicFold, ...]
    fold_bindings: tuple[FoldBindingResult, ...]
    no_entry_artifacts: tuple[StructuralNoEntryResult, ...]
    failure_codes: tuple[str, ...]


def _extract_positive_int(config: dict[str, int], key: str) -> int | None:
    value = config.get(key)
    if not isinstance(value, int) or value <= 0:
        return None
    return value


def _effective_feature_dimension(projection: CausalEligibilityProjection) -> int:
    effective = 0
    for feature_index in range(len(projection.required_feature_ids)):
        seen: set[float] = set()
        non_numeric = False
        for row in projection.required_feature_values_by_observation:
            value = row[feature_index]
            if isinstance(value, str):
                non_numeric = True
                break
            if value is None:
                continue
            seen.add(float(value))
        if not non_numeric and len(seen) > 1:
            effective += 1
    return effective


def _minima_failure_code(
    *,
    projection: CausalEligibilityProjection,
    minima_config_by_fold: dict[str, dict[str, int]] | None,
) -> str | None:
    if minima_config_by_fold is None:
        return "MISSING_MODEL_CONFIGURATION"
    config = minima_config_by_fold.get(projection.fold_id)
    if config is None:
        return "MISSING_MODEL_CONFIGURATION"

    n_clusters = _extract_positive_int(config, "n_clusters")
    if n_clusters is None:
        return "MISSING_MODEL_CONFIGURATION"
    m3_required = required_m3_training_minimum(n_clusters=n_clusters)
    if projection.stage == ProjectionStage.TRAIN and projection.eligible_count < m3_required:
        return "M3_TRAINING_MINIMUM_UNSATISFIED"

    if projection.stage in {ProjectionStage.VALIDATION, ProjectionStage.TEST}:
        heldout_required = required_m3_heldout_minimum(n_clusters=n_clusters)
        if projection.eligible_count < heldout_required:
            return "M3_HELDOUT_MINIMUM_UNSATISFIED"

    n_components = _extract_positive_int(config, "n_components")
    if n_components is None:
        return "MISSING_MODEL_CONFIGURATION"
    d_eff = _effective_feature_dimension(projection)
    if d_eff <= 0:
        return "M5_EFFECTIVE_DIMENSION_UNPROVEN"
    m5_required = required_m5_training_minimum(n_components=n_components, effective_feature_dimension=d_eff)
    if projection.stage == ProjectionStage.TRAIN and projection.eligible_count < m5_required:
        return "M5_TRAINING_MINIMUM_UNSATISFIED"

    return None


def assemble_phase2_structural(
    *,
    total_sessions: int,
    policy: M10Policy,
    plan: ExperimentPlan,
    projections: tuple[CausalEligibilityProjection, ...],
    evidence_by_fold: dict[str, FoldEvidenceArtifact],
    evidence_dependency_identities_by_fold: dict[str, tuple[str, ...]],
    minima_config_by_fold: dict[str, dict[str, int]] | None = None,
) -> StructuralAssemblyResult:
    folds = build_expanding_fold_intervals(total_sessions=total_sessions, policy=policy)
    by_fold = {item.fold_id: item for item in folds}
    fold_order = {item.fold_id: index for index, item in enumerate(folds)}

    fold_bindings: list[FoldBindingResult] = []
    no_entries: list[StructuralNoEntryResult] = []
    failure_codes: list[str] = []
    seen_projection_folds: set[str] = set()
    ordered_projections = sorted(projections, key=lambda item: (fold_order.get(item.fold_id, 10**9), item.fold_id, item.stage.value, item.projection_digest))

    for projection in ordered_projections:
        if projection.fold_id in seen_projection_folds:
            failure_codes.append("DUPLICATE_FOLD_PROJECTION")
            continue
        seen_projection_folds.add(projection.fold_id)

        if projection.fold_id.startswith("fold_") and "_tr_" not in projection.fold_id:
            failure_codes.append("NONCANONICAL_FOLD_ID")
            continue
        fold = by_fold.get(projection.fold_id)
        if fold is None:
            failure_codes.append("UNKNOWN_FOLD_ID")
            continue

        minimum_failure = _minima_failure_code(projection=projection, minima_config_by_fold=minima_config_by_fold)
        if minimum_failure is not None:
            failure_codes.append(minimum_failure)
            continue

        evidence = evidence_by_fold.get(projection.fold_id)
        if evidence is None:
            failure_codes.append("MISSING_FOLD_EVIDENCE")
            continue
        deps = evidence_dependency_identities_by_fold.get(projection.fold_id)
        if deps is None:
            failure_codes.append("MISSING_EVIDENCE_DEPENDENCIES")
            continue

        binding = build_fold_input_binding(
            plan=plan,
            fold_id=projection.fold_id,
            interval=fold.interval,
            projection=projection,
            evidence=evidence,
            evidence_dependency_identities=deps,
        )
        fold_bindings.append(binding)

        # Structural-only abstention artifact; no M9 execution path.
        no_entry = build_structural_no_entry(
            experiment_id=plan.experiment_id,
            fold_id=projection.fold_id,
            fold_input_identity=binding.fold_input.input_digest or "",
            decision_session=fold.interval.test_start,
            boundary_b=fold.interval.test_end,
            reason_code="STRUCTURAL_PHASE2_NO_ENTRY",
            evidence_dependency_identities=deps,
        )
        no_entries.append(no_entry)

    return StructuralAssemblyResult(
        folds=folds,
        fold_bindings=tuple(fold_bindings),
        no_entry_artifacts=tuple(no_entries),
        failure_codes=tuple(failure_codes),
    )

from __future__ import annotations

from typing import Any

from regimequant.walkforward.causal_projection import CausalEligibilityProjection
from regimequant.walkforward.phase3_contracts import (
    Phase3BindingValidation,
    Phase3CausalDataset,
    Phase3FailureCode,
    Phase3ValidationError,
    compute_phase3_row_lineage_identity_payload,
)
from regimequant.walkforward.phase3_identity import (
    compute_phase2_projection_identity_digest,
    compute_phase3_binding_identity_digest,
)
from regimequant.walkforward.identity import compute_identity_digest


ROW_LINEAGE_DOMAIN = "regimequant.walkforward.phase3a.row_lineage"


def _require_real_projection(projection: object) -> CausalEligibilityProjection:
    if not isinstance(projection, CausalEligibilityProjection):
        raise Phase3ValidationError(
            Phase3FailureCode.PROJECTION_TYPE_MISMATCH,
            "projection must be a real CausalEligibilityProjection instance",
            missing_evidence=("projection",),
        )
    return projection


def _projection_field(projection: CausalEligibilityProjection, name: str, *, required: bool = True) -> Any:
    if hasattr(projection, name):
        return getattr(projection, name)
    if required:
        raise Phase3ValidationError(
            Phase3FailureCode.PROJECTION_EVIDENCE_INSUFFICIENT,
            f"projection missing required field '{name}'",
            missing_evidence=(name,),
        )
    return None


def _expected_ids_from_projection(projection: CausalEligibilityProjection) -> tuple[str, ...]:
    if hasattr(projection, "ordered_eligible_observation_ids"):
        value = getattr(projection, "ordered_eligible_observation_ids")
        if isinstance(value, tuple):
            return value
        if isinstance(value, list):
            return tuple(str(v) for v in value)
    if hasattr(projection, "eligible_observation_ids"):
        value = getattr(projection, "eligible_observation_ids")
        if isinstance(value, tuple):
            return value
        if isinstance(value, list):
            return tuple(str(v) for v in value)
    raise Phase3ValidationError(
        Phase3FailureCode.PROJECTION_EVIDENCE_INSUFFICIENT,
        "projection missing eligible observation ordering evidence",
        missing_evidence=("ordered_eligible_observation_ids", "eligible_observation_ids"),
    )


def _expected_required_features_from_projection(projection: CausalEligibilityProjection) -> tuple[str, ...]:
    if hasattr(projection, "required_feature_ids"):
        value = getattr(projection, "required_feature_ids")
        if isinstance(value, tuple):
            return value
        if isinstance(value, list):
            return tuple(str(v) for v in value)
    if hasattr(projection, "required_feature_columns"):
        value = getattr(projection, "required_feature_columns")
        if isinstance(value, tuple):
            return value
        if isinstance(value, list):
            return tuple(str(v) for v in value)
    raise Phase3ValidationError(
        Phase3FailureCode.PROJECTION_EVIDENCE_INSUFFICIENT,
        "projection missing required feature set evidence",
        missing_evidence=("required_feature_ids", "required_feature_columns"),
    )


def _normalized_projection_feature_values(values: tuple[float | None | str, ...]) -> tuple[float | None | str, ...]:
    normalized: list[float | None | str] = []
    for value in values:
        if value is None or isinstance(value, str):
            normalized.append(value)
        else:
            normalized.append(float(value))
    return tuple(normalized)


def _stage_matches_projection(dataset: Phase3CausalDataset, projection: CausalEligibilityProjection) -> bool:
    if hasattr(projection, "stage"):
        return str(projection.stage.value) == dataset.stage.value
    if hasattr(projection, "stage_name"):
        return str(getattr(projection, "stage_name")) == dataset.stage.value
    return False


def _compute_expected_row_lineage_identity(
    *,
    canonical_observation_id: str,
    required_feature_ids: tuple[str, ...],
    feature_values_in_required_order: tuple[float, ...],
) -> str:
    payload = compute_phase3_row_lineage_identity_payload(
        canonical_observation_id=canonical_observation_id,
        required_feature_ids=required_feature_ids,
        feature_values_in_required_order=feature_values_in_required_order,
    )
    return compute_identity_digest(payload, domain=ROW_LINEAGE_DOMAIN)


def _validate_dataset_against_projection(dataset: Phase3CausalDataset, projection: CausalEligibilityProjection) -> None:
    projection = _require_real_projection(projection)
    expected_projection_digest = compute_phase2_projection_identity_digest(projection)

    if projection.projection_digest != expected_projection_digest:
        raise Phase3ValidationError(
            Phase3FailureCode.PROJECTION_CONTENT_TAMPERING,
            "projection.projection_digest does not match material projection payload",
        )
    if dataset.phase2_projection_digest != expected_projection_digest:
        raise Phase3ValidationError(
            Phase3FailureCode.PROJECTION_DIGEST_MISMATCH,
            "dataset phase2_projection_digest does not match recomputed frozen projection digest",
        )
    if dataset.phase2_projection_digest != projection.projection_digest:
        raise Phase3ValidationError(
            Phase3FailureCode.PROJECTION_DIGEST_MISMATCH,
            "dataset phase2_projection_digest does not match projection.projection_digest",
        )

    if dataset.source_dataset_identity != projection.source_dataset_identity:
        raise Phase3ValidationError(
            Phase3FailureCode.DATASET_IDENTITY_MISMATCH,
            "dataset source_dataset_identity does not match projection",
        )
    if dataset.session_calendar_id != projection.session_calendar_id:
        raise Phase3ValidationError(
            Phase3FailureCode.CALENDAR_IDENTITY_MISMATCH,
            "dataset session_calendar_id does not match projection",
        )
    if dataset.session_calendar_version != projection.session_calendar_version:
        raise Phase3ValidationError(
            Phase3FailureCode.CALENDAR_VERSION_MISMATCH,
            "dataset session_calendar_version does not match projection",
        )
    if dataset.session_sequence_digest != projection.session_sequence_digest:
        raise Phase3ValidationError(
            Phase3FailureCode.SESSION_SEQUENCE_MISMATCH,
            "dataset session_sequence_digest does not match projection",
        )
    if dataset.fold_id != projection.fold_id:
        raise Phase3ValidationError(
            Phase3FailureCode.FOLD_MISMATCH,
            f"dataset fold_id={dataset.fold_id} does not match projection fold_id={projection.fold_id}",
        )
    if dataset.fold_interval != projection.fold_interval:
        raise Phase3ValidationError(
            Phase3FailureCode.FOLD_MISMATCH,
            "dataset fold_interval does not match projection fold_interval",
        )
    if dataset.stage.value != projection.stage.value:
        raise Phase3ValidationError(
            Phase3FailureCode.STAGE_MISMATCH,
            "dataset stage does not match projection stage",
        )
    if dataset.stage_information_cutoff_utc != projection.stage_information_cutoff_utc:
        raise Phase3ValidationError(
            Phase3FailureCode.PROJECTION_CONTENT_TAMPERING,
            "dataset stage cutoff does not match projection stage_information_cutoff_utc",
        )

    expected_required_feature_ids = _expected_required_features_from_projection(projection)
    if dataset.required_feature_ids != expected_required_feature_ids:
        raise Phase3ValidationError(
            Phase3FailureCode.FEATURE_COLUMN_ORDER_MISMATCH,
            "dataset required_feature_ids does not match projection-required ordering",
        )

    expected_eligible_ids = _expected_ids_from_projection(projection)
    if dataset.eligible_observation_ids != expected_eligible_ids:
        dataset_ids = set(dataset.eligible_observation_ids)
        expected_ids = set(expected_eligible_ids)
        if dataset_ids != expected_ids:
            if expected_ids - dataset_ids:
                raise Phase3ValidationError(
                    Phase3FailureCode.MISSING_ELIGIBLE_ROW,
                    "dataset missing eligible projection rows",
                )
            raise Phase3ValidationError(
                Phase3FailureCode.EXTRA_OBSERVATION_ROW,
                "dataset contains rows not present in projection eligibility",
            )
        raise Phase3ValidationError(
            Phase3FailureCode.REORDERED_OBSERVATION_ROWS,
            "dataset eligible observation ordering deviates from projection",
        )

    projection_evidence = {}
    for idx, observation_id in enumerate(projection.canonical_observation_ids):
        projection_evidence[observation_id] = {
            "observation_timestamp_utc": projection.observation_timestamps_utc[idx],
            "availability_timestamp_utc": projection.availability_timestamps_utc[idx],
            "feature_values": _normalized_projection_feature_values(
                projection.required_feature_values_by_observation[idx]
            ),
        }

    for row in dataset.ordered_rows:
        if row.canonical_observation_id not in projection_evidence:
            raise Phase3ValidationError(
                Phase3FailureCode.MISSING_ELIGIBLE_ROW,
                f"row {row.canonical_observation_id} missing from frozen projection evidence",
            )
        expected = projection_evidence[row.canonical_observation_id]
        if row.observation_timestamp_utc != expected["observation_timestamp_utc"]:
            raise Phase3ValidationError(
                Phase3FailureCode.PROJECTION_CONTENT_TAMPERING,
                f"row {row.canonical_observation_id} observation timestamp differs from frozen projection",
            )
        if row.availability_timestamp_utc != expected["availability_timestamp_utc"]:
            raise Phase3ValidationError(
                Phase3FailureCode.UNVERIFIED_AVAILABILITY,
                f"row {row.canonical_observation_id} availability differs from frozen projection",
            )
        if row.feature_values_in_required_order != tuple(
            float(item) for item in expected["feature_values"] if item is not None and not isinstance(item, str)
        ):
            raise Phase3ValidationError(
                Phase3FailureCode.FEATURE_LINEAGE_MISMATCH,
                f"row {row.canonical_observation_id} feature values differ from frozen projection",
            )
        expected_lineage = _compute_expected_row_lineage_identity(
            canonical_observation_id=row.canonical_observation_id,
            required_feature_ids=dataset.required_feature_ids,
            feature_values_in_required_order=row.feature_values_in_required_order,
        )
        if row.feature_lineage_identity != expected_lineage:
            raise Phase3ValidationError(
                Phase3FailureCode.FEATURE_LINEAGE_MISMATCH,
                f"row {row.canonical_observation_id} has mismatched feature_lineage_identity",
            )


def reconstruct_causal_matrix(*, dataset: Phase3CausalDataset, projection: CausalEligibilityProjection) -> Any:
    validate_phase3_causal_binding(dataset=dataset, projection=projection)
    import pandas as pd

    columns = ["canonical_observation_id", "session_index", "observation_timestamp_utc", "availability_timestamp_utc", *dataset.required_feature_ids]
    matrix_rows = []
    for row in dataset.ordered_rows:
        record = {
            "canonical_observation_id": row.canonical_observation_id,
            "session_index": row.session_index,
            "observation_timestamp_utc": row.observation_timestamp_utc,
            "availability_timestamp_utc": row.availability_timestamp_utc,
        }
        for feature_id, value in zip(dataset.required_feature_ids, row.feature_values_in_required_order):
            record[feature_id] = value
        matrix_rows.append(record)
    return pd.DataFrame(matrix_rows, columns=columns)


def validate_phase3_causal_binding(
    *,
    dataset: Phase3CausalDataset,
    projection: CausalEligibilityProjection,
) -> Phase3BindingValidation:
    _validate_dataset_against_projection(dataset, projection)
    binding_identity_digest = compute_phase3_binding_identity_digest(
        dataset_identity_digest=dataset.dataset_identity_digest,
        projection_digest=dataset.phase2_projection_digest,
        fold_id=dataset.fold_id,
        stage=dataset.stage,
        required_feature_ids=dataset.required_feature_ids,
        eligible_observation_ids=dataset.eligible_observation_ids,
    )
    return Phase3BindingValidation(
        projection_digest=dataset.phase2_projection_digest,
        dataset_identity_digest=dataset.dataset_identity_digest,
        binding_identity_digest=binding_identity_digest,
        verified=True,
    )


def validate_phase3_causal_binding_fail_closed(
    *,
    dataset: Phase3CausalDataset,
    projection: CausalEligibilityProjection,
) -> Phase3BindingValidation:
    try:
        return validate_phase3_causal_binding(dataset=dataset, projection=projection)
    except Phase3ValidationError as exc:
        return Phase3BindingValidation(
            projection_digest=dataset.phase2_projection_digest,
            dataset_identity_digest=dataset.dataset_identity_digest,
            binding_identity_digest="UNVERIFIED",
            verified=False,
            failure_code=exc.code,
            failure_detail=str(exc),
            missing_evidence=exc.missing_evidence,
        )

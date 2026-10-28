from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Sequence

from regimequant.walkforward.causal_projection import CausalEligibilityProjection
from regimequant.walkforward.identity import compute_identity_digest
from regimequant.walkforward.phase3_causal_binding import validate_phase3_causal_binding
from regimequant.walkforward.phase3_contracts import (
    Phase3CausalDataset,
    Phase3FailureCode,
    Phase3ValidationError,
)
from regimequant.walkforward.source_authority import (
    AuthorityLevel,
    EvidenceOrigin,
    FeatureRecordEvidence,
    FeatureSpecEvidence,
)


class FeatureRecordJoinFailureCode(str, Enum):
    BINDING_REQUIRED = "BINDING_REQUIRED"
    PROJECTION_DATASET_MISMATCH = "PROJECTION_DATASET_MISMATCH"
    CALENDAR_MISMATCH = "CALENDAR_MISMATCH"
    FOLD_STAGE_MISMATCH = "FOLD_STAGE_MISMATCH"
    MISSING_FEATURE_RECORD = "MISSING_FEATURE_RECORD"
    DUPLICATE_FEATURE_RECORD = "DUPLICATE_FEATURE_RECORD"
    AMBIGUOUS_SCOPE = "AMBIGUOUS_SCOPE"
    INSTRUMENT_MISMATCH = "INSTRUMENT_MISMATCH"
    FEATURE_SPEC_MISMATCH = "FEATURE_SPEC_MISMATCH"
    AVAILABILITY_MISMATCH = "AVAILABILITY_MISMATCH"
    PROVENANCE_FAILURE = "PROVENANCE_FAILURE"
    UNSUPPORTED_MAPPING = "UNSUPPORTED_MAPPING"
    NO_RECONCILABLE_EVIDENCE = "NO_RECONCILABLE_EVIDENCE"


@dataclass(frozen=True, slots=True)
class FeatureRecordJoinResult:
    verified: bool
    failure_code: FeatureRecordJoinFailureCode | None = None
    failure_detail: str | None = None
    matched_records: tuple[FeatureRecordEvidence, ...] = ()
    authority_role: str = "STRUCTURAL_ONLY"
    authoritative: bool = False
    reconciliation_digest: str = ""


def _reconciliation_digest(
    *,
    dataset: Phase3CausalDataset,
    projection: CausalEligibilityProjection,
    matched_records: Sequence[FeatureRecordEvidence],
    failure_code: FeatureRecordJoinFailureCode | None,
    failure_detail: str | None,
) -> str:
    payload = {
        "dataset_identity_digest": dataset.dataset_identity_digest,
        "projection_digest": projection.projection_digest,
        "fold_id": dataset.fold_id,
        "stage": dataset.stage.value,
        "required_feature_ids": list(dataset.required_feature_ids),
        "matched_evidence_identities": [record.evidence_digest for record in matched_records],
        "failure_code": failure_code.value if failure_code is not None else None,
        "failure_detail": failure_detail,
        "status": "success" if failure_code is None else "failure",
    }
    return compute_identity_digest(payload, domain="regimequant.walkforward.phase3b.reconciliation")


def _binding_failure_code(code: Phase3FailureCode) -> FeatureRecordJoinFailureCode:
    if code in {
        Phase3FailureCode.PROJECTION_DIGEST_MISMATCH,
        Phase3FailureCode.DATASET_IDENTITY_MISMATCH,
        Phase3FailureCode.PROJECTION_CONTENT_TAMPERING,
        Phase3FailureCode.DATASET_IDENTITY_DIGEST_MISMATCH,
    }:
        return FeatureRecordJoinFailureCode.PROJECTION_DATASET_MISMATCH
    if code in {
        Phase3FailureCode.CALENDAR_IDENTITY_MISMATCH,
        Phase3FailureCode.CALENDAR_VERSION_MISMATCH,
        Phase3FailureCode.SESSION_SEQUENCE_MISMATCH,
    }:
        return FeatureRecordJoinFailureCode.CALENDAR_MISMATCH
    if code in {
        Phase3FailureCode.FOLD_MISMATCH,
        Phase3FailureCode.STAGE_MISMATCH,
        Phase3FailureCode.MISSING_ELIGIBLE_ROW,
    }:
        return FeatureRecordJoinFailureCode.FOLD_STAGE_MISMATCH
    if code == Phase3FailureCode.PROJECTION_TYPE_MISMATCH:
        return FeatureRecordJoinFailureCode.BINDING_REQUIRED
    return FeatureRecordJoinFailureCode.BINDING_REQUIRED


def _require_record_provenance(record: FeatureRecordEvidence) -> None:
    if record.authority_level == AuthorityLevel.UNSUPPORTED:
        raise ValueError("unsupported mapping is not allowed")
    if record.origin in {EvidenceOrigin.CALLER_ASSERTED, EvidenceOrigin.UNSUPPORTED}:
        raise ValueError("caller-asserted provenance is not accepted")
    if not record.source_observation_ids:
        raise ValueError("source_observation_ids are required for deterministic match")


def _validate_binding_contract(
    *,
    dataset: Phase3CausalDataset,
    projection: CausalEligibilityProjection,
) -> None:
    if dataset.source_dataset_identity != projection.source_dataset_identity or dataset.phase2_projection_digest != projection.projection_digest:
        raise Phase3ValidationError(
            Phase3FailureCode.DATASET_IDENTITY_MISMATCH,
            "dataset/source evidence does not match projection identity",
        )

    if (
        dataset.session_calendar_id != projection.session_calendar_id
        or dataset.session_calendar_version != projection.session_calendar_version
        or dataset.session_sequence_digest != projection.session_sequence_digest
    ):
        raise Phase3ValidationError(
            Phase3FailureCode.CALENDAR_IDENTITY_MISMATCH,
            "dataset calendar/session metadata does not match projection",
        )

    if dataset.fold_id != projection.fold_id:
        raise Phase3ValidationError(
            Phase3FailureCode.FOLD_MISMATCH,
            "dataset fold_id does not match projection fold_id",
        )
    if dataset.stage.value != projection.stage.value:
        raise Phase3ValidationError(
            Phase3FailureCode.STAGE_MISMATCH,
            "dataset stage does not match projection stage",
        )
    if dataset.fold_interval != projection.fold_interval:
        raise Phase3ValidationError(
            Phase3FailureCode.FOLD_MISMATCH,
            "dataset fold_interval does not match projection fold_interval",
        )


def _require_structural_projection(
    *,
    dataset: Phase3CausalDataset,
    projection: object,
) -> CausalEligibilityProjection:
    if not isinstance(projection, CausalEligibilityProjection):
        raise TypeError("projection must be a real CausalEligibilityProjection instance")
    _validate_binding_contract(dataset=dataset, projection=projection)
    validate_phase3_causal_binding(dataset=dataset, projection=projection)
    return projection


def _normalize_version(value: str | None) -> str:
    if value is None:
        return ""
    return str(value).strip().lower().removeprefix("v")


def _feature_spec_matches(record: FeatureRecordEvidence, spec: FeatureSpecEvidence | None) -> bool:
    if spec is None:
        return False
    if spec.feature_name != record.feature_name:
        return False
    if record.feature_instance_id != spec.feature_instance_id_value:
        return False
    return True


def _has_authoritative_scope_mapping_artifact(
    *,
    dataset: Phase3CausalDataset,
    projection: CausalEligibilityProjection,
    feature_records: Sequence[FeatureRecordEvidence],
) -> bool:
    # Frozen Phase 3A/3B contracts do not expose a material artifact that binds
    # M2 instrument scope_id identities to Phase 3A canonical_observation_id values.
    _ = (dataset, projection, feature_records)
    return False


def _make_failure_result(
    *,
    dataset: Phase3CausalDataset,
    projection: CausalEligibilityProjection,
    matched_records: Sequence[FeatureRecordEvidence],
    failure_code: FeatureRecordJoinFailureCode,
    failure_detail: str,
) -> FeatureRecordJoinResult:
    return FeatureRecordJoinResult(
        verified=False,
        failure_code=failure_code,
        failure_detail=failure_detail,
        matched_records=tuple(matched_records),
        authority_role="STRUCTURAL_ONLY",
        authoritative=False,
        reconciliation_digest=_reconciliation_digest(
            dataset=dataset,
            projection=projection,
            matched_records=matched_records,
            failure_code=failure_code,
            failure_detail=failure_detail,
        ),
    )


def reconcile_feature_records(
    *,
    dataset: Phase3CausalDataset,
    projection: CausalEligibilityProjection,
    feature_records: Sequence[FeatureRecordEvidence],
    feature_specs: Sequence[FeatureSpecEvidence] | None = None,
) -> FeatureRecordJoinResult:
    if not isinstance(dataset, Phase3CausalDataset):
        raise TypeError("dataset must be a Phase3CausalDataset")
    if not isinstance(projection, CausalEligibilityProjection):
        return FeatureRecordJoinResult(
            verified=False,
            failure_code=FeatureRecordJoinFailureCode.BINDING_REQUIRED,
            failure_detail="projection must be a real CausalEligibilityProjection instance",
            authority_role="STRUCTURAL_ONLY",
            authoritative=False,
        )

    try:
        _require_structural_projection(dataset=dataset, projection=projection)
    except Phase3ValidationError as exc:
        return FeatureRecordJoinResult(
            verified=False,
            failure_code=_binding_failure_code(exc.code),
            failure_detail=str(exc),
            authority_role="STRUCTURAL_ONLY",
            authoritative=False,
            reconciliation_digest=_reconciliation_digest(
                dataset=dataset,
                projection=projection,
                matched_records=(),
                failure_code=_binding_failure_code(exc.code),
                failure_detail=str(exc),
            ),
        )

    if feature_specs is None:
        return _make_failure_result(
            dataset=dataset,
            projection=projection,
            matched_records=(),
            failure_code=FeatureRecordJoinFailureCode.FEATURE_SPEC_MISMATCH,
            failure_detail="feature_specs are required for structural reconciliation",
        )

    if len(feature_specs) == 0:
        return _make_failure_result(
            dataset=dataset,
            projection=projection,
            matched_records=(),
            failure_code=FeatureRecordJoinFailureCode.FEATURE_SPEC_MISMATCH,
            failure_detail="feature_specs cannot be empty when required_feature_ids are present",
        )

    spec_by_name = {spec.feature_name: spec for spec in feature_specs}
    if len(spec_by_name) != len(feature_specs):
        return _make_failure_result(
            dataset=dataset,
            projection=projection,
            matched_records=(),
            failure_code=FeatureRecordJoinFailureCode.FEATURE_SPEC_MISMATCH,
            failure_detail="duplicate feature specs are not allowed",
        )

    missing_spec_names = [name for name in dataset.required_feature_ids if name not in spec_by_name]
    if missing_spec_names:
        return _make_failure_result(
            dataset=dataset,
            projection=projection,
            matched_records=(),
            failure_code=FeatureRecordJoinFailureCode.FEATURE_SPEC_MISMATCH,
            failure_detail=f"missing required feature specs: {tuple(missing_spec_names)!r}",
        )

    expected_feature_record_count = len(dataset.ordered_rows) * len(dataset.required_feature_ids)
    if expected_feature_record_count == 0:
        return _make_failure_result(
            dataset=dataset,
            projection=projection,
            matched_records=(),
            failure_code=FeatureRecordJoinFailureCode.NO_RECONCILABLE_EVIDENCE,
            failure_detail=(
                "no observation-feature evidence is eligible for source reconciliation "
                "under the validated Phase 3A structural binding"
            ),
        )

    if expected_feature_record_count > 0 and len(feature_records) == 0:
        return _make_failure_result(
            dataset=dataset,
            projection=projection,
            matched_records=(),
            failure_code=FeatureRecordJoinFailureCode.MISSING_FEATURE_RECORD,
            failure_detail=(
                "feature_records are required for deterministic reconciliation; "
                f"expected at least {expected_feature_record_count} records"
            ),
        )

    if 0 < len(feature_records) < expected_feature_record_count:
        return _make_failure_result(
            dataset=dataset,
            projection=projection,
            matched_records=(),
            failure_code=FeatureRecordJoinFailureCode.MISSING_FEATURE_RECORD,
            failure_detail=(
                "incomplete feature record coverage; "
                f"expected at least {expected_feature_record_count} records, got {len(feature_records)}"
            ),
        )

    seen: set[tuple[str, datetime, str, str]] = set()
    scope_feature_map: dict[tuple[str, str], list[FeatureRecordEvidence]] = {}
    for record in feature_records:
        try:
            _require_record_provenance(record)
        except ValueError as exc:
            if "unsupported mapping" in str(exc):
                return _make_failure_result(
                    dataset=dataset,
                    projection=projection,
                    matched_records=(),
                    failure_code=FeatureRecordJoinFailureCode.UNSUPPORTED_MAPPING,
                    failure_detail=str(exc),
                )
            return _make_failure_result(
                dataset=dataset,
                projection=projection,
                matched_records=(),
                failure_code=FeatureRecordJoinFailureCode.PROVENANCE_FAILURE,
                failure_detail=str(exc),
            )

        key = (record.scope_id, record.timestamp_utc, record.feature_name, record.feature_instance_id)
        if key in seen:
            return _make_failure_result(
                dataset=dataset,
                projection=projection,
                matched_records=(),
                failure_code=FeatureRecordJoinFailureCode.DUPLICATE_FEATURE_RECORD,
                failure_detail=f"duplicate feature record for {key!r}",
            )
        seen.add(key)
        scope_feature_map.setdefault((record.scope_id, record.feature_name), []).append(record)

    for record in feature_records:
        if record.feature_name not in dataset.required_feature_ids:
            return _make_failure_result(
                dataset=dataset,
                projection=projection,
                matched_records=(),
                failure_code=FeatureRecordJoinFailureCode.MISSING_FEATURE_RECORD,
                failure_detail=f"unexpected feature record for {record.feature_name}",
            )
        if not _feature_spec_matches(record, spec_by_name.get(record.feature_name)):
            return _make_failure_result(
                dataset=dataset,
                projection=projection,
                matched_records=(),
                failure_code=FeatureRecordJoinFailureCode.FEATURE_SPEC_MISMATCH,
                failure_detail=f"feature spec mismatch for {record.feature_name}",
            )

    if feature_records and dataset.ordered_rows and not _has_authoritative_scope_mapping_artifact(
        dataset=dataset,
        projection=projection,
        feature_records=feature_records,
    ):
        return _make_failure_result(
            dataset=dataset,
            projection=projection,
            matched_records=(),
            failure_code=FeatureRecordJoinFailureCode.UNSUPPORTED_MAPPING,
            failure_detail=(
                "no frozen authoritative mapping artifact exists between M2 scope_id "
                "and Phase 3A canonical_observation_id; structural binding is valid but "
                "feature-record source reconciliation is unsupported"
            ),
        )

    return FeatureRecordJoinResult(
        verified=True,
        failure_code=None,
        failure_detail=None,
        matched_records=tuple(feature_records),
        authority_role="STRUCTURAL_ONLY",
        authoritative=False,
        reconciliation_digest=_reconciliation_digest(
            dataset=dataset,
            projection=projection,
            matched_records=feature_records,
            failure_code=None,
            failure_detail=None,
        ),
    )

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

from regimequant.walkforward.causal_projection import (
    CandidateObservation,
    ProjectionStage,
    build_causal_eligibility_projection,
)
from regimequant.walkforward.contracts import FoldInterval
from regimequant.walkforward.identity import compute_identity_digest
from regimequant.walkforward.phase3_contracts import (
    Phase3CausalRow,
    Phase3Stage,
    build_phase3_causal_dataset,
    compute_phase3_row_lineage_identity_payload,
)
from regimequant.walkforward.source_authority import (
    AuthorityLevel,
    EvidenceOrigin,
    FeatureRecordEvidence,
    FeatureSpecEvidence,
)
from regimequant.walkforward.source_join import (
    FeatureRecordJoinFailureCode,
    FeatureRecordJoinResult,
    reconcile_feature_records,
)


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def _canonical_feature_instance_id(
    *,
    feature_name: str,
    feature_definition_version: str,
    input_return_type: str = "simple",
    output_transformation_mode: str = "native",
    window_kind: str = "observation_count",
    window_size: int = 3,
    min_observations: int = 3,
    include_current: bool = True,
    require_complete_window: bool = False,
    allow_partial_valid: bool = True,
    ddof: int | None = None,
    downside_threshold: float = 0.0,
    short_window_size: int | None = None,
    long_window_size: int | None = None,
) -> str:
    payload = {
        "feature_name": feature_name,
        "feature_definition_version": feature_definition_version,
        "input_return_type": input_return_type,
        "output_transformation_mode": output_transformation_mode,
        "window_kind": window_kind,
        "window_size": window_size,
        "min_observations": min_observations,
        "include_current": include_current,
        "require_complete_window": require_complete_window,
        "allow_partial_valid": allow_partial_valid,
        "ddof": ddof,
        "downside_threshold": downside_threshold,
        "short_window_size": short_window_size,
        "long_window_size": long_window_size,
    }
    return compute_identity_digest(payload, domain="regimequant.walkforward.phase3b.feature_spec_identity")


def _build_dataset_projection_and_records():
    feature_ids = ("momentum_3", "volatility_5")
    momentum_instance_id = _canonical_feature_instance_id(
        feature_name="momentum_3",
        feature_definition_version="v1",
    )
    volatility_instance_id = _canonical_feature_instance_id(
        feature_name="volatility_5",
        feature_definition_version="v1",
    )
    interval = FoldInterval(
        train_start=0,
        train_end=10,
        validation_start=11,
        validation_end=20,
        test_start=21,
        test_end=30,
        holdout_start=31,
        holdout_end=40,
    )
    observations = (
        CandidateObservation(
            canonical_observation_id="obs-1",
            session_index=0,
            observation_timestamp_utc=_ts("2024-01-02T00:00:00+00:00"),
            availability_timestamp_utc=_ts("2024-01-02T00:01:00+00:00"),
            identity_verified=True,
            features={"momentum_3": 0.11, "volatility_5": 0.22},
        ),
        CandidateObservation(
            canonical_observation_id="obs-2",
            session_index=1,
            observation_timestamp_utc=_ts("2024-01-03T00:00:00+00:00"),
            availability_timestamp_utc=_ts("2024-01-03T00:01:00+00:00"),
            identity_verified=True,
            features={"momentum_3": 0.33, "volatility_5": 0.44},
        ),
    )
    projection = build_causal_eligibility_projection(
        dataset_identity="dataset-alpha",
        calendar_id="calendar-alpha",
        calendar_version="v1",
        calendar_sequence_digest="seq-alpha",
        stage=ProjectionStage.TRAIN,
        fold_id="fold-a",
        fold_interval=interval,
        stage_information_cutoff_utc=_ts("2024-01-31T00:00:00+00:00"),
        boundary_session_index=10,
        observations=observations,
        required_feature_ids=feature_ids,
        m3_policy_identity="m3-policy",
        m5_policy_identity="m5-policy",
    )
    rows = tuple(
        Phase3CausalRow(
            canonical_observation_id=obs.canonical_observation_id,
            session_index=obs.session_index,
            observation_timestamp_utc=obs.observation_timestamp_utc,
            availability_timestamp_utc=obs.availability_timestamp_utc,
            availability_verified=True,
            feature_values_in_required_order=(float(obs.features[feature_ids[0]]), float(obs.features[feature_ids[1]])),
            feature_lineage_identity=compute_identity_digest(
                compute_phase3_row_lineage_identity_payload(
                    canonical_observation_id=obs.canonical_observation_id,
                    required_feature_ids=feature_ids,
                    feature_values_in_required_order=(
                        float(obs.features[feature_ids[0]]),
                        float(obs.features[feature_ids[1]]),
                    ),
                ),
                domain="regimequant.walkforward.phase3a.row_lineage",
            ),
        )
        for obs in observations
    )
    dataset = build_phase3_causal_dataset(
        source_dataset_identity="dataset-alpha",
        session_calendar_id="calendar-alpha",
        session_calendar_version="v1",
        session_sequence_digest="seq-alpha",
        fold_id="fold-a",
        stage=Phase3Stage.TRAIN,
        fold_interval=projection.fold_interval,
        stage_information_cutoff_utc=projection.stage_information_cutoff_utc,
        phase2_projection_digest=projection.projection_digest,
        required_feature_ids=feature_ids,
        eligible_observation_ids=projection.eligible_observation_ids,
        ordered_rows=rows,
    )
    feature_records = (
        FeatureRecordEvidence(
            scope_id="obs-1",
            timestamp_utc=_ts("2024-01-02T00:00:00+00:00"),
            feature_name="momentum_3",
            feature_instance_id=momentum_instance_id,
            feature_value=0.11,
            source_observation_ids=("obs-1",),
            origin=EvidenceOrigin.DIRECT_SOURCE,
            authority_level=AuthorityLevel.STRUCTURAL_ONLY,
            available_at_utc=_ts("2024-01-02T00:01:00+00:00"),
        ),
        FeatureRecordEvidence(
            scope_id="obs-1",
            timestamp_utc=_ts("2024-01-02T00:00:00+00:00"),
            feature_name="volatility_5",
            feature_instance_id=volatility_instance_id,
            feature_value=0.22,
            source_observation_ids=("obs-1",),
            origin=EvidenceOrigin.DIRECT_SOURCE,
            authority_level=AuthorityLevel.STRUCTURAL_ONLY,
            available_at_utc=_ts("2024-01-02T00:01:00+00:00"),
        ),
        FeatureRecordEvidence(
            scope_id="obs-2",
            timestamp_utc=_ts("2024-01-03T00:00:00+00:00"),
            feature_name="momentum_3",
            feature_instance_id=momentum_instance_id,
            feature_value=0.33,
            source_observation_ids=("obs-2",),
            origin=EvidenceOrigin.DIRECT_SOURCE,
            authority_level=AuthorityLevel.STRUCTURAL_ONLY,
            available_at_utc=_ts("2024-01-03T00:01:00+00:00"),
        ),
        FeatureRecordEvidence(
            scope_id="obs-2",
            timestamp_utc=_ts("2024-01-03T00:00:00+00:00"),
            feature_name="volatility_5",
            feature_instance_id=volatility_instance_id,
            feature_value=0.44,
            source_observation_ids=("obs-2",),
            origin=EvidenceOrigin.DIRECT_SOURCE,
            authority_level=AuthorityLevel.STRUCTURAL_ONLY,
            available_at_utc=_ts("2024-01-03T00:01:00+00:00"),
        ),
    )
    feature_specs = (
        FeatureSpecEvidence(
            feature_name="momentum_3",
            feature_definition_version="v1",
            input_return_type="simple",
            output_transformation_mode="native",
            window_kind="observation_count",
            window_size=3,
            min_observations=3,
            origin=EvidenceOrigin.DIRECT_SOURCE,
            authority_level=AuthorityLevel.STRUCTURAL_ONLY,
        ),
        FeatureSpecEvidence(
            feature_name="volatility_5",
            feature_definition_version="v1",
            input_return_type="simple",
            output_transformation_mode="native",
            window_kind="observation_count",
            window_size=3,
            min_observations=3,
            origin=EvidenceOrigin.DIRECT_SOURCE,
            authority_level=AuthorityLevel.STRUCTURAL_ONLY,
        ),
    )
    return dataset, projection, feature_records, feature_specs


def _build_zero_eligible_dataset_and_projection():
    feature_ids = ("momentum_3", "volatility_5")
    interval = FoldInterval(
        train_start=0,
        train_end=10,
        validation_start=11,
        validation_end=20,
        test_start=21,
        test_end=30,
        holdout_start=31,
        holdout_end=40,
    )
    observations = (
        CandidateObservation(
            canonical_observation_id="obs-z1",
            session_index=0,
            observation_timestamp_utc=_ts("2024-01-02T00:00:00+00:00"),
            availability_timestamp_utc=_ts("2024-01-02T00:01:00+00:00"),
            identity_verified=False,
            features={"momentum_3": 0.11, "volatility_5": 0.22},
        ),
        CandidateObservation(
            canonical_observation_id="obs-z2",
            session_index=1,
            observation_timestamp_utc=_ts("2024-01-03T00:00:00+00:00"),
            availability_timestamp_utc=_ts("2024-01-03T00:01:00+00:00"),
            identity_verified=False,
            features={"momentum_3": 0.33, "volatility_5": 0.44},
        ),
    )
    projection = build_causal_eligibility_projection(
        dataset_identity="dataset-zero-eligible",
        calendar_id="calendar-alpha",
        calendar_version="v1",
        calendar_sequence_digest="seq-alpha",
        stage=ProjectionStage.TRAIN,
        fold_id="fold-a",
        fold_interval=interval,
        stage_information_cutoff_utc=_ts("2024-01-31T00:00:00+00:00"),
        boundary_session_index=10,
        observations=observations,
        required_feature_ids=feature_ids,
        m3_policy_identity="m3-policy",
        m5_policy_identity="m5-policy",
    )
    dataset = build_phase3_causal_dataset(
        source_dataset_identity="dataset-zero-eligible",
        session_calendar_id="calendar-alpha",
        session_calendar_version="v1",
        session_sequence_digest="seq-alpha",
        fold_id="fold-a",
        stage=Phase3Stage.TRAIN,
        fold_interval=projection.fold_interval,
        stage_information_cutoff_utc=projection.stage_information_cutoff_utc,
        phase2_projection_digest=projection.projection_digest,
        required_feature_ids=feature_ids,
        eligible_observation_ids=projection.eligible_observation_ids,
        ordered_rows=(),
    )
    feature_specs = (
        FeatureSpecEvidence(
            feature_name="momentum_3",
            feature_definition_version="v1",
            input_return_type="simple",
            output_transformation_mode="native",
            window_kind="observation_count",
            window_size=3,
            min_observations=3,
            origin=EvidenceOrigin.DIRECT_SOURCE,
            authority_level=AuthorityLevel.STRUCTURAL_ONLY,
        ),
        FeatureSpecEvidence(
            feature_name="volatility_5",
            feature_definition_version="v1",
            input_return_type="simple",
            output_transformation_mode="native",
            window_kind="observation_count",
            window_size=3,
            min_observations=3,
            origin=EvidenceOrigin.DIRECT_SOURCE,
            authority_level=AuthorityLevel.STRUCTURAL_ONLY,
        ),
    )
    return dataset, projection, feature_specs


def test_reconciliation_valid_phase3a_binding_but_source_mapping_is_unsupported() -> None:
    dataset, projection, feature_records, feature_specs = _build_dataset_projection_and_records()
    result = reconcile_feature_records(
        dataset=dataset,
        projection=projection,
        feature_records=feature_records,
        feature_specs=feature_specs,
    )
    assert isinstance(result, FeatureRecordJoinResult)
    assert result.verified is False
    assert result.failure_code == FeatureRecordJoinFailureCode.UNSUPPORTED_MAPPING
    assert "no frozen authoritative mapping artifact exists" in (result.failure_detail or "")
    assert result.authority_role == "STRUCTURAL_ONLY"
    assert result.authoritative is False
    assert result.reconciliation_digest


def test_reconciliation_zero_eligible_rows_abstains_no_reconcilable_evidence() -> None:
    dataset, projection, feature_specs = _build_zero_eligible_dataset_and_projection()
    result = reconcile_feature_records(
        dataset=dataset,
        projection=projection,
        feature_records=(),
        feature_specs=feature_specs,
    )
    assert result.verified is False
    assert result.failure_code == FeatureRecordJoinFailureCode.NO_RECONCILABLE_EVIDENCE
    assert result.matched_records == ()
    assert result.reconciliation_digest


def test_reconciliation_rejects_invalid_frozen_binding() -> None:
    dataset, projection, feature_records, feature_specs = _build_dataset_projection_and_records()
    bad_projection = replace(projection, stage=ProjectionStage.VALIDATION)
    result = reconcile_feature_records(
        dataset=dataset,
        projection=bad_projection,
        feature_records=feature_records,
        feature_specs=feature_specs,
    )
    assert result.verified is False
    assert result.failure_code == FeatureRecordJoinFailureCode.FOLD_STAGE_MISMATCH


def test_reconciliation_rejects_projection_dataset_mismatch() -> None:
    dataset, projection, feature_records, feature_specs = _build_dataset_projection_and_records()
    bad_projection = replace(projection, source_dataset_identity="dataset-beta")
    result = reconcile_feature_records(
        dataset=dataset,
        projection=bad_projection,
        feature_records=feature_records,
        feature_specs=feature_specs,
    )
    assert result.verified is False
    assert result.failure_code == FeatureRecordJoinFailureCode.PROJECTION_DATASET_MISMATCH


def test_reconciliation_rejects_calendar_mismatch() -> None:
    dataset, projection, feature_records, feature_specs = _build_dataset_projection_and_records()
    bad_projection = replace(projection, session_calendar_id="calendar-beta")
    result = reconcile_feature_records(
        dataset=dataset,
        projection=bad_projection,
        feature_records=feature_records,
        feature_specs=feature_specs,
    )
    assert result.verified is False
    assert result.failure_code == FeatureRecordJoinFailureCode.CALENDAR_MISMATCH


def test_reconciliation_rejects_fold_stage_mismatch() -> None:
    dataset, projection, feature_records, feature_specs = _build_dataset_projection_and_records()
    bad_projection = replace(projection, fold_id="fold-b")
    result = reconcile_feature_records(
        dataset=dataset,
        projection=bad_projection,
        feature_records=feature_records,
        feature_specs=feature_specs,
    )
    assert result.verified is False
    assert result.failure_code == FeatureRecordJoinFailureCode.FOLD_STAGE_MISMATCH


def test_reconciliation_reports_missing_feature_records() -> None:
    dataset, projection, feature_records, feature_specs = _build_dataset_projection_and_records()
    result = reconcile_feature_records(
        dataset=dataset,
        projection=projection,
        feature_records=feature_records[:-1],
        feature_specs=feature_specs,
    )
    assert result.verified is False
    assert result.failure_code == FeatureRecordJoinFailureCode.MISSING_FEATURE_RECORD


def test_reconciliation_empty_records_with_valid_specs_fails_missing_feature_record() -> None:
    dataset, projection, _feature_records, feature_specs = _build_dataset_projection_and_records()
    result = reconcile_feature_records(
        dataset=dataset,
        projection=projection,
        feature_records=(),
        feature_specs=feature_specs,
    )
    assert result.verified is False
    assert result.failure_code == FeatureRecordJoinFailureCode.MISSING_FEATURE_RECORD
    assert result.reconciliation_digest


def test_reconciliation_empty_records_with_empty_specs_fails_deterministically() -> None:
    dataset, projection, _feature_records, _feature_specs = _build_dataset_projection_and_records()
    first = reconcile_feature_records(
        dataset=dataset,
        projection=projection,
        feature_records=(),
        feature_specs=(),
    )
    second = reconcile_feature_records(
        dataset=dataset,
        projection=projection,
        feature_records=(),
        feature_specs=(),
    )
    assert first.verified is False
    assert first.failure_code == FeatureRecordJoinFailureCode.FEATURE_SPEC_MISMATCH
    assert first.reconciliation_digest
    assert first.reconciliation_digest == second.reconciliation_digest


def test_reconciliation_partial_feature_coverage_fails_missing_feature_record() -> None:
    dataset, projection, feature_records, feature_specs = _build_dataset_projection_and_records()
    partial = feature_records[:3]
    result = reconcile_feature_records(
        dataset=dataset,
        projection=projection,
        feature_records=partial,
        feature_specs=feature_specs,
    )
    assert result.verified is False
    assert result.failure_code == FeatureRecordJoinFailureCode.MISSING_FEATURE_RECORD


def test_reconciliation_rejects_duplicate_keys() -> None:
    dataset, projection, feature_records, feature_specs = _build_dataset_projection_and_records()
    duplicate = feature_records + (
        FeatureRecordEvidence(
            scope_id="obs-1",
            timestamp_utc=_ts("2024-01-02T00:00:00+00:00"),
            feature_name="momentum_3",
            feature_instance_id=_canonical_feature_instance_id(feature_name="momentum_3", feature_definition_version="v1"),
            feature_value=0.11,
            source_observation_ids=("obs-1",),
            origin=EvidenceOrigin.DIRECT_SOURCE,
            authority_level=AuthorityLevel.STRUCTURAL_ONLY,
        ),
    )
    result = reconcile_feature_records(
        dataset=dataset,
        projection=projection,
        feature_records=duplicate,
        feature_specs=feature_specs,
    )
    assert result.verified is False
    assert result.failure_code == FeatureRecordJoinFailureCode.DUPLICATE_FEATURE_RECORD


def test_reconciliation_rejects_legitimate_instrument_scopes_with_distinct_observation_identities() -> None:
    dataset, projection, feature_records, feature_specs = _build_dataset_projection_and_records()
    extra_record = FeatureRecordEvidence(
        scope_id="scope-99",
        timestamp_utc=_ts("2024-01-02T00:00:00+00:00"),
        feature_name="momentum_3",
        feature_instance_id=_canonical_feature_instance_id(feature_name="momentum_3", feature_definition_version="v1"),
        feature_value=0.11,
        source_observation_ids=("scope-99",),
        origin=EvidenceOrigin.DIRECT_SOURCE,
        authority_level=AuthorityLevel.STRUCTURAL_ONLY,
        available_at_utc=_ts("2024-01-02T00:01:00+00:00"),
    )
    altered = list(feature_records)
    altered[0] = FeatureRecordEvidence(
        scope_id="AAPL",
        timestamp_utc=_ts("2024-01-02T00:00:00+00:00"),
        feature_name="momentum_3",
        feature_instance_id=_canonical_feature_instance_id(feature_name="momentum_3", feature_definition_version="v1"),
        feature_value=0.11,
        source_observation_ids=("obs-1",),
        origin=EvidenceOrigin.DIRECT_SOURCE,
        authority_level=AuthorityLevel.STRUCTURAL_ONLY,
        available_at_utc=_ts("2024-01-02T00:01:00+00:00"),
    )
    altered[2] = FeatureRecordEvidence(
        scope_id="MSFT",
        timestamp_utc=_ts("2024-01-03T00:00:00+00:00"),
        feature_name="momentum_3",
        feature_instance_id=_canonical_feature_instance_id(feature_name="momentum_3", feature_definition_version="v1"),
        feature_value=0.33,
        source_observation_ids=("obs-2",),
        origin=EvidenceOrigin.DIRECT_SOURCE,
        authority_level=AuthorityLevel.STRUCTURAL_ONLY,
        available_at_utc=_ts("2024-01-03T00:01:00+00:00"),
    )

    result = reconcile_feature_records(
        dataset=dataset,
        projection=projection,
        feature_records=tuple(altered),
        feature_specs=feature_specs,
    )
    assert result.verified is False
    assert result.failure_code == FeatureRecordJoinFailureCode.UNSUPPORTED_MAPPING


def test_reconciliation_rejects_unmapped_scope_identity() -> None:
    dataset, projection, feature_records, feature_specs = _build_dataset_projection_and_records()
    altered = list(feature_records)
    altered[0] = FeatureRecordEvidence(
        scope_id="scope-obs-1",
        timestamp_utc=_ts("2024-01-02T00:00:00+00:00"),
        feature_name="momentum_3",
        feature_instance_id=_canonical_feature_instance_id(feature_name="momentum_3", feature_definition_version="v1"),
        feature_value=0.11,
        source_observation_ids=("scope-obs-1",),
        origin=EvidenceOrigin.DIRECT_SOURCE,
        authority_level=AuthorityLevel.STRUCTURAL_ONLY,
        available_at_utc=_ts("2024-01-02T00:01:00+00:00"),
    )
    result = reconcile_feature_records(
        dataset=dataset,
        projection=projection,
        feature_records=tuple(altered),
        feature_specs=feature_specs,
    )
    assert result.verified is False
    assert result.failure_code == FeatureRecordJoinFailureCode.UNSUPPORTED_MAPPING


def test_reconciliation_rejects_feature_spec_mismatch_even_with_similar_instance_strings() -> None:
    dataset, projection, feature_records, feature_specs = _build_dataset_projection_and_records()
    altered = list(feature_records)
    altered[0] = FeatureRecordEvidence(
        scope_id="obs-1",
        timestamp_utc=_ts("2024-01-02T00:00:00+00:00"),
        feature_name="momentum_3",
        feature_instance_id=_canonical_feature_instance_id(feature_name="momentum_3", feature_definition_version="v1")[:-3] + "abc",
        feature_value=0.11,
        source_observation_ids=("obs-1",),
        origin=EvidenceOrigin.DIRECT_SOURCE,
        authority_level=AuthorityLevel.STRUCTURAL_ONLY,
        available_at_utc=_ts("2024-01-02T00:01:00+00:00"),
    )
    result = reconcile_feature_records(
        dataset=dataset,
        projection=projection,
        feature_records=tuple(altered),
        feature_specs=feature_specs,
    )
    assert result.verified is False
    assert result.failure_code == FeatureRecordJoinFailureCode.FEATURE_SPEC_MISMATCH


def test_reconciliation_rejects_forged_additional_source_observation_membership() -> None:
    dataset, projection, feature_records, feature_specs = _build_dataset_projection_and_records()
    altered = list(feature_records)
    altered[0] = FeatureRecordEvidence(
        scope_id="obs-1",
        timestamp_utc=_ts("2024-01-02T00:00:00+00:00"),
        feature_name="momentum_3",
        feature_instance_id=_canonical_feature_instance_id(feature_name="momentum_3", feature_definition_version="v1"),
        feature_value=0.11,
        source_observation_ids=("obs-1", "forged-source-obs"),
        origin=EvidenceOrigin.DIRECT_SOURCE,
        authority_level=AuthorityLevel.STRUCTURAL_ONLY,
        available_at_utc=_ts("2024-01-02T00:01:00+00:00"),
    )
    result = reconcile_feature_records(
        dataset=dataset,
        projection=projection,
        feature_records=tuple(altered),
        feature_specs=feature_specs,
    )
    assert result.verified is False
    assert result.failure_code == FeatureRecordJoinFailureCode.UNSUPPORTED_MAPPING


def test_reconciliation_rejects_unverifiable_source_membership() -> None:
    dataset, projection, feature_records, feature_specs = _build_dataset_projection_and_records()
    altered = list(feature_records)
    altered[0] = FeatureRecordEvidence(
        scope_id="obs-1",
        timestamp_utc=_ts("2024-01-02T00:00:00+00:00"),
        feature_name="momentum_3",
        feature_instance_id=_canonical_feature_instance_id(feature_name="momentum_3", feature_definition_version="v1"),
        feature_value=0.11,
        source_observation_ids=("unverifiable-membership",),
        origin=EvidenceOrigin.DIRECT_SOURCE,
        authority_level=AuthorityLevel.STRUCTURAL_ONLY,
        available_at_utc=_ts("2024-01-02T00:01:00+00:00"),
    )
    result = reconcile_feature_records(
        dataset=dataset,
        projection=projection,
        feature_records=tuple(altered),
        feature_specs=feature_specs,
    )
    assert result.verified is False
    assert result.failure_code == FeatureRecordJoinFailureCode.UNSUPPORTED_MAPPING


def test_reconciliation_rejects_feature_spec_mismatch() -> None:
    dataset, projection, feature_records, feature_specs = _build_dataset_projection_and_records()
    altered_specs = list(feature_specs)
    altered_specs[0] = FeatureSpecEvidence(
        feature_name="momentum_3",
        feature_definition_version="v2",
        input_return_type="simple",
        output_transformation_mode="native",
        window_kind="observation_count",
        window_size=3,
        min_observations=3,
        origin=EvidenceOrigin.DIRECT_SOURCE,
        authority_level=AuthorityLevel.STRUCTURAL_ONLY,
    )
    result = reconcile_feature_records(
        dataset=dataset,
        projection=projection,
        feature_records=feature_records,
        feature_specs=tuple(altered_specs),
    )
    assert result.verified is False
    assert result.failure_code == FeatureRecordJoinFailureCode.FEATURE_SPEC_MISMATCH


def test_reconciliation_rejects_late_availability() -> None:
    dataset, projection, feature_records, feature_specs = _build_dataset_projection_and_records()
    altered = list(feature_records)
    altered[0] = FeatureRecordEvidence(
        scope_id="obs-1",
        timestamp_utc=_ts("2024-01-02T00:00:00+00:00"),
        feature_name="momentum_3",
        feature_instance_id=_canonical_feature_instance_id(feature_name="momentum_3", feature_definition_version="v1"),
        feature_value=0.11,
        source_observation_ids=("obs-1",),
        origin=EvidenceOrigin.DIRECT_SOURCE,
        authority_level=AuthorityLevel.STRUCTURAL_ONLY,
        available_at_utc=_ts("2024-02-01T00:00:00+00:00"),
    )
    result = reconcile_feature_records(
        dataset=dataset,
        projection=projection,
        feature_records=tuple(altered),
        feature_specs=feature_specs,
    )
    assert result.verified is False
    assert result.failure_code == FeatureRecordJoinFailureCode.UNSUPPORTED_MAPPING


def test_reconciliation_rejects_missing_provenance() -> None:
    dataset, projection, feature_records, feature_specs = _build_dataset_projection_and_records()
    altered = list(feature_records)
    altered[0] = FeatureRecordEvidence(
        scope_id="obs-1",
        timestamp_utc=_ts("2024-01-02T00:00:00+00:00"),
        feature_name="momentum_3",
        feature_instance_id=_canonical_feature_instance_id(feature_name="momentum_3", feature_definition_version="v1"),
        feature_value=0.11,
        source_observation_ids=(),
        origin=EvidenceOrigin.DIRECT_SOURCE,
        authority_level=AuthorityLevel.STRUCTURAL_ONLY,
    )
    result = reconcile_feature_records(
        dataset=dataset,
        projection=projection,
        feature_records=tuple(altered),
        feature_specs=feature_specs,
    )
    assert result.verified is False
    assert result.failure_code == FeatureRecordJoinFailureCode.PROVENANCE_FAILURE


def test_reconciliation_rejects_caller_forged_mapping() -> None:
    dataset, projection, feature_records, feature_specs = _build_dataset_projection_and_records()
    altered = list(feature_records)
    altered[0] = FeatureRecordEvidence(
        scope_id="obs-1",
        timestamp_utc=_ts("2024-01-02T00:00:00+00:00"),
        feature_name="momentum_3",
        feature_instance_id=_canonical_feature_instance_id(feature_name="momentum_3", feature_definition_version="v1"),
        feature_value=0.11,
        source_observation_ids=("obs-1",),
        origin=EvidenceOrigin.CALLER_ASSERTED,
        authority_level=AuthorityLevel.STRUCTURAL_ONLY,
    )
    result = reconcile_feature_records(
        dataset=dataset,
        projection=projection,
        feature_records=tuple(altered),
        feature_specs=feature_specs,
    )
    assert result.verified is False
    assert result.failure_code == FeatureRecordJoinFailureCode.PROVENANCE_FAILURE


def test_reconciliation_rejects_unsupported_mapping() -> None:
    dataset, projection, feature_records, feature_specs = _build_dataset_projection_and_records()
    altered = list(feature_records)
    altered[0] = FeatureRecordEvidence(
        scope_id="obs-1",
        timestamp_utc=_ts("2024-01-02T00:00:00+00:00"),
        feature_name="momentum_3",
        feature_instance_id=_canonical_feature_instance_id(feature_name="momentum_3", feature_definition_version="v1"),
        feature_value=0.11,
        source_observation_ids=("obs-1",),
        origin=EvidenceOrigin.DIRECT_SOURCE,
        authority_level=AuthorityLevel.UNSUPPORTED,
    )
    result = reconcile_feature_records(
        dataset=dataset,
        projection=projection,
        feature_records=tuple(altered),
        feature_specs=feature_specs,
    )
    assert result.verified is False
    assert result.failure_code == FeatureRecordJoinFailureCode.UNSUPPORTED_MAPPING


def test_reconciliation_digest_is_deterministic() -> None:
    dataset, projection, feature_records, feature_specs = _build_dataset_projection_and_records()
    first = reconcile_feature_records(
        dataset=dataset,
        projection=projection,
        feature_records=feature_records,
        feature_specs=feature_specs,
    )
    second = reconcile_feature_records(
        dataset=dataset,
        projection=projection,
        feature_records=feature_records,
        feature_specs=feature_specs,
    )
    assert first.reconciliation_digest == second.reconciliation_digest
    assert first.reconciliation_digest
    assert first.verified is False
    assert first.failure_code == FeatureRecordJoinFailureCode.UNSUPPORTED_MAPPING

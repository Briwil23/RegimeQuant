from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from regimequant.walkforward.causal_projection import (
    CandidateObservation,
    ProjectionStage,
    build_causal_eligibility_projection,
)
from regimequant.walkforward.fold_geometry import FoldInterval
from regimequant.walkforward.identity import compute_identity_digest
from regimequant.walkforward.phase3_causal_binding import (
    ROW_LINEAGE_DOMAIN,
    reconstruct_causal_matrix,
    validate_phase3_causal_binding,
    validate_phase3_causal_binding_fail_closed,
)
from regimequant.walkforward.phase3_contracts import (
    Phase3CausalDataset,
    Phase3CausalRow,
    Phase3FailureCode,
    Phase3FrozenModelConfig,
    Phase3M3ComparatorConfig,
    Phase3M5PrimaryConfig,
    Phase3Stage,
    Phase3ValidationError,
    build_phase3_causal_dataset,
    build_phase3_frozen_model_config,
    compute_phase3_causal_dataset_identity_payload,
    compute_phase3_row_lineage_identity_payload,
)
from regimequant.walkforward.phase3_identity import (
    PHASE2_PROJECTION_DOMAIN,
    PHASE3_BINDING_DOMAIN,
    PHASE3_DATASET_DOMAIN,
    PHASE3_MODEL_CONFIG_DOMAIN,
    compute_phase2_projection_identity_digest,
    compute_phase3_binding_identity_digest,
    compute_phase3_dataset_identity_digest,
)


def _dt(hour: int) -> datetime:
    return datetime(2024, 1, 2, hour, 0, tzinfo=timezone.utc)


def _interval() -> FoldInterval:
    return FoldInterval(
        train_start=0,
        train_end=3,
        validation_start=3,
        validation_end=5,
        test_start=5,
        test_end=7,
        holdout_start=7,
        holdout_end=8,
    )


def _lineage(obs_id: str, features: tuple[str, ...], values: tuple[float, ...]) -> str:
    payload = compute_phase3_row_lineage_identity_payload(
        canonical_observation_id=obs_id,
        required_feature_ids=features,
        feature_values_in_required_order=values,
    )
    return compute_identity_digest(payload, domain=ROW_LINEAGE_DOMAIN)


def _stage_boundary(stage: Phase3Stage, interval: FoldInterval) -> int:
    if stage == Phase3Stage.TRAIN:
        return interval.train_end
    if stage == Phase3Stage.VALIDATION:
        return interval.validation_end
    return interval.test_end


def _projected_observations(rows: tuple[Phase3CausalRow, ...], features: tuple[str, ...]) -> tuple[CandidateObservation, ...]:
    observations: list[CandidateObservation] = []
    for row in rows:
        feature_map = {feature_id: value for feature_id, value in zip(features, row.feature_values_in_required_order)}
        observations.append(
            CandidateObservation(
                canonical_observation_id=row.canonical_observation_id,
                session_index=row.session_index,
                observation_timestamp_utc=row.observation_timestamp_utc,
                availability_timestamp_utc=row.availability_timestamp_utc,
                identity_verified=True,
                features=feature_map,
            )
        )
    return tuple(observations)


def _dataset(*, stage: Phase3Stage = Phase3Stage.TRAIN, ids: tuple[str, ...] = ("obs0", "obs1", "obs2")) -> tuple[Phase3CausalDataset, object]:
    features = ("f1", "f2")
    interval = _interval()
    if stage == Phase3Stage.TRAIN:
        start_index = interval.train_start
    elif stage == Phase3Stage.VALIDATION:
        start_index = interval.validation_start
    else:
        start_index = interval.test_start
    rows = []
    for idx, obs in enumerate(ids):
        values = (float(idx + 1), float((idx + 1) * 2))
        rows.append(
            Phase3CausalRow(
                canonical_observation_id=obs,
                session_index=start_index + idx,
                observation_timestamp_utc=_dt(1 + idx),
                availability_timestamp_utc=_dt(1 + idx),
                availability_verified=True,
                feature_values_in_required_order=values,
                feature_lineage_identity=_lineage(obs, features, values),
            )
        )
    projection = build_causal_eligibility_projection(
        dataset_identity="source:dataset:v1",
        calendar_id="calendar-1",
        calendar_version="v1",
        calendar_sequence_digest="seq:abc",
        stage=ProjectionStage(stage.value),
        fold_id="fold-001",
        fold_interval=interval,
        stage_information_cutoff_utc=_dt(8),
        boundary_session_index=_stage_boundary(stage, interval),
        observations=_projected_observations(tuple(rows), features),
        required_feature_ids=features,
        m3_policy_identity="m3:test-policy",
        m5_policy_identity="m5:test-policy",
    )
    dataset = build_phase3_causal_dataset(
        source_dataset_identity="source:dataset:v1",
        session_calendar_id="calendar-1",
        session_calendar_version="v1",
        session_sequence_digest="seq:abc",
        fold_id="fold-001",
        stage=stage,
        fold_interval=interval,
        stage_information_cutoff_utc=_dt(8),
        phase2_projection_digest=projection.projection_digest,
        required_feature_ids=features,
        eligible_observation_ids=ids,
        ordered_rows=tuple(rows),
    )
    return dataset, projection


def test_dataset_identity_is_domain_separated_and_deterministic() -> None:
    dataset, _ = _dataset()
    payload = compute_phase3_causal_dataset_identity_payload(dataset)
    expected = compute_identity_digest(payload, domain=PHASE3_DATASET_DOMAIN)
    assert dataset.dataset_identity_digest == expected
    assert dataset.dataset_identity_digest == compute_phase3_dataset_identity_digest(dataset)


def test_model_config_identity_is_domain_separated_and_deterministic() -> None:
    config = build_phase3_frozen_model_config(
        m5_primary=Phase3M5PrimaryConfig(n_components=3),
        m3_comparator=Phase3M3ComparatorConfig(n_clusters=3),
    )
    expected = compute_identity_digest(
        {
            "m5_is_primary": True,
            "m5_primary": {
                "n_components": 3,
                "covariance_type": "diag",
                "random_state": 42,
                "n_init": 10,
                "max_iter": 500,
                "tol": 1e-4,
                "reg_covar": 1e-6,
            },
            "m3_comparator": {"n_clusters": 3},
        },
        domain=PHASE3_MODEL_CONFIG_DOMAIN,
    )
    assert config.config_identity_digest == expected


def test_phase2_projection_digest_matches_material() -> None:
    dataset, projection = _dataset()
    expected = compute_phase2_projection_identity_digest(projection)
    assert projection.projection_digest == expected
    assert dataset.phase2_projection_digest == expected


def test_binding_accepts_real_projection_dataset_pair() -> None:
    dataset, projection = _dataset()
    result = validate_phase3_causal_binding(dataset=dataset, projection=projection)
    assert result.verified is True
    assert result.binding_identity_digest == compute_phase3_binding_identity_digest(
        dataset_identity_digest=dataset.dataset_identity_digest,
        projection_digest=dataset.phase2_projection_digest,
        fold_id=dataset.fold_id,
        stage=dataset.stage,
        required_feature_ids=dataset.required_feature_ids,
        eligible_observation_ids=dataset.eligible_observation_ids,
    )


def test_binding_rejects_calendar_version_mismatch() -> None:
    dataset, projection = _dataset()
    tampered_dataset = build_phase3_causal_dataset(
        source_dataset_identity=dataset.source_dataset_identity,
        session_calendar_id=dataset.session_calendar_id,
        session_calendar_version="v9",
        session_sequence_digest=dataset.session_sequence_digest,
        fold_id=dataset.fold_id,
        stage=dataset.stage,
        fold_interval=dataset.fold_interval,
        stage_information_cutoff_utc=dataset.stage_information_cutoff_utc,
        phase2_projection_digest=dataset.phase2_projection_digest,
        required_feature_ids=dataset.required_feature_ids,
        eligible_observation_ids=dataset.eligible_observation_ids,
        ordered_rows=dataset.ordered_rows,
    )
    result = validate_phase3_causal_binding_fail_closed(dataset=tampered_dataset, projection=projection)
    assert result.verified is False
    assert result.failure_code == Phase3FailureCode.CALENDAR_VERSION_MISMATCH


def test_binding_rejects_fold_id_mismatch() -> None:
    dataset, projection = _dataset()
    tampered_dataset = build_phase3_causal_dataset(
        source_dataset_identity=dataset.source_dataset_identity,
        session_calendar_id=dataset.session_calendar_id,
        session_calendar_version=dataset.session_calendar_version,
        session_sequence_digest=dataset.session_sequence_digest,
        fold_id="fold-999",
        stage=dataset.stage,
        fold_interval=dataset.fold_interval,
        stage_information_cutoff_utc=dataset.stage_information_cutoff_utc,
        phase2_projection_digest=dataset.phase2_projection_digest,
        required_feature_ids=dataset.required_feature_ids,
        eligible_observation_ids=dataset.eligible_observation_ids,
        ordered_rows=dataset.ordered_rows,
    )
    result = validate_phase3_causal_binding_fail_closed(dataset=tampered_dataset, projection=projection)
    assert result.verified is False
    assert result.failure_code == Phase3FailureCode.FOLD_MISMATCH


def test_binding_rejects_stage_mismatch() -> None:
    train_dataset, projection = _dataset(stage=Phase3Stage.TRAIN, ids=("obs0", "obs1"))
    validation_dataset = build_phase3_causal_dataset(
        source_dataset_identity=train_dataset.source_dataset_identity,
        session_calendar_id=train_dataset.session_calendar_id,
        session_calendar_version=train_dataset.session_calendar_version,
        session_sequence_digest=train_dataset.session_sequence_digest,
        fold_id=train_dataset.fold_id,
        stage=Phase3Stage.VALIDATION,
        fold_interval=train_dataset.fold_interval,
        stage_information_cutoff_utc=train_dataset.stage_information_cutoff_utc,
        phase2_projection_digest=projection.projection_digest,
        required_feature_ids=train_dataset.required_feature_ids,
        eligible_observation_ids=train_dataset.eligible_observation_ids,
        ordered_rows=(
            Phase3CausalRow(
                canonical_observation_id=train_dataset.ordered_rows[0].canonical_observation_id,
                session_index=3,
                observation_timestamp_utc=train_dataset.ordered_rows[0].observation_timestamp_utc,
                availability_timestamp_utc=train_dataset.ordered_rows[0].availability_timestamp_utc,
                availability_verified=True,
                feature_values_in_required_order=train_dataset.ordered_rows[0].feature_values_in_required_order,
                feature_lineage_identity=train_dataset.ordered_rows[0].feature_lineage_identity,
            ),
            Phase3CausalRow(
                canonical_observation_id=train_dataset.ordered_rows[1].canonical_observation_id,
                session_index=4,
                observation_timestamp_utc=train_dataset.ordered_rows[1].observation_timestamp_utc,
                availability_timestamp_utc=train_dataset.ordered_rows[1].availability_timestamp_utc,
                availability_verified=True,
                feature_values_in_required_order=train_dataset.ordered_rows[1].feature_values_in_required_order,
                feature_lineage_identity=train_dataset.ordered_rows[1].feature_lineage_identity,
            ),
        ),
    )
    result = validate_phase3_causal_binding_fail_closed(dataset=validation_dataset, projection=projection)
    assert result.verified is False
    assert result.failure_code == Phase3FailureCode.STAGE_MISMATCH


def test_dataset_rejects_availability_after_cutoff() -> None:
    dataset, _ = _dataset()
    delayed_row = Phase3CausalRow(
        canonical_observation_id=dataset.ordered_rows[0].canonical_observation_id,
        session_index=dataset.ordered_rows[0].session_index,
        observation_timestamp_utc=dataset.ordered_rows[0].observation_timestamp_utc,
        availability_timestamp_utc=dataset.stage_information_cutoff_utc + timedelta(minutes=1),
        availability_verified=True,
        feature_values_in_required_order=dataset.ordered_rows[0].feature_values_in_required_order,
        feature_lineage_identity=dataset.ordered_rows[0].feature_lineage_identity,
    )
    with pytest.raises(Phase3ValidationError) as exc:
        build_phase3_causal_dataset(
            source_dataset_identity=dataset.source_dataset_identity,
            session_calendar_id=dataset.session_calendar_id,
            session_calendar_version=dataset.session_calendar_version,
            session_sequence_digest=dataset.session_sequence_digest,
            fold_id=dataset.fold_id,
            stage=dataset.stage,
            fold_interval=dataset.fold_interval,
            stage_information_cutoff_utc=dataset.stage_information_cutoff_utc,
            phase2_projection_digest=dataset.phase2_projection_digest,
            required_feature_ids=dataset.required_feature_ids,
            eligible_observation_ids=(delayed_row.canonical_observation_id,),
            ordered_rows=(delayed_row,),
        )
    assert exc.value.code == Phase3FailureCode.LATE_AVAILABILITY


def test_binding_rejects_required_feature_ordering_mismatch() -> None:
    dataset, projection = _dataset()
    mismatched_dataset = build_phase3_causal_dataset(
        source_dataset_identity=dataset.source_dataset_identity,
        session_calendar_id=dataset.session_calendar_id,
        session_calendar_version=dataset.session_calendar_version,
        session_sequence_digest=dataset.session_sequence_digest,
        fold_id=dataset.fold_id,
        stage=dataset.stage,
        fold_interval=dataset.fold_interval,
        stage_information_cutoff_utc=dataset.stage_information_cutoff_utc,
        phase2_projection_digest=dataset.phase2_projection_digest,
        required_feature_ids=("f2", "f1"),
        eligible_observation_ids=dataset.eligible_observation_ids,
        ordered_rows=(
            Phase3CausalRow(
                canonical_observation_id=dataset.ordered_rows[0].canonical_observation_id,
                session_index=dataset.ordered_rows[0].session_index,
                observation_timestamp_utc=dataset.ordered_rows[0].observation_timestamp_utc,
                availability_timestamp_utc=dataset.ordered_rows[0].availability_timestamp_utc,
                availability_verified=True,
                feature_values_in_required_order=(dataset.ordered_rows[0].feature_values_in_required_order[1], dataset.ordered_rows[0].feature_values_in_required_order[0]),
                feature_lineage_identity=_lineage(
                    dataset.ordered_rows[0].canonical_observation_id,
                    ("f2", "f1"),
                    (dataset.ordered_rows[0].feature_values_in_required_order[1], dataset.ordered_rows[0].feature_values_in_required_order[0]),
                ),
            ),
            *dataset.ordered_rows[1:],
        ),
    )
    result = validate_phase3_causal_binding_fail_closed(dataset=mismatched_dataset, projection=projection)
    assert result.verified is False
    assert result.failure_code == Phase3FailureCode.FEATURE_COLUMN_ORDER_MISMATCH


def test_binding_rejects_missing_eligible_observation() -> None:
    dataset, projection = _dataset()
    missing_dataset = build_phase3_causal_dataset(
        source_dataset_identity=dataset.source_dataset_identity,
        session_calendar_id=dataset.session_calendar_id,
        session_calendar_version=dataset.session_calendar_version,
        session_sequence_digest=dataset.session_sequence_digest,
        fold_id=dataset.fold_id,
        stage=dataset.stage,
        fold_interval=dataset.fold_interval,
        stage_information_cutoff_utc=dataset.stage_information_cutoff_utc,
        phase2_projection_digest=dataset.phase2_projection_digest,
        required_feature_ids=dataset.required_feature_ids,
        eligible_observation_ids=(dataset.eligible_observation_ids[0],),
        ordered_rows=(dataset.ordered_rows[0],),
    )
    result = validate_phase3_causal_binding_fail_closed(dataset=missing_dataset, projection=projection)
    assert result.verified is False
    assert result.failure_code == Phase3FailureCode.MISSING_ELIGIBLE_ROW


def test_binding_rejects_extra_eligible_observation() -> None:
    dataset, projection = _dataset()
    extra_row = Phase3CausalRow(
        canonical_observation_id="obs_extra",
        session_index=dataset.ordered_rows[0].session_index,
        observation_timestamp_utc=_dt(1),
        availability_timestamp_utc=_dt(1),
        availability_verified=True,
        feature_values_in_required_order=(3.0, 6.0),
        feature_lineage_identity=_lineage("obs_extra", dataset.required_feature_ids, (3.0, 6.0)),
    )
    extra_dataset = build_phase3_causal_dataset(
        source_dataset_identity=dataset.source_dataset_identity,
        session_calendar_id=dataset.session_calendar_id,
        session_calendar_version=dataset.session_calendar_version,
        session_sequence_digest=dataset.session_sequence_digest,
        fold_id=dataset.fold_id,
        stage=dataset.stage,
        fold_interval=dataset.fold_interval,
        stage_information_cutoff_utc=dataset.stage_information_cutoff_utc,
        phase2_projection_digest=dataset.phase2_projection_digest,
        required_feature_ids=dataset.required_feature_ids,
        eligible_observation_ids=(*dataset.eligible_observation_ids, "obs_extra"),
        ordered_rows=(*dataset.ordered_rows, extra_row),
    )
    result = validate_phase3_causal_binding_fail_closed(dataset=extra_dataset, projection=projection)
    assert result.verified is False
    assert result.failure_code == Phase3FailureCode.EXTRA_OBSERVATION_ROW


def test_binding_rejects_simple_namespace_projection() -> None:
    dataset, _ = _dataset()
    projection = SimpleNamespace(
        fold_id=dataset.fold_id,
        stage=dataset.stage,
        source_dataset_identity=dataset.source_dataset_identity,
        session_calendar_id=dataset.session_calendar_id,
        session_calendar_version=dataset.session_calendar_version,
        session_sequence_digest=dataset.session_sequence_digest,
        projection_digest=dataset.phase2_projection_digest,
        required_feature_ids=dataset.required_feature_ids,
        eligible_observation_ids=dataset.eligible_observation_ids,
    )
    result = validate_phase3_causal_binding_fail_closed(dataset=dataset, projection=projection)
    assert result.verified is False
    assert result.failure_code == Phase3FailureCode.PROJECTION_TYPE_MISMATCH


def test_binding_rejects_dictionary_projection() -> None:
    dataset, _ = _dataset()
    result = validate_phase3_causal_binding_fail_closed(
        dataset=dataset,
        projection={
            "fold_id": dataset.fold_id,
            "stage": dataset.stage,
            "projection_digest": dataset.phase2_projection_digest,
        },
    )
    assert result.verified is False
    assert result.failure_code == Phase3FailureCode.PROJECTION_TYPE_MISMATCH


def test_binding_rejects_forged_projection_digest() -> None:
    dataset, projection = _dataset()
    object.__setattr__(projection, "projection_digest", "forged-digest")
    result = validate_phase3_causal_binding_fail_closed(dataset=dataset, projection=projection)
    assert result.verified is False
    assert result.failure_code == Phase3FailureCode.PROJECTION_CONTENT_TAMPERING


def test_binding_rejects_tampered_projection_material() -> None:
    dataset, projection = _dataset()
    object.__setattr__(projection, "fold_id", "tampered-fold")
    result = validate_phase3_causal_binding_fail_closed(dataset=dataset, projection=projection)
    assert result.verified is False
    assert result.failure_code == Phase3FailureCode.PROJECTION_CONTENT_TAMPERING


def test_binding_rejects_mismatched_dataset_projection_digest() -> None:
    dataset, projection = _dataset()
    tampered_dataset = build_phase3_causal_dataset(
        source_dataset_identity=dataset.source_dataset_identity,
        session_calendar_id=dataset.session_calendar_id,
        session_calendar_version=dataset.session_calendar_version,
        session_sequence_digest=dataset.session_sequence_digest,
        fold_id=dataset.fold_id,
        stage=dataset.stage,
        fold_interval=dataset.fold_interval,
        stage_information_cutoff_utc=dataset.stage_information_cutoff_utc,
        phase2_projection_digest="forged-digest",
        required_feature_ids=dataset.required_feature_ids,
        eligible_observation_ids=dataset.eligible_observation_ids,
        ordered_rows=dataset.ordered_rows,
    )
    result = validate_phase3_causal_binding_fail_closed(dataset=tampered_dataset, projection=projection)
    assert result.verified is False
    assert result.failure_code == Phase3FailureCode.PROJECTION_DIGEST_MISMATCH


def test_binding_rejects_reordered_eligible_observations() -> None:
    dataset, projection = _dataset(ids=("obs0", "obs1", "obs2"))
    reordered = build_phase3_causal_dataset(
        source_dataset_identity=dataset.source_dataset_identity,
        session_calendar_id=dataset.session_calendar_id,
        session_calendar_version=dataset.session_calendar_version,
        session_sequence_digest=dataset.session_sequence_digest,
        fold_id=dataset.fold_id,
        stage=dataset.stage,
        fold_interval=dataset.fold_interval,
        stage_information_cutoff_utc=dataset.stage_information_cutoff_utc,
        phase2_projection_digest=dataset.phase2_projection_digest,
        required_feature_ids=dataset.required_feature_ids,
        eligible_observation_ids=("obs2", "obs1", "obs0"),
        ordered_rows=(dataset.ordered_rows[2], dataset.ordered_rows[1], dataset.ordered_rows[0]),
    )
    result = validate_phase3_causal_binding_fail_closed(dataset=reordered, projection=projection)
    assert result.verified is False
    assert result.failure_code == Phase3FailureCode.REORDERED_OBSERVATION_ROWS


def test_binding_rejects_mismatched_feature_lineage() -> None:
    dataset, projection = _dataset()
    tampered_row = Phase3CausalRow(
        canonical_observation_id=dataset.ordered_rows[0].canonical_observation_id,
        session_index=dataset.ordered_rows[0].session_index,
        observation_timestamp_utc=dataset.ordered_rows[0].observation_timestamp_utc,
        availability_timestamp_utc=dataset.ordered_rows[0].availability_timestamp_utc,
        availability_verified=True,
        feature_values_in_required_order=dataset.ordered_rows[0].feature_values_in_required_order,
        feature_lineage_identity="tampered-lineage",
    )
    tampered = build_phase3_causal_dataset(
        source_dataset_identity=dataset.source_dataset_identity,
        session_calendar_id=dataset.session_calendar_id,
        session_calendar_version=dataset.session_calendar_version,
        session_sequence_digest=dataset.session_sequence_digest,
        fold_id=dataset.fold_id,
        stage=dataset.stage,
        fold_interval=dataset.fold_interval,
        stage_information_cutoff_utc=dataset.stage_information_cutoff_utc,
        phase2_projection_digest=dataset.phase2_projection_digest,
        required_feature_ids=dataset.required_feature_ids,
        eligible_observation_ids=dataset.eligible_observation_ids,
        ordered_rows=(tampered_row, *dataset.ordered_rows[1:]),
    )
    result = validate_phase3_causal_binding_fail_closed(dataset=tampered, projection=projection)
    assert result.verified is False
    assert result.failure_code == Phase3FailureCode.FEATURE_LINEAGE_MISMATCH


def test_binding_rejects_mismatched_availability() -> None:
    dataset, projection = _dataset()
    altered_projection = projection
    shifted = projection.availability_timestamps_utc[0] + timedelta(minutes=5)
    object.__setattr__(altered_projection, "availability_timestamps_utc", (shifted, *projection.availability_timestamps_utc[1:]))
    result = validate_phase3_causal_binding_fail_closed(dataset=dataset, projection=altered_projection)
    assert result.verified is False
    assert result.failure_code == Phase3FailureCode.PROJECTION_CONTENT_TAMPERING


def test_unbound_matrix_reconstruction_is_rejected() -> None:
    dataset, _ = _dataset()
    with pytest.raises(Phase3ValidationError) as exc:
        reconstruct_causal_matrix(dataset=dataset, projection=SimpleNamespace())
    assert exc.value.code == Phase3FailureCode.PROJECTION_TYPE_MISMATCH


def test_real_projection_reconstructs_matrix() -> None:
    dataset, projection = _dataset()
    matrix = reconstruct_causal_matrix(dataset=dataset, projection=projection)
    assert isinstance(matrix, pd.DataFrame)
    assert list(matrix.columns) == ["canonical_observation_id", "session_index", "observation_timestamp_utc", "availability_timestamp_utc", "f1", "f2"]
    assert list(matrix["canonical_observation_id"]) == list(dataset.eligible_observation_ids)


def test_reconstruction_is_deterministic() -> None:
    dataset, projection = _dataset()
    first = reconstruct_causal_matrix(dataset=dataset, projection=projection)
    second = reconstruct_causal_matrix(dataset=dataset, projection=projection)
    pd.testing.assert_frame_equal(first, second)


def test_dataset_rejects_nonfinite_feature_values() -> None:
    with pytest.raises(Phase3ValidationError) as exc:
        Phase3CausalRow(
            canonical_observation_id="obs0",
            session_index=0,
            observation_timestamp_utc=_dt(1),
            availability_timestamp_utc=_dt(1),
            availability_verified=True,
            feature_values_in_required_order=(1.0, float("nan")),
            feature_lineage_identity="x",
        )
    assert exc.value.code == Phase3FailureCode.NONFINITE_FEATURE_VALUE


def test_dataset_rejects_unverified_availability() -> None:
    with pytest.raises(Phase3ValidationError) as exc:
        Phase3CausalRow(
            canonical_observation_id="obs0",
            session_index=0,
            observation_timestamp_utc=_dt(1),
            availability_timestamp_utc=_dt(1),
            availability_verified=False,
            feature_values_in_required_order=(1.0, 2.0),
            feature_lineage_identity="x",
        )
    assert exc.value.code == Phase3FailureCode.UNVERIFIED_AVAILABILITY


def test_projection_content_tampering_config_is_rejected() -> None:
    payload = {
        "m5_primary": Phase3M5PrimaryConfig(n_components=3),
        "m3_comparator": None,
        "m5_is_primary": True,
    }
    with pytest.raises(Phase3ValidationError) as exc:
        Phase3FrozenModelConfig(config_identity_digest="tampered", **payload)
    assert exc.value.code == Phase3FailureCode.PROJECTION_CONTENT_TAMPERING


def test_binding_identity_changes_when_order_changes() -> None:
    dataset, _ = _dataset(ids=("obs0", "obs1", "obs2"))
    digest_a = compute_phase3_binding_identity_digest(
        dataset_identity_digest=dataset.dataset_identity_digest,
        projection_digest=dataset.phase2_projection_digest,
        fold_id=dataset.fold_id,
        stage=dataset.stage,
        required_feature_ids=dataset.required_feature_ids,
        eligible_observation_ids=dataset.eligible_observation_ids,
    )
    digest_b = compute_phase3_binding_identity_digest(
        dataset_identity_digest=dataset.dataset_identity_digest,
        projection_digest=dataset.phase2_projection_digest,
        fold_id=dataset.fold_id,
        stage=dataset.stage,
        required_feature_ids=dataset.required_feature_ids,
        eligible_observation_ids=("obs1", "obs0", "obs2"),
    )
    assert digest_a != digest_b


def test_binding_helper_matches_direct_compute_identity_digest() -> None:
    dataset, _ = _dataset()
    helper = compute_phase3_binding_identity_digest(
        dataset_identity_digest=dataset.dataset_identity_digest,
        projection_digest=dataset.phase2_projection_digest,
        fold_id=dataset.fold_id,
        stage=dataset.stage,
        required_feature_ids=dataset.required_feature_ids,
        eligible_observation_ids=dataset.eligible_observation_ids,
    )
    direct = compute_identity_digest(
        {
            "dataset_identity_digest": dataset.dataset_identity_digest,
            "projection_digest": dataset.phase2_projection_digest,
            "fold_id": dataset.fold_id,
            "stage": dataset.stage.value,
            "required_feature_ids": list(dataset.required_feature_ids),
            "eligible_observation_ids": list(dataset.eligible_observation_ids),
        },
        domain=PHASE3_BINDING_DOMAIN,
    )
    assert helper == direct


def test_phase2_projection_identity_domain_is_correct() -> None:
    _, projection = _dataset()
    payload = {
        "schema": projection.schema,
        "version": projection.version,
        "source_dataset_identity": projection.source_dataset_identity,
        "session_calendar_id": projection.session_calendar_id,
        "session_calendar_version": projection.session_calendar_version,
        "session_sequence_digest": projection.session_sequence_digest,
        "stage": projection.stage.value,
        "fold_id": projection.fold_id,
        "fold_interval": {
            "train_start": projection.fold_interval.train_start,
            "train_end": projection.fold_interval.train_end,
            "validation_start": projection.fold_interval.validation_start,
            "validation_end": projection.fold_interval.validation_end,
            "test_start": projection.fold_interval.test_start,
            "test_end": projection.fold_interval.test_end,
            "holdout_start": projection.fold_interval.holdout_start,
            "holdout_end": projection.fold_interval.holdout_end,
        },
        "stage_information_cutoff_utc": projection.stage_information_cutoff_utc.isoformat(),
        "boundary_session_index": projection.boundary_session_index,
        "required_feature_ids": list(projection.required_feature_ids),
        "canonical_observation_ids": list(projection.canonical_observation_ids),
        "observation_timestamps_utc": [item.isoformat() for item in projection.observation_timestamps_utc],
        "availability_timestamps_utc": [None if item is None else item.isoformat() for item in projection.availability_timestamps_utc],
        "required_feature_values_by_observation": [list(item) for item in projection.required_feature_values_by_observation],
        "eligible_observation_ids": list(projection.eligible_observation_ids),
        "excluded_observations": [
            {"observation_id": item.observation_id, "reason_code": item.reason_code.value, "detail": item.detail}
            for item in projection.excluded_observations
        ],
        "m3_policy_identity": projection.m3_policy_identity,
        "m5_policy_identity": projection.m5_policy_identity,
    }
    assert compute_phase2_projection_identity_digest(projection) == compute_identity_digest(payload, domain=PHASE2_PROJECTION_DOMAIN)


def test_dataset_cutoff_accepts_equal_timestamps() -> None:
    cutoff = _dt(5)
    features = ("f1", "f2")
    values = (1.0, 2.0)
    row = Phase3CausalRow(
        canonical_observation_id="obs0",
        session_index=0,
        observation_timestamp_utc=cutoff,
        availability_timestamp_utc=cutoff,
        availability_verified=True,
        feature_values_in_required_order=values,
        feature_lineage_identity=_lineage("obs0", features, values),
    )
    ds = build_phase3_causal_dataset(
        source_dataset_identity="s",
        session_calendar_id="c",
        session_calendar_version="v",
        session_sequence_digest="seq",
        fold_id="f",
        stage=Phase3Stage.TRAIN,
        fold_interval=_interval(),
        stage_information_cutoff_utc=cutoff,
        phase2_projection_digest="pd",
        required_feature_ids=features,
        eligible_observation_ids=("obs0",),
        ordered_rows=(row,),
    )
    assert ds.stage_information_cutoff_utc == cutoff

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from regimequant.features.definitions import FeatureSpec
from regimequant.walkforward.phase3_contracts import Phase3Stage
from regimequant.walkforward.source_authority import (
    AuthorityLevel,
    EvidenceOrigin,
    FeatureRecordEvidence,
    FeatureSpecEvidence,
    FoldStageEvidence,
    OriginalProvenanceEvidence,
    SourceAuthorityFailure,
    SourceAuthorityFailureCode,
    SourceRecordEvidence,
    authority_level_from_evidence,
    evidence_failures_for_historical_artifact,
    recompute_local_evidence_digest,
    require_unique_evidence_keys,
    require_unique_feature_keys,
    validate_local_evidence_digest,
)
from regimequant.walkforward.contracts import FoldInterval


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def test_source_record_evidence_digest_and_boundaries() -> None:
    record = SourceRecordEvidence(
        source_dataset_identity="src:dataset:alpha",
        instrument="AAPL",
        source_observation_id="obs-001",
        observation_timestamp_utc=_ts("2024-01-02T00:00:00+00:00"),
        availability_timestamp_utc=_ts("2024-01-02T00:01:00+00:00"),
        availability_status="verified_timestamp",
        price_type="close",
        adjustment_state="raw",
        source_provider_id="provider-x",
        source_artifact_id="artifact-1",
        source_artifact_hash="sha256:abc",
        origin=EvidenceOrigin.DIRECT_SOURCE,
        authority_level=AuthorityLevel.STRUCTURAL_ONLY,
    )
    assert record.evidence_digest
    assert record.evidence_key == ("src:dataset:alpha", "AAPL", "obs-001")
    assert authority_level_from_evidence(evidence=record) == AuthorityLevel.STRUCTURAL_ONLY


def test_source_evidence_requires_nonempty_identity() -> None:
    with pytest.raises(ValueError):
        SourceRecordEvidence(
            source_dataset_identity="",
            instrument="AAPL",
            source_observation_id="obs-001",
            observation_timestamp_utc=_ts("2024-01-02T00:00:00+00:00"),
        )


def test_duplicate_source_keys_fail() -> None:
    rows = [
        SourceRecordEvidence(
            source_dataset_identity="dataset",
            instrument="AAPL",
            source_observation_id="obs-1",
            observation_timestamp_utc=_ts("2024-01-02T00:00:00+00:00"),
        ),
        SourceRecordEvidence(
            source_dataset_identity="dataset",
            instrument="AAPL",
            source_observation_id="obs-1",
            observation_timestamp_utc=_ts("2024-01-02T00:00:00+00:00"),
        ),
    ]
    with pytest.raises(ValueError):
        require_unique_evidence_keys(rows)


def test_feature_spec_evidence_is_deterministic() -> None:
    spec = FeatureSpec(
        feature_name="rolling_mean_return",
        feature_definition_version="1",
        window_size=3,
        min_observations=3,
        include_current=True,
        allow_partial_valid=True,
    )
    evidence = FeatureSpecEvidence(
        feature_name=spec.feature_name,
        feature_definition_version=spec.feature_definition_version,
        input_return_type=spec.input_return_type.value,
        output_transformation_mode=spec.output_transformation_mode,
        window_kind=spec.window_kind,
        window_size=spec.window_size,
        min_observations=spec.min_observations,
        include_current=spec.include_current,
        require_complete_window=spec.require_complete_window,
        allow_partial_valid=spec.allow_partial_valid,
        ddof=spec.ddof,
        downside_threshold=spec.downside_threshold,
        short_window_size=spec.short_window_size,
        long_window_size=spec.long_window_size,
        source_spec=spec,
        origin=EvidenceOrigin.DIRECT_SOURCE,
        authority_level=AuthorityLevel.STRUCTURAL_ONLY,
    )
    assert evidence.feature_instance_id_value
    assert evidence.evidence_digest


def test_feature_spec_evidence_rejects_promoted_authority() -> None:
    with pytest.raises(ValueError):
        FeatureSpecEvidence(
            feature_name="rolling_mean_return",
            feature_definition_version="1",
            input_return_type="simple",
            output_transformation_mode="native",
            window_kind="observation_count",
            window_size=3,
            min_observations=3,
            include_current=True,
            require_complete_window=False,
            allow_partial_valid=True,
            authority_level=AuthorityLevel.FEATURE_RECORD_BOUND,
        )


def test_original_provenance_evidence_requires_fields() -> None:
    with pytest.raises(ValueError):
        OriginalProvenanceEvidence(provider_id="")


def test_feature_record_evidence_rejects_promoted_authority() -> None:
    with pytest.raises(ValueError):
        FeatureRecordEvidence(
            scope_id="AAPL",
            timestamp_utc=_ts("2024-01-02T00:00:00+00:00"),
            feature_name="rolling_mean_return",
            feature_instance_id="m2v1_sha256_test",
            feature_value=0.5,
            authority_level=AuthorityLevel.FEATURE_RECORD_BOUND,
        )


def test_fold_stage_evidence_has_frozen_structure() -> None:
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
    evidence = FoldStageEvidence(
        source_dataset_identity="dataset-a",
        session_calendar_id="calendar-a",
        session_calendar_version="v1",
        session_sequence_digest="seq-1",
        fold_id="fold-a",
        stage="TRAIN",
        fold_interval=interval,
        stage_information_cutoff_utc=_ts("2024-01-31T00:00:00+00:00"),
        boundary_session_index=10,
        projection_digest="proj-digest",
        dataset_identity_digest="dataset-digest",
        origin=EvidenceOrigin.DIRECT_SOURCE,
        authority_level=AuthorityLevel.STRUCTURAL_ONLY,
    )
    assert evidence.evidence_digest


def test_must_not_promote_authority_from_evidence() -> None:
    record = SourceRecordEvidence(
        source_dataset_identity="dataset",
        instrument="AAPL",
        source_observation_id="obs-7",
        observation_timestamp_utc=_ts("2024-01-02T00:00:00+00:00"),
    )
    assert authority_level_from_evidence(evidence=record, require_reconciliation_proof=False) == AuthorityLevel.STRUCTURAL_ONLY
    assert authority_level_from_evidence(evidence=record, require_reconciliation_proof=True) == AuthorityLevel.STRUCTURAL_ONLY


def test_historical_artifact_failure_is_explicit() -> None:
    failures = evidence_failures_for_historical_artifact()
    assert failures[0].code == SourceAuthorityFailureCode.UNSUPPORTED_HISTORICAL_ARTIFACT
    assert "historical artifact" in failures[0].message


def test_duplicate_feature_evidence_keys_fail() -> None:
    rows = [
        FeatureRecordEvidence(
            scope_id="AAPL",
            timestamp_utc=_ts("2024-01-02T00:00:00+00:00"),
            feature_name="rolling_mean_return",
            feature_instance_id="feature-1",
            feature_value=0.1,
            source_observation_ids=("obs-1", "obs-2"),
        ),
        FeatureRecordEvidence(
            scope_id="AAPL",
            timestamp_utc=_ts("2024-01-02T00:00:00+00:00"),
            feature_name="rolling_mean_return",
            feature_instance_id="feature-1",
            feature_value=0.1,
            source_observation_ids=("obs-1", "obs-2"),
        ),
    ]
    with pytest.raises(ValueError):
        require_unique_feature_keys(rows)


def test_tampered_digest_is_not_accepted_by_design() -> None:
    record = SourceRecordEvidence(
        source_dataset_identity="dataset",
        instrument="AAPL",
        source_observation_id="obs-9",
        observation_timestamp_utc=_ts("2024-01-02T00:00:00+00:00"),
    )
    payload = {
        "source_dataset_identity": record.source_dataset_identity,
        "instrument": record.instrument,
        "source_observation_id": record.source_observation_id,
        "observation_timestamp_utc": record.observation_timestamp_utc,
        "availability_timestamp_utc": record.availability_timestamp_utc,
        "availability_status": record.availability_status,
        "price_type": record.price_type,
        "adjustment_state": record.adjustment_state,
        "source_provider_id": record.source_provider_id,
        "source_artifact_id": record.source_artifact_id,
        "source_artifact_hash": record.source_artifact_hash,
        "origin": record.origin.value,
        "authority_level": record.authority_level.value,
        "evidence_digest": record.evidence_digest,
    }
    expected = recompute_local_evidence_digest(payload, domain="regimequant.walkforward.phase3b.source_record_evidence")
    assert expected == record.evidence_digest
    assert validate_local_evidence_digest(
        payload,
        domain="regimequant.walkforward.phase3b.source_record_evidence",
        supplied_digest=record.evidence_digest,
    ) == record.evidence_digest
    with pytest.raises(ValueError):
        validate_local_evidence_digest(
            payload,
            domain="regimequant.walkforward.phase3b.source_record_evidence",
            supplied_digest="tampered:bad-digest",
        )


def test_reordered_semantically_identical_payload_keeps_deterministic_digest() -> None:
    left = {
        "source_dataset_identity": "dataset",
        "instrument": "AAPL",
        "source_observation_id": "obs-7",
        "observation_timestamp_utc": _ts("2024-01-02T00:00:00+00:00"),
        "availability_status": "verified_timestamp",
        "price_type": "close",
        "adjustment_state": "raw",
        "origin": "DIRECT_SOURCE",
        "authority_level": "STRUCTURAL_ONLY",
    }
    right = {
        "authority_level": "STRUCTURAL_ONLY",
        "price_type": "close",
        "source_dataset_identity": "dataset",
        "origin": "DIRECT_SOURCE",
        "observation_timestamp_utc": _ts("2024-01-02T00:00:00+00:00"),
        "availability_status": "verified_timestamp",
        "source_observation_id": "obs-7",
        "instrument": "AAPL",
        "adjustment_state": "raw",
    }
    assert recompute_local_evidence_digest(left, domain="regimequant.walkforward.phase3b.source_record_evidence") == recompute_local_evidence_digest(
        right,
        domain="regimequant.walkforward.phase3b.source_record_evidence",
    )


def test_digest_excludes_its_own_field_from_identity_material() -> None:
    payload = {
        "feature_name": "rolling_mean_return",
        "feature_definition_version": "1",
        "window_size": 3,
        "min_observations": 3,
        "evidence_digest": "junk",
    }
    digest = recompute_local_evidence_digest(payload, domain="regimequant.walkforward.phase3b.feature_spec_evidence")
    assert digest != "junk"


def test_upstream_opaque_identifiers_remain_non_digest_semantics() -> None:
    record = SourceRecordEvidence(
        source_dataset_identity="dataset",
        instrument="AAPL",
        source_observation_id="obs-opaque-9",
        observation_timestamp_utc=_ts("2024-01-02T00:00:00+00:00"),
        source_provider_id="provider-opaque",
        source_artifact_id="artifact-opaque",
        source_artifact_hash="sha256:opaque",
    )
    assert record.source_observation_id != record.evidence_digest
    assert record.source_artifact_hash != record.evidence_digest
    assert record.source_provider_id != record.evidence_digest


def test_naive_timestamp_is_rejected() -> None:
    naive = datetime(2024, 1, 2, 0, 0, 0)
    with pytest.raises(ValueError):
        SourceRecordEvidence(
            source_dataset_identity="dataset",
            instrument="AAPL",
            source_observation_id="obs-10",
            observation_timestamp_utc=naive,
        )


def test_invalid_provenance_fields_rejected() -> None:
    with pytest.raises(ValueError):
        OriginalProvenanceEvidence(provider_id="provider", availability_policy_id="")


def test_structural_only_is_fallback_state() -> None:
    assert AuthorityLevel.STRUCTURAL_ONLY.value == "STRUCTURAL_ONLY"
    assert AuthorityLevel.UNSUPPORTED.value == "UNSUPPORTED"


def test_failure_type_is_immutable() -> None:
    failure = SourceAuthorityFailure(
        code=SourceAuthorityFailureCode.PROVENANCE_MISSING,
        message="missing provenance",
        missing_evidence=("provenance",),
    )
    assert failure.code == SourceAuthorityFailureCode.PROVENANCE_MISSING
    assert failure.missing_evidence == ("provenance",)

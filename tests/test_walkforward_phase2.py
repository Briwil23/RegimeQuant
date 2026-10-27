from __future__ import annotations

from datetime import datetime, timedelta, timezone
from math import inf

import pytest

from regimequant.cointegration.contracts import DecisionEnvelope
from regimequant.signals.m8_contracts import M8AbstainReason
from regimequant.signals.m8_eligibility import M8EligibilityDecision
from regimequant.walkforward.causal_projection import (
    CandidateObservation,
    ExclusionReason,
    ProjectionStage,
    build_causal_eligibility_projection,
    required_m3_heldout_minimum,
    required_m3_training_minimum,
    required_m5_training_minimum,
)
from regimequant.walkforward.contracts import ExperimentPlan, FoldEvidenceArtifact, FoldInterval, M10Policy
from regimequant.walkforward.evidence_authority import (
    EvidenceAuthorityState,
    classify_evidence_authority,
    classify_from_m8_eligibility,
    to_validation_result,
)
from regimequant.walkforward.fold_contract_binding import build_fold_input_binding
from regimequant.walkforward.fold_geometry import FoldGeometryStatus, build_expanding_fold_intervals, classify_fold_geometry
from regimequant.walkforward.no_entry_structural import build_structural_no_entry
from regimequant.walkforward.phase2_orchestrator_structural import assemble_phase2_structural


def _policy() -> M10Policy:
    return M10Policy()


def _plan() -> ExperimentPlan:
    return ExperimentPlan(
        schema="regimequant.walkforward.experiment_plan",
        experiment_id="exp_phase2",
        experiment_version="m10.v1",
        regime_primary="M5",
        regime_comparator="M3",
        calendar_id="XNYS",
        calendar_version="2026.10",
        dataset_identity="dataset:phase2",
        feature_spec_set_hash="feature_hash_001",
        model_policy_identity="m3_policy_v1",
        formation_policy_identity="m5_policy_v1",
        family_policy_identity="family_policy_v1",
        final_holdout_sessions=20,
        created_at=datetime(2026, 10, 27, tzinfo=timezone.utc),
        folds=(),
        plan_digest=None,
        structural_only=True,
        scientific_certification_status="STRUCTURAL_ONLY",
    )


def _interval() -> FoldInterval:
    return FoldInterval(
        train_start=0,
        train_end=120,
        validation_start=120,
        validation_end=140,
        test_start=140,
        test_end=160,
        holdout_start=160,
        holdout_end=180,
    )


def _canonical_fold(total_sessions: int = 180, index: int = 0):
    folds = build_expanding_fold_intervals(total_sessions=total_sessions, policy=_policy())
    return folds[index]


def _obs(
    obs_id: str,
    session_index: int,
    *,
    obs_dt: datetime,
    avail_dt: datetime | None,
    identity_verified: bool = True,
    features: dict[str, float | int | None] | None = None,
) -> CandidateObservation:
    if features is None:
        features = {"f1": 1.0, "f2": 2.0}
    return CandidateObservation(
        canonical_observation_id=obs_id,
        session_index=session_index,
        observation_timestamp_utc=obs_dt,
        availability_timestamp_utc=avail_dt,
        identity_verified=identity_verified,
        features=features,
    )


def test_exact_n_boundary_cases() -> None:
    policy = _policy()
    assert len(build_expanding_fold_intervals(total_sessions=159, policy=policy)) == 0
    assert len(build_expanding_fold_intervals(total_sessions=160, policy=policy)) == 0
    assert len(build_expanding_fold_intervals(total_sessions=179, policy=policy)) == 0
    assert len(build_expanding_fold_intervals(total_sessions=180, policy=policy)) == 1
    assert len(build_expanding_fold_intervals(total_sessions=199, policy=policy)) == 1
    assert len(build_expanding_fold_intervals(total_sessions=200, policy=policy)) == 2
    assert len(build_expanding_fold_intervals(total_sessions=220, policy=policy)) == 3


def test_half_open_interval_correctness() -> None:
    fold = build_expanding_fold_intervals(total_sessions=180, policy=_policy())[0]
    interval = fold.interval
    assert interval.train_end == interval.validation_start
    assert interval.validation_end == interval.test_start


def test_final_holdout_isolation() -> None:
    fold = build_expanding_fold_intervals(total_sessions=220, policy=_policy())[-1]
    assert fold.interval.test_end <= fold.interval.holdout_start
    assert fold.interval.holdout_start == 200
    assert fold.interval.holdout_end == 220


def test_deterministic_fold_identifiers() -> None:
    a = build_expanding_fold_intervals(total_sessions=220, policy=_policy())
    b = build_expanding_fold_intervals(total_sessions=220, policy=_policy())
    assert [x.fold_id for x in a] == [x.fold_id for x in b]


def test_no_admissible_fold_classification() -> None:
    status = classify_fold_geometry(total_sessions=179, policy=_policy())
    assert status == FoldGeometryStatus.NO_ADMISSIBLE_FOLD


def test_invalid_session_count_rejected() -> None:
    with pytest.raises(ValueError):
        build_expanding_fold_intervals(total_sessions=-1, policy=_policy())


def test_raw_vs_eligible_count_distinction() -> None:
    interval = _interval()
    cutoff = datetime(2026, 1, 1, 10, tzinfo=timezone.utc)
    obs = (
        _obs("o1", 0, obs_dt=cutoff - timedelta(minutes=2), avail_dt=cutoff - timedelta(minutes=1), features={"f1": 1.0}),
        _obs("o2", 1, obs_dt=cutoff - timedelta(minutes=2), avail_dt=cutoff - timedelta(minutes=1), features={"f1": None}),
    )
    projection = build_causal_eligibility_projection(
        dataset_identity="dataset:phase2",
        calendar_id="XNYS",
        calendar_version="2026.10",
        calendar_sequence_digest="cal_digest_1",
        stage=ProjectionStage.TRAIN,
        fold_id="fold_0000",
        fold_interval=interval,
        stage_information_cutoff_utc=cutoff,
        boundary_session_index=interval.train_end,
        observations=obs,
        required_feature_ids=("f1",),
        m3_policy_identity="m3_policy_v1",
        m5_policy_identity="m5_policy_v1",
    )
    assert len(projection.canonical_observation_ids) == 2
    assert projection.eligible_count == 1


def test_duplicate_observation_rejection() -> None:
    interval = _interval()
    cutoff = datetime(2026, 1, 1, 10, tzinfo=timezone.utc)
    dup = _obs("dup", 0, obs_dt=cutoff, avail_dt=cutoff)
    with pytest.raises(ValueError):
        build_causal_eligibility_projection(
            dataset_identity="dataset:phase2",
            calendar_id="XNYS",
            calendar_version="2026.10",
            calendar_sequence_digest="cal_digest_1",
            stage=ProjectionStage.TRAIN,
            fold_id="fold_0000",
            fold_interval=interval,
            stage_information_cutoff_utc=cutoff,
            boundary_session_index=interval.train_end,
            observations=(dup, dup),
            required_feature_ids=("f1",),
            m3_policy_identity="m3_policy_v1",
            m5_policy_identity="m5_policy_v1",
        )


def test_missing_availability_exclusion() -> None:
    interval = _interval()
    cutoff = datetime(2026, 1, 1, 10, tzinfo=timezone.utc)
    projection = build_causal_eligibility_projection(
        dataset_identity="dataset:phase2",
        calendar_id="XNYS",
        calendar_version="2026.10",
        calendar_sequence_digest="cal_digest_1",
        stage=ProjectionStage.TRAIN,
        fold_id="fold_0000",
        fold_interval=interval,
        stage_information_cutoff_utc=cutoff,
        boundary_session_index=interval.train_end,
        observations=(_obs("o1", 0, obs_dt=cutoff - timedelta(minutes=2), avail_dt=None),),
        required_feature_ids=("f1",),
        m3_policy_identity="m3_policy_v1",
        m5_policy_identity="m5_policy_v1",
    )
    assert projection.excluded_observations[0].reason_code == ExclusionReason.AVAILABILITY_MISSING


def test_late_availability_exclusion() -> None:
    interval = _interval()
    cutoff = datetime(2026, 1, 1, 10, tzinfo=timezone.utc)
    projection = build_causal_eligibility_projection(
        dataset_identity="dataset:phase2",
        calendar_id="XNYS",
        calendar_version="2026.10",
        calendar_sequence_digest="cal_digest_1",
        stage=ProjectionStage.TRAIN,
        fold_id="fold_0000",
        fold_interval=interval,
        stage_information_cutoff_utc=cutoff,
        boundary_session_index=interval.train_end,
        observations=(_obs("o1", 0, obs_dt=cutoff - timedelta(minutes=2), avail_dt=cutoff + timedelta(minutes=1)),),
        required_feature_ids=("f1",),
        m3_policy_identity="m3_policy_v1",
        m5_policy_identity="m5_policy_v1",
    )
    assert projection.excluded_observations[0].reason_code == ExclusionReason.AVAILABILITY_LATE


def test_future_observation_exclusion() -> None:
    interval = _interval()
    cutoff = datetime(2026, 1, 1, 10, tzinfo=timezone.utc)
    projection = build_causal_eligibility_projection(
        dataset_identity="dataset:phase2",
        calendar_id="XNYS",
        calendar_version="2026.10",
        calendar_sequence_digest="cal_digest_1",
        stage=ProjectionStage.TRAIN,
        fold_id="fold_0000",
        fold_interval=interval,
        stage_information_cutoff_utc=cutoff,
        boundary_session_index=interval.train_end,
        observations=(_obs("o1", 0, obs_dt=cutoff + timedelta(minutes=1), avail_dt=cutoff + timedelta(minutes=1)),),
        required_feature_ids=("f1",),
        m3_policy_identity="m3_policy_v1",
        m5_policy_identity="m5_policy_v1",
    )
    assert projection.excluded_observations[0].reason_code == ExclusionReason.OBS_AFTER_CUTOFF


def test_stage_interval_mismatch_exclusion() -> None:
    interval = _interval()
    cutoff = datetime(2026, 1, 1, 10, tzinfo=timezone.utc)
    projection = build_causal_eligibility_projection(
        dataset_identity="dataset:phase2",
        calendar_id="XNYS",
        calendar_version="2026.10",
        calendar_sequence_digest="cal_digest_1",
        stage=ProjectionStage.VALIDATION,
        fold_id="fold_0000",
        fold_interval=interval,
        stage_information_cutoff_utc=cutoff,
        boundary_session_index=interval.validation_end,
        observations=(_obs("o1", 0, obs_dt=cutoff - timedelta(minutes=2), avail_dt=cutoff - timedelta(minutes=1)),),
        required_feature_ids=("f1",),
        m3_policy_identity="m3_policy_v1",
        m5_policy_identity="m5_policy_v1",
    )
    assert projection.excluded_observations[0].reason_code == ExclusionReason.STAGE_INTERVAL_MISMATCH


def test_nonfinite_feature_exclusion() -> None:
    interval = _interval()
    cutoff = datetime(2026, 1, 1, 10, tzinfo=timezone.utc)
    projection = build_causal_eligibility_projection(
        dataset_identity="dataset:phase2",
        calendar_id="XNYS",
        calendar_version="2026.10",
        calendar_sequence_digest="cal_digest_1",
        stage=ProjectionStage.TRAIN,
        fold_id="fold_0000",
        fold_interval=interval,
        stage_information_cutoff_utc=cutoff,
        boundary_session_index=interval.train_end,
        observations=(_obs("o1", 0, obs_dt=cutoff - timedelta(minutes=2), avail_dt=cutoff - timedelta(minutes=1), features={"f1": inf}),),
        required_feature_ids=("f1",),
        m3_policy_identity="m3_policy_v1",
        m5_policy_identity="m5_policy_v1",
    )
    assert projection.excluded_observations[0].reason_code == ExclusionReason.FEATURE_NONFINITE


def test_missing_identity_exclusion() -> None:
    interval = _interval()
    cutoff = datetime(2026, 1, 1, 10, tzinfo=timezone.utc)
    projection = build_causal_eligibility_projection(
        dataset_identity="dataset:phase2",
        calendar_id="XNYS",
        calendar_version="2026.10",
        calendar_sequence_digest="cal_digest_1",
        stage=ProjectionStage.TRAIN,
        fold_id="fold_0000",
        fold_interval=interval,
        stage_information_cutoff_utc=cutoff,
        boundary_session_index=interval.train_end,
        observations=(_obs("o1", 0, obs_dt=cutoff - timedelta(minutes=2), avail_dt=cutoff - timedelta(minutes=1), identity_verified=False),),
        required_feature_ids=("f1",),
        m3_policy_identity="m3_policy_v1",
        m5_policy_identity="m5_policy_v1",
    )
    assert projection.excluded_observations[0].reason_code == ExclusionReason.IDENTITY_UNVERIFIED


def test_deterministic_projection_digest() -> None:
    interval = _interval()
    cutoff = datetime(2026, 1, 1, 10, tzinfo=timezone.utc)
    obs = (_obs("o1", 0, obs_dt=cutoff - timedelta(minutes=2), avail_dt=cutoff - timedelta(minutes=1)),)
    a = build_causal_eligibility_projection(
        dataset_identity="dataset:phase2",
        calendar_id="XNYS",
        calendar_version="2026.10",
        calendar_sequence_digest="cal_digest_1",
        stage=ProjectionStage.TRAIN,
        fold_id="fold_0000",
        fold_interval=interval,
        stage_information_cutoff_utc=cutoff,
        boundary_session_index=interval.train_end,
        observations=obs,
        required_feature_ids=("f1",),
        m3_policy_identity="m3_policy_v1",
        m5_policy_identity="m5_policy_v1",
    )
    b = build_causal_eligibility_projection(
        dataset_identity="dataset:phase2",
        calendar_id="XNYS",
        calendar_version="2026.10",
        calendar_sequence_digest="cal_digest_1",
        stage=ProjectionStage.TRAIN,
        fold_id="fold_0000",
        fold_interval=interval,
        stage_information_cutoff_utc=cutoff,
        boundary_session_index=interval.train_end,
        observations=obs,
        required_feature_ids=("f1",),
        m3_policy_identity="m3_policy_v1",
        m5_policy_identity="m5_policy_v1",
    )
    assert a.projection_digest == b.projection_digest


def test_observation_reordering_invariance_under_canonical_ordering() -> None:
    interval = _interval()
    cutoff = datetime(2026, 1, 1, 10, tzinfo=timezone.utc)
    o1 = _obs("a", 0, obs_dt=cutoff - timedelta(minutes=2), avail_dt=cutoff - timedelta(minutes=1))
    o2 = _obs("b", 1, obs_dt=cutoff - timedelta(minutes=2), avail_dt=cutoff - timedelta(minutes=1))
    a = build_causal_eligibility_projection(
        dataset_identity="dataset:phase2",
        calendar_id="XNYS",
        calendar_version="2026.10",
        calendar_sequence_digest="cal_digest_1",
        stage=ProjectionStage.TRAIN,
        fold_id="fold_0000",
        fold_interval=interval,
        stage_information_cutoff_utc=cutoff,
        boundary_session_index=interval.train_end,
        observations=(o2, o1),
        required_feature_ids=("f1",),
        m3_policy_identity="m3_policy_v1",
        m5_policy_identity="m5_policy_v1",
    )
    b = build_causal_eligibility_projection(
        dataset_identity="dataset:phase2",
        calendar_id="XNYS",
        calendar_version="2026.10",
        calendar_sequence_digest="cal_digest_1",
        stage=ProjectionStage.TRAIN,
        fold_id="fold_0000",
        fold_interval=interval,
        stage_information_cutoff_utc=cutoff,
        boundary_session_index=interval.train_end,
        observations=(o1, o2),
        required_feature_ids=("f1",),
        m3_policy_identity="m3_policy_v1",
        m5_policy_identity="m5_policy_v1",
    )
    assert a.projection_digest == b.projection_digest


def test_material_identity_tampering_detection() -> None:
    interval = _interval()
    cutoff = datetime(2026, 1, 1, 10, tzinfo=timezone.utc)
    base = _obs("o1", 0, obs_dt=cutoff - timedelta(minutes=2), avail_dt=cutoff - timedelta(minutes=1))
    tampered = _obs("o1", 0, obs_dt=cutoff - timedelta(minutes=2), avail_dt=cutoff - timedelta(minutes=1), features={"f1": 99.0, "f2": 2.0})
    a = build_causal_eligibility_projection(
        dataset_identity="dataset:phase2",
        calendar_id="XNYS",
        calendar_version="2026.10",
        calendar_sequence_digest="cal_digest_1",
        stage=ProjectionStage.TRAIN,
        fold_id="fold_0000",
        fold_interval=interval,
        stage_information_cutoff_utc=cutoff,
        boundary_session_index=interval.train_end,
        observations=(base,),
        required_feature_ids=("f1",),
        m3_policy_identity="m3_policy_v1",
        m5_policy_identity="m5_policy_v1",
    )
    b = build_causal_eligibility_projection(
        dataset_identity="dataset:phase2",
        calendar_id="XNYS",
        calendar_version="2026.10",
        calendar_sequence_digest="cal_digest_1",
        stage=ProjectionStage.TRAIN,
        fold_id="fold_0000",
        fold_interval=interval,
        stage_information_cutoff_utc=cutoff,
        boundary_session_index=interval.train_end,
        observations=(tampered,),
        required_feature_ids=("f1",),
        m3_policy_identity="m3_policy_v1",
        m5_policy_identity="m5_policy_v1",
    )
    assert a.projection_digest != b.projection_digest


def test_m3_sample_minimum_enforcement() -> None:
    assert required_m3_training_minimum(n_clusters=3) == 30
    assert required_m3_training_minimum(n_clusters=5) == 50
    assert required_m3_heldout_minimum(n_clusters=3) == 20


def test_m5_sample_minimum_enforcement() -> None:
    assert required_m5_training_minimum(n_components=3, effective_feature_dimension=2) == 20
    assert required_m5_training_minimum(n_components=5, effective_feature_dimension=1) == 10


def test_missing_model_policy_evidence() -> None:
    decision = classify_evidence_authority(
        has_evidence=False,
        causal_timing_verified=False,
        identity_verified=False,
        evidence_late=False,
        family_authority_independently_verified=False,
        insufficient_history=False,
        caller_family_completeness_claim=False,
    )
    assert decision.state == EvidenceAuthorityState.MISSING_EVIDENCE


def test_family_digest_is_not_authority() -> None:
    decision = classify_evidence_authority(
        has_evidence=True,
        causal_timing_verified=True,
        identity_verified=True,
        evidence_late=False,
        family_authority_independently_verified=False,
        insufficient_history=False,
        caller_family_completeness_claim=False,
    )
    assert decision.state == EvidenceAuthorityState.UNVERIFIED_FAMILY_AUTHORITY


def test_caller_family_completeness_rejection() -> None:
    decision = classify_evidence_authority(
        has_evidence=True,
        causal_timing_verified=True,
        identity_verified=True,
        evidence_late=False,
        family_authority_independently_verified=False,
        insufficient_history=False,
        caller_family_completeness_claim=True,
    )
    assert decision.state == EvidenceAuthorityState.UNVERIFIED_FAMILY_AUTHORITY


def test_late_decision_evidence_classification() -> None:
    envelope = DecisionEnvelope(
        mode="production",
        cutoff=datetime(2026, 1, 1, 10, tzinfo=timezone.utc),
        availability_verified=False,
        source_digest="prod",
        evidence_digest="ev",
        decision_digest="dd",
    )
    eligibility = M8EligibilityDecision(eligible=False, proof=None, reason=M8AbstainReason.EVIDENCE_LATE)
    decision = classify_from_m8_eligibility(envelope=envelope, eligibility=eligibility)
    assert decision.state == EvidenceAuthorityState.LATE_EVIDENCE


def test_invalid_family_identity_classification() -> None:
    envelope = DecisionEnvelope(
        mode="replay",
        cutoff=datetime(2026, 1, 1, 10, tzinfo=timezone.utc),
        availability_verified=True,
        source_digest="replay",
        evidence_digest="ev",
        decision_digest="dd",
    )
    eligibility = M8EligibilityDecision(eligible=False, proof=None, reason=M8AbstainReason.BY_INCONSISTENT_EVIDENCE)
    decision = classify_from_m8_eligibility(envelope=envelope, eligibility=eligibility)
    assert decision.state == EvidenceAuthorityState.INVALID_IDENTITY


def test_parent_child_identity_binding() -> None:
    plan = _plan()
    interval = _interval()
    cutoff = datetime(2026, 1, 1, 10, tzinfo=timezone.utc)
    projection = build_causal_eligibility_projection(
        dataset_identity=plan.dataset_identity,
        calendar_id=plan.calendar_id,
        calendar_version=plan.calendar_version,
        calendar_sequence_digest="cal_digest_1",
        stage=ProjectionStage.TRAIN,
        fold_id="fold_0000",
        fold_interval=interval,
        stage_information_cutoff_utc=cutoff,
        boundary_session_index=interval.train_end,
        observations=(_obs("o1", 0, obs_dt=cutoff - timedelta(minutes=2), avail_dt=cutoff - timedelta(minutes=1)),),
        required_feature_ids=("f1",),
        m3_policy_identity=plan.model_policy_identity,
        m5_policy_identity=plan.formation_policy_identity,
    )
    evidence = FoldEvidenceArtifact(
        family_provenance_digest="family:1",
        calendar_provenance_digest="calendar:1",
        model_provenance_digest="model:1",
        validation_digest="validation:1",
        evidence_digest="evidence:1",
        provenance_state="UNVERIFIED",
        family_provenance_state="UNVERIFIED",
    )
    binding = build_fold_input_binding(
        plan=plan,
        fold_id="fold_0000",
        interval=interval,
        projection=projection,
        evidence=evidence,
        evidence_dependency_identities=(
            "family:1",
            "calendar:1",
            "model:1",
            "validation:1",
            "evidence:1",
            "cal_digest_1",
        ),
    )
    assert binding.fold_input.plan_id == plan.experiment_id
    assert binding.fold_input.input_digest is not None


def test_fold_input_mismatch_rejection() -> None:
    plan = _plan()
    interval = _interval()
    cutoff = datetime(2026, 1, 1, 10, tzinfo=timezone.utc)
    projection = build_causal_eligibility_projection(
        dataset_identity="wrong:dataset",
        calendar_id=plan.calendar_id,
        calendar_version=plan.calendar_version,
        calendar_sequence_digest="cal_digest_1",
        stage=ProjectionStage.TRAIN,
        fold_id="fold_0000",
        fold_interval=interval,
        stage_information_cutoff_utc=cutoff,
        boundary_session_index=interval.train_end,
        observations=(_obs("o1", 0, obs_dt=cutoff - timedelta(minutes=2), avail_dt=cutoff - timedelta(minutes=1)),),
        required_feature_ids=("f1",),
        m3_policy_identity=plan.model_policy_identity,
        m5_policy_identity=plan.formation_policy_identity,
    )
    evidence = FoldEvidenceArtifact(
        family_provenance_digest="family:1",
        calendar_provenance_digest="calendar:1",
        model_provenance_digest="model:1",
        validation_digest="validation:1",
        evidence_digest="evidence:1",
    )
    with pytest.raises(ValueError):
        build_fold_input_binding(
            plan=plan,
            fold_id="fold_0000",
            interval=interval,
            projection=projection,
            evidence=evidence,
            evidence_dependency_identities=("dep:1",),
        )


def test_no_entry_invoked_m9_false() -> None:
    result = build_structural_no_entry(
        experiment_id="exp_phase2",
        fold_id="fold_0000",
        fold_input_identity="input:1",
        decision_session=140,
        boundary_b=160,
        reason_code="NO_ENTRY",
        evidence_dependency_identities=("dep:1",),
    )
    assert result.artifact.invoked_m9 is False


def test_no_entry_deterministic_identity() -> None:
    a = build_structural_no_entry(
        experiment_id="exp_phase2",
        fold_id="fold_0000",
        fold_input_identity="input:1",
        decision_session=140,
        boundary_b=160,
        reason_code="NO_ENTRY",
        evidence_dependency_identities=("dep:1",),
    )
    b = build_structural_no_entry(
        experiment_id="exp_phase2",
        fold_id="fold_0000",
        fold_input_identity="input:1",
        decision_session=140,
        boundary_b=160,
        reason_code="NO_ENTRY",
        evidence_dependency_identities=("dep:1",),
    )
    assert a.artifact.no_entry_digest == b.artifact.no_entry_digest


def test_no_m9_execution() -> None:
    import regimequant.walkforward.no_entry_structural as module

    assert "m9" not in module.__dict__


def test_no_model_fitting() -> None:
    import regimequant.walkforward.phase2_orchestrator_structural as module

    names = set(module.__dict__.keys())
    assert "fit_m3_regime_model" not in names
    assert "fit_m5_regime_model" not in names


def test_no_scientific_pass_promotion() -> None:
    decision = classify_evidence_authority(
        has_evidence=True,
        causal_timing_verified=True,
        identity_verified=True,
        evidence_late=False,
        family_authority_independently_verified=False,
        insufficient_history=False,
        caller_family_completeness_claim=True,
    )
    result = to_validation_result(decision)
    assert result.authoritative is False
    assert result.status_role == "STRUCTURAL_ONLY"


def test_caller_family_authority_spoof_rejected() -> None:
    decision = classify_evidence_authority(
        has_evidence=True,
        causal_timing_verified=True,
        identity_verified=True,
        evidence_late=False,
        family_authority_independently_verified=True,
        insufficient_history=False,
        caller_family_completeness_claim=False,
    )
    assert decision.state == EvidenceAuthorityState.UNVERIFIED_FAMILY_AUTHORITY
    assert decision.scientific_status == "MISSING_CAUSAL_EVIDENCE"


def test_caller_causal_timing_spoof_rejected_without_family_proof() -> None:
    decision = classify_evidence_authority(
        has_evidence=True,
        causal_timing_verified=True,
        identity_verified=True,
        evidence_late=False,
        family_authority_independently_verified=False,
        insufficient_history=False,
        caller_family_completeness_claim=False,
    )
    result = to_validation_result(decision)
    assert result.valid is False
    assert result.scientific_status == "MISSING_CAUSAL_EVIDENCE"


def test_no_scientific_valid_promotion_from_caller_assertions() -> None:
    decision = classify_evidence_authority(
        has_evidence=True,
        causal_timing_verified=True,
        identity_verified=True,
        evidence_late=False,
        family_authority_independently_verified=True,
        insufficient_history=False,
        caller_family_completeness_claim=True,
    )
    result = to_validation_result(decision)
    assert result.valid is False
    assert result.scientific_status != "VALID_RESEARCH_EXECUTION"


def test_non_numeric_feature_exclusion() -> None:
    interval = _interval()
    cutoff = datetime(2026, 1, 1, 10, tzinfo=timezone.utc)
    projection = build_causal_eligibility_projection(
        dataset_identity="dataset:phase2",
        calendar_id="XNYS",
        calendar_version="2026.10",
        calendar_sequence_digest="cal_digest_1",
        stage=ProjectionStage.TRAIN,
        fold_id="fold_0000",
        fold_interval=interval,
        stage_information_cutoff_utc=cutoff,
        boundary_session_index=interval.train_end,
        observations=(
            _obs(
                "o1",
                0,
                obs_dt=cutoff - timedelta(minutes=2),
                avail_dt=cutoff - timedelta(minutes=1),
                features={"f1": "bad"},
            ),
        ),
        required_feature_ids=("f1",),
        m3_policy_identity="m3_policy_v1",
        m5_policy_identity="m5_policy_v1",
    )
    assert projection.eligible_count == 0
    assert projection.excluded_observations[0].reason_code == ExclusionReason.FEATURE_NONFINITE


def test_train_boundary_mismatch_rejected() -> None:
    interval = _interval()
    cutoff = datetime(2026, 1, 1, 10, tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        build_causal_eligibility_projection(
            dataset_identity="dataset:phase2",
            calendar_id="XNYS",
            calendar_version="2026.10",
            calendar_sequence_digest="cal_digest_1",
            stage=ProjectionStage.TRAIN,
            fold_id="fold_0000",
            fold_interval=interval,
            stage_information_cutoff_utc=cutoff,
            boundary_session_index=interval.train_end - 1,
            observations=(_obs("o1", 0, obs_dt=cutoff - timedelta(minutes=2), avail_dt=cutoff - timedelta(minutes=1)),),
            required_feature_ids=("f1",),
            m3_policy_identity="m3_policy_v1",
            m5_policy_identity="m5_policy_v1",
        )


def test_validation_boundary_mismatch_rejected() -> None:
    interval = _interval()
    cutoff = datetime(2026, 1, 1, 10, tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        build_causal_eligibility_projection(
            dataset_identity="dataset:phase2",
            calendar_id="XNYS",
            calendar_version="2026.10",
            calendar_sequence_digest="cal_digest_1",
            stage=ProjectionStage.VALIDATION,
            fold_id="fold_0000",
            fold_interval=interval,
            stage_information_cutoff_utc=cutoff,
            boundary_session_index=interval.validation_end - 1,
            observations=(
                _obs(
                    "o1",
                    interval.validation_start,
                    obs_dt=cutoff - timedelta(minutes=2),
                    avail_dt=cutoff - timedelta(minutes=1),
                ),
            ),
            required_feature_ids=("f1",),
            m3_policy_identity="m3_policy_v1",
            m5_policy_identity="m5_policy_v1",
        )


def test_test_boundary_mismatch_rejected() -> None:
    interval = _interval()
    cutoff = datetime(2026, 1, 1, 10, tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        build_causal_eligibility_projection(
            dataset_identity="dataset:phase2",
            calendar_id="XNYS",
            calendar_version="2026.10",
            calendar_sequence_digest="cal_digest_1",
            stage=ProjectionStage.TEST,
            fold_id="fold_0000",
            fold_interval=interval,
            stage_information_cutoff_utc=cutoff,
            boundary_session_index=interval.test_end - 1,
            observations=(
                _obs(
                    "o1",
                    interval.test_start,
                    obs_dt=cutoff - timedelta(minutes=2),
                    avail_dt=cutoff - timedelta(minutes=1),
                ),
            ),
            required_feature_ids=("f1",),
            m3_policy_identity="m3_policy_v1",
            m5_policy_identity="m5_policy_v1",
        )


def test_calendar_version_mismatch_rejected() -> None:
    plan = _plan()
    interval = _interval()
    cutoff = datetime(2026, 1, 1, 10, tzinfo=timezone.utc)
    projection = build_causal_eligibility_projection(
        dataset_identity=plan.dataset_identity,
        calendar_id=plan.calendar_id,
        calendar_version="2026.11",
        calendar_sequence_digest="cal_digest_1",
        stage=ProjectionStage.TRAIN,
        fold_id="fold_0000",
        fold_interval=interval,
        stage_information_cutoff_utc=cutoff,
        boundary_session_index=interval.train_end,
        observations=(_obs("o1", 0, obs_dt=cutoff - timedelta(minutes=2), avail_dt=cutoff - timedelta(minutes=1)),),
        required_feature_ids=("f1",),
        m3_policy_identity=plan.model_policy_identity,
        m5_policy_identity=plan.formation_policy_identity,
    )
    evidence = FoldEvidenceArtifact(
        family_provenance_digest="family:1",
        calendar_provenance_digest="calendar:1",
        model_provenance_digest="model:1",
        validation_digest="validation:1",
        evidence_digest="evidence:1",
    )
    with pytest.raises(ValueError):
        build_fold_input_binding(
            plan=plan,
            fold_id="fold_0000",
            interval=interval,
            projection=projection,
            evidence=evidence,
            evidence_dependency_identities=(
                "family:1",
                "calendar:1",
                "model:1",
                "validation:1",
                "evidence:1",
                "cal_digest_1",
            ),
        )


def test_calendar_sequence_digest_mismatch_rejected() -> None:
    plan = _plan()
    interval = _interval()
    cutoff = datetime(2026, 1, 1, 10, tzinfo=timezone.utc)
    projection = build_causal_eligibility_projection(
        dataset_identity=plan.dataset_identity,
        calendar_id=plan.calendar_id,
        calendar_version=plan.calendar_version,
        calendar_sequence_digest="cal_digest_actual",
        stage=ProjectionStage.TRAIN,
        fold_id="fold_0000",
        fold_interval=interval,
        stage_information_cutoff_utc=cutoff,
        boundary_session_index=interval.train_end,
        observations=(_obs("o1", 0, obs_dt=cutoff - timedelta(minutes=2), avail_dt=cutoff - timedelta(minutes=1)),),
        required_feature_ids=("f1",),
        m3_policy_identity=plan.model_policy_identity,
        m5_policy_identity=plan.formation_policy_identity,
    )
    evidence = FoldEvidenceArtifact(
        family_provenance_digest="family:1",
        calendar_provenance_digest="calendar:1",
        model_provenance_digest="model:1",
        validation_digest="validation:1",
        evidence_digest="evidence:1",
    )
    with pytest.raises(ValueError):
        build_fold_input_binding(
            plan=plan,
            fold_id="fold_0000",
            interval=interval,
            projection=projection,
            evidence=evidence,
            evidence_dependency_identities=(
                "family:1",
                "calendar:1",
                "model:1",
                "validation:1",
                "evidence:1",
                "cal_digest_other",
            ),
        )


def test_unbound_dependency_identity_rejected() -> None:
    plan = _plan()
    interval = _interval()
    cutoff = datetime(2026, 1, 1, 10, tzinfo=timezone.utc)
    projection = build_causal_eligibility_projection(
        dataset_identity=plan.dataset_identity,
        calendar_id=plan.calendar_id,
        calendar_version=plan.calendar_version,
        calendar_sequence_digest="cal_digest_1",
        stage=ProjectionStage.TRAIN,
        fold_id="fold_0000",
        fold_interval=interval,
        stage_information_cutoff_utc=cutoff,
        boundary_session_index=interval.train_end,
        observations=(_obs("o1", 0, obs_dt=cutoff - timedelta(minutes=2), avail_dt=cutoff - timedelta(minutes=1)),),
        required_feature_ids=("f1",),
        m3_policy_identity=plan.model_policy_identity,
        m5_policy_identity=plan.formation_policy_identity,
    )
    evidence = FoldEvidenceArtifact(
        family_provenance_digest="family:1",
        calendar_provenance_digest="calendar:1",
        model_provenance_digest="model:1",
        validation_digest="validation:1",
        evidence_digest="evidence:1",
    )
    with pytest.raises(ValueError):
        build_fold_input_binding(
            plan=plan,
            fold_id="fold_0000",
            interval=interval,
            projection=projection,
            evidence=evidence,
            evidence_dependency_identities=("family:1",),
        )


def test_repeat_run_determinism() -> None:
    plan = _plan()
    fold = _canonical_fold(total_sessions=180)
    interval = fold.interval
    cutoff = datetime(2026, 1, 1, 10, tzinfo=timezone.utc)
    projection = build_causal_eligibility_projection(
        dataset_identity=plan.dataset_identity,
        calendar_id=plan.calendar_id,
        calendar_version=plan.calendar_version,
        calendar_sequence_digest="cal_digest_1",
        stage=ProjectionStage.TRAIN,
        fold_id=fold.fold_id,
        fold_interval=interval,
        stage_information_cutoff_utc=cutoff,
        boundary_session_index=interval.train_end,
        observations=tuple(
            _obs(
                f"o{i}",
                i,
                obs_dt=cutoff - timedelta(minutes=2),
                avail_dt=cutoff - timedelta(minutes=1),
                features={"f1": float(i), "f2": float(i + 1)},
            )
            for i in range(interval.train_start, interval.train_end)
        ),
        required_feature_ids=("f1",),
        m3_policy_identity=plan.model_policy_identity,
        m5_policy_identity=plan.formation_policy_identity,
    )
    evidence = FoldEvidenceArtifact(
        family_provenance_digest="family:1",
        calendar_provenance_digest="calendar:1",
        model_provenance_digest="model:1",
        validation_digest="validation:1",
        evidence_digest="evidence:1",
    )
    a = assemble_phase2_structural(
        total_sessions=180,
        policy=_policy(),
        plan=plan,
        projections=(projection,),
        evidence_by_fold={fold.fold_id: evidence},
        evidence_dependency_identities_by_fold={
            fold.fold_id: (
                "family:1",
                "calendar:1",
                "model:1",
                "validation:1",
                "evidence:1",
                "cal_digest_1",
            )
        },
        minima_config_by_fold={fold.fold_id: {"n_clusters": 3, "n_components": 1}},
    )
    b = assemble_phase2_structural(
        total_sessions=180,
        policy=_policy(),
        plan=plan,
        projections=(projection,),
        evidence_by_fold={fold.fold_id: evidence},
        evidence_dependency_identities_by_fold={
            fold.fold_id: (
                "family:1",
                "calendar:1",
                "model:1",
                "validation:1",
                "evidence:1",
                "cal_digest_1",
            )
        },
        minima_config_by_fold={fold.fold_id: {"n_clusters": 3, "n_components": 1}},
    )
    assert [f.fold_id for f in a.folds] == [f.fold_id for f in b.folds]
    assert a.fold_bindings[0].fold_input.input_digest == b.fold_bindings[0].fold_input.input_digest
    assert a.no_entry_artifacts[0].artifact.no_entry_digest == b.no_entry_artifacts[0].artifact.no_entry_digest


def test_m3_training_below_minimum_rejected() -> None:
    plan = _plan()
    fold = _canonical_fold(total_sessions=180)
    interval = fold.interval
    cutoff = datetime(2026, 1, 1, 10, tzinfo=timezone.utc)
    projection = build_causal_eligibility_projection(
        dataset_identity=plan.dataset_identity,
        calendar_id=plan.calendar_id,
        calendar_version=plan.calendar_version,
        calendar_sequence_digest="cal_digest_1",
        stage=ProjectionStage.TRAIN,
        fold_id=fold.fold_id,
        fold_interval=interval,
        stage_information_cutoff_utc=cutoff,
        boundary_session_index=interval.train_end,
        observations=tuple(
            _obs(
                f"o{i}",
                i,
                obs_dt=cutoff - timedelta(minutes=2),
                avail_dt=cutoff - timedelta(minutes=1),
                features={"f1": float(i), "f2": float(i + 1)},
            )
            for i in range(20)
        ),
        required_feature_ids=("f1", "f2"),
        m3_policy_identity=plan.model_policy_identity,
        m5_policy_identity=plan.formation_policy_identity,
    )
    evidence = FoldEvidenceArtifact(
        family_provenance_digest="family:1",
        calendar_provenance_digest="calendar:1",
        model_provenance_digest="model:1",
        validation_digest="validation:1",
        evidence_digest="evidence:1",
    )
    result = assemble_phase2_structural(
        total_sessions=180,
        policy=_policy(),
        plan=plan,
        projections=(projection,),
        evidence_by_fold={fold.fold_id: evidence},
        evidence_dependency_identities_by_fold={
            fold.fold_id: (
                "family:1",
                "calendar:1",
                "model:1",
                "validation:1",
                "evidence:1",
                "cal_digest_1",
            )
        },
        minima_config_by_fold={fold.fold_id: {"n_clusters": 3, "n_components": 1}},
    )
    assert "M3_TRAINING_MINIMUM_UNSATISFIED" in result.failure_codes
    assert result.fold_bindings == ()


def test_m3_validation_below_minimum_rejected() -> None:
    plan = _plan()
    fold = _canonical_fold(total_sessions=180)
    interval = fold.interval
    cutoff = datetime(2026, 1, 1, 10, tzinfo=timezone.utc)
    projection = build_causal_eligibility_projection(
        dataset_identity=plan.dataset_identity,
        calendar_id=plan.calendar_id,
        calendar_version=plan.calendar_version,
        calendar_sequence_digest="cal_digest_1",
        stage=ProjectionStage.VALIDATION,
        fold_id=fold.fold_id,
        fold_interval=interval,
        stage_information_cutoff_utc=cutoff,
        boundary_session_index=interval.validation_end,
        observations=tuple(
            _obs(
                f"v{i}",
                interval.validation_start + i,
                obs_dt=cutoff - timedelta(minutes=2),
                avail_dt=cutoff - timedelta(minutes=1),
            )
            for i in range(10)
        ),
        required_feature_ids=("f1",),
        m3_policy_identity=plan.model_policy_identity,
        m5_policy_identity=plan.formation_policy_identity,
    )
    evidence = FoldEvidenceArtifact(
        family_provenance_digest="family:1",
        calendar_provenance_digest="calendar:1",
        model_provenance_digest="model:1",
        validation_digest="validation:1",
        evidence_digest="evidence:1",
    )
    result = assemble_phase2_structural(
        total_sessions=180,
        policy=_policy(),
        plan=plan,
        projections=(projection,),
        evidence_by_fold={fold.fold_id: evidence},
        evidence_dependency_identities_by_fold={
            fold.fold_id: (
                "family:1",
                "calendar:1",
                "model:1",
                "validation:1",
                "evidence:1",
                "cal_digest_1",
            )
        },
        minima_config_by_fold={fold.fold_id: {"n_clusters": 6, "n_components": 1}},
    )
    assert "M3_HELDOUT_MINIMUM_UNSATISFIED" in result.failure_codes


def test_m3_test_below_minimum_rejected() -> None:
    plan = _plan()
    fold = _canonical_fold(total_sessions=180)
    interval = fold.interval
    cutoff = datetime(2026, 1, 1, 10, tzinfo=timezone.utc)
    projection = build_causal_eligibility_projection(
        dataset_identity=plan.dataset_identity,
        calendar_id=plan.calendar_id,
        calendar_version=plan.calendar_version,
        calendar_sequence_digest="cal_digest_1",
        stage=ProjectionStage.TEST,
        fold_id=fold.fold_id,
        fold_interval=interval,
        stage_information_cutoff_utc=cutoff,
        boundary_session_index=interval.test_end,
        observations=tuple(
            _obs(
                f"t{i}",
                interval.test_start + i,
                obs_dt=cutoff - timedelta(minutes=2),
                avail_dt=cutoff - timedelta(minutes=1),
            )
            for i in range(5)
        ),
        required_feature_ids=("f1",),
        m3_policy_identity=plan.model_policy_identity,
        m5_policy_identity=plan.formation_policy_identity,
    )
    evidence = FoldEvidenceArtifact(
        family_provenance_digest="family:1",
        calendar_provenance_digest="calendar:1",
        model_provenance_digest="model:1",
        validation_digest="validation:1",
        evidence_digest="evidence:1",
    )
    result = assemble_phase2_structural(
        total_sessions=180,
        policy=_policy(),
        plan=plan,
        projections=(projection,),
        evidence_by_fold={fold.fold_id: evidence},
        evidence_dependency_identities_by_fold={
            fold.fold_id: (
                "family:1",
                "calendar:1",
                "model:1",
                "validation:1",
                "evidence:1",
                "cal_digest_1",
            )
        },
        minima_config_by_fold={fold.fold_id: {"n_clusters": 10, "n_components": 1}},
    )
    assert "M3_HELDOUT_MINIMUM_UNSATISFIED" in result.failure_codes


def test_m5_training_below_minimum_rejected() -> None:
    plan = _plan()
    fold = _canonical_fold(total_sessions=180)
    interval = fold.interval
    cutoff = datetime(2026, 1, 1, 10, tzinfo=timezone.utc)
    projection = build_causal_eligibility_projection(
        dataset_identity=plan.dataset_identity,
        calendar_id=plan.calendar_id,
        calendar_version=plan.calendar_version,
        calendar_sequence_digest="cal_digest_1",
        stage=ProjectionStage.TRAIN,
        fold_id=fold.fold_id,
        fold_interval=interval,
        stage_information_cutoff_utc=cutoff,
        boundary_session_index=interval.train_end,
        observations=tuple(
            _obs(
                f"o{i}",
                i,
                obs_dt=cutoff - timedelta(minutes=2),
                avail_dt=cutoff - timedelta(minutes=1),
                features={"f1": float(i), "f2": float(i + 1)},
            )
            for i in range(40)
        ),
        required_feature_ids=("f1", "f2"),
        m3_policy_identity=plan.model_policy_identity,
        m5_policy_identity=plan.formation_policy_identity,
    )
    evidence = FoldEvidenceArtifact(
        family_provenance_digest="family:1",
        calendar_provenance_digest="calendar:1",
        model_provenance_digest="model:1",
        validation_digest="validation:1",
        evidence_digest="evidence:1",
    )
    result = assemble_phase2_structural(
        total_sessions=180,
        policy=_policy(),
        plan=plan,
        projections=(projection,),
        evidence_by_fold={fold.fold_id: evidence},
        evidence_dependency_identities_by_fold={
            fold.fold_id: (
                "family:1",
                "calendar:1",
                "model:1",
                "validation:1",
                "evidence:1",
                "cal_digest_1",
            )
        },
        minima_config_by_fold={fold.fold_id: {"n_clusters": 1, "n_components": 21}},
    )
    assert "M5_TRAINING_MINIMUM_UNSATISFIED" in result.failure_codes


def test_missing_model_configuration_rejected() -> None:
    plan = _plan()
    fold = _canonical_fold(total_sessions=180)
    interval = fold.interval
    cutoff = datetime(2026, 1, 1, 10, tzinfo=timezone.utc)
    projection = build_causal_eligibility_projection(
        dataset_identity=plan.dataset_identity,
        calendar_id=plan.calendar_id,
        calendar_version=plan.calendar_version,
        calendar_sequence_digest="cal_digest_1",
        stage=ProjectionStage.TRAIN,
        fold_id=fold.fold_id,
        fold_interval=interval,
        stage_information_cutoff_utc=cutoff,
        boundary_session_index=interval.train_end,
        observations=tuple(
            _obs(
                f"o{i}",
                i,
                obs_dt=cutoff - timedelta(minutes=2),
                avail_dt=cutoff - timedelta(minutes=1),
                features={"f1": float(i), "f2": float(i + 1)},
            )
            for i in range(50)
        ),
        required_feature_ids=("f1", "f2"),
        m3_policy_identity=plan.model_policy_identity,
        m5_policy_identity=plan.formation_policy_identity,
    )
    evidence = FoldEvidenceArtifact(
        family_provenance_digest="family:1",
        calendar_provenance_digest="calendar:1",
        model_provenance_digest="model:1",
        validation_digest="validation:1",
        evidence_digest="evidence:1",
    )
    result = assemble_phase2_structural(
        total_sessions=180,
        policy=_policy(),
        plan=plan,
        projections=(projection,),
        evidence_by_fold={fold.fold_id: evidence},
        evidence_dependency_identities_by_fold={
            fold.fold_id: (
                "family:1",
                "calendar:1",
                "model:1",
                "validation:1",
                "evidence:1",
                "cal_digest_1",
            )
        },
        minima_config_by_fold=None,
    )
    assert "MISSING_MODEL_CONFIGURATION" in result.failure_codes


def test_duplicate_projection_rejection() -> None:
    plan = _plan()
    fold = _canonical_fold(total_sessions=180)
    interval = fold.interval
    cutoff = datetime(2026, 1, 1, 10, tzinfo=timezone.utc)
    projection = build_causal_eligibility_projection(
        dataset_identity=plan.dataset_identity,
        calendar_id=plan.calendar_id,
        calendar_version=plan.calendar_version,
        calendar_sequence_digest="cal_digest_1",
        stage=ProjectionStage.TRAIN,
        fold_id=fold.fold_id,
        fold_interval=interval,
        stage_information_cutoff_utc=cutoff,
        boundary_session_index=interval.train_end,
        observations=tuple(
            _obs(
                f"o{i}",
                i,
                obs_dt=cutoff - timedelta(minutes=2),
                avail_dt=cutoff - timedelta(minutes=1),
                features={"f1": float(i), "f2": float(i + 1)},
            )
            for i in range(60)
        ),
        required_feature_ids=("f1", "f2"),
        m3_policy_identity=plan.model_policy_identity,
        m5_policy_identity=plan.formation_policy_identity,
    )
    evidence = FoldEvidenceArtifact(
        family_provenance_digest="family:1",
        calendar_provenance_digest="calendar:1",
        model_provenance_digest="model:1",
        validation_digest="validation:1",
        evidence_digest="evidence:1",
    )
    result = assemble_phase2_structural(
        total_sessions=180,
        policy=_policy(),
        plan=plan,
        projections=(projection, projection),
        evidence_by_fold={fold.fold_id: evidence},
        evidence_dependency_identities_by_fold={
            fold.fold_id: (
                "family:1",
                "calendar:1",
                "model:1",
                "validation:1",
                "evidence:1",
                "cal_digest_1",
            )
        },
        minima_config_by_fold={fold.fold_id: {"n_clusters": 2, "n_components": 1}},
    )
    assert "DUPLICATE_FOLD_PROJECTION" in result.failure_codes


def test_short_alias_rejected() -> None:
    plan = _plan()
    fold = _canonical_fold(total_sessions=180)
    interval = fold.interval
    cutoff = datetime(2026, 1, 1, 10, tzinfo=timezone.utc)
    projection = build_causal_eligibility_projection(
        dataset_identity=plan.dataset_identity,
        calendar_id=plan.calendar_id,
        calendar_version=plan.calendar_version,
        calendar_sequence_digest="cal_digest_1",
        stage=ProjectionStage.TRAIN,
        fold_id="fold_0000",
        fold_interval=interval,
        stage_information_cutoff_utc=cutoff,
        boundary_session_index=interval.train_end,
        observations=tuple(
            _obs(
                f"o{i}",
                i,
                obs_dt=cutoff - timedelta(minutes=2),
                avail_dt=cutoff - timedelta(minutes=1),
            )
            for i in range(40)
        ),
        required_feature_ids=("f1",),
        m3_policy_identity=plan.model_policy_identity,
        m5_policy_identity=plan.formation_policy_identity,
    )
    evidence = FoldEvidenceArtifact(
        family_provenance_digest="family:1",
        calendar_provenance_digest="calendar:1",
        model_provenance_digest="model:1",
        validation_digest="validation:1",
        evidence_digest="evidence:1",
    )
    result = assemble_phase2_structural(
        total_sessions=180,
        policy=_policy(),
        plan=plan,
        projections=(projection,),
        evidence_by_fold={fold.fold_id: evidence},
        evidence_dependency_identities_by_fold={
            fold.fold_id: (
                "family:1",
                "calendar:1",
                "model:1",
                "validation:1",
                "evidence:1",
                "cal_digest_1",
            )
        },
        minima_config_by_fold={fold.fold_id: {"n_clusters": 2, "n_components": 1}},
    )
    assert "NONCANONICAL_FOLD_ID" in result.failure_codes


def test_permuted_projection_ordering_stable() -> None:
    plan = _plan()
    folds = build_expanding_fold_intervals(total_sessions=200, policy=_policy())
    cutoff = datetime(2026, 1, 1, 10, tzinfo=timezone.utc)

    def make_projection(fold, prefix: str):
        return build_causal_eligibility_projection(
            dataset_identity=plan.dataset_identity,
            calendar_id=plan.calendar_id,
            calendar_version=plan.calendar_version,
            calendar_sequence_digest=f"cal_digest_{prefix}",
            stage=ProjectionStage.TRAIN,
            fold_id=fold.fold_id,
            fold_interval=fold.interval,
            stage_information_cutoff_utc=cutoff,
            boundary_session_index=fold.interval.train_end,
            observations=tuple(
                _obs(
                    f"{prefix}_{i}",
                    i,
                    obs_dt=cutoff - timedelta(minutes=2),
                    avail_dt=cutoff - timedelta(minutes=1),
                    features={"f1": float(i), "f2": float(i + 1)},
                )
                for i in range(fold.interval.train_start, fold.interval.train_end)
            ),
            required_feature_ids=("f1", "f2"),
            m3_policy_identity=plan.model_policy_identity,
            m5_policy_identity=plan.formation_policy_identity,
        )

    p0 = make_projection(folds[0], "a")
    p1 = make_projection(folds[1], "b")
    evidence = {
        folds[0].fold_id: FoldEvidenceArtifact(
            family_provenance_digest="family:0",
            calendar_provenance_digest="calendar:0",
            model_provenance_digest="model:0",
            validation_digest="validation:0",
            evidence_digest="evidence:0",
        ),
        folds[1].fold_id: FoldEvidenceArtifact(
            family_provenance_digest="family:1",
            calendar_provenance_digest="calendar:1",
            model_provenance_digest="model:1",
            validation_digest="validation:1",
            evidence_digest="evidence:1",
        ),
    }
    deps = {
        folds[0].fold_id: ("family:0", "calendar:0", "model:0", "validation:0", "evidence:0", "cal_digest_a"),
        folds[1].fold_id: ("family:1", "calendar:1", "model:1", "validation:1", "evidence:1", "cal_digest_b"),
    }
    minima = {
        folds[0].fold_id: {"n_clusters": 3, "n_components": 1},
        folds[1].fold_id: {"n_clusters": 3, "n_components": 1},
    }

    a = assemble_phase2_structural(
        total_sessions=200,
        policy=_policy(),
        plan=plan,
        projections=(p1, p0),
        evidence_by_fold=evidence,
        evidence_dependency_identities_by_fold=deps,
        minima_config_by_fold=minima,
    )
    b = assemble_phase2_structural(
        total_sessions=200,
        policy=_policy(),
        plan=plan,
        projections=(p0, p1),
        evidence_by_fold=evidence,
        evidence_dependency_identities_by_fold=deps,
        minima_config_by_fold=minima,
    )
    assert [item.fold_input.fold_id for item in a.fold_bindings] == [item.fold_input.fold_id for item in b.fold_bindings]
    assert [item.fold_input.input_digest for item in a.fold_bindings] == [item.fold_input.input_digest for item in b.fold_bindings]
    assert [item.artifact.no_entry_digest for item in a.no_entry_artifacts] == [item.artifact.no_entry_digest for item in b.no_entry_artifacts]

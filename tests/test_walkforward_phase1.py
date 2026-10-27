from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from regimequant.walkforward import (
    M10Policy,
    ExperimentPlan,
    ExperimentResult,
    FoldCertification,
    FoldEvidenceArtifact,
    FoldInput,
    FoldInterval,
    FoldOutput,
    NoEntryArtifact,
    ValidationResult,
)
from regimequant.walkforward.identity import (
    build_experiment_result_digest,
    build_fold_input_digest,
    build_no_entry_digest,
    canonical_json,
    canonical_decimal_string,
    compute_identity_digest,
    hash_identity_payload,
    normalize_timestamp_utc,
)
from regimequant.walkforward.manifest import (
    M10_RATIFIED_POLICY,
    build_experiment_plan,
    validate_experiment_plan,
)


@pytest.fixture
def policy() -> M10Policy:
    return M10Policy(
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


@pytest.fixture
def plan(policy: M10Policy) -> ExperimentPlan:
    return build_experiment_plan(
        experiment_id="exp_001",
        experiment_version="m10.v1",
        regime_primary="M5",
        regime_comparator="M3",
        calendar_id="XNYS",
        calendar_version="2025.01",
        dataset_identity="dataset:sp500:2025.01",
        feature_spec_set_hash="spec_hash_001",
        model_policy_identity="model_policy_001",
        formation_policy_identity="formation_policy_001",
        family_policy_identity="family_policy_001",
        final_holdout_sessions=policy.final_holdout_sessions,
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )


def test_frozen_slotted_contracts_are_immutable() -> None:
    policy = M10Policy(
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
    with pytest.raises(FrozenInstanceError):
        policy.initial_training_sessions = 121


def test_ratified_policy_values(policy: M10Policy) -> None:
    assert M10_RATIFIED_POLICY == policy
    assert policy.initial_training_sessions == 120
    assert policy.validation_sessions == 20
    assert policy.outer_test_sessions == 20
    assert policy.stride_sessions == 20
    assert policy.final_holdout_sessions == 20
    assert policy.formation_min_aligned_levels == 253
    assert policy.discovery_freshness_limit_completed_sessions == 20
    assert policy.discovery_refresh_sessions == 20
    assert policy.by_q_threshold == 0.05
    assert policy.empty_signal_policy == "valid_abstention_only"
    assert policy.research_mode == "research-only"


def test_unsupported_policy_rejected() -> None:
    with pytest.raises(ValueError):
        M10Policy(
            initial_training_sessions=120,
            validation_sessions=20,
            outer_test_sessions=20,
            stride_sessions=20,
            final_holdout_sessions=20,
            formation_min_aligned_levels=253,
            discovery_freshness_limit_completed_sessions=20,
            discovery_refresh_sessions=20,
            by_q_threshold=0.1,
            empty_signal_policy="valid_abstention_only",
            research_mode="production",
        )


def test_research_only_enforced(policy: M10Policy) -> None:
    with pytest.raises(ValueError):
        M10Policy(
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
            research_mode="production",
        )


def test_missing_identity_fields_are_rejected() -> None:
    with pytest.raises(ValueError):
        build_experiment_plan(
            experiment_id="",
            experiment_version="m10.v1",
            regime_primary="M5",
            regime_comparator="M3",
            calendar_id="XNYS",
            calendar_version="2025.01",
            dataset_identity="dataset:sp500:2025.01",
            feature_spec_set_hash="spec_hash_001",
            model_policy_identity="model_policy_001",
            formation_policy_identity="formation_policy_001",
            family_policy_identity="",
            final_holdout_sessions=20,
        )


def test_missing_plan_schema_rejected() -> None:
    with pytest.raises(ValueError):
        ExperimentPlan(
            schema="wrong.schema",
            experiment_id="exp_001",
            experiment_version="m10.v1",
            regime_primary="M5",
            regime_comparator="M3",
            calendar_id="XNYS",
            calendar_version="2025.01",
            dataset_identity="dataset:sp500:2025.01",
            feature_spec_set_hash="spec_hash_001",
            model_policy_identity="model_policy_001",
            formation_policy_identity="formation_policy_001",
            family_policy_identity="family_policy_001",
            final_holdout_sessions=20,
            created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )


def test_unsupported_schema_version_rejected() -> None:
    with pytest.raises(ValueError):
        ExperimentPlan(
            schema="regimequant.walkforward.experiment_plan",
            experiment_id="exp_001",
            experiment_version="m10.v99",
            regime_primary="M5",
            regime_comparator="M3",
            calendar_id="XNYS",
            calendar_version="2025.01",
            dataset_identity="dataset:sp500:2025.01",
            feature_spec_set_hash="spec_hash_001",
            model_policy_identity="model_policy_001",
            formation_policy_identity="formation_policy_001",
            family_policy_identity="family_policy_001",
            final_holdout_sessions=20,
            created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )


def test_manifest_remains_structural_only() -> None:
    plan = build_experiment_plan(
        experiment_id="exp_001",
        experiment_version="m10.v1",
        regime_primary="M5",
        regime_comparator="M3",
        calendar_id="XNYS",
        calendar_version="2025.01",
        dataset_identity="dataset:sp500:2025.01",
        feature_spec_set_hash="spec_hash_001",
        model_policy_identity="model_policy_001",
        formation_policy_identity="formation_policy_001",
        family_policy_identity="family_policy_001",
        final_holdout_sessions=20,
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    assert plan.structural_only is True
    assert plan.scientific_certification_status == "STRUCTURAL_ONLY"


def test_no_entry_prohibits_invoked_m9_true() -> None:
    with pytest.raises(ValueError):
        build_no_entry_digest(
            experiment_id="exp_123",
            fold_id="fold_1",
            fold_input_identity="fold_input_hash",
            decision_session=160,
            boundary_b=180,
            reason_code="INSUFFICIENT_HISTORY",
            evidence_dependency_identities=("family:1", "calendar:1"),
            invoked_m9=True,
        )


def test_no_entry_digest_contains_false_invariant() -> None:
    digest = build_no_entry_digest(
        experiment_id="exp_123",
        fold_id="fold_1",
        fold_input_identity="fold_input_hash",
        decision_session=160,
        boundary_b=180,
        reason_code="INSUFFICIENT_HISTORY",
        evidence_dependency_identities=("family:1", "calendar:1"),
        invoked_m9=False,
    )
    assert isinstance(digest, str)
    assert len(digest) == 64


def test_identical_inputs_produce_identical_digests() -> None:
    a = build_no_entry_digest(
        experiment_id="exp_123",
        fold_id="fold_1",
        fold_input_identity="fold_input_hash",
        decision_session=160,
        boundary_b=180,
        reason_code="INSUFFICIENT_HISTORY",
        evidence_dependency_identities=("family:1", "calendar:1"),
        invoked_m9=False,
    )
    b = build_no_entry_digest(
        experiment_id="exp_123",
        fold_id="fold_1",
        fold_input_identity="fold_input_hash",
        decision_session=160,
        boundary_b=180,
        reason_code="INSUFFICIENT_HISTORY",
        evidence_dependency_identities=("family:1", "calendar:1"),
        invoked_m9=False,
    )
    assert a == b


def test_duplicate_fold_rejection() -> None:
    interval = FoldInterval(
        train_start=0,
        train_end=120,
        validation_start=120,
        validation_end=140,
        test_start=140,
        test_end=160,
        holdout_start=160,
        holdout_end=180,
    )
    input_a = FoldInput(
        schema="regimequant.walkforward.fold_input",
        version="m10.v1",
        plan_id="plan_1",
        fold_id="fold_1",
        interval=interval,
        dataset_identity="dataset:sp500:2025.01",
        feature_spec_set_hash="spec_hash_001",
        calendar_identity="calendar:XNYS",
        evidence=FoldEvidenceArtifact(
            family_provenance_digest="family:1",
            calendar_provenance_digest="calendar:1",
            model_provenance_digest="model:1",
            validation_digest="valid:1",
            evidence_digest="evidence:1",
            provenance_state="CALLER_SUPPLIED",
            family_provenance_state="UNVERIFIED",
        ),
        model_policy_identity="model_policy_001",
        formation_policy_identity="formation_policy_001",
        family_policy_identity="family_policy_001",
        input_digest="digest_a",
    )
    input_b = FoldInput(
        schema="regimequant.walkforward.fold_input",
        version="m10.v1",
        plan_id="plan_1",
        fold_id="fold_1",
        interval=interval,
        dataset_identity="dataset:sp500:2025.01",
        feature_spec_set_hash="spec_hash_001",
        calendar_identity="calendar:XNYS",
        evidence=input_a.evidence,
        model_policy_identity="model_policy_001",
        formation_policy_identity="formation_policy_001",
        family_policy_identity="family_policy_001",
        input_digest="digest_b",
    )
    with pytest.raises(ValueError):
        validate_experiment_plan(
            ExperimentPlan(
                experiment_id="exp_001",
                experiment_version="m10.v1",
                regime_primary="M5",
                regime_comparator="M3",
                calendar_id="XNYS",
                calendar_version="2025.01",
                dataset_identity="dataset:sp500:2025.01",
                feature_spec_set_hash="spec_hash_001",
                model_policy_identity="model_policy_001",
                formation_policy_identity="formation_policy_001",
                family_policy_identity="family_policy_001",
                final_holdout_sessions=20,
                created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
                folds=(input_a, input_b),
            )
        )


def test_invalid_interval_rejected() -> None:
    with pytest.raises(ValueError):
        FoldInterval(
            train_start=0,
            train_end=120,
            validation_start=100,
            validation_end=140,
            test_start=140,
            test_end=160,
            holdout_start=160,
            holdout_end=180,
        )


def test_manifest_is_deterministic(plan: ExperimentPlan) -> None:
    manifest = build_experiment_plan(
        experiment_id=plan.experiment_id,
        experiment_version=plan.experiment_version,
        regime_primary=plan.regime_primary,
        regime_comparator=plan.regime_comparator,
        calendar_id=plan.calendar_id,
        calendar_version=plan.calendar_version,
        dataset_identity=plan.dataset_identity,
        feature_spec_set_hash=plan.feature_spec_set_hash,
        model_policy_identity=plan.model_policy_identity,
        formation_policy_identity=plan.formation_policy_identity,
        family_policy_identity=plan.family_policy_identity,
        final_holdout_sessions=plan.final_holdout_sessions,
        created_at=plan.created_at,
    )
    assert manifest == plan


def test_canonical_json_is_stable() -> None:
    payload = {"b": 2, "a": [3, 1, 2], "nested": {"z": 1, "y": 2}}
    first = canonical_json(payload)
    second = canonical_json({"nested": {"y": 2, "z": 1}, "a": [3, 1, 2], "b": 2})
    assert first == second


def test_dictionary_key_order_is_independent() -> None:
    a = {"outer": {"b": 1, "a": 2}, "items": [3, 1, 2]}
    b = {"items": [3, 1, 2], "outer": {"a": 2, "b": 1}}
    assert canonical_json(a) == canonical_json(b)


def test_decimal_canonicalization() -> None:
    assert canonical_decimal_string(Decimal("0.1000")) == "0.1"
    assert canonical_decimal_string(Decimal("1E+3")) == "1000"


def test_utc_timestamp_normalization() -> None:
    ts = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    assert normalize_timestamp_utc(ts) == "2025-01-01T12:00:00Z"


def test_naive_timestamp_rejected() -> None:
    with pytest.raises(ValueError):
        normalize_timestamp_utc(datetime(2025, 1, 1, 12, 0, 0))


def test_nan_and_infinity_rejected() -> None:
    with pytest.raises(ValueError):
        canonical_decimal_string(Decimal("NaN"))
    with pytest.raises(ValueError):
        canonical_decimal_string(Decimal("Infinity"))


def test_digest_self_exclusion() -> None:
    payload = {"schema": "regimequant.walkforward", "version": "m10.v1", "digest": "self"}
    digest = compute_identity_digest(payload, domain="regimequant.walkforward.test")
    assert digest == compute_identity_digest({"schema": "regimequant.walkforward", "version": "m10.v1"}, domain="regimequant.walkforward.test")


def test_domain_separation() -> None:
    payload = {"experiment_id": "exp_1"}
    assert compute_identity_digest(payload, domain="domain_a") != compute_identity_digest(payload, domain="domain_b")


def test_parent_child_binding() -> None:
    plan = build_experiment_plan(
        experiment_id="exp_001",
        experiment_version="m10.v1",
        regime_primary="M5",
        regime_comparator="M3",
        calendar_id="XNYS",
        calendar_version="2025.01",
        dataset_identity="dataset:sp500:2025.01",
        feature_spec_set_hash="spec_hash_001",
        model_policy_identity="model_policy_001",
        formation_policy_identity="formation_policy_001",
        family_policy_identity="family_policy_001",
        final_holdout_sessions=20,
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    interval = FoldInterval(
        train_start=0,
        train_end=120,
        validation_start=120,
        validation_end=140,
        test_start=140,
        test_end=160,
        holdout_start=160,
        holdout_end=180,
    )
    evidence = FoldEvidenceArtifact(
        family_provenance_digest="family:1",
        calendar_provenance_digest="calendar:1",
        model_provenance_digest="model:1",
        validation_digest="valid:1",
        evidence_digest="evidence:1",
        provenance_state="CALLER_SUPPLIED",
        family_provenance_state="MISSING",
    )
    fold_input = FoldInput(
        schema="regimequant.walkforward.fold_input",
        version="m10.v1",
        plan_id=plan.experiment_id,
        fold_id="fold_1",
        interval=interval,
        dataset_identity=plan.dataset_identity,
        feature_spec_set_hash=plan.feature_spec_set_hash,
        calendar_identity=plan.calendar_id,
        evidence=evidence,
        model_policy_identity=plan.model_policy_identity,
        formation_policy_identity=plan.formation_policy_identity,
        family_policy_identity=plan.family_policy_identity,
        input_digest="digest_1",
    )
    fold_output = FoldOutput(
        schema="regimequant.walkforward.fold_output",
        version="m10.v1",
        plan_id=plan.experiment_id,
        fold_id=fold_input.fold_id,
        model_digest="model_digest_1",
        family_digest="family_digest_1",
        no_trade_digest="no_trade_digest_1",
        accounting_digest="accounting_digest_1",
        decision_digest="decision_digest_1",
        output_digest="output_digest_1",
    )
    result = ExperimentResult(
        schema="regimequant.walkforward.experiment_result",
        version="m10.v1",
        experiment_id=plan.experiment_id,
        plan_id=plan.experiment_id,
        fold_inputs=(fold_input,),
        fold_outputs=(fold_output,),
        scientific_status="VALID_RESEARCH_EXECUTION",
        accounting_status="CERTIFIED",
        metric_status="METRICS_MISSING",
        result_digest="result_digest_1",
    )
    assert result.plan_id == plan.experiment_id
    assert fold_output.fold_id == fold_input.fold_id


def test_identity_tampering_changes_digest() -> None:
    payload = {"experiment_id": "exp_1", "fold_id": "fold_1", "sessions": 120}
    digest_a = compute_identity_digest(payload, domain="domain.xyz")
    payload_b = dict(payload)
    payload_b["sessions"] = 121
    digest_b = compute_identity_digest(payload_b, domain="domain.xyz")
    assert digest_a != digest_b


def test_deterministic_rerun_identity() -> None:
    payload = {"experiment_id": "exp_1", "fold_id": "fold_1", "list": [1, 2, 3]}
    assert compute_identity_digest(payload, domain="regimequant.test") == compute_identity_digest({"fold_id": "fold_1", "experiment_id": "exp_1", "list": [1, 2, 3]}, domain="regimequant.test")


def test_no_entry_invoked_m9_false() -> None:
    artifact = NoEntryArtifact(
        schema="regimequant.walkforward.no_entry",
        version="m10.v1",
        experiment_id="exp_123",
        fold_id="fold_1",
        fold_input_identity="fold_input_hash",
        decision_session=160,
        boundary_b=180,
        reason_code="INSUFFICIENT_HISTORY",
        evidence_dependency_identities=("family:1", "calendar:1"),
        invoked_m9=False,
        no_entry_digest="digest_1",
    )
    assert artifact.invoked_m9 is False
    assert artifact.reason_code == "INSUFFICIENT_HISTORY"


def test_no_entry_identity_reproducible() -> None:
    artifact_a = build_no_entry_digest(
        experiment_id="exp_123",
        fold_id="fold_1",
        fold_input_identity="fold_input_hash",
        decision_session=160,
        boundary_b=180,
        reason_code="INSUFFICIENT_HISTORY",
        evidence_dependency_identities=("family:1", "calendar:1"),
        invoked_m9=False,
    )
    artifact_b = build_no_entry_digest(
        experiment_id="exp_123",
        fold_id="fold_1",
        fold_input_identity="fold_input_hash",
        decision_session=160,
        boundary_b=180,
        reason_code="INSUFFICIENT_HISTORY",
        evidence_dependency_identities=("family:1", "calendar:1"),
        invoked_m9=False,
    )
    assert artifact_a == artifact_b


def test_missing_family_provenance_is_not_scientific_certificate() -> None:
    result = ValidationResult(
        valid=True,
        scientific_status="VALID_RESEARCH_EXECUTION",
        metric_status="METRICS_MISSING",
        reason_code="MISSING_FAMILY_PROVENANCE",
        details="Structural validation only; not certified.",
        authoritative=False,
        status_role="STRUCTURAL_ONLY",
    )
    assert result.valid is True
    assert result.authoritative is False
    assert result.status_role == "STRUCTURAL_ONLY"
    assert result.metric_status == "METRICS_MISSING"


def test_no_executable_m9_integration() -> None:
    policy = M10Policy(
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
    assert policy.research_mode == "research-only"
    assert "m9" not in __import__("regimequant.walkforward").__dict__


def test_no_frozen_module_modification() -> None:
    import regimequant.backtesting.m9_engine as engine

    assert hasattr(engine, "run_research_backtest")
    assert callable(engine.run_research_backtest)


def test_bind_child_to_parent_identity() -> None:
    plan = build_experiment_plan(
        experiment_id="exp-parent",
        experiment_version="m10.v1",
        regime_primary="M5",
        regime_comparator="M3",
        calendar_id="XNYS",
        calendar_version="2025.01",
        dataset_identity="dataset:sp500:2025.01",
        feature_spec_set_hash="spec_hash_001",
        model_policy_identity="model_policy_001",
        formation_policy_identity="formation_policy_001",
        family_policy_identity="family_policy_001",
        final_holdout_sessions=20,
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    interval = FoldInterval(
        train_start=0,
        train_end=120,
        validation_start=120,
        validation_end=140,
        test_start=140,
        test_end=160,
        holdout_start=160,
        holdout_end=180,
    )
    fold_input = FoldInput(
        schema="regimequant.walkforward.fold_input",
        version="m10.v1",
        plan_id=plan.experiment_id,
        fold_id="fold_1",
        interval=interval,
        dataset_identity=plan.dataset_identity,
        feature_spec_set_hash=plan.feature_spec_set_hash,
        calendar_identity=plan.calendar_id,
        evidence=FoldEvidenceArtifact(
            family_provenance_digest="family:1",
            calendar_provenance_digest="calendar:1",
            model_provenance_digest="model:1",
            validation_digest="valid:1",
            evidence_digest="evidence:1",
            provenance_state="CALLER_SUPPLIED",
            family_provenance_state="CALLER_SUPPLIED",
        ),
        model_policy_identity=plan.model_policy_identity,
        formation_policy_identity=plan.formation_policy_identity,
        family_policy_identity=plan.family_policy_identity,
        input_digest=build_fold_input_digest(
            plan_id=plan.experiment_id,
            fold_id="fold_1",
            interval=interval,
            dataset_identity=plan.dataset_identity,
            feature_spec_set_hash=plan.feature_spec_set_hash,
            calendar_identity=plan.calendar_id,
            evidence=FoldEvidenceArtifact(
                family_provenance_digest="family:1",
                calendar_provenance_digest="calendar:1",
                model_provenance_digest="model:1",
                validation_digest="valid:1",
                evidence_digest="evidence:1",
                provenance_state="CALLER_SUPPLIED",
                family_provenance_state="CALLER_SUPPLIED",
            ),
            model_policy_identity=plan.model_policy_identity,
            formation_policy_identity=plan.formation_policy_identity,
            family_policy_identity=plan.family_policy_identity,
        ),
    )
    assert fold_input.plan_id == plan.experiment_id
    assert fold_input.fold_id == "fold_1"


def test_experiment_result_digest_uses_distinct_domain() -> None:
    digest = build_experiment_result_digest(
        experiment_id="exp_1",
        plan_id="plan_1",
        fold_input_identities=("fold_input_1", "fold_input_2"),
        fold_output_identities=("fold_output_1", "fold_output_2"),
        scientific_status="VALID_RESEARCH_EXECUTION",
        accounting_status="CERTIFIED",
        metric_status="METRICS_VALID",
    )
    assert isinstance(digest, str)
    assert len(digest) == 64

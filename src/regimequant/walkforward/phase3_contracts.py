from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from math import isfinite
from typing import Literal

from regimequant.walkforward.contracts import FoldInterval
from regimequant.walkforward.identity import compute_identity_digest


class Phase3Stage(str, Enum):
    TRAIN = "TRAIN"
    VALIDATION = "VALIDATION"
    TEST = "TEST"


class Phase3FailureCode(str, Enum):
    BINDING_REQUIRED = "BINDING_REQUIRED"
    MISSING_ELIGIBLE_ROW = "MISSING_ELIGIBLE_ROW"
    EXTRA_OBSERVATION_ROW = "EXTRA_OBSERVATION_ROW"
    DUPLICATE_OBSERVATION_ID = "DUPLICATE_OBSERVATION_ID"
    REORDERED_OBSERVATION_ROWS = "REORDERED_OBSERVATION_ROWS"
    MISSING_REQUIRED_FEATURE_COLUMNS = "MISSING_REQUIRED_FEATURE_COLUMNS"
    EXTRA_FEATURE_COLUMNS = "EXTRA_FEATURE_COLUMNS"
    FEATURE_COLUMN_ORDER_MISMATCH = "FEATURE_COLUMN_ORDER_MISMATCH"
    NONFINITE_FEATURE_VALUE = "NONFINITE_FEATURE_VALUE"
    NONNUMERIC_FEATURE_VALUE = "NONNUMERIC_FEATURE_VALUE"
    MISSING_FEATURE_LINEAGE = "MISSING_FEATURE_LINEAGE"
    UNVERIFIED_AVAILABILITY = "UNVERIFIED_AVAILABILITY"
    LATE_AVAILABILITY = "LATE_AVAILABILITY"
    FUTURE_OBSERVATION = "FUTURE_OBSERVATION"
    OBS_OUTSIDE_STAGE_INTERVAL = "OBS_OUTSIDE_STAGE_INTERVAL"
    STAGE_MISMATCH = "STAGE_MISMATCH"
    FOLD_MISMATCH = "FOLD_MISMATCH"
    CALENDAR_IDENTITY_MISMATCH = "CALENDAR_IDENTITY_MISMATCH"
    CALENDAR_VERSION_MISMATCH = "CALENDAR_VERSION_MISMATCH"
    SESSION_SEQUENCE_MISMATCH = "SESSION_SEQUENCE_MISMATCH"
    PROJECTION_DIGEST_MISMATCH = "PROJECTION_DIGEST_MISMATCH"
    DATASET_IDENTITY_MISMATCH = "DATASET_IDENTITY_MISMATCH"
    UNSUPPORTED_TIMESTAMP_TIMEZONE = "UNSUPPORTED_TIMESTAMP_TIMEZONE"
    DATASET_IDENTITY_DIGEST_MISMATCH = "DATASET_IDENTITY_DIGEST_MISMATCH"
    FEATURE_LINEAGE_MISMATCH = "FEATURE_LINEAGE_MISMATCH"
    PROJECTION_CONTENT_TAMPERING = "PROJECTION_CONTENT_TAMPERING"
    PROJECTION_TYPE_MISMATCH = "PROJECTION_TYPE_MISMATCH"
    PROJECTION_EVIDENCE_INSUFFICIENT = "PROJECTION_EVIDENCE_INSUFFICIENT"


class Phase3ValidationError(ValueError):
    def __init__(self, code: Phase3FailureCode, message: str, *, missing_evidence: tuple[str, ...] = ()) -> None:
        super().__init__(f"{code.value}: {message}")
        self.code = code
        self.missing_evidence = tuple(missing_evidence)


def _as_utc(value: datetime, field_name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise Phase3ValidationError(
            Phase3FailureCode.UNSUPPORTED_TIMESTAMP_TIMEZONE,
            f"{field_name} must be timezone-aware",
        )
    return value.astimezone(timezone.utc)


def _require_non_empty(value: str, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must be a non-empty string")
    return text


def _stage_bounds(stage: Phase3Stage, interval: FoldInterval) -> tuple[int, int]:
    if stage == Phase3Stage.TRAIN:
        return interval.train_start, interval.train_end
    if stage == Phase3Stage.VALIDATION:
        return interval.validation_start, interval.validation_end
    return interval.test_start, interval.test_end


@dataclass(frozen=True, slots=True)
class Phase3CausalRow:
    canonical_observation_id: str
    session_index: int
    observation_timestamp_utc: datetime
    availability_timestamp_utc: datetime
    availability_verified: bool
    feature_values_in_required_order: tuple[float, ...]
    feature_lineage_identity: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "canonical_observation_id", _require_non_empty(self.canonical_observation_id, "canonical_observation_id"))
        if not isinstance(self.session_index, int) or self.session_index < 0:
            raise ValueError("session_index must be a non-negative integer")
        object.__setattr__(self, "observation_timestamp_utc", _as_utc(self.observation_timestamp_utc, "observation_timestamp_utc"))
        object.__setattr__(self, "availability_timestamp_utc", _as_utc(self.availability_timestamp_utc, "availability_timestamp_utc"))
        if self.availability_verified is not True:
            raise Phase3ValidationError(
                Phase3FailureCode.UNVERIFIED_AVAILABILITY,
                "availability_verified must be True",
            )
        if not isinstance(self.feature_values_in_required_order, tuple):
            raise ValueError("feature_values_in_required_order must be a tuple")
        if len(self.feature_values_in_required_order) == 0:
            raise Phase3ValidationError(
                Phase3FailureCode.MISSING_REQUIRED_FEATURE_COLUMNS,
                "feature_values_in_required_order cannot be empty",
            )
        normalized: list[float] = []
        for value in self.feature_values_in_required_order:
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise Phase3ValidationError(
                    Phase3FailureCode.NONNUMERIC_FEATURE_VALUE,
                    "feature values must be numeric",
                ) from exc
            if not isfinite(numeric):
                raise Phase3ValidationError(
                    Phase3FailureCode.NONFINITE_FEATURE_VALUE,
                    "feature values must be finite",
                )
            normalized.append(numeric)
        object.__setattr__(self, "feature_values_in_required_order", tuple(normalized))
        object.__setattr__(self, "feature_lineage_identity", _require_non_empty(self.feature_lineage_identity, "feature_lineage_identity"))


@dataclass(frozen=True, slots=True)
class Phase3CausalDataset:
    source_dataset_identity: str
    session_calendar_id: str
    session_calendar_version: str
    session_sequence_digest: str
    fold_id: str
    stage: Phase3Stage
    fold_interval: FoldInterval
    stage_information_cutoff_utc: datetime
    phase2_projection_digest: str
    required_feature_ids: tuple[str, ...]
    eligible_observation_ids: tuple[str, ...]
    ordered_rows: tuple[Phase3CausalRow, ...]
    dataset_identity_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_dataset_identity", _require_non_empty(self.source_dataset_identity, "source_dataset_identity"))
        object.__setattr__(self, "session_calendar_id", _require_non_empty(self.session_calendar_id, "session_calendar_id"))
        object.__setattr__(self, "session_calendar_version", _require_non_empty(self.session_calendar_version, "session_calendar_version"))
        object.__setattr__(self, "session_sequence_digest", _require_non_empty(self.session_sequence_digest, "session_sequence_digest"))
        object.__setattr__(self, "fold_id", _require_non_empty(self.fold_id, "fold_id"))
        object.__setattr__(self, "phase2_projection_digest", _require_non_empty(self.phase2_projection_digest, "phase2_projection_digest"))
        object.__setattr__(self, "dataset_identity_digest", _require_non_empty(self.dataset_identity_digest, "dataset_identity_digest"))
        object.__setattr__(self, "stage_information_cutoff_utc", _as_utc(self.stage_information_cutoff_utc, "stage_information_cutoff_utc"))

        if not isinstance(self.stage, Phase3Stage):
            raise ValueError("stage must be a Phase3Stage")
        if not isinstance(self.fold_interval, FoldInterval):
            raise ValueError("fold_interval must be a FoldInterval")
        if not isinstance(self.required_feature_ids, tuple) or len(self.required_feature_ids) == 0:
            raise Phase3ValidationError(
                Phase3FailureCode.MISSING_REQUIRED_FEATURE_COLUMNS,
                "required_feature_ids must be a non-empty tuple",
            )
        if len(set(self.required_feature_ids)) != len(self.required_feature_ids):
            raise Phase3ValidationError(
                Phase3FailureCode.EXTRA_FEATURE_COLUMNS,
                "required_feature_ids must be unique",
            )
        if not isinstance(self.eligible_observation_ids, tuple):
            raise ValueError("eligible_observation_ids must be a tuple")
        if not isinstance(self.ordered_rows, tuple):
            raise ValueError("ordered_rows must be a tuple")

        row_ids = tuple(row.canonical_observation_id for row in self.ordered_rows)
        if len(set(row_ids)) != len(row_ids):
            raise Phase3ValidationError(
                Phase3FailureCode.DUPLICATE_OBSERVATION_ID,
                "ordered_rows contains duplicate canonical_observation_id",
            )
        if row_ids != self.eligible_observation_ids:
            raise Phase3ValidationError(
                Phase3FailureCode.REORDERED_OBSERVATION_ROWS,
                "ordered_rows canonical_observation_id order must exactly match eligible_observation_ids",
            )

        lower, upper = _stage_bounds(self.stage, self.fold_interval)
        for row in self.ordered_rows:
            if row.session_index < lower or row.session_index >= upper:
                raise Phase3ValidationError(
                    Phase3FailureCode.OBS_OUTSIDE_STAGE_INTERVAL,
                    f"row {row.canonical_observation_id} is outside stage interval",
                )
            if row.observation_timestamp_utc > self.stage_information_cutoff_utc:
                raise Phase3ValidationError(
                    Phase3FailureCode.FUTURE_OBSERVATION,
                    f"row {row.canonical_observation_id} observation timestamp exceeds stage cutoff",
                )
            if row.availability_timestamp_utc > self.stage_information_cutoff_utc:
                raise Phase3ValidationError(
                    Phase3FailureCode.LATE_AVAILABILITY,
                    f"row {row.canonical_observation_id} availability timestamp exceeds stage cutoff",
                )
            if len(row.feature_values_in_required_order) != len(self.required_feature_ids):
                raise Phase3ValidationError(
                    Phase3FailureCode.FEATURE_COLUMN_ORDER_MISMATCH,
                    f"row {row.canonical_observation_id} feature length does not match required_feature_ids",
                )

        expected = compute_phase3_causal_dataset_identity_payload(self)
        expected_digest = compute_identity_digest(expected, domain="regimequant.walkforward.phase3a.causal_dataset")
        if self.dataset_identity_digest != expected_digest:
            raise Phase3ValidationError(
                Phase3FailureCode.DATASET_IDENTITY_DIGEST_MISMATCH,
                "dataset_identity_digest does not match material dataset payload",
            )


@dataclass(frozen=True, slots=True)
class Phase3M5PrimaryConfig:
    n_components: int
    covariance_type: Literal["diag"] = "diag"
    random_state: int = 42
    n_init: int = 10
    max_iter: int = 500
    tol: float = 1e-4
    reg_covar: float = 1e-6

    def __post_init__(self) -> None:
        if self.n_components < 2:
            raise ValueError("n_components must be at least 2")
        if self.n_init <= 0:
            raise ValueError("n_init must be positive")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if self.tol <= 0:
            raise ValueError("tol must be positive")
        if self.reg_covar <= 0:
            raise ValueError("reg_covar must be positive")


@dataclass(frozen=True, slots=True)
class Phase3M3ComparatorConfig:
    n_clusters: int

    def __post_init__(self) -> None:
        if self.n_clusters <= 0:
            raise ValueError("n_clusters must be positive")


@dataclass(frozen=True, slots=True)
class Phase3FrozenModelConfig:
    m5_primary: Phase3M5PrimaryConfig
    m3_comparator: Phase3M3ComparatorConfig | None = None
    m5_is_primary: Literal[True] = True
    config_identity_digest: str = ""

    def __post_init__(self) -> None:
        if self.m5_is_primary is not True:
            raise ValueError("m5_is_primary must remain True")
        object.__setattr__(self, "config_identity_digest", _require_non_empty(self.config_identity_digest, "config_identity_digest"))
        expected = compute_phase3_model_config_identity_payload(self)
        expected_digest = compute_identity_digest(expected, domain="regimequant.walkforward.phase3a.model_config")
        if self.config_identity_digest != expected_digest:
            raise Phase3ValidationError(
                Phase3FailureCode.PROJECTION_CONTENT_TAMPERING,
                "config_identity_digest does not match material model configuration payload",
            )


@dataclass(frozen=True, slots=True)
class Phase3FoldModelArtifacts:
    m5_primary_artifact_identity: str
    m3_comparator_artifact_identity: str | None
    training_dataset_identity_digest: str
    model_config_identity_digest: str
    replay_fit_completed_at_utc: datetime
    authority_role: Literal["STRUCTURAL_ONLY"] = "STRUCTURAL_ONLY"
    authoritative: Literal[False] = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "m5_primary_artifact_identity", _require_non_empty(self.m5_primary_artifact_identity, "m5_primary_artifact_identity"))
        if self.m3_comparator_artifact_identity is not None:
            object.__setattr__(self, "m3_comparator_artifact_identity", _require_non_empty(self.m3_comparator_artifact_identity, "m3_comparator_artifact_identity"))
        object.__setattr__(self, "training_dataset_identity_digest", _require_non_empty(self.training_dataset_identity_digest, "training_dataset_identity_digest"))
        object.__setattr__(self, "model_config_identity_digest", _require_non_empty(self.model_config_identity_digest, "model_config_identity_digest"))
        object.__setattr__(self, "replay_fit_completed_at_utc", _as_utc(self.replay_fit_completed_at_utc, "replay_fit_completed_at_utc"))
        if self.authoritative is not False:
            raise ValueError("Phase3FoldModelArtifacts must remain non-authoritative")


@dataclass(frozen=True, slots=True)
class Phase3ValidationAssessment:
    m5_primary_artifact_identity: str
    validation_dataset_identity_digest: str
    assessment_completed_at_utc: datetime
    assessment_identity: str
    authority_role: Literal["STRUCTURAL_ONLY"] = "STRUCTURAL_ONLY"
    authoritative: Literal[False] = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "m5_primary_artifact_identity", _require_non_empty(self.m5_primary_artifact_identity, "m5_primary_artifact_identity"))
        object.__setattr__(self, "validation_dataset_identity_digest", _require_non_empty(self.validation_dataset_identity_digest, "validation_dataset_identity_digest"))
        object.__setattr__(self, "assessment_identity", _require_non_empty(self.assessment_identity, "assessment_identity"))
        object.__setattr__(self, "assessment_completed_at_utc", _as_utc(self.assessment_completed_at_utc, "assessment_completed_at_utc"))
        if self.authoritative is not False:
            raise ValueError("Phase3ValidationAssessment must remain non-authoritative")


@dataclass(frozen=True, slots=True)
class Phase3TestInference:
    m5_primary_artifact_identity: str
    test_dataset_identity_digest: str
    completed_validation_assessment_identity: str
    first_test_decision_timestamp_utc: datetime
    inference_identity: str
    authority_role: Literal["STRUCTURAL_ONLY"] = "STRUCTURAL_ONLY"
    authoritative: Literal[False] = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "m5_primary_artifact_identity", _require_non_empty(self.m5_primary_artifact_identity, "m5_primary_artifact_identity"))
        object.__setattr__(self, "test_dataset_identity_digest", _require_non_empty(self.test_dataset_identity_digest, "test_dataset_identity_digest"))
        object.__setattr__(self, "completed_validation_assessment_identity", _require_non_empty(self.completed_validation_assessment_identity, "completed_validation_assessment_identity"))
        object.__setattr__(self, "inference_identity", _require_non_empty(self.inference_identity, "inference_identity"))
        object.__setattr__(self, "first_test_decision_timestamp_utc", _as_utc(self.first_test_decision_timestamp_utc, "first_test_decision_timestamp_utc"))
        if self.authoritative is not False:
            raise ValueError("Phase3TestInference must remain non-authoritative")


@dataclass(frozen=True, slots=True)
class Phase3BindingValidation:
    projection_digest: str
    dataset_identity_digest: str
    binding_identity_digest: str
    verified: bool
    failure_code: Phase3FailureCode | None = None
    failure_detail: str | None = None
    missing_evidence: tuple[str, ...] = ()
    authority_role: Literal["STRUCTURAL_ONLY"] = "STRUCTURAL_ONLY"
    authoritative: Literal[False] = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "projection_digest", _require_non_empty(self.projection_digest, "projection_digest"))
        object.__setattr__(self, "dataset_identity_digest", _require_non_empty(self.dataset_identity_digest, "dataset_identity_digest"))
        object.__setattr__(self, "binding_identity_digest", _require_non_empty(self.binding_identity_digest, "binding_identity_digest"))
        if self.verified:
            if self.failure_code is not None or self.failure_detail is not None or self.missing_evidence:
                raise ValueError("verified binding cannot include failure fields")
        else:
            if self.failure_code is None:
                raise ValueError("unverified binding must include failure_code")
        if self.authoritative is not False:
            raise ValueError("Phase3BindingValidation must remain non-authoritative")


def compute_phase3_row_lineage_identity_payload(
    *,
    canonical_observation_id: str,
    required_feature_ids: tuple[str, ...],
    feature_values_in_required_order: tuple[float, ...],
) -> dict[str, object]:
    return {
        "canonical_observation_id": canonical_observation_id,
        "required_feature_ids": list(required_feature_ids),
        "feature_values_in_required_order": [float(value) for value in feature_values_in_required_order],
    }


def compute_phase3_causal_dataset_identity_payload(dataset: Phase3CausalDataset) -> dict[str, object]:
    return compute_phase3_causal_dataset_identity_payload_from_material(
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
        ordered_rows=dataset.ordered_rows,
    )


def compute_phase3_causal_dataset_identity_payload_from_material(
    *,
    source_dataset_identity: str,
    session_calendar_id: str,
    session_calendar_version: str,
    session_sequence_digest: str,
    fold_id: str,
    stage: Phase3Stage,
    fold_interval: FoldInterval,
    stage_information_cutoff_utc: datetime,
    phase2_projection_digest: str,
    required_feature_ids: tuple[str, ...],
    eligible_observation_ids: tuple[str, ...],
    ordered_rows: tuple[Phase3CausalRow, ...],
) -> dict[str, object]:
    return {
        "source_dataset_identity": source_dataset_identity,
        "session_calendar_id": session_calendar_id,
        "session_calendar_version": session_calendar_version,
        "session_sequence_digest": session_sequence_digest,
        "fold_id": fold_id,
        "stage": stage.value,
        "fold_interval": {
            "train_start": fold_interval.train_start,
            "train_end": fold_interval.train_end,
            "validation_start": fold_interval.validation_start,
            "validation_end": fold_interval.validation_end,
            "test_start": fold_interval.test_start,
            "test_end": fold_interval.test_end,
            "holdout_start": fold_interval.holdout_start,
            "holdout_end": fold_interval.holdout_end,
        },
        "stage_information_cutoff_utc": stage_information_cutoff_utc,
        "phase2_projection_digest": phase2_projection_digest,
        "required_feature_ids": list(required_feature_ids),
        "eligible_observation_ids": list(eligible_observation_ids),
        "ordered_rows": [
            {
                "canonical_observation_id": row.canonical_observation_id,
                "session_index": row.session_index,
                "observation_timestamp_utc": row.observation_timestamp_utc,
                "availability_timestamp_utc": row.availability_timestamp_utc,
                "availability_verified": row.availability_verified,
                "feature_values_in_required_order": list(row.feature_values_in_required_order),
                "feature_lineage_identity": row.feature_lineage_identity,
            }
            for row in ordered_rows
        ],
    }


def compute_phase3_model_config_identity_payload(config: Phase3FrozenModelConfig) -> dict[str, object]:
    return compute_phase3_model_config_identity_payload_from_material(
        m5_primary=config.m5_primary,
        m3_comparator=config.m3_comparator,
    )


def compute_phase3_model_config_identity_payload_from_material(
    *,
    m5_primary: Phase3M5PrimaryConfig,
    m3_comparator: Phase3M3ComparatorConfig | None,
) -> dict[str, object]:
    return {
        "m5_is_primary": True,
        "m5_primary": {
            "n_components": m5_primary.n_components,
            "covariance_type": m5_primary.covariance_type,
            "random_state": m5_primary.random_state,
            "n_init": m5_primary.n_init,
            "max_iter": m5_primary.max_iter,
            "tol": m5_primary.tol,
            "reg_covar": m5_primary.reg_covar,
        },
        "m3_comparator": None
        if m3_comparator is None
        else {
            "n_clusters": m3_comparator.n_clusters,
        },
    }


def compute_phase3_binding_identity_payload(
    *,
    dataset_identity_digest: str,
    projection_digest: str,
    fold_id: str,
    stage: Phase3Stage,
    required_feature_ids: tuple[str, ...],
    eligible_observation_ids: tuple[str, ...],
) -> dict[str, object]:
    return {
        "dataset_identity_digest": dataset_identity_digest,
        "projection_digest": projection_digest,
        "fold_id": fold_id,
        "stage": stage.value,
        "required_feature_ids": list(required_feature_ids),
        "eligible_observation_ids": list(eligible_observation_ids),
    }


def build_phase3_causal_dataset(
    *,
    source_dataset_identity: str,
    session_calendar_id: str,
    session_calendar_version: str,
    session_sequence_digest: str,
    fold_id: str,
    stage: Phase3Stage,
    fold_interval: FoldInterval,
    stage_information_cutoff_utc: datetime,
    phase2_projection_digest: str,
    required_feature_ids: tuple[str, ...],
    eligible_observation_ids: tuple[str, ...],
    ordered_rows: tuple[Phase3CausalRow, ...],
) -> Phase3CausalDataset:
    payload = compute_phase3_causal_dataset_identity_payload_from_material(
        source_dataset_identity=source_dataset_identity,
        session_calendar_id=session_calendar_id,
        session_calendar_version=session_calendar_version,
        session_sequence_digest=session_sequence_digest,
        fold_id=fold_id,
        stage=stage,
        fold_interval=fold_interval,
        stage_information_cutoff_utc=stage_information_cutoff_utc,
        phase2_projection_digest=phase2_projection_digest,
        required_feature_ids=required_feature_ids,
        eligible_observation_ids=eligible_observation_ids,
        ordered_rows=ordered_rows,
    )
    digest = compute_identity_digest(payload, domain="regimequant.walkforward.phase3a.causal_dataset")
    return Phase3CausalDataset(
        source_dataset_identity=source_dataset_identity,
        session_calendar_id=session_calendar_id,
        session_calendar_version=session_calendar_version,
        session_sequence_digest=session_sequence_digest,
        fold_id=fold_id,
        stage=stage,
        fold_interval=fold_interval,
        stage_information_cutoff_utc=stage_information_cutoff_utc,
        phase2_projection_digest=phase2_projection_digest,
        required_feature_ids=required_feature_ids,
        eligible_observation_ids=eligible_observation_ids,
        ordered_rows=ordered_rows,
        dataset_identity_digest=digest,
    )


def build_phase3_frozen_model_config(
    *,
    m5_primary: Phase3M5PrimaryConfig,
    m3_comparator: Phase3M3ComparatorConfig | None,
) -> Phase3FrozenModelConfig:
    payload = compute_phase3_model_config_identity_payload_from_material(
        m5_primary=m5_primary,
        m3_comparator=m3_comparator,
    )
    digest = compute_identity_digest(payload, domain="regimequant.walkforward.phase3a.model_config")
    return Phase3FrozenModelConfig(
        m5_primary=m5_primary,
        m3_comparator=m3_comparator,
        m5_is_primary=True,
        config_identity_digest=digest,
    )

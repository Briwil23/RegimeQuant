from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from math import isfinite
from typing import Mapping

from regimequant.walkforward.contracts import FoldInterval
from regimequant.walkforward.identity import compute_identity_digest


class ProjectionStage(str, Enum):
    TRAIN = "TRAIN"
    VALIDATION = "VALIDATION"
    TEST = "TEST"


class ExclusionReason(str, Enum):
    OBS_AFTER_CUTOFF = "OBS_AFTER_CUTOFF"
    AVAILABILITY_MISSING = "AVAILABILITY_MISSING"
    AVAILABILITY_LATE = "AVAILABILITY_LATE"
    IDENTITY_UNVERIFIED = "IDENTITY_UNVERIFIED"
    DUPLICATE_KEY = "DUPLICATE_KEY"
    FEATURE_MISSING = "FEATURE_MISSING"
    FEATURE_NONFINITE = "FEATURE_NONFINITE"
    STAGE_INTERVAL_MISMATCH = "STAGE_INTERVAL_MISMATCH"


@dataclass(frozen=True, slots=True)
class ExcludedObservation:
    observation_id: str
    reason_code: ExclusionReason
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class CandidateObservation:
    canonical_observation_id: str
    session_index: int
    observation_timestamp_utc: datetime
    availability_timestamp_utc: datetime | None
    identity_verified: bool
    features: Mapping[str, float | int | None]


@dataclass(frozen=True, slots=True)
class CausalEligibilityProjection:
    schema: str
    version: str
    source_dataset_identity: str
    session_calendar_id: str
    session_calendar_version: str
    session_sequence_digest: str
    stage: ProjectionStage
    fold_id: str
    fold_interval: FoldInterval
    stage_information_cutoff_utc: datetime
    boundary_session_index: int
    required_feature_ids: tuple[str, ...]
    canonical_observation_ids: tuple[str, ...]
    observation_timestamps_utc: tuple[datetime, ...]
    availability_timestamps_utc: tuple[datetime | None, ...]
    required_feature_values_by_observation: tuple[tuple[float | None | str, ...], ...]
    eligible_observation_ids: tuple[str, ...]
    excluded_observations: tuple[ExcludedObservation, ...]
    m3_policy_identity: str
    m5_policy_identity: str
    projection_digest: str

    @property
    def eligible_count(self) -> int:
        return len(self.eligible_observation_ids)

    @property
    def excluded_count(self) -> int:
        return len(self.excluded_observations)


def _stage_bounds(stage: ProjectionStage, interval: FoldInterval) -> tuple[int, int]:
    if stage == ProjectionStage.TRAIN:
        return interval.train_start, interval.train_end
    if stage == ProjectionStage.VALIDATION:
        return interval.validation_start, interval.validation_end
    return interval.test_start, interval.test_end


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(timezone.utc)


def _validate_base_identities(*, dataset_identity: str, calendar_id: str, calendar_version: str, calendar_sequence_digest: str, m3_policy_identity: str, m5_policy_identity: str) -> None:
    for name, value in {
        "source_dataset_identity": dataset_identity,
        "session_calendar_id": calendar_id,
        "session_calendar_version": calendar_version,
        "session_sequence_digest": calendar_sequence_digest,
        "m3_policy_identity": m3_policy_identity,
        "m5_policy_identity": m5_policy_identity,
    }.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")


def build_causal_eligibility_projection(
    *,
    dataset_identity: str,
    calendar_id: str,
    calendar_version: str,
    calendar_sequence_digest: str,
    stage: ProjectionStage,
    fold_id: str,
    fold_interval: FoldInterval,
    stage_information_cutoff_utc: datetime,
    boundary_session_index: int,
    observations: tuple[CandidateObservation, ...],
    required_feature_ids: tuple[str, ...],
    m3_policy_identity: str,
    m5_policy_identity: str,
) -> CausalEligibilityProjection:
    _validate_base_identities(
        dataset_identity=dataset_identity,
        calendar_id=calendar_id,
        calendar_version=calendar_version,
        calendar_sequence_digest=calendar_sequence_digest,
        m3_policy_identity=m3_policy_identity,
        m5_policy_identity=m5_policy_identity,
    )
    if not isinstance(fold_id, str) or not fold_id.strip():
        raise ValueError("fold_id must be a non-empty string")

    cutoff_utc = _as_utc(stage_information_cutoff_utc)
    lower, upper = _stage_bounds(stage, fold_interval)
    expected_boundary = upper
    if boundary_session_index != expected_boundary:
        raise ValueError(
            f"boundary_session_index must equal stage boundary {expected_boundary} for stage {stage.value}"
        )

    ordered = sorted(
        observations,
        key=lambda item: (
            item.canonical_observation_id,
            item.session_index,
            item.observation_timestamp_utc,
        ),
    )

    seen_ids: set[str] = set()
    canonical_ids: list[str] = []
    observation_times: list[datetime] = []
    availability_times: list[datetime | None] = []
    feature_lineage: list[tuple[float | None | str, ...]] = []
    eligible_ids: list[str] = []
    excluded: list[ExcludedObservation] = []
    feature_ids_ordered = tuple(str(item) for item in required_feature_ids)

    for obs in ordered:
        if obs.canonical_observation_id in seen_ids:
            raise ValueError("duplicate canonical observation id detected")
        seen_ids.add(obs.canonical_observation_id)

        obs_time = _as_utc(obs.observation_timestamp_utc)
        avail_time = None if obs.availability_timestamp_utc is None else _as_utc(obs.availability_timestamp_utc)

        canonical_ids.append(obs.canonical_observation_id)
        observation_times.append(obs_time)
        availability_times.append(avail_time)

        feature_snapshot: list[float | None | str] = []
        for feature_id in feature_ids_ordered:
            raw_value = obs.features.get(feature_id)
            if raw_value is None:
                feature_snapshot.append(None)
                continue
            try:
                numeric = float(raw_value)
            except (TypeError, ValueError):
                feature_snapshot.append("NONNUMERIC")
                continue
            if not isfinite(numeric):
                feature_snapshot.append("NONFINITE")
            else:
                feature_snapshot.append(numeric)
        feature_lineage.append(tuple(feature_snapshot))

        exclusion: ExcludedObservation | None = None
        if not obs.identity_verified or not obs.canonical_observation_id.strip():
            exclusion = ExcludedObservation(obs.canonical_observation_id, ExclusionReason.IDENTITY_UNVERIFIED, "identity not verified")
        elif obs.session_index < lower or obs.session_index >= upper:
            exclusion = ExcludedObservation(obs.canonical_observation_id, ExclusionReason.STAGE_INTERVAL_MISMATCH, "outside stage interval")
        elif obs_time > cutoff_utc:
            exclusion = ExcludedObservation(obs.canonical_observation_id, ExclusionReason.OBS_AFTER_CUTOFF, "observation timestamp exceeds cutoff")
        elif avail_time is None:
            exclusion = ExcludedObservation(obs.canonical_observation_id, ExclusionReason.AVAILABILITY_MISSING, "availability timestamp missing")
        elif avail_time > cutoff_utc:
            exclusion = ExcludedObservation(obs.canonical_observation_id, ExclusionReason.AVAILABILITY_LATE, "availability timestamp exceeds cutoff")
        else:
            for feature_id in feature_ids_ordered:
                if feature_id not in obs.features or obs.features[feature_id] is None:
                    exclusion = ExcludedObservation(obs.canonical_observation_id, ExclusionReason.FEATURE_MISSING, f"missing feature: {feature_id}")
                    break
                try:
                    value = float(obs.features[feature_id])
                except (TypeError, ValueError):
                    exclusion = ExcludedObservation(
                        obs.canonical_observation_id,
                        ExclusionReason.FEATURE_NONFINITE,
                        f"non-numeric feature: {feature_id}",
                    )
                    break
                if not isfinite(value):
                    exclusion = ExcludedObservation(obs.canonical_observation_id, ExclusionReason.FEATURE_NONFINITE, f"non-finite feature: {feature_id}")
                    break

        if exclusion is None:
            eligible_ids.append(obs.canonical_observation_id)
        else:
            excluded.append(exclusion)

    payload = {
        "schema": "regimequant.walkforward.phase2.causal_projection",
        "version": "m10.phase2.v1",
        "source_dataset_identity": dataset_identity,
        "session_calendar_id": calendar_id,
        "session_calendar_version": calendar_version,
        "session_sequence_digest": calendar_sequence_digest,
        "stage": stage.value,
        "fold_id": fold_id,
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
        "stage_information_cutoff_utc": cutoff_utc.isoformat(),
        "boundary_session_index": boundary_session_index,
        "required_feature_ids": list(feature_ids_ordered),
        "canonical_observation_ids": canonical_ids,
        "observation_timestamps_utc": [dt.isoformat() for dt in observation_times],
        "availability_timestamps_utc": [None if dt is None else dt.isoformat() for dt in availability_times],
        "required_feature_values_by_observation": [list(item) for item in feature_lineage],
        "eligible_observation_ids": eligible_ids,
        "excluded_observations": [
            {"observation_id": item.observation_id, "reason_code": item.reason_code.value, "detail": item.detail}
            for item in excluded
        ],
        "m3_policy_identity": m3_policy_identity,
        "m5_policy_identity": m5_policy_identity,
    }
    projection_digest = compute_identity_digest(payload, domain="regimequant.walkforward.phase2.causal_projection")

    return CausalEligibilityProjection(
        schema="regimequant.walkforward.phase2.causal_projection",
        version="m10.phase2.v1",
        source_dataset_identity=dataset_identity,
        session_calendar_id=calendar_id,
        session_calendar_version=calendar_version,
        session_sequence_digest=calendar_sequence_digest,
        stage=stage,
        fold_id=fold_id,
        fold_interval=fold_interval,
        stage_information_cutoff_utc=cutoff_utc,
        boundary_session_index=boundary_session_index,
        required_feature_ids=feature_ids_ordered,
        canonical_observation_ids=tuple(canonical_ids),
        observation_timestamps_utc=tuple(observation_times),
        availability_timestamps_utc=tuple(availability_times),
        required_feature_values_by_observation=tuple(feature_lineage),
        eligible_observation_ids=tuple(eligible_ids),
        excluded_observations=tuple(excluded),
        m3_policy_identity=m3_policy_identity,
        m5_policy_identity=m5_policy_identity,
        projection_digest=projection_digest,
    )


def required_m3_training_minimum(*, n_clusters: int) -> int:
    if n_clusters <= 0:
        raise ValueError("n_clusters must be positive")
    return max(30, 10 * n_clusters)


def required_m3_heldout_minimum(*, n_clusters: int) -> int:
    if n_clusters <= 0:
        raise ValueError("n_clusters must be positive")
    return max(20, 2 * n_clusters)


def required_m5_training_minimum(*, n_components: int, effective_feature_dimension: int) -> int:
    if n_components <= 0:
        raise ValueError("n_components must be positive")
    if effective_feature_dimension <= 0:
        raise ValueError("effective_feature_dimension must be positive")
    return max(2 * n_components, 10 * effective_feature_dimension)

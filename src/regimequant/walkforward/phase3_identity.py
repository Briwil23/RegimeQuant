from __future__ import annotations

from regimequant.walkforward.causal_projection import CausalEligibilityProjection
from regimequant.walkforward.identity import compute_identity_digest
from regimequant.walkforward.phase3_contracts import (
    Phase3CausalDataset,
    Phase3FrozenModelConfig,
    Phase3Stage,
    compute_phase3_binding_identity_payload,
    compute_phase3_causal_dataset_identity_payload,
    compute_phase3_model_config_identity_payload,
)


PHASE2_PROJECTION_DOMAIN = "regimequant.walkforward.phase2.causal_projection"
PHASE3_DATASET_DOMAIN = "regimequant.walkforward.phase3a.causal_dataset"
PHASE3_MODEL_CONFIG_DOMAIN = "regimequant.walkforward.phase3a.model_config"
PHASE3_BINDING_DOMAIN = "regimequant.walkforward.phase3a.binding"


def compute_phase2_projection_identity_digest(projection: CausalEligibilityProjection) -> str:
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
    return compute_identity_digest(payload, domain=PHASE2_PROJECTION_DOMAIN)


def compute_phase3_dataset_identity_digest(dataset: Phase3CausalDataset) -> str:
    payload = compute_phase3_causal_dataset_identity_payload(dataset)
    return compute_identity_digest(payload, domain=PHASE3_DATASET_DOMAIN)


def compute_phase3_model_config_identity_digest(config: Phase3FrozenModelConfig) -> str:
    payload = compute_phase3_model_config_identity_payload(config)
    return compute_identity_digest(payload, domain=PHASE3_MODEL_CONFIG_DOMAIN)


def compute_phase3_binding_identity_digest(
    *,
    dataset_identity_digest: str,
    projection_digest: str,
    fold_id: str,
    stage: Phase3Stage,
    required_feature_ids: tuple[str, ...],
    eligible_observation_ids: tuple[str, ...],
) -> str:
    payload = compute_phase3_binding_identity_payload(
        dataset_identity_digest=dataset_identity_digest,
        projection_digest=projection_digest,
        fold_id=fold_id,
        stage=stage,
        required_feature_ids=required_feature_ids,
        eligible_observation_ids=eligible_observation_ids,
    )
    return compute_identity_digest(payload, domain=PHASE3_BINDING_DOMAIN)

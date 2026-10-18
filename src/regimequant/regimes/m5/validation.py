from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .contracts import M5AvailabilityMode, M5ModelArtifact, M5ValidationError, _as_float_list, _coerce_utc_timestamp


def validate_m5_feature_provenance(frame: pd.DataFrame) -> dict[str, Any]:
    provenance = frame.attrs.get("provenance")
    if not isinstance(provenance, dict):
        raise M5ValidationError("M5 feature frame is missing attrs.provenance")
    required = {"feature_spec_set_hash", "generated_at_utc", "metadata_durability"}
    missing = sorted(required - set(provenance.keys()))
    if missing:
        raise M5ValidationError(f"missing required provenance fields: {missing!r}")
    if not isinstance(provenance.get("feature_spec_set_hash"), str) or not provenance["feature_spec_set_hash"]:
        raise M5ValidationError("attrs.provenance.feature_spec_set_hash must be a non-empty string")
    if provenance.get("generated_at_utc") is None:
        raise M5ValidationError("attrs.provenance.generated_at_utc is required")
    _coerce_utc_timestamp(provenance["generated_at_utc"], allow_null=False)
    if not isinstance(provenance.get("metadata_durability"), str) or not provenance["metadata_durability"]:
        raise M5ValidationError("attrs.provenance.metadata_durability must be a non-empty string")
    return provenance


def validate_m5_probability_vector(values: Any) -> tuple[float, ...]:
    numeric = _as_float_list(values, name="probability_vector")
    if not numeric:
        raise M5ValidationError("probability_vector is empty")
    if any(not math.isfinite(v) for v in numeric):
        raise M5ValidationError("probability_vector contains non-finite values")
    if any(v < 0.0 for v in numeric):
        raise M5ValidationError("probability_vector contains negative probabilities")
    total = float(sum(numeric))
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise M5ValidationError("probability_vector must sum to 1.0 within tolerance")
    return tuple(numeric)


def validate_m5_artifact(artifact: Any) -> M5ModelArtifact:
    if not isinstance(artifact, M5ModelArtifact):
        raise M5ValidationError("artifact must be an M5ModelArtifact instance")
    if not artifact.training_input_digest:
        raise M5ValidationError("training_input_digest is required")
    if not artifact.effective_feature_ids:
        raise M5ValidationError("effective_feature_ids must not be empty")
    if not artifact.means or not artifact.weights:
        raise M5ValidationError("model parameters are missing")
    if artifact.training_cutoff is None:
        raise M5ValidationError("training_cutoff is required")
    if artifact.training_information_cutoff is None:
        raise M5ValidationError("training_information_cutoff is required")
    if artifact.requested_component_count < 2:
        raise M5ValidationError("requested_component_count must be at least 2")
    if artifact.effective_component_count != len(artifact.weights):
        raise M5ValidationError("effective_component_count must match the weight count")
    if len(artifact.means) != len(artifact.weights):
        raise M5ValidationError("mean count must match weight count")
    if len(artifact.diagonal_covariances) != len(artifact.weights):
        raise M5ValidationError("covariance count must match weight count")
    for index, mean in enumerate(artifact.means):
        if len(mean) != len(artifact.scaler_mean):
            raise M5ValidationError(f"mean dimension mismatch at component {index}")
        if not all(math.isfinite(v) for v in mean):
            raise M5ValidationError(f"component {index} contains a non-finite mean")
    for index, cov in enumerate(artifact.diagonal_covariances):
        if len(cov) != len(artifact.scaler_mean):
            raise M5ValidationError(f"covariance dimension mismatch at component {index}")
        if any(not math.isfinite(v) or v <= 0.0 for v in cov):
            raise M5ValidationError(f"component {index} has invalid covariance")
    if any(not math.isfinite(v) or v <= 0.0 for v in artifact.weights):
        raise M5ValidationError("mixture weights must be finite and positive")
    if any(v < 0.01 for v in artifact.weights):
        raise M5ValidationError("component weights below the scientific floor are not allowed")
    if not artifact.converged:
        raise M5ValidationError("artifact converged flag must be true")
    if any(v < 2 for v in artifact.hard_occupancy.values()):
        raise M5ValidationError("hard occupancy below the required minimum for every component")
    if any(v < max(2, len(artifact.scaler_mean)) for v in artifact.effective_occupancy.values()):
        raise M5ValidationError("effective occupancy below the required minimum for every component")
    # duplicate-component rejection with absolute tolerance
    for i in range(len(artifact.weights)):
        for j in range(i + 1, len(artifact.weights)):
            weight_delta = abs(artifact.weights[i] - artifact.weights[j])
            mean_delta = max(abs(a - b) for a, b in zip(artifact.means[i], artifact.means[j]))
            cov_delta = max(abs(a - b) for a, b in zip(artifact.diagonal_covariances[i], artifact.diagonal_covariances[j]))
            if max(weight_delta, mean_delta, cov_delta) <= 1e-10:
                raise M5ValidationError("duplicate components are not allowed in the fitted artifact")
    if artifact.availability_mode == M5AvailabilityMode.AS_OF_PRODUCTION:
        if artifact.actual_creation_timestamp_utc.tzinfo is None or artifact.actual_creation_timestamp_utc.tz is None:
            raise M5ValidationError("actual_creation_timestamp_utc must be timezone-aware")
        if artifact.effective_model_availability is None:
            raise M5ValidationError("effective_model_availability is required for production assignment")
        if artifact.effective_model_availability < artifact.actual_creation_timestamp_utc:
            raise M5ValidationError("effective_model_availability cannot precede actual_creation_timestamp_utc")
        if artifact.training_information_cutoff is not None and artifact.effective_model_availability < artifact.training_information_cutoff:
            raise M5ValidationError("effective_model_availability cannot precede training_information_cutoff")
    if artifact.availability_mode == M5AvailabilityMode.HISTORICAL_REPLAY:
        if artifact.simulated_replay_availability is None:
            raise M5ValidationError("simulated_replay_availability is required for historical replay")
        if artifact.training_information_cutoff is not None and artifact.simulated_replay_availability < artifact.training_information_cutoff:
            raise M5ValidationError("simulated_replay_availability cannot precede training_information_cutoff")
    expected_model_id = artifact.model_id
    if not expected_model_id:
        raise M5ValidationError("artifact model_id is required")
    return artifact


def validate_m5_assignment_matrix(artifact: M5ModelArtifact, matrix: pd.DataFrame) -> None:
    if matrix is None or matrix.empty:
        raise M5ValidationError("feature matrix is empty")
    if "scope_id" not in matrix.columns:
        raise M5ValidationError("feature_matrix must include a scope_id column")
    if "timestamp" not in matrix.columns:
        raise M5ValidationError("feature_matrix must include a timestamp column")
    if "feature_vector_available_at" not in matrix.columns:
        raise M5ValidationError("feature_matrix must include feature_vector_available_at")
    if "feature_observation_available" not in matrix.columns:
        raise M5ValidationError("feature_matrix must include feature_observation_available")
    if matrix.duplicated(subset=["scope_id", "timestamp"]).any():
        raise M5ValidationError("duplicate assignment keys are not allowed")
    if artifact.feature_spec_hash is not None:
        provenance = matrix.attrs.get("provenance", {})
        if not isinstance(provenance, dict):
            raise M5ValidationError("feature_matrix provenance is required")
        matrix_hash = provenance.get("feature_spec_set_hash")
        if matrix_hash is not None and matrix_hash != artifact.feature_spec_hash:
            raise M5ValidationError("artifact feature_spec_hash does not match the feature matrix provenance")
    for feature_id in artifact.effective_feature_ids:
        if feature_id not in matrix.columns:
            raise M5ValidationError(f"feature_matrix is missing the artifact effective feature id: {feature_id}")

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .contracts import M5AssignmentStatus, M5AvailabilityMode, M5ModelArtifact, M5ModelUnavailableError, M5ValidationError
from .validation import validate_m5_artifact, validate_m5_assignment_matrix, validate_m5_probability_vector


def _normalize_utc(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None or ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _component_log_probability(x: np.ndarray, weights: np.ndarray, means: np.ndarray, covariances: np.ndarray) -> np.ndarray:
    log_prob: list[float] = []
    for idx in range(len(weights)):
        mean = means[idx]
        variance = covariances[idx]
        mahalanobis = ((x - mean) ** 2) / variance
        log_term = -0.5 * (len(x) * math.log(2.0 * math.pi) + np.sum(np.log(variance)) + np.sum(mahalanobis))
        log_prob.append(math.log(weights[idx]) + log_term)
    return np.asarray(log_prob, dtype=float)


def assign_m5_regimes(
    artifact: M5ModelArtifact,
    feature_matrix: pd.DataFrame,
    *,
    decision_time: pd.Timestamp,
    availability_mode: M5AvailabilityMode | str | None = None,
) -> pd.DataFrame:
    validate_m5_artifact(artifact)
    validate_m5_assignment_matrix(artifact, feature_matrix)

    mode = artifact.availability_mode if availability_mode is None else M5AvailabilityMode(availability_mode)
    if mode != artifact.availability_mode:
        raise M5ValidationError("artifact and invocation availability modes are incompatible")

    decision_ts = _normalize_utc(decision_time)
    if artifact.availability_mode == M5AvailabilityMode.AS_OF_PRODUCTION:
        if artifact.effective_model_availability is None:
            raise M5ValidationError("effective_model_availability is required")
    if artifact.availability_mode == M5AvailabilityMode.HISTORICAL_REPLAY:
        if artifact.simulated_replay_availability is None:
            raise M5ValidationError("simulated_replay_availability is required for historical replay")

    matrix = feature_matrix.copy()
    rows: list[dict[str, Any]] = []
    weights = np.asarray(artifact.weights, dtype=float)
    means = np.asarray(artifact.means, dtype=float)
    covariances = np.asarray(artifact.diagonal_covariances, dtype=float)
    scaler_mean = np.asarray(artifact.scaler_mean, dtype=float)
    scaler_scale = np.asarray(artifact.scaler_scale, dtype=float)

    for _, row in matrix.iterrows():
        package: dict[str, Any] = {
            "instrument": str(row.get("scope_id", artifact.instrument)),
            "observation_timestamp_utc": _normalize_utc(row["timestamp"]),
            "decision_timestamp_utc": decision_ts,
            "model_id": artifact.model_id,
            "assignment_status": M5AssignmentStatus.ABSTAINED.value,
            "most_likely_component": None,
            "probability_vector": None,
            "log_likelihood": None,
            "entropy_bits": None,
            "feature_vector_available_at_utc": None,
            "abstention_reason": None,
            "provenance_digest": artifact.training_input_digest,
        }

        if str(row.get("scope_id", artifact.instrument)) != artifact.instrument:
            package["abstention_reason"] = "instrument_mismatch"
            rows.append(package)
            continue

        feature_available = row.get("feature_vector_available_at")
        if feature_available is None or pd.isna(feature_available):
            package["abstention_reason"] = "missing_feature_vector_availability"
            rows.append(package)
            continue
        feature_available_ts = _normalize_utc(feature_available)
        package["feature_vector_available_at_utc"] = feature_available_ts
        if feature_available_ts > decision_ts:
            package["abstention_reason"] = "late_feature_availability"
            rows.append(package)
            continue

        if not bool(row.get("feature_observation_available", False)):
            package["abstention_reason"] = "missing_or_unverified_feature_availability"
            rows.append(package)
            continue

        if artifact.availability_mode == M5AvailabilityMode.AS_OF_PRODUCTION and (
            decision_ts < artifact.actual_creation_timestamp_utc
            or decision_ts < artifact.training_information_cutoff
            or decision_ts < artifact.effective_model_availability
        ):
            package["abstention_reason"] = "model_unavailable"
            rows.append(package)
            continue
        if artifact.availability_mode == M5AvailabilityMode.HISTORICAL_REPLAY and (
            decision_ts < artifact.training_information_cutoff
            or decision_ts < artifact.simulated_replay_availability
        ):
            package["abstention_reason"] = "model_unavailable"
            rows.append(package)
            continue

        x_values: list[float] = []
        missing_or_undefined = False
        for feature_id in artifact.effective_feature_ids:
            value = row.get(feature_id)
            if value is None or pd.isna(value):
                missing_or_undefined = True
                break
            x_values.append(float(value))
        if missing_or_undefined:
            package["abstention_reason"] = "undefined_feature"
            rows.append(package)
            continue

        x = np.asarray(x_values, dtype=float)
        x_scaled = (x - scaler_mean) / scaler_scale
        log_probs = np.asarray([_component_log_probability(x_scaled, weights, means, covariances)[idx] for idx in range(len(weights))], dtype=float)
        max_log = np.max(log_probs)
        shifted = np.exp(log_probs - max_log)
        probs = shifted / shifted.sum()
        try:
            validate_m5_probability_vector(probs.tolist())
        except M5ValidationError:
            package["abstention_reason"] = "invalid_probability_vector"
            rows.append(package)
            continue

        component_index = int(np.argmax(probs))
        entropy = -float(np.sum(probs * np.log2(probs + 1e-300)))
        package["assignment_status"] = M5AssignmentStatus.ASSIGNED.value
        package["most_likely_component"] = component_index
        package["probability_vector"] = [float(v) for v in probs.tolist()]
        package["log_likelihood"] = float(max_log + math.log(shifted.sum()))
        package["entropy_bits"] = entropy
        package["abstention_reason"] = None
        rows.append(package)

    output = pd.DataFrame(
        rows,
        columns=[
            "instrument",
            "observation_timestamp_utc",
            "decision_timestamp_utc",
            "model_id",
            "assignment_status",
            "most_likely_component",
            "probability_vector",
            "log_likelihood",
            "entropy_bits",
            "feature_vector_available_at_utc",
            "abstention_reason",
            "provenance_digest",
        ],
    )
    if not output.empty:
        output["assignment_status"] = output["assignment_status"].astype(str)
        output["observation_timestamp_utc"] = pd.to_datetime(output["observation_timestamp_utc"], utc=True)
        output["decision_timestamp_utc"] = pd.to_datetime(output["decision_timestamp_utc"], utc=True)
        output["feature_vector_available_at_utc"] = pd.to_datetime(output["feature_vector_available_at_utc"], utc=True, errors="coerce")
        output = output.sort_values(["instrument", "observation_timestamp_utc"], kind="mergesort").reset_index(drop=True)
    return output

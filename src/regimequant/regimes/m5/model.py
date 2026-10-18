from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .assignment import _component_log_probability
from .contracts import M5AvailabilityMode, M5ModelArtifact, M5ValidationError, canonical_json_dumps
from .validation import validate_m5_artifact, validate_m5_feature_provenance


def _normalize_utc(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None or ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _safe_float_array(values: Any, *, name: str) -> np.ndarray:
    if values is None:
        raise M5ValidationError(f"{name} is required")
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        array = array.reshape(1)
    if not np.all(np.isfinite(array)):
        raise M5ValidationError(f"{name} contains non-finite values")
    return array


def _build_training_digest(frame: pd.DataFrame, requested_features: list[str], feature_spec_hash: str | None) -> str:
    ordered = frame.loc[:, requested_features].copy()
    ordered = ordered.sort_index(axis=1)
    ordered = ordered.sort_values(by=list(ordered.columns[:1]), kind="mergesort").copy() if len(ordered.columns) > 0 else ordered
    payload = {
        "feature_spec_hash": feature_spec_hash,
        "requested_feature_ids": sorted(requested_features),
        "row_count": int(len(ordered)),
        "columns": [str(c) for c in ordered.columns],
        "data": np.asarray(ordered.astype(float).to_numpy(), dtype=float).tolist(),
    }
    canonical_json = canonical_json_dumps(payload)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def compute_m5_descriptive_diagnostics(
    artifact: M5ModelArtifact,
    feature_matrix: pd.DataFrame,
    *,
    interval: tuple[pd.Timestamp, pd.Timestamp],
    label: str = "validation",
) -> dict[str, Any]:
    """Compute frozen descriptive diagnostics for a held-out interval without refitting."""

    validate_m5_artifact(artifact)
    if feature_matrix is None or feature_matrix.empty:
        raise M5ValidationError("feature_matrix is empty")
    if "scope_id" not in feature_matrix.columns or "timestamp" not in feature_matrix.columns:
        raise M5ValidationError("feature_matrix must include scope_id and timestamp columns")

    start_ts = _normalize_utc(interval[0])
    end_ts = _normalize_utc(interval[1])
    if start_ts >= end_ts:
        raise M5ValidationError("diagnostic interval must be strictly ordered")

    eligible = feature_matrix[feature_matrix["scope_id"] == artifact.instrument].copy()
    eligible = eligible[(eligible["timestamp"] >= start_ts) & (eligible["timestamp"] < end_ts)].copy()
    eligible = eligible[eligible["feature_vector_available_at"].notna()].copy()
    eligible = eligible[eligible["feature_vector_available_at"] <= artifact.training_information_cutoff].copy()
    eligible = eligible[eligible["feature_observation_available"] == True].copy()
    eligible = eligible.sort_values(["scope_id", "timestamp"], kind="mergesort").copy()

    if eligible.empty:
        return {
            "diagnostic_kind": "descriptive",
            "metric_kind": "descriptive",
            "label": label,
            "interval": (start_ts.isoformat(), end_ts.isoformat()),
            "eligible_sample_count": 0,
            "total_log_likelihood": None,
            "mean_log_likelihood": None,
            "mean_posterior_entropy": None,
            "hard_occupancy": {str(i): 0 for i in range(len(artifact.weights))},
            "effective_occupancy": {str(i): 0.0 for i in range(len(artifact.weights))},
            "component_concentration": None,
            "numerical_validity": False,
            "descriptive_only": True,
            "refit_performed": False,
        }

    scaler_mean = np.asarray(artifact.scaler_mean, dtype=float)
    scaler_scale = np.asarray(artifact.scaler_scale, dtype=float)
    weights = np.asarray(artifact.weights, dtype=float)
    means = np.asarray(artifact.means, dtype=float)
    covariances = np.asarray(artifact.diagonal_covariances, dtype=float)
    per_row_logits: list[np.ndarray] = []
    per_row_probs: list[np.ndarray] = []
    log_likelihoods: list[float] = []
    entropy_values: list[float] = []
    hard_counts = {str(i): 0 for i in range(len(weights))}
    effective_counts = {str(i): 0.0 for i in range(len(weights))}
    numerical_valid = True

    for _, row in eligible.iterrows():
        values: list[float] = []
        for feature_id in artifact.effective_feature_ids:
            value = row.get(feature_id)
            if value is None or pd.isna(value):
                numerical_valid = False
                break
            values.append(float(value))
        if not numerical_valid:
            break
        x = np.asarray(values, dtype=float)
        x_scaled = (x - scaler_mean) / scaler_scale
        if not np.all(np.isfinite(x_scaled)):
            numerical_valid = False
            break
        log_probs = np.asarray([
            _component_log_probability(x_scaled, weights, means, covariances)[idx]
            for idx in range(len(weights))
        ], dtype=float)
        max_log = np.max(log_probs)
        shifted = np.exp(log_probs - max_log)
        probs = shifted / shifted.sum()
        if not np.all(np.isfinite(probs)) or np.any(probs < 0.0):
            numerical_valid = False
            break
        if not np.isclose(float(probs.sum()), 1.0, atol=1e-8, rtol=0.0):
            numerical_valid = False
            break
        assignment_idx = int(np.argmax(probs))
        hard_counts[str(assignment_idx)] += 1
        for idx, value in enumerate(probs):
            effective_counts[str(idx)] += float(value)
        total_log = float(max_log + math.log(shifted.sum()))
        log_likelihoods.append(total_log)
        entropy_values.append(float(-np.sum(probs * np.log2(probs + 1e-300))))
        per_row_probs.append(probs)

    if not numerical_valid and len(log_likelihoods) == 0:
        return {
            "diagnostic_kind": "descriptive",
            "metric_kind": "descriptive",
            "label": label,
            "interval": (start_ts.isoformat(), end_ts.isoformat()),
            "eligible_sample_count": int(len(eligible)),
            "total_log_likelihood": None,
            "mean_log_likelihood": None,
            "mean_posterior_entropy": None,
            "hard_occupancy": {str(i): int(hard_counts.get(str(i), 0)) for i in range(len(weights))},
            "effective_occupancy": {str(i): float(effective_counts.get(str(i), 0.0)) for i in range(len(weights))},
            "component_concentration": None,
            "numerical_validity": False,
            "descriptive_only": True,
            "refit_performed": False,
        }

    concentration = float(np.mean(np.max(np.asarray(per_row_probs, dtype=float), axis=1))) if per_row_probs else None
    diagnostics = {
        "diagnostic_kind": "descriptive",
        "metric_kind": "descriptive",
        "label": label,
        "interval": (start_ts.isoformat(), end_ts.isoformat()),
        "eligible_sample_count": int(len(eligible)),
        "total_log_likelihood": float(np.sum(log_likelihoods)) if log_likelihoods else None,
        "mean_log_likelihood": float(np.mean(log_likelihoods)) if log_likelihoods else None,
        "mean_posterior_entropy": float(np.mean(entropy_values)) if entropy_values else None,
        "hard_occupancy": {str(i): int(hard_counts.get(str(i), 0)) for i in range(len(weights))},
        "effective_occupancy": {str(i): float(effective_counts.get(str(i), 0.0)) for i in range(len(weights))},
        "component_concentration": concentration,
        "numerical_validity": bool(numerical_valid),
        "descriptive_only": True,
        "refit_performed": False,
    }
    return diagnostics


def fit_m5_regime_model(
    feature_matrix: pd.DataFrame,
    *,
    instrument: str,
    requested_feature_ids: Iterable[str],
    training_interval: tuple[pd.Timestamp, pd.Timestamp] | None,
    validation_interval: tuple[pd.Timestamp, pd.Timestamp] | None,
    test_interval: tuple[pd.Timestamp, pd.Timestamp] | None,
    training_information_cutoff: pd.Timestamp,
    training_cutoff: pd.Timestamp,
    availability_mode: M5AvailabilityMode = M5AvailabilityMode.AS_OF_PRODUCTION,
    simulated_replay_availability: pd.Timestamp | None = None,
    n_components: int = 3,
    covariance_type: str = "diag",
    random_state: int = 42,
    n_init: int = 10,
    max_iter: int = 500,
    tol: float = 1e-4,
    reg_covar: float = 1e-6,
) -> M5ModelArtifact:
    if feature_matrix is None or feature_matrix.empty:
        raise M5ValidationError("feature_matrix is empty")
    if training_interval is None:
        raise M5ValidationError("training_interval is required")
    if validation_interval is None:
        raise M5ValidationError("validation_interval is required")
    if test_interval is None:
        raise M5ValidationError("test_interval is required")
    if n_components < 2:
        raise M5ValidationError("n_components must be at least 2")
    if covariance_type != "diag":
        raise M5ValidationError("M5 v1 requires diagonal covariance GaussianMixture")

    validate_m5_feature_provenance(feature_matrix)
    provenance = feature_matrix.attrs.get("provenance", {})
    feature_spec_hash = provenance.get("feature_spec_set_hash")
    requested = tuple(str(feature) for feature in requested_feature_ids)
    required = ["scope_id", "timestamp", "feature_vector_available_at", "feature_observation_available"] + list(requested)
    missing = [name for name in required if name not in feature_matrix.columns]
    if missing:
        raise M5ValidationError(f"feature_matrix missing required columns: {missing!r}")

    start_ts, end_ts = training_interval
    start_ts = _normalize_utc(start_ts)
    end_ts = _normalize_utc(end_ts)
    if start_ts >= end_ts:
        raise M5ValidationError("training_interval must be strictly ordered")
    train_frame = feature_matrix[feature_matrix["scope_id"] == instrument].copy()
    train_frame = train_frame[(train_frame["timestamp"] >= start_ts) & (train_frame["timestamp"] < end_ts)].copy()
    train_frame = train_frame[train_frame["feature_vector_available_at"].notna()].copy()
    train_frame = train_frame[train_frame["feature_vector_available_at"] <= _normalize_utc(training_information_cutoff)].copy()
    train_frame = train_frame[train_frame["feature_observation_available"] == True].copy()
    train_frame = train_frame.sort_values(["scope_id", "timestamp"], kind="mergesort").copy()
    if train_frame.empty:
        raise M5ValidationError("no causally eligible training rows remain")

    if train_frame.duplicated(subset=["scope_id", "timestamp"]).any():
        raise M5ValidationError("duplicate training keys are not allowed")

    value_matrix = train_frame.loc[:, list(requested)].copy()
    value_matrix = value_matrix.apply(pd.to_numeric, errors="coerce")
    value_matrix = value_matrix.dropna(axis=0, how="any").copy()
    if value_matrix.empty:
        raise M5ValidationError("no valid training rows remain after dropping missing feature values")
    constant_mask = value_matrix.var(axis=0, ddof=0) > 0
    effective_feature_ids = [requested[idx] for idx, is_kept in enumerate(constant_mask) if bool(is_kept)]
    dropped = [requested[idx] for idx, is_kept in enumerate(constant_mask) if not bool(is_kept)]
    if not effective_feature_ids:
        raise M5ValidationError("all requested features are constant")

    X = value_matrix.loc[:, effective_feature_ids].to_numpy(dtype=float)
    if X.shape[0] < max(2 * n_components, 10 * X.shape[1]):
        raise M5ValidationError("minimum eligible training rows are not satisfied")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        reg_covar=reg_covar,
    )
    gmm.fit(X_scaled)
    if not bool(gmm.converged_):
        raise M5ValidationError("GaussianMixture did not converge")

    weights = gmm.weights_
    means = gmm.means_
    covs = gmm.covariances_
    if len(weights) != n_components:
        raise M5ValidationError("GaussianMixture produced a component count mismatch")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise M5ValidationError("mixture weights are invalid")
    if not np.all(np.isfinite(means)):
        raise M5ValidationError("component means contain non-finite values")
    if not np.all(np.isfinite(covs)):
        raise M5ValidationError("component covariances contain non-finite values")
    if np.any(covs <= 0):
        raise M5ValidationError("component covariance entries must be positive")
    if not np.all(np.isfinite(scaler.mean_)) or not np.all(np.isfinite(scaler.scale_)):
        raise M5ValidationError("scaler parameters are non-finite")
    if np.any(scaler.scale_ <= 0.0):
        raise M5ValidationError("scaler scale values must be positive")

    # Canonical ordering by weights descending, means descending, covariance ascending, original index ascending.
    order = sorted(range(n_components), key=lambda idx: (-weights[idx],) + tuple(-float(v) for v in means[idx]) + tuple(float(v) for v in covs[idx]) + (idx,))
    ordered_weights = weights[order]
    ordered_means = means[order]
    ordered_covs = covs[order]

    for i in range(len(ordered_weights)):
        for j in range(i + 1, len(ordered_weights)):
            base_delta = abs(ordered_weights[i] - ordered_weights[j])
            mean_delta = max(abs(a - b) for a, b in zip(ordered_means[i], ordered_means[j]))
            cov_delta = max(abs(a - b) for a, b in zip(ordered_covs[i], ordered_covs[j]))
            if max(base_delta, mean_delta, cov_delta) <= 1e-10:
                raise M5ValidationError("duplicate components are not allowed")

    hard_occupancy = {str(i): int(np.sum(gmm.predict(X_scaled) == i)) for i in range(n_components)}
    effective_occupancy = {str(i): float(np.sum(gmm.predict_proba(X_scaled)[:, i])) for i in range(n_components)}
    if any(value < 2 for value in hard_occupancy.values()):
        raise M5ValidationError("hard occupancy below 2")
    if any(value < max(2, X.shape[1]) for value in effective_occupancy.values()):
        raise M5ValidationError("effective occupancy below max(2, d_eff)")
    if any(value <= 0 for value in effective_occupancy.values()):
        raise M5ValidationError("zero effective occupancy")

    warnings: list[str] = []
    for idx, weight in enumerate(ordered_weights):
        if weight < 0.01:
            warnings.append(f"component_{idx}_weight_below_0_01")

    training_input_digest = _build_training_digest(train_frame, list(effective_feature_ids), feature_spec_hash)
    artifact = M5ModelArtifact(
        instrument=instrument,
        requested_feature_ids=requested,
        effective_feature_ids=tuple(effective_feature_ids),
        dropped_constant_feature_ids=tuple(dropped),
        feature_order=tuple(effective_feature_ids),
        feature_provenance=provenance,
        feature_spec_hash=feature_spec_hash,
        scaler_mean=tuple(float(v) for v in scaler.mean_),
        scaler_scale=tuple(float(v) for v in scaler.scale_),
        training_interval=(start_ts, end_ts),
        validation_interval=validation_interval,
        test_interval=test_interval,
        training_cutoff=_normalize_utc(training_cutoff),
        training_information_cutoff=_normalize_utc(training_information_cutoff),
        requested_component_count=n_components,
        effective_component_count=n_components,
        covariance_type=covariance_type,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        reg_covar=reg_covar,
        means=tuple(tuple(float(v) for v in row) for row in ordered_means),
        diagonal_covariances=tuple(tuple(float(v) for v in row) for row in ordered_covs),
        weights=tuple(float(v) for v in ordered_weights),
        hard_occupancy={str(i): int(hard_occupancy[str(i)]) for i in range(n_components)},
        effective_occupancy={str(i): float(effective_occupancy[str(i)]) for i in range(n_components)},
        converged=bool(gmm.converged_),
        diagnostics={
            "log_likelihood": float(gmm.lower_bound_),
            "n_iter": int(gmm.n_iter_),
            "train_rows": int(len(train_frame)),
            "feature_dimension": int(X.shape[1]),
        },
        warnings=tuple(warnings),
        provenance={
            "feature_spec_set_hash": feature_spec_hash,
            "training_input_digest": training_input_digest,
            "requested_feature_ids": list(requested),
            "effective_feature_ids": list(effective_feature_ids),
            "dropped_constant_feature_ids": list(dropped),
        },
        availability_mode=availability_mode,
            actual_creation_timestamp_utc=max(pd.Timestamp.now(tz="UTC"), _normalize_utc(training_cutoff), _normalize_utc(training_information_cutoff)),
        simulated_replay_availability=_normalize_utc(simulated_replay_availability) if simulated_replay_availability is not None else None,
        effective_model_availability=max(pd.Timestamp.now(tz="UTC"), _normalize_utc(training_cutoff), _normalize_utc(training_information_cutoff)),
        training_input_digest=training_input_digest,
        model_id="",
    )

    if availability_mode == M5AvailabilityMode.AS_OF_PRODUCTION:
        if artifact.effective_model_availability is None:
            raise M5ValidationError("effective_model_availability is required for production mode")
        if artifact.effective_model_availability < artifact.actual_creation_timestamp_utc:
            raise M5ValidationError("production model availability cannot precede creation time")
        if artifact.training_information_cutoff is not None and artifact.effective_model_availability < artifact.training_information_cutoff:
            raise M5ValidationError("production model availability cannot precede training information cutoff")
    if availability_mode == M5AvailabilityMode.HISTORICAL_REPLAY:
        if artifact.simulated_replay_availability is None:
            raise M5ValidationError("simulated_replay_availability is required for historical replay")
        if artifact.training_information_cutoff is not None and artifact.simulated_replay_availability < artifact.training_information_cutoff:
            raise M5ValidationError("simulated_replay_availability cannot precede training information cutoff")

    return artifact

from __future__ import annotations

import math
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .contracts import M3AvailabilityMode, M3ModelArtifact, M3ValidationError
from .splits import build_m3_split_plan


def _coerce_finite_float_list(values: Any, *, name: str) -> list[float]:
    if isinstance(values, np.ndarray):
        values = values.tolist()
    if not isinstance(values, (list, tuple)):
        raise M3ValidationError(f"{name} must be a list-like numeric payload")
    cleaned: list[float] = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise M3ValidationError(f"{name} contains a non-numeric value") from exc
        if not math.isfinite(numeric):
            raise M3ValidationError(f"{name} contains a non-finite numeric value")
        cleaned.append(numeric)
    return cleaned


def validate_m3_model_artifact(artifact: Any) -> M3ModelArtifact:
    if not isinstance(artifact, M3ModelArtifact):
        raise M3ValidationError("artifact must be an M3ModelArtifact instance")
    if not artifact.model_id.startswith("m3v1_sha256_"):
        raise M3ValidationError("artifact model_id is not an approved M3 v1 identity")
    if artifact.training_cutoff is None:
        raise M3ValidationError("training cutoff is required")
    if not artifact.effective_feature_instance_ids:
        raise M3ValidationError("effective feature ids must not be empty")
    if not artifact.canonical_centroids:
        raise M3ValidationError("model requires canonical centroids")
    if not isinstance(artifact.scaler_parameters, dict):
        raise M3ValidationError("scaler_parameters must be a dictionary")
    if "mean_" not in artifact.scaler_parameters or "scale_" not in artifact.scaler_parameters:
        raise M3ValidationError("scaler_parameters must include mean_ and scale_")

    mean_values = _coerce_finite_float_list(artifact.scaler_parameters["mean_"], name="scaler_parameters['mean_']")
    scale_values = _coerce_finite_float_list(artifact.scaler_parameters["scale_"], name="scaler_parameters['scale_']")
    if not mean_values or not scale_values:
        raise M3ValidationError("scaler_parameters cannot be empty")
    if len(mean_values) != len(scale_values):
        raise M3ValidationError("scaler mean_ and scale_ lengths must match")
    if any(value <= 0.0 for value in scale_values):
        raise M3ValidationError("scaler scale_ values must be strictly positive")
    if len(artifact.effective_feature_instance_ids) != len(mean_values):
        raise M3ValidationError("scaler feature count does not match the effective feature set")

    centroid_array = np.asarray(artifact.canonical_centroids, dtype=float)
    if centroid_array.ndim == 1:
        centroid_array = centroid_array.reshape(1, -1)
    if centroid_array.size == 0:
        raise M3ValidationError("model requires canonical centroids")
    if not np.all(np.isfinite(centroid_array)):
        raise M3ValidationError("canonical centroids contain non-finite values")
    if centroid_array.shape[1] != len(mean_values):
        raise M3ValidationError("centroid dimension mismatch with scaler parameter count")
    if np.unique(centroid_array, axis=0).shape[0] != centroid_array.shape[0]:
        raise M3ValidationError("duplicate centroids detected in the artifact")

    if artifact.training_occupancy:
        if not isinstance(artifact.training_occupancy, dict):
            raise M3ValidationError("training_occupancy must be a dictionary")
        required_keys = {str(index) for index in range(len(centroid_array))}
        extra_or_missing = set(artifact.training_occupancy.keys()) ^ required_keys
        if extra_or_missing:
            raise M3ValidationError("training_occupancy must exactly match the canonical centroid index set")
        values = []
        for key, value in artifact.training_occupancy.items():
            try:
                numeric = int(value)
            except (TypeError, ValueError) as exc:
                raise M3ValidationError("training_occupancy values must be integer counts") from exc
            if numeric <= 0:
                raise M3ValidationError("empty cluster detected in the artifact occupancy")
            values.append(numeric)
        if sum(values) <= 0:
            raise M3ValidationError("training_occupancy cannot be empty")

    if artifact.actual_creation_timestamp_utc.tzinfo is None or artifact.actual_creation_timestamp_utc.tz is None:
        raise M3ValidationError("artifact actual_creation_timestamp_utc must be timezone-aware")
    if artifact.assignment_mode == M3AvailabilityMode.AS_OF_PRODUCTION:
        if artifact.effective_model_availability is None:
            raise M3ValidationError("production model availability is required")
        if artifact.effective_model_availability < artifact.actual_creation_timestamp_utc:
            raise M3ValidationError("production model availability cannot precede the actual artifact creation time")
        if artifact.training_information_cutoff is not None and artifact.effective_model_availability < artifact.training_information_cutoff:
            raise M3ValidationError("production model availability cannot precede the training information cutoff")
    if artifact.assignment_mode == M3AvailabilityMode.HISTORICAL_REPLAY:
        if artifact.simulated_replay_availability is None:
            raise M3ValidationError("simulated_replay_availability is required for historical replay")
        if artifact.training_information_cutoff is not None and artifact.simulated_replay_availability < artifact.training_information_cutoff:
            raise M3ValidationError("replay availability must be at or after the training information cutoff")
    expected_id = M3ModelArtifact(
        model_contract_version=artifact.model_contract_version,
        instrument=artifact.instrument,
        requested_feature_instance_ids=artifact.requested_feature_instance_ids,
        effective_feature_instance_ids=artifact.effective_feature_instance_ids,
        dropped_constant_feature_ids=artifact.dropped_constant_feature_ids,
        feature_spec_set_hash=artifact.feature_spec_set_hash,
        parent_provenance_digest=artifact.parent_provenance_digest,
        return_type=artifact.return_type,
        price_type=artifact.price_type,
        adjustment_state=artifact.adjustment_state,
        training_interval=artifact.training_interval,
        training_cutoff=artifact.training_cutoff,
        training_information_cutoff=artifact.training_information_cutoff,
        validation_interval=artifact.validation_interval,
        test_interval=artifact.test_interval,
        kmeans_config=artifact.kmeans_config,
        scaler_parameters=artifact.scaler_parameters,
        canonical_centroids=artifact.canonical_centroids,
        training_occupancy=artifact.training_occupancy,
        assignment_mode=artifact.assignment_mode,
        actual_creation_timestamp_utc=artifact.actual_creation_timestamp_utc,
        simulated_replay_availability=artifact.simulated_replay_availability,
        effective_model_availability=artifact.effective_model_availability,
        seed=artifact.seed,
    )
    if artifact.model_id != expected_id.model_id:
        raise M3ValidationError("artifact model_id is inconsistent with the identity-bearing payload")
    return artifact


def _stable_float(value: float, *, places: int = 12) -> float:
    return float(np.round(float(value), places))


def _normalize_utc(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None or ts.tz is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _validate_interval(name: str, interval: tuple[pd.Timestamp, pd.Timestamp] | None) -> tuple[pd.Timestamp, pd.Timestamp]:
    if interval is None:
        raise M3ValidationError(f"{name}_interval is required")
    start, end = interval
    start_ts = _normalize_utc(start)
    end_ts = _normalize_utc(end)
    if start_ts >= end_ts:
        raise M3ValidationError(f"{name}_interval must be strictly ordered")
    return start_ts, end_ts


def _eligible_interval_rows(
    matrix: pd.DataFrame,
    *,
    instrument: str,
    interval: tuple[pd.Timestamp, pd.Timestamp],
    information_cutoff: pd.Timestamp,
    required_features: tuple[str, ...],
) -> pd.DataFrame:
    start_ts, end_ts = interval
    rows = matrix[matrix["scope_id"] == instrument].copy()
    rows = rows[(rows["timestamp"] >= start_ts) & (rows["timestamp"] < end_ts)].copy()
    if rows.empty:
        return rows
    if rows["scope_id"].nunique() != 1:
        raise M3ValidationError("feature matrix contains multiple instruments; fit_m3_regime_model requires a single instrument")
    if rows.duplicated(subset=["scope_id", "timestamp"]).any():
        raise M3ValidationError("duplicate observation keys are not allowed inside an interval")
    rows = rows[rows["feature_observation_available"].fillna(False)].copy()
    rows = rows[rows["feature_vector_available_at"].notna()].copy()
    rows = rows[rows["feature_vector_available_at"] <= information_cutoff].copy()
    rows = rows[rows[list(required_features)].notna().all(axis=1)].copy()
    if rows.empty:
        return rows
    return rows


def fit_m3_regime_model(
    feature_matrix: pd.DataFrame,
    *,
    instrument: str,
    requested_feature_instance_ids: Iterable[str],
    training_cutoff: pd.Timestamp,
    training_interval: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    validation_interval: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    test_interval: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    training_information_cutoff: pd.Timestamp | None = None,
    validation_information_cutoff: pd.Timestamp | None = None,
    test_information_cutoff: pd.Timestamp | None = None,
    feature_spec_set_hash: str | None = None,
    parent_provenance_digest: str | None = None,
    return_type: str | None = None,
    price_type: str | None = None,
    adjustment_state: str | None = None,
    availability_mode: M3AvailabilityMode = M3AvailabilityMode.AS_OF_PRODUCTION,
    simulated_replay_availability: pd.Timestamp | None = None,
    n_clusters: int = 3,
) -> M3ModelArtifact:
    if feature_matrix.empty:
        raise M3ValidationError("feature matrix is empty")
    if n_clusters <= 0:
        raise M3ValidationError("n_clusters must be positive")
    if training_interval is None:
        raise M3ValidationError("training_interval is required")
    if validation_interval is None:
        raise M3ValidationError("validation_interval is required")
    if test_interval is None:
        raise M3ValidationError("test_interval is required")

    matrix = feature_matrix.copy()
    if "scope_id" in matrix.columns:
        unique_instruments = sorted({str(value) for value in matrix["scope_id"].dropna().unique().tolist()})
        if len(unique_instruments) > 1:
            raise M3ValidationError("feature matrix contains multiple instruments; fit_m3_regime_model requires a single instrument")
        if unique_instruments and instrument != unique_instruments[0]:
            raise M3ValidationError("feature matrix instrument does not match the requested model instrument")

    requested = tuple(dict.fromkeys(str(value) for value in requested_feature_instance_ids))
    if not requested:
        raise M3ValidationError("requested_feature_instance_ids must not be empty")

    missing = [feature for feature in requested if feature not in matrix.columns]
    if missing:
        raise M3ValidationError(f"requested features missing from matrix: {missing!r}")

    if "feature_vector_available_at" not in matrix.columns:
        raise M3ValidationError("feature matrix is missing feature_vector_available_at")
    if "feature_observation_available" not in matrix.columns:
        raise M3ValidationError("feature matrix is missing feature_observation_available")

    matrix["timestamp"] = pd.to_datetime(matrix["timestamp"], errors="coerce")
    matrix["timestamp"] = [ts.tz_localize("UTC") if ts.tzinfo is None or ts.tz is None else ts.tz_convert("UTC") for ts in matrix["timestamp"]]
    matrix["feature_vector_available_at"] = pd.to_datetime(matrix["feature_vector_available_at"], errors="coerce")
    matrix["feature_vector_available_at"] = [
        ts.tz_localize("UTC") if ts.tzinfo is None or ts.tz is None else ts.tz_convert("UTC")
        for ts in matrix["feature_vector_available_at"]
    ]

    training_cutoff_ts = _normalize_utc(training_cutoff)
    valid_timestamps = matrix["timestamp"].dropna()
    if valid_timestamps.empty:
        raise M3ValidationError("feature matrix does not contain valid timestamps")

    if training_interval is None:
        train_start = valid_timestamps.min()
        train_end = min(valid_timestamps.max(), training_cutoff_ts)
        training_interval = (train_start, train_end)

    training_interval_ts = _validate_interval("training", training_interval)
    training_information_cutoff_ts = _normalize_utc(training_information_cutoff) if training_information_cutoff is not None else training_cutoff_ts
    if training_interval_ts[1] > training_information_cutoff_ts:
        raise M3ValidationError("train_end must not exceed the training information cutoff")
    if training_cutoff_ts is not None and training_interval_ts[1] > training_cutoff_ts:
        raise M3ValidationError("training interval must end at or before the training cutoff")

    validation_interval_ts = _validate_interval("validation", validation_interval) if validation_interval is not None else None
    test_interval_ts = _validate_interval("test", test_interval) if test_interval is not None else None
    if validation_interval_ts is not None and test_interval_ts is not None:
        if validation_interval_ts[0] < training_interval_ts[1]:
            raise M3ValidationError("validation interval must start at or after the training interval end")
        if test_interval_ts[0] < validation_interval_ts[1]:
            raise M3ValidationError("test interval must start at or after the validation interval end")
        if training_interval_ts[1] > validation_interval_ts[0]:
            raise M3ValidationError("training and validation intervals overlap")
        if validation_interval_ts[1] > test_interval_ts[0]:
            raise M3ValidationError("validation and test intervals overlap")

    validation_information_cutoff_ts = _normalize_utc(validation_information_cutoff) if validation_information_cutoff is not None else (validation_interval_ts[1] if validation_interval_ts is not None else training_information_cutoff_ts)
    test_information_cutoff_ts = _normalize_utc(test_information_cutoff) if test_information_cutoff is not None else (test_interval_ts[1] if test_interval_ts is not None else training_information_cutoff_ts)
    if validation_interval_ts is not None and validation_information_cutoff_ts < validation_interval_ts[0]:
        raise M3ValidationError("validation information cutoff cannot precede the validation interval")
    if test_interval_ts is not None and test_information_cutoff_ts < test_interval_ts[0]:
        raise M3ValidationError("test information cutoff cannot precede the test interval")

    for label, interval, info_cutoff in [
        ("training", training_interval_ts, training_information_cutoff_ts),
        ("validation", validation_interval_ts, validation_information_cutoff_ts),
        ("test", test_interval_ts, test_information_cutoff_ts),
    ]:
        if interval is None:
            continue
        interval_rows = matrix[(matrix["scope_id"] == instrument) & (matrix["timestamp"] >= interval[0]) & (matrix["timestamp"] < interval[1])].copy()
        if interval_rows.empty:
            continue
        late_rows = interval_rows[interval_rows["feature_vector_available_at"] > info_cutoff]
        if not late_rows.empty:
            raise M3ValidationError(f"{label} feature availability violates the cutoff window")

    train_rows = _eligible_interval_rows(
        matrix,
        instrument=instrument,
        interval=training_interval_ts,
        information_cutoff=training_information_cutoff_ts,
        required_features=tuple(requested),
    )
    train_rows = train_rows.sort_values(["scope_id", "timestamp", "feature_vector_available_at"], kind="mergesort").reset_index(drop=True)
    validation_rows = _eligible_interval_rows(
        matrix,
        instrument=instrument,
        interval=validation_interval_ts,
        information_cutoff=validation_information_cutoff_ts,
        required_features=tuple(requested),
    ) if validation_interval_ts is not None else pd.DataFrame(columns=matrix.columns)
    test_rows = _eligible_interval_rows(
        matrix,
        instrument=instrument,
        interval=test_interval_ts,
        information_cutoff=test_information_cutoff_ts,
        required_features=tuple(requested),
    ) if test_interval_ts is not None else pd.DataFrame(columns=matrix.columns)

    train_count = len(train_rows)
    validation_count = len(validation_rows)
    test_count = len(test_rows)
    required_train_count = max(30, 10 * n_clusters)
    required_validation_count = max(20, 2 * n_clusters)
    required_test_count = max(20, 2 * n_clusters)
    if train_count < required_train_count:
        raise M3ValidationError(f"training requires at least {required_train_count} causally eligible observations")
    if validation_interval_ts is not None and validation_count < required_validation_count:
        raise M3ValidationError(f"validation requires at least {required_validation_count} causally eligible observations")
    if test_interval_ts is not None and test_count < required_test_count:
        raise M3ValidationError(f"test requires at least {required_test_count} causally eligible observations")

    if train_rows.empty:
        raise M3ValidationError("no training rows are eligible before the training cutoff")
    if validation_interval_ts is not None and validation_rows.empty:
        raise M3ValidationError("no validation rows are causally eligible for the held-out gate")
    if test_interval_ts is not None and test_rows.empty:
        raise M3ValidationError("no test rows are causally eligible for the held-out gate")

    selected = train_rows.loc[:, list(requested)].copy()
    selected = selected.apply(pd.to_numeric, errors="coerce")
    if selected.isna().all(axis=None):
        raise M3ValidationError("training matrix has no valid numeric data")
    invalid_rows = selected.isna().any(axis=1)
    if invalid_rows.any():
        raise M3ValidationError("training matrix contains non-causally eligible rows with missing feature values")

    constant_cols = [col for col in selected.columns if selected[col].nunique(dropna=True) <= 1]
    effective = [col for col in requested if col not in constant_cols]
    if not effective:
        raise M3ValidationError("all requested features are constant in the training window")
    selected = selected.loc[:, effective]

    scaler = StandardScaler()
    scaled = scaler.fit_transform(selected)
    if scaled.shape[0] < max(2, n_clusters):
        raise M3ValidationError("training set is too small for the requested regime count")

    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=10,
        max_iter=300,
        tol=1e-4,
        random_state=42,
        algorithm="lloyd",
    )
    labels = kmeans.fit_predict(scaled)
    if len(np.unique(labels)) < n_clusters:
        raise M3ValidationError("KMeans produced fewer occupied clusters than requested")
    centroids = np.asarray(kmeans.cluster_centers_, dtype=float)
    centroid_order = sorted(range(len(centroids)), key=lambda idx: tuple(float(value) for value in centroids[idx]))
    ordered = np.asarray([centroids[idx] for idx in centroid_order], dtype=float)
    if len(np.unique(ordered, axis=0)) != len(ordered):
        raise M3ValidationError("duplicate centroids detected after canonical ordering")

    remapped_labels = np.empty_like(labels, dtype=int)
    for canonical_idx, original_idx in enumerate(centroid_order):
        remapped_labels[labels == original_idx] = canonical_idx
    labels = remapped_labels
    training_occupancy = {str(cluster_id): int((labels == cluster_id).sum()) for cluster_id in range(n_clusters)}
    if any(count == 0 for count in training_occupancy.values()):
        raise M3ValidationError("empty cluster detected in the fitted regime model")

    actual_creation_timestamp_utc = pd.Timestamp(datetime.now(timezone.utc)).tz_convert("UTC")
    if availability_mode == M3AvailabilityMode.HISTORICAL_REPLAY:
        if simulated_replay_availability is None:
            raise M3ValidationError("simulated_replay_availability is required for historical replay")
        effective_model_availability = _normalize_utc(simulated_replay_availability)
        if effective_model_availability < training_information_cutoff_ts:
            raise M3ValidationError("replay availability must be at or after the training information cutoff")
    else:
        effective_model_availability = max(actual_creation_timestamp_utc, training_information_cutoff_ts)
    stable_mean = [_stable_float(value) for value in scaler.mean_.tolist()]
    stable_scale = [_stable_float(value) for value in scaler.scale_.tolist()]
    stable_centroids = tuple(tuple(_stable_float(value) for value in row) for row in ordered)
    artifact = M3ModelArtifact(
        model_contract_version="m3v1",
        instrument=instrument,
        requested_feature_instance_ids=requested,
        effective_feature_instance_ids=tuple(effective),
        dropped_constant_feature_ids=tuple(constant_cols),
        feature_spec_set_hash=feature_spec_set_hash,
        parent_provenance_digest=parent_provenance_digest,
        return_type=return_type,
        price_type=price_type,
        adjustment_state=adjustment_state,
        training_interval=training_interval_ts,
        training_cutoff=training_cutoff_ts,
        training_information_cutoff=training_information_cutoff_ts,
        validation_interval=validation_interval_ts,
        test_interval=test_interval_ts,
        kmeans_config={
            "init": "k-means++",
            "n_init": 10,
            "max_iter": 300,
            "tol": 1e-4,
            "random_state": 42,
            "algorithm": "lloyd",
            "n_clusters": n_clusters,
        },
        scaler_parameters={
            "mean_": stable_mean,
            "scale_": stable_scale,
        },
        canonical_centroids=stable_centroids,
        training_occupancy=training_occupancy,
        assignment_mode=availability_mode,
        actual_creation_timestamp_utc=actual_creation_timestamp_utc,
        simulated_replay_availability=simulated_replay_availability,
        effective_model_availability=effective_model_availability,
        seed=42,
    )
    return validate_m3_model_artifact(artifact)

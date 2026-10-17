from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .contracts import M3AssignmentStatus, M3AvailabilityMode, M3ModelArtifact, M3ValidationError, validate_m2_feature_provenance
from .model import validate_m3_model_artifact


def _validate_artifact_matrix_compatibility(artifact: M3ModelArtifact, feature_matrix: pd.DataFrame) -> None:
    validate_m2_feature_provenance(feature_matrix)
    provenance = feature_matrix.attrs.get("provenance")
    if not isinstance(provenance, dict):
        raise M3ValidationError("M2 feature frame is missing attrs.provenance")

    if artifact.feature_spec_set_hash is None:
        raise M3ValidationError("artifact feature_spec_set_hash is required")
    if provenance.get("feature_spec_set_hash") is not None and provenance.get("feature_spec_set_hash") != artifact.feature_spec_set_hash:
        raise M3ValidationError("artifact feature_spec_set_hash does not match the feature matrix provenance")

    matrix_parent_digest = provenance.get("parent_return_provenance_digest")
    if matrix_parent_digest is None:
        matrix_parent_digest = provenance.get("parent_provenance_digest")
    if artifact.parent_provenance_digest is not None and matrix_parent_digest is not None and matrix_parent_digest != artifact.parent_provenance_digest:
        raise M3ValidationError("artifact parent_provenance_digest does not match the feature matrix provenance")

    for key in ("return_type", "price_type", "adjustment_state"):
        expected = getattr(artifact, key)
        actual = provenance.get(key)
        if actual is not None and expected is not None and str(actual) != str(expected):
            raise M3ValidationError(f"artifact {key} does not match the feature matrix provenance")

    feature_columns = {
        column for column in feature_matrix.columns
        if column not in {"scope_id", "timestamp", "feature_vector_available_at", "feature_observation_available"}
    }
    missing_features = [feature for feature in artifact.effective_feature_instance_ids if feature not in feature_columns]
    if missing_features:
        raise M3ValidationError(f"feature_matrix is missing the artifact effective feature ids: {missing_features!r}")


def assign_m3_regimes(
    artifact: M3ModelArtifact,
    feature_matrix: pd.DataFrame,
    *,
    decision_time: pd.Timestamp,
    assignment_mode: M3AvailabilityMode | str = M3AvailabilityMode.AS_OF_PRODUCTION,
) -> pd.DataFrame:
    validate_m3_model_artifact(artifact)
    if not isinstance(artifact, M3ModelArtifact):
        raise M3ValidationError("artifact must be an M3ModelArtifact")

    _validate_artifact_matrix_compatibility(artifact, feature_matrix)

    if isinstance(assignment_mode, str):
        assignment_mode = M3AvailabilityMode(assignment_mode)
    if assignment_mode != artifact.assignment_mode:
        raise M3ValidationError("assignment_mode must match the artifact's declared availability mode")

    matrix = feature_matrix.copy()
    if "timestamp" not in matrix.columns:
        raise M3ValidationError("feature_matrix must include a timestamp column")
    if "scope_id" not in matrix.columns:
        raise M3ValidationError("feature_matrix must include a scope_id column")
    if "feature_vector_available_at" not in matrix.columns:
        raise M3ValidationError("feature_matrix must include feature_vector_available_at")

    decision_ts = pd.Timestamp(decision_time)
    if decision_ts.tzinfo is None or decision_ts.tz is None:
        decision_ts = decision_ts.tz_localize("UTC")
    else:
        decision_ts = decision_ts.tz_convert("UTC")

    model_available = artifact.effective_model_availability
    if model_available is None:
        raise M3ValidationError("artifact effective_model_availability is required for assignment")
    model_available_ts = pd.Timestamp(model_available)
    if model_available_ts.tzinfo is None or model_available_ts.tz is None:
        model_available_ts = model_available_ts.tz_localize("UTC")
    else:
        model_available_ts = model_available_ts.tz_convert("UTC")

    if assignment_mode == M3AvailabilityMode.AS_OF_PRODUCTION:
        creation_ts = pd.Timestamp(artifact.actual_creation_timestamp_utc)
        if creation_ts.tzinfo is None or creation_ts.tz is None:
            creation_ts = creation_ts.tz_localize("UTC")
        else:
            creation_ts = creation_ts.tz_convert("UTC")
        if decision_ts < creation_ts:
            model_available_ts = creation_ts

    if assignment_mode == M3AvailabilityMode.HISTORICAL_REPLAY:
        if artifact.simulated_replay_availability is None:
            raise M3ValidationError("simulated_replay_availability is required for historical replay")
        replay_ts = pd.Timestamp(artifact.simulated_replay_availability)
        if replay_ts.tzinfo is None or replay_ts.tz is None:
            replay_ts = replay_ts.tz_localize("UTC")
        else:
            replay_ts = replay_ts.tz_convert("UTC")
        available_ts = replay_ts
    else:
        available_ts = model_available_ts

    centroids = artifact.centroid_matrix()
    rows: list[dict[str, Any]] = []
    if available_ts > decision_ts:
        for _, row in matrix.iterrows():
            rows.append(
                {
                    "instrument": row.get("scope_id", artifact.instrument),
                    "observation_timestamp": pd.Timestamp(row["timestamp"]),
                    "decision_timestamp": pd.Timestamp(decision_time),
                    "model_id": artifact.model_id,
                    "cluster_id": None,
                    "assignment_status": "abstained",
                    "abstention_reason": "model_unavailable",
                    "feature_vector_available_at": pd.Timestamp(row["feature_vector_available_at"]) if row.get("feature_vector_available_at") is not None and not pd.isna(row.get("feature_vector_available_at")) else None,
                    "feature_spec_set_hash": artifact.feature_spec_set_hash,
                    "model_training_cutoff": artifact.training_cutoff,
                }
            )
        output = pd.DataFrame(rows, columns=[
            "instrument",
            "observation_timestamp",
            "decision_timestamp",
            "model_id",
            "cluster_id",
            "assignment_status",
            "abstention_reason",
            "feature_vector_available_at",
            "feature_spec_set_hash",
            "model_training_cutoff",
        ])
        output["cluster_id"] = pd.Series(output["cluster_id"], dtype="Int64")
        output["observation_timestamp"] = pd.to_datetime(output["observation_timestamp"], utc=True)
        output["decision_timestamp"] = pd.to_datetime(output["decision_timestamp"], utc=True)
        output["feature_vector_available_at"] = pd.to_datetime(output["feature_vector_available_at"], utc=True, errors="coerce")
        output["assignment_status"] = output["assignment_status"].astype(str)
        if not output.empty:
            output = output.sort_values(["instrument", "observation_timestamp"], kind="mergesort").reset_index(drop=True)
        return output
    for _, row in matrix.iterrows():
        selected = []
        is_valid = True
        abstention_reason = None
        row_scope = row.get("scope_id")
        if row_scope is not None and str(row_scope) != str(artifact.instrument):
            is_valid = False
            abstention_reason = "instrument_mismatch"
        feature_vector_available_at = row.get("feature_vector_available_at")
        if feature_vector_available_at is None or pd.isna(feature_vector_available_at):
            is_valid = False
            abstention_reason = "missing_feature_vector_availability"
        else:
            feature_vector_available_at = pd.Timestamp(feature_vector_available_at)
            if feature_vector_available_at.tzinfo is None or feature_vector_available_at.tz is None:
                feature_vector_available_at = feature_vector_available_at.tz_localize("UTC")
            else:
                feature_vector_available_at = feature_vector_available_at.tz_convert("UTC")
            if feature_vector_available_at > decision_ts:
                is_valid = False
                abstention_reason = "late_feature_availability"

        for feature_id in artifact.effective_feature_instance_ids:
            if feature_id not in row.index:
                is_valid = False
                abstention_reason = "missing_selected_feature"
                break
            value = row[feature_id]
            if pd.isna(value):
                is_valid = False
                abstention_reason = "undefined_feature"
                break
            selected.append(float(value))

        if is_valid and not row.get("feature_observation_available", False):
            is_valid = False
            abstention_reason = "undefined_feature"

        if is_valid:
            sample = np.asarray(selected, dtype=float).reshape(1, -1)
            distances = np.linalg.norm(sample - centroids, axis=1)
            cluster_id = int(np.argmin(distances))
            rows.append(
                {
                    "instrument": row.get("scope_id", artifact.instrument),
                    "observation_timestamp": pd.Timestamp(row["timestamp"]),
                    "decision_timestamp": pd.Timestamp(decision_time),
                    "model_id": artifact.model_id,
                    "cluster_id": cluster_id,
                    "assignment_status": "assigned",
                    "abstention_reason": None,
                    "feature_vector_available_at": pd.Timestamp(feature_vector_available_at) if feature_vector_available_at is not None else None,
                    "feature_spec_set_hash": artifact.feature_spec_set_hash,
                    "model_training_cutoff": artifact.training_cutoff,
                }
            )
        else:
            rows.append(
                {
                    "instrument": row.get("scope_id", artifact.instrument),
                    "observation_timestamp": pd.Timestamp(row["timestamp"]),
                    "decision_timestamp": pd.Timestamp(decision_time),
                    "model_id": artifact.model_id,
                    "cluster_id": None,
                    "assignment_status": "abstained",
                    "abstention_reason": abstention_reason or "unknown_abstention",
                    "feature_vector_available_at": pd.Timestamp(feature_vector_available_at) if feature_vector_available_at is not None else None,
                    "feature_spec_set_hash": artifact.feature_spec_set_hash,
                    "model_training_cutoff": artifact.training_cutoff,
                }
            )

    output = pd.DataFrame(rows, columns=[
        "instrument",
        "observation_timestamp",
        "decision_timestamp",
        "model_id",
        "cluster_id",
        "assignment_status",
        "abstention_reason",
        "feature_vector_available_at",
        "feature_spec_set_hash",
        "model_training_cutoff",
    ])
    if not output.empty:
        output["cluster_id"] = output["cluster_id"].astype("Int64")
        output["observation_timestamp"] = pd.to_datetime(output["observation_timestamp"], utc=True)
        output["decision_timestamp"] = pd.to_datetime(output["decision_timestamp"], utc=True)
        output["feature_vector_available_at"] = pd.to_datetime(output["feature_vector_available_at"], utc=True, errors="coerce")
        output["assignment_status"] = output["assignment_status"].astype(str)
    if not output.empty:
        output = output.sort_values(["instrument", "observation_timestamp"], kind="mergesort").reset_index(drop=True)
        if output.duplicated(subset=["instrument", "observation_timestamp", "decision_timestamp", "model_id"]).any():
            raise M3ValidationError("duplicate assignment keys are not allowed")
    assigned = output[output["assignment_status"] == "assigned"] if not output.empty else output
    if not assigned.empty:
        if assigned["cluster_id"].isna().any():
            raise M3ValidationError("assigned rows cannot carry null cluster_id")
        if assigned["abstention_reason"].notna().any():
            raise M3ValidationError("assigned rows cannot carry a non-null abstention_reason")
    abstained = output[output["assignment_status"] == "abstained"] if not output.empty else output
    if not abstained.empty:
        if abstained["cluster_id"].notna().any():
            raise M3ValidationError("abstained rows cannot carry a non-null cluster_id")
        if abstained["abstention_reason"].isna().any():
            raise M3ValidationError("abstained rows require a non-null abstention_reason")
    return output

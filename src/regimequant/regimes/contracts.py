from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable

import numpy as np
import pandas as pd


class M3Error(Exception):
    """Base exception for M3 regime-discovery failures."""


class M3ValidationError(M3Error):
    """Raised when an M3 artifact or feature matrix violates the contract."""


class M3ModelUnavailableError(M3Error):
    """Raised when a model is unavailable for assignment at the requested time."""


class M3AvailabilityMode(str, Enum):
    AS_OF_PRODUCTION = "AS_OF_PRODUCTION"
    HISTORICAL_REPLAY = "HISTORICAL_REPLAY"


class M3AssignmentStatus(str, Enum):
    ASSIGNED = "assigned"
    ABSTAINED = "abstained"


M3_MODEL_CONTRACT_VERSION = "m3v1"

REQUIRED_M2_FEATURE_COLUMNS = {
    "scope_type",
    "scope_id",
    "timestamp",
    "feature_name",
    "feature_instance_id",
    "feature_value",
    "math_status",
    "undefined_reason",
    "input_availability_status",
    "available_at",
    "input_return_type",
    "price_type",
    "adjustment_state",
    "window_kind",
    "window_size",
    "min_observations",
    "observation_count_used",
    "feature_definition_version",
}


def _coerce_utc_timestamp(value: Any, *, allow_null: bool) -> pd.Timestamp | None:
    if value is None or value is pd.NaT or pd.isna(value):
        if allow_null:
            return None
        raise M3ValidationError("timestamp field cannot be null")
    ts = pd.Timestamp(value)
    if ts.tzinfo is None or ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def validate_m2_feature_provenance(frame: pd.DataFrame) -> dict[str, Any]:
    provenance = frame.attrs.get("provenance")
    if not isinstance(provenance, dict):
        raise M3ValidationError("M2 feature frame is missing attrs.provenance")
    required = {"feature_spec_set_hash", "generated_at_utc", "metadata_durability"}
    if "parent_return_provenance_digest" in provenance:
        required.add("parent_return_provenance_digest")
    missing = sorted(required - set(provenance.keys()))
    if missing:
        raise M3ValidationError(f"missing required provenance fields: {missing!r}")
    if not isinstance(provenance["feature_spec_set_hash"], str) or not provenance["feature_spec_set_hash"]:
        raise M3ValidationError("attrs.provenance.feature_spec_set_hash must be a non-empty string")
    if provenance.get("parent_return_provenance_digest") is not None and not isinstance(provenance["parent_return_provenance_digest"], str):
        raise M3ValidationError("attrs.provenance.parent_return_provenance_digest must be a string or null")
    generated_at_utc = provenance.get("generated_at_utc")
    if generated_at_utc is None:
        raise M3ValidationError("attrs.provenance.generated_at_utc is required")
    _coerce_utc_timestamp(generated_at_utc, allow_null=False)
    if not isinstance(provenance.get("metadata_durability"), str) or not provenance["metadata_durability"]:
        raise M3ValidationError("attrs.provenance.metadata_durability must be a non-empty string")
    return provenance


def _normalize_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise M3ValidationError("non-finite numeric value in model identity payload")
        return value
    if isinstance(value, np.floating):
        if not math.isfinite(float(value)):
            raise M3ValidationError("non-finite numeric value in model identity payload")
        return float(value)
    if isinstance(value, tuple):
        return [_normalize_json_value(v) for v in value]
    if isinstance(value, list):
        return [_normalize_json_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _normalize_json_value(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return [_normalize_json_value(v) for v in value.tolist()]
    return str(value)


def canonical_json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(_normalize_json_value(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _freeze_identity_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _freeze_identity_value(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_freeze_identity_value(v) for v in value)
    if isinstance(value, list):
        return tuple(_freeze_identity_value(v) for v in value)
    if isinstance(value, np.ndarray):
        return tuple(_freeze_identity_value(v) for v in value.tolist())
    if isinstance(value, pd.Timestamp):
        return value
    return value


@dataclass(frozen=True, slots=True)
class M3ModelArtifact:
    model_contract_version: str = "m3v1"
    instrument: str = ""
    requested_feature_instance_ids: tuple[str, ...] = ()
    effective_feature_instance_ids: tuple[str, ...] = ()
    dropped_constant_feature_ids: tuple[str, ...] = ()
    feature_spec_set_hash: str | None = None
    parent_provenance_digest: str | None = None
    return_type: str | None = None
    price_type: str | None = None
    adjustment_state: str | None = None
    training_interval: tuple[pd.Timestamp, pd.Timestamp] = field(default_factory=lambda: (pd.Timestamp.min.tz_localize("UTC"), pd.Timestamp.max.tz_localize("UTC")))
    training_cutoff: pd.Timestamp | None = None
    training_information_cutoff: pd.Timestamp | None = None
    validation_interval: tuple[pd.Timestamp, pd.Timestamp] | None = None
    test_interval: tuple[pd.Timestamp, pd.Timestamp] | None = None
    kmeans_config: dict[str, Any] = field(default_factory=dict)
    scaler_parameters: dict[str, Any] = field(default_factory=dict)
    canonical_centroids: tuple[tuple[float, ...], ...] = ()
    training_occupancy: dict[str, int] = field(default_factory=dict)
    assignment_mode: M3AvailabilityMode = M3AvailabilityMode.AS_OF_PRODUCTION
    actual_creation_timestamp_utc: pd.Timestamp = field(default_factory=lambda: pd.Timestamp(datetime.now(timezone.utc)))
    simulated_replay_availability: pd.Timestamp | None = None
    effective_model_availability: pd.Timestamp | None = None
    seed: int = 42
    model_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "requested_feature_instance_ids", tuple(str(value) for value in self.requested_feature_instance_ids))
        object.__setattr__(self, "effective_feature_instance_ids", tuple(str(value) for value in self.effective_feature_instance_ids))
        object.__setattr__(self, "dropped_constant_feature_ids", tuple(str(value) for value in self.dropped_constant_feature_ids))
        object.__setattr__(self, "kmeans_config", _freeze_identity_value(dict(self.kmeans_config)))
        object.__setattr__(self, "scaler_parameters", _freeze_identity_value(dict(self.scaler_parameters)))
        object.__setattr__(self, "canonical_centroids", tuple(tuple(float(v) for v in row) for row in self.canonical_centroids))
        object.__setattr__(self, "training_occupancy", {str(k): int(v) for k, v in self.training_occupancy.items()})
        object.__setattr__(self, "model_id", self._compute_model_id())

    def _compute_model_id(self) -> str:
        payload = {
            "model_contract_version": self.model_contract_version,
            "instrument": self.instrument,
            "assignment_mode": self.assignment_mode.value,
            "requested_feature_instance_ids": list(self.requested_feature_instance_ids),
            "effective_feature_instance_ids": list(self.effective_feature_instance_ids),
            "dropped_constant_feature_ids": list(self.dropped_constant_feature_ids),
            "feature_spec_set_hash": self.feature_spec_set_hash,
            "parent_provenance_digest": self.parent_provenance_digest,
            "return_type": self.return_type,
            "price_type": self.price_type,
            "adjustment_state": self.adjustment_state,
            "training_interval": [self.training_interval[0].isoformat(), self.training_interval[1].isoformat()] if self.training_interval else None,
            "training_cutoff": self.training_cutoff.isoformat() if self.training_cutoff is not None else None,
            "training_information_cutoff": self.training_information_cutoff.isoformat() if self.training_information_cutoff is not None else None,
            "validation_interval": [self.validation_interval[0].isoformat(), self.validation_interval[1].isoformat()] if self.validation_interval else None,
            "test_interval": [self.test_interval[0].isoformat(), self.test_interval[1].isoformat()] if self.test_interval else None,
            "kmeans_config": self.kmeans_config,
            "scaler_parameters": self.scaler_parameters,
            "canonical_centroids": list(self.canonical_centroids),
            "training_occupancy": self.training_occupancy,
            "seed": self.seed,
        }
        digest = hashlib.sha256(canonical_json_dumps(payload).encode("utf-8")).hexdigest()
        return f"m3v1_sha256_{digest}"

    def centroid_matrix(self) -> np.ndarray:
        return np.asarray(self.canonical_centroids, dtype=float)


def validate_m3_feature_row(row: dict[str, Any]) -> None:
    required = REQUIRED_M2_FEATURE_COLUMNS
    missing = sorted(required - set(row.keys()))
    if missing:
        raise M3ValidationError(f"missing required M2 feature columns: {missing!r}")

    scope_type = str(row.get("scope_type", ""))
    if scope_type != "instrument":
        raise M3ValidationError("M3 requires scope_type == 'instrument'")

    for key in ("scope_id", "feature_name", "feature_instance_id", "input_return_type", "price_type", "adjustment_state"):
        if row.get(key) is None or str(row.get(key, "")).strip() == "":
            raise M3ValidationError(f"M2 feature row has empty required field: {key!r}")

    for key in ("window_size", "min_observations"):
        if key in row and row.get(key) is not None and not pd.isna(row.get(key)):
            try:
                value = int(row[key])
            except (TypeError, ValueError):
                raise M3ValidationError(f"M2 feature row field {key!r} must be integer-like")
            if value <= 0:
                raise M3ValidationError(f"M2 feature row field {key!r} must be positive")

    observation_count_used = row.get("observation_count_used")
    if observation_count_used is not None and not pd.isna(observation_count_used):
        try:
            value = int(observation_count_used)
        except (TypeError, ValueError):
            raise M3ValidationError("M2 feature row field 'observation_count_used' must be integer-like")
        if row.get("math_status") == "defined" and value <= 0:
            raise M3ValidationError("M2 feature row field 'observation_count_used' must be positive for defined rows")
        if row.get("math_status") == "undefined" and value < 0:
            raise M3ValidationError("M2 feature row field 'observation_count_used' cannot be negative for undefined rows")

    timestamp = _coerce_utc_timestamp(row.get("timestamp"), allow_null=False)
    if row.get("available_at") is not None and not pd.isna(row.get("available_at")):
        _coerce_utc_timestamp(row.get("available_at"), allow_null=False)

    math_status = str(row["math_status"])
    availability_status = str(row["input_availability_status"])
    if math_status not in {"defined", "undefined"}:
        raise M3ValidationError(f"illegal math_status: {math_status!r}")
    if availability_status not in {"verified_timestamp", "unknown_unverified"}:
        raise M3ValidationError(f"illegal input_availability_status: {availability_status!r}")

    value = row["feature_value"]
    if math_status == "defined":
        if value is None or pd.isna(value):
            raise M3ValidationError("defined feature_value cannot be null")
        if not np.isfinite(float(value)):
            raise M3ValidationError("defined feature_value must be finite")
        if row.get("undefined_reason") is not None and not pd.isna(row.get("undefined_reason")):
            raise M3ValidationError("defined feature rows must not carry undefined_reason")
        if availability_status == "verified_timestamp" and (row.get("available_at") is None or pd.isna(row.get("available_at"))):
            raise M3ValidationError("verified feature rows require available_at")
    else:
        if value is not None and not pd.isna(value):
            raise M3ValidationError("undefined feature_value must be null")
        if row.get("undefined_reason") is None or pd.isna(row.get("undefined_reason")):
            raise M3ValidationError("undefined feature rows require undefined_reason")
        if row.get("available_at") is not None and not pd.isna(row.get("available_at")):
            raise M3ValidationError("undefined feature rows must not carry available_at")

    if row.get("feature_name") not in {"rolling_mean_return", "cumulative_return_simple_equivalent", "lagged_momentum_simple_equivalent", "rolling_sample_std", "downside_deviation", "realized_variance", "short_long_volatility_ratio", "downside_observation_frequency"}:
        raise M3ValidationError(f"unsupported M2 feature_name: {row.get('feature_name')!r}")

    if str(row.get("feature_definition_version", "")).strip() == "":
        raise M3ValidationError("feature_definition_version is required")


def build_m3_feature_identity_payload(model: M3ModelArtifact) -> dict[str, Any]:
    return {
        "model_contract_version": model.model_contract_version,
        "instrument": model.instrument,
        "assignment_mode": model.assignment_mode.value,
        "requested_feature_instance_ids": list(model.requested_feature_instance_ids),
        "effective_feature_instance_ids": list(model.effective_feature_instance_ids),
        "dropped_constant_feature_ids": list(model.dropped_constant_feature_ids),
        "feature_spec_set_hash": model.feature_spec_set_hash,
        "parent_provenance_digest": model.parent_provenance_digest,
        "return_type": model.return_type,
        "price_type": model.price_type,
        "adjustment_state": model.adjustment_state,
        "training_interval": [model.training_interval[0].isoformat(), model.training_interval[1].isoformat()] if model.training_interval else None,
        "training_cutoff": model.training_cutoff.isoformat() if model.training_cutoff is not None else None,
        "training_information_cutoff": model.training_information_cutoff.isoformat() if model.training_information_cutoff is not None else None,
        "validation_interval": [model.validation_interval[0].isoformat(), model.validation_interval[1].isoformat()] if model.validation_interval else None,
        "test_interval": [model.test_interval[0].isoformat(), model.test_interval[1].isoformat()] if model.test_interval else None,
        "kmeans_config": model.kmeans_config,
        "scaler_parameters": model.scaler_parameters,
        "canonical_centroids": list(model.canonical_centroids),
        "training_occupancy": model.training_occupancy,
        "seed": model.seed,
    }

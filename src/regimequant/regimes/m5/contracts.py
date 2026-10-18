from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


class M5Error(Exception):
    """Base exception for M5 regime-modeling failures."""


class M5ValidationError(M5Error):
    """Raised when an M5 artifact or feature matrix violates the contract."""


class M5ModelUnavailableError(M5Error):
    """Raised when an M5 artifact is unavailable for assignment."""


class M5AvailabilityMode(str, Enum):
    AS_OF_PRODUCTION = "AS_OF_PRODUCTION"
    HISTORICAL_REPLAY = "HISTORICAL_REPLAY"


class M5AssignmentStatus(str, Enum):
    ASSIGNED = "assigned"
    ABSTAINED = "abstained"


M5_MODEL_CONTRACT_VERSION = "m5v1"


def _scientific_identity_payload(artifact: "M5ModelArtifact") -> dict[str, Any]:
    payload = {
        "model_contract_version": artifact.model_contract_version,
        "instrument": artifact.instrument,
        "requested_feature_ids": list(artifact.requested_feature_ids),
        "effective_feature_ids": list(artifact.effective_feature_ids),
        "dropped_constant_feature_ids": list(artifact.dropped_constant_feature_ids),
        "feature_order": list(artifact.feature_order),
        "feature_spec_hash": artifact.feature_spec_hash,
        "training_input_digest": artifact.training_input_digest,
        "training_interval": [artifact.training_interval[0].isoformat(), artifact.training_interval[1].isoformat()],
        "validation_interval": [artifact.validation_interval[0].isoformat(), artifact.validation_interval[1].isoformat()] if artifact.validation_interval else None,
        "test_interval": [artifact.test_interval[0].isoformat(), artifact.test_interval[1].isoformat()] if artifact.test_interval else None,
        "training_cutoff": artifact.training_cutoff.isoformat() if artifact.training_cutoff is not None else None,
        "training_information_cutoff": artifact.training_information_cutoff.isoformat() if artifact.training_information_cutoff is not None else None,
        "requested_component_count": artifact.requested_component_count,
        "effective_component_count": artifact.effective_component_count,
        "covariance_type": artifact.covariance_type,
        "random_state": artifact.random_state,
        "n_init": artifact.n_init,
        "max_iter": artifact.max_iter,
        "tol": artifact.tol,
        "reg_covar": artifact.reg_covar,
        "means": list(artifact.means),
        "diagonal_covariances": list(artifact.diagonal_covariances),
        "weights": list(artifact.weights),
        "hard_occupancy": dict(artifact.hard_occupancy),
        "effective_occupancy": dict(artifact.effective_occupancy),
        "availability_mode": artifact.availability_mode.value,
    }
    return payload


def _canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _canonicalize(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_canonicalize(v) for v in value.tolist()]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def canonical_json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(_canonicalize(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


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
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


@dataclass(frozen=True, slots=True)
class M5ModelArtifact:
    model_contract_version: str = M5_MODEL_CONTRACT_VERSION
    instrument: str = ""
    requested_feature_ids: tuple[str, ...] = ()
    effective_feature_ids: tuple[str, ...] = ()
    dropped_constant_feature_ids: tuple[str, ...] = ()
    feature_order: tuple[str, ...] = ()
    feature_provenance: dict[str, Any] = field(default_factory=dict)
    feature_spec_hash: str | None = None
    scaler_mean: tuple[float, ...] = ()
    scaler_scale: tuple[float, ...] = ()
    training_interval: tuple[pd.Timestamp, pd.Timestamp] = field(
        default_factory=lambda: (pd.Timestamp.min.tz_localize("UTC"), pd.Timestamp.max.tz_localize("UTC"))
    )
    validation_interval: tuple[pd.Timestamp, pd.Timestamp] | None = None
    test_interval: tuple[pd.Timestamp, pd.Timestamp] | None = None
    training_cutoff: pd.Timestamp | None = None
    training_information_cutoff: pd.Timestamp | None = None
    requested_component_count: int = 0
    effective_component_count: int = 0
    covariance_type: str = "diag"
    random_state: int = 42
    n_init: int = 10
    max_iter: int = 500
    tol: float = 1e-4
    reg_covar: float = 1e-6
    means: tuple[tuple[float, ...], ...] = ()
    diagonal_covariances: tuple[tuple[float, ...], ...] = ()
    weights: tuple[float, ...] = ()
    hard_occupancy: dict[str, int] = field(default_factory=dict)
    effective_occupancy: dict[str, float] = field(default_factory=dict)
    converged: bool = True
    diagnostics: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)
    availability_mode: M5AvailabilityMode = M5AvailabilityMode.AS_OF_PRODUCTION
    actual_creation_timestamp_utc: pd.Timestamp = field(default_factory=lambda: pd.Timestamp(datetime.now(timezone.utc)))
    simulated_replay_availability: pd.Timestamp | None = None
    effective_model_availability: pd.Timestamp | None = None
    training_input_digest: str = ""
    model_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "requested_feature_ids", tuple(str(value) for value in self.requested_feature_ids))
        object.__setattr__(self, "effective_feature_ids", tuple(str(value) for value in self.effective_feature_ids))
        object.__setattr__(self, "dropped_constant_feature_ids", tuple(str(value) for value in self.dropped_constant_feature_ids))
        object.__setattr__(self, "feature_order", tuple(str(value) for value in self.feature_order))
        object.__setattr__(self, "feature_provenance", _freeze_identity_value(dict(self.feature_provenance)))
        object.__setattr__(self, "scaler_mean", tuple(float(v) for v in self.scaler_mean))
        object.__setattr__(self, "scaler_scale", tuple(float(v) for v in self.scaler_scale))
        object.__setattr__(self, "means", tuple(tuple(float(v) for v in row) for row in self.means))
        object.__setattr__(self, "diagonal_covariances", tuple(tuple(float(v) for v in row) for row in self.diagonal_covariances))
        object.__setattr__(self, "weights", tuple(float(v) for v in self.weights))
        object.__setattr__(self, "hard_occupancy", {str(k): int(v) for k, v in self.hard_occupancy.items()})
        object.__setattr__(self, "effective_occupancy", {str(k): float(v) for k, v in self.effective_occupancy.items()})
        object.__setattr__(self, "diagnostics", _freeze_identity_value(dict(self.diagnostics)))
        object.__setattr__(self, "warnings", tuple(str(v) for v in self.warnings))
        object.__setattr__(self, "provenance", _freeze_identity_value(dict(self.provenance)))
        object.__setattr__(self, "model_id", self._compute_model_id())

    def _compute_model_id(self) -> str:
        payload = _scientific_identity_payload(self)
        digest = hashlib.sha256(canonical_json_dumps(payload).encode("utf-8")).hexdigest()
        return f"m5v1_sha256_{digest}"


def _coerce_utc_timestamp(value: Any, *, allow_null: bool) -> pd.Timestamp | None:
    if value is None or value is pd.NaT or pd.isna(value):
        if allow_null:
            return None
        raise M5ValidationError("timestamp field cannot be null")
    ts = pd.Timestamp(value)
    if ts.tzinfo is None or ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _as_float_list(value: Any, *, name: str) -> list[float]:
    if isinstance(value, np.ndarray):
        values = value.tolist()
    elif isinstance(value, tuple):
        values = list(value)
    elif isinstance(value, list):
        values = value
    else:
        raise M5ValidationError(f"{name} must be list-like")
    result: list[float] = []
    for item in values:
        try:
            numeric = float(item)
        except (TypeError, ValueError) as exc:
            raise M5ValidationError(f"{name} contains a non-numeric value") from exc
        if not math.isfinite(numeric):
            raise M5ValidationError(f"{name} contains a non-finite value")
        result.append(numeric)
    return result

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

M4ReportVersion = Literal["m4v1"]
M4EvaluationMode = Literal["AS_OF_PRODUCTION", "HISTORICAL_REPLAY"]


@dataclass(frozen=True, slots=True)
class M4AssignmentStatus:
    assignment_status: str
    cluster_id: int | None
    observation_timestamp_utc: pd.Timestamp
    abstention_reason: str | None = None


@dataclass(frozen=True, slots=True)
class M4Coverage:
    requested_count: int
    assigned_count: int
    abstained_count: int
    assigned_share: float | None = None
    abstained_share: float | None = None


@dataclass(frozen=True, slots=True)
class M4ClusterOccupancy:
    cluster_id: int
    assigned_count: int
    assigned_share: float | None = None


@dataclass(frozen=True, slots=True)
class M4CentroidDiagnostics:
    cluster_id: int
    mean_assigned_centroid_distance: float | None = None
    centroid_margin_mean: float | None = None
    observed_cluster_mean_displacement: float | None = None
    undefined_reason: str | None = None


@dataclass(frozen=True, slots=True)
class M4TransitionEdge:
    source_cluster: int
    target_cluster: int
    count: int


@dataclass(frozen=True, slots=True)
class M4TransitionProbability:
    source_cluster: int
    target_cluster: int
    probability: float | None = None
    undefined_reason: str | None = None


@dataclass(frozen=True, slots=True)
class M4OverallPersistence:
    overall_persistence: float | None = None
    undefined_reason: str | None = None


@dataclass(frozen=True, slots=True)
class M4UndefinedMetric:
    metric_name: str
    reason: str


@dataclass(frozen=True, slots=True)
class M4RegimeStabilityReport:
    report_version: str = "m4v1"
    metric_version: str = "m4v1"
    hash_prefix: str = "sha256:"
    report_id: str = ""
    model_id: str = ""
    instrument: str = ""
    evaluation_mode: str = "AS_OF_PRODUCTION"
    decision_timestamp_utc: pd.Timestamp | None = None
    evaluation_window_start_utc: pd.Timestamp | None = None
    evaluation_window_end_utc: pd.Timestamp | None = None
    model_training_cutoff_utc: pd.Timestamp | None = None
    simulated_replay_availability_utc: pd.Timestamp | None = None
    effective_feature_ids: tuple[str, ...] = ()
    effective_feature_vectors: tuple[tuple[float, ...], ...] = ()
    observation_timestamps: tuple[pd.Timestamp, ...] = ()
    feature_vector_available_at: tuple[pd.Timestamp, ...] = ()
    assignment_statuses: tuple[M4AssignmentStatus, ...] = ()
    coverage: M4Coverage | None = None
    cluster_occupancy: tuple[M4ClusterOccupancy, ...] = ()
    centroid_diagnostics: tuple[M4CentroidDiagnostics, ...] = ()
    cluster_self_transition_probability: tuple[M4TransitionProbability, ...] = ()
    transition_counts: tuple[M4TransitionEdge, ...] = ()
    transition_probabilities: tuple[M4TransitionProbability, ...] = ()
    occupancy_entropy: float | None = None
    overall_persistence: M4OverallPersistence | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    undefined_metrics: tuple[M4UndefinedMetric, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "report_version": self.report_version,
            "metric_version": self.metric_version,
            "hash_prefix": self.hash_prefix,
            "report_id": self.report_id,
            "model_id": self.model_id,
            "instrument": self.instrument,
            "evaluation_mode": self.evaluation_mode,
            "decision_timestamp_utc": self.decision_timestamp_utc.isoformat() if self.decision_timestamp_utc is not None else None,
            "evaluation_window_start_utc": self.evaluation_window_start_utc.isoformat() if self.evaluation_window_start_utc is not None else None,
            "evaluation_window_end_utc": self.evaluation_window_end_utc.isoformat() if self.evaluation_window_end_utc is not None else None,
            "model_training_cutoff_utc": self.model_training_cutoff_utc.isoformat() if self.model_training_cutoff_utc is not None else None,
            "simulated_replay_availability_utc": self.simulated_replay_availability_utc.isoformat() if self.simulated_replay_availability_utc is not None else None,
            "effective_feature_ids": list(self.effective_feature_ids),
            "effective_feature_vectors": [list(vector) for vector in self.effective_feature_vectors],
            "observation_timestamps": [ts.isoformat() for ts in self.observation_timestamps],
            "feature_vector_available_at": [ts.isoformat() for ts in self.feature_vector_available_at],
            "assignment_statuses": [
                {
                    "assignment_status": item.assignment_status,
                    "cluster_id": item.cluster_id,
                    "observation_timestamp_utc": item.observation_timestamp_utc.isoformat(),
                    "abstention_reason": item.abstention_reason,
                }
                for item in self.assignment_statuses
            ],
            "coverage": {
                "requested_count": self.coverage.requested_count if self.coverage is not None else None,
                "assigned_count": self.coverage.assigned_count if self.coverage is not None else None,
                "abstained_count": self.coverage.abstained_count if self.coverage is not None else None,
                "assigned_share": self.coverage.assigned_share if self.coverage is not None else None,
                "abstained_share": self.coverage.abstained_share if self.coverage is not None else None,
            },
            "cluster_occupancy": [
                {
                    "cluster_id": item.cluster_id,
                    "assigned_count": item.assigned_count,
                    "assigned_share": item.assigned_share,
                }
                for item in self.cluster_occupancy
            ],
            "centroid_diagnostics": [
                {
                    "cluster_id": item.cluster_id,
                    "mean_assigned_centroid_distance": item.mean_assigned_centroid_distance,
                    "centroid_margin_mean": item.centroid_margin_mean,
                    "observed_cluster_mean_displacement": item.observed_cluster_mean_displacement,
                    "undefined_reason": item.undefined_reason,
                }
                for item in self.centroid_diagnostics
            ],
            "cluster_self_transition_probability": [
                {
                    "source_cluster": item.source_cluster,
                    "target_cluster": item.target_cluster,
                    "probability": item.probability,
                    "undefined_reason": item.undefined_reason,
                }
                for item in self.cluster_self_transition_probability
            ],
            "transition_counts": [
                {"source_cluster": item.source_cluster, "target_cluster": item.target_cluster, "count": item.count}
                for item in self.transition_counts
            ],
            "transition_probabilities": [
                {
                    "source_cluster": item.source_cluster,
                    "target_cluster": item.target_cluster,
                    "probability": item.probability,
                    "undefined_reason": item.undefined_reason,
                }
                for item in self.transition_probabilities
            ],
            "occupancy_entropy": self.occupancy_entropy,
            "overall_persistence": {
                "overall_persistence": self.overall_persistence.overall_persistence if self.overall_persistence is not None else None,
                "undefined_reason": self.overall_persistence.undefined_reason if self.overall_persistence is not None else None,
            },
            "provenance": self.provenance,
            "undefined_metrics": [
                {"metric_name": item.metric_name, "reason": item.reason}
                for item in self.undefined_metrics
            ],
        }
        return payload

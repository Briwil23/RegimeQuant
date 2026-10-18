from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from regimequant.regimes.assignment import assign_m3_regimes
from regimequant.regimes.contracts import M3AvailabilityMode, M3ModelArtifact

from .contracts import (
    M4AssignmentStatus,
    M4CentroidDiagnostics,
    M4ClusterOccupancy,
    M4Coverage,
    M4OverallPersistence,
    M4RegimeStabilityReport,
    M4TransitionEdge,
    M4TransitionProbability,
    M4UndefinedMetric,
)
from .identity import compute_m4_report_identity


def _normalize_utc(ts: Any) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None or ts.tz is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _safe_float(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if np.isnan(value) or np.isinf(value):
        return None
    return float(np.round(value, 12))


def compute_requested_domain(
    feature_matrix: pd.DataFrame,
    *,
    instrument: str,
    evaluation_window_start_utc: pd.Timestamp,
    evaluation_window_end_utc: pd.Timestamp,
) -> pd.DataFrame:
    if feature_matrix is None or feature_matrix.empty:
        return pd.DataFrame(columns=feature_matrix.columns if feature_matrix is not None else [])
    matrix = feature_matrix.copy()
    if "scope_id" not in matrix.columns or "timestamp" not in matrix.columns:
        raise ValueError("feature matrix must include scope_id and timestamp")
    matrix["timestamp"] = pd.to_datetime(matrix["timestamp"], errors="coerce")
    matrix["timestamp"] = [
        ts.tz_localize("UTC") if ts.tzinfo is None or ts.tz is None else ts.tz_convert("UTC") for ts in matrix["timestamp"]
    ]
    start = _normalize_utc(evaluation_window_start_utc)
    end = _normalize_utc(evaluation_window_end_utc)
    result = matrix[(matrix["scope_id"] == instrument) & (matrix["timestamp"] >= start) & (matrix["timestamp"] < end)].copy()
    return result.sort_values(["timestamp"], kind="mergesort").reset_index(drop=True)


def _build_assignment_statuses(assignments: pd.DataFrame) -> tuple[M4AssignmentStatus, ...]:
    statuses: list[M4AssignmentStatus] = []
    for _, row in assignments.sort_values(["observation_timestamp"], kind="mergesort").iterrows():
        status = str(row.get("assignment_status", "abstained"))
        cluster_id = row.get("cluster_id")
        ts = _normalize_utc(row.get("observation_timestamp"))
        statuses.append(
            M4AssignmentStatus(
                assignment_status=status, 
                cluster_id=None if pd.isna(cluster_id) else int(cluster_id),
                observation_timestamp_utc=ts,
                abstention_reason=row.get("abstention_reason"),
            )
        )
    return tuple(statuses)


def _cluster_occupancy(assignments: pd.DataFrame, n_clusters: int) -> tuple[M4ClusterOccupancy, ...]:
    assigned = assignments[assignments["assignment_status"] == "assigned"].copy()
    total = len(assigned)
    occupancy: list[M4ClusterOccupancy] = []
    counts = {idx: 0 for idx in range(n_clusters)}
    for _, row in assigned.iterrows():
        cluster_id = int(row["cluster_id"])
        counts[cluster_id] = counts.get(cluster_id, 0) + 1
    for cluster_id in range(n_clusters):
        count = counts.get(cluster_id, 0)
        share = None if total == 0 else count / total
        occupancy.append(M4ClusterOccupancy(cluster_id=cluster_id, assigned_count=count, assigned_share=_safe_float(share)))
    return tuple(occupancy)


def _transition_metrics(assignments: pd.DataFrame) -> tuple[dict[int, int], tuple[M4TransitionEdge, ...], tuple[M4TransitionProbability, ...], list[M4UndefinedMetric]]:
    ordered = assignments.sort_values(["observation_timestamp"], kind="mergesort").reset_index(drop=True)
    counts: dict[tuple[int, int], int] = defaultdict(int)
    valid_edges = 0
    previous: dict[str, Any] | None = None
    for _, row in ordered.iterrows():
        current = row
        if previous is not None and previous.get("assignment_status") == "assigned" and current.get("assignment_status") == "assigned":
            source_cluster = int(previous["cluster_id"])
            target_cluster = int(current["cluster_id"])
            counts[(source_cluster, target_cluster)] += 1
            valid_edges += 1
        previous = current
    edges = tuple(M4TransitionEdge(source_cluster=src, target_cluster=tgt, count=count) for (src, tgt), count in sorted(counts.items()))
    probs: list[M4TransitionProbability] = []
    undefined: list[M4UndefinedMetric] = []
    outgoing_totals: dict[int, int] = defaultdict(int)
    for edge in edges:
        outgoing_totals[edge.source_cluster] += edge.count
    for src in sorted({edge.source_cluster for edge in edges}):
        total_out = outgoing_totals.get(src, 0)
        for edge in edges:
            if edge.source_cluster != src:
                continue
            prob = None if total_out == 0 else edge.count / total_out
            probs.append(M4TransitionProbability(source_cluster=src, target_cluster=edge.target_cluster, probability=_safe_float(prob), undefined_reason=None if prob is not None else "zero_outgoing_total"))
        if total_out == 0:
            undefined.append(M4UndefinedMetric(metric_name=f"transition_probabilities_source_{src}", reason="zero_outgoing_total"))
    return dict((pair, counts[pair]) for pair in counts.keys()), edges, tuple(probs), undefined


def _occupancy_entropy(occupancy: tuple[M4ClusterOccupancy, ...]) -> tuple[float | None, list[M4UndefinedMetric]]:
    assigned_total = sum(item.assigned_count for item in occupancy)
    if assigned_total <= 0:
        return None, [M4UndefinedMetric(metric_name="occupancy_entropy", reason="assigned_total_is_zero")]
    entropy = 0.0
    for item in occupancy:
        if item.assigned_count <= 0:
            continue
        p = item.assigned_count / assigned_total
        entropy -= p * np.log2(p)
    return _safe_float(entropy), []


def _centroid_diagnostics(
    assignments: pd.DataFrame,
    artifact: M3ModelArtifact,
    n_clusters: int,
) -> tuple[tuple[M4CentroidDiagnostics, ...], list[M4UndefinedMetric]]:
    centroids = np.asarray(artifact.canonical_centroids, dtype=float)
    scaler_mean = np.asarray(artifact.scaler_parameters["mean_"], dtype=float)
    scaler_scale = np.asarray(artifact.scaler_parameters["scale_"], dtype=float)
    selected = assignments[assignments["assignment_status"] == "assigned"].copy()
    diagnostics: list[M4CentroidDiagnostics] = []
    undefined: list[M4UndefinedMetric] = []
    feature_columns = list(artifact.effective_feature_instance_ids)
    data = []
    for _, row in selected.iterrows():
        values = []
        for feature in feature_columns:
            values.append(float(row[feature]))
        data.append(np.asarray(values, dtype=float))
    if not data:
        for cluster_id in range(n_clusters):
            diagnostics.append(M4CentroidDiagnostics(cluster_id=cluster_id, mean_assigned_centroid_distance=None, centroid_margin_mean=None, observed_cluster_mean_displacement=None, undefined_reason="no_assigned_observations"))
            undefined.append(M4UndefinedMetric(metric_name=f"centroid_diagnostics_cluster_{cluster_id}", reason="no_assigned_observations"))
        return tuple(diagnostics), undefined
    X = np.asarray(data, dtype=float)
    standardized = (X - scaler_mean) / scaler_scale
    for cluster_id in range(n_clusters):
        cluster_rows = selected[selected["cluster_id"] == cluster_id].copy()
        if cluster_rows.empty:
            diagnostics.append(
                M4CentroidDiagnostics(
                    cluster_id=cluster_id,
                    mean_assigned_centroid_distance=None,
                    centroid_margin_mean=None,
                    observed_cluster_mean_displacement=None,
                    undefined_reason="cluster_has_no_assigned_observations",
                )
            )
            undefined.append(M4UndefinedMetric(metric_name=f"centroid_diagnostics_cluster_{cluster_id}", reason="cluster_has_no_assigned_observations"))
            continue
        cluster_values = (
            cluster_rows[feature_columns].to_numpy(dtype=float) - scaler_mean
        ) / scaler_scale
        if cluster_values.size == 0:
            cluster_values = np.asarray([[]], dtype=float)
        centroid = centroids[cluster_id]
        distances = np.linalg.norm(cluster_values - centroid, axis=1)
        mean_dist = float(np.mean(distances)) if len(distances) else None
        empirical_mean = np.mean(cluster_values, axis=0) if len(cluster_values) else None
        mean_displacement = None if empirical_mean is None else float(np.linalg.norm(empirical_mean - centroid))
        if n_clusters <= 1:
            centroid_margin = None
            margin_reason = "k_equals_1_has_no_alternative_cluster"
        else:
            margin_values = []
            for observation in cluster_values:
                alt_distances = [float(np.linalg.norm(observation - centroids[j])) for j in range(n_clusters) if j != cluster_id]
                assigned_dist = float(np.linalg.norm(observation - centroid))
                if not alt_distances:
                    margin_reason = "no_alternative_cluster_available"
                    margin_values = []
                    break
                margin_values.append(min(alt_distances) - assigned_dist)
            if margin_values:
                centroid_margin = float(np.mean(margin_values))
                margin_reason = None
            else:
                centroid_margin = None
                margin_reason = "no_alternative_cluster_available"
        if centroid_margin is None:
            undefined.append(M4UndefinedMetric(metric_name=f"centroid_margin_mean_cluster_{cluster_id}", reason=margin_reason or "undefined"))
        if mean_dist is None:
            undefined.append(M4UndefinedMetric(metric_name=f"mean_assigned_centroid_distance_cluster_{cluster_id}", reason="no_assigned_observations"))
        if mean_displacement is None:
            undefined.append(M4UndefinedMetric(metric_name=f"observed_cluster_mean_displacement_cluster_{cluster_id}", reason="no_assigned_observations"))
        diagnostics.append(
            M4CentroidDiagnostics(
                cluster_id=cluster_id,
                mean_assigned_centroid_distance=_safe_float(mean_dist),
                centroid_margin_mean=_safe_float(centroid_margin),
                observed_cluster_mean_displacement=_safe_float(mean_displacement),
                undefined_reason=None if centroid_margin is not None and mean_dist is not None and mean_displacement is not None else margin_reason,
            )
        )
    return tuple(diagnostics), undefined


def compute_m4_regime_stability_report(
    feature_matrix: pd.DataFrame,
    artifact: M3ModelArtifact,
    *,
    instrument: str,
    evaluation_window_start_utc: pd.Timestamp,
    evaluation_window_end_utc: pd.Timestamp,
    decision_timestamp_utc: pd.Timestamp,
    evaluation_mode: str = "AS_OF_PRODUCTION",
    simulated_replay_availability_utc: pd.Timestamp | None = None,
    model_training_cutoff_utc: pd.Timestamp | None = None,
) -> M4RegimeStabilityReport:
    if feature_matrix is None or feature_matrix.empty:
        raise ValueError("feature_matrix must not be empty")
    requested = compute_requested_domain(
        feature_matrix,
        instrument=instrument,
        evaluation_window_start_utc=evaluation_window_start_utc,
        evaluation_window_end_utc=evaluation_window_end_utc,
    )
    if evaluation_mode not in {"AS_OF_PRODUCTION", "HISTORICAL_REPLAY"}:
        raise ValueError("evaluation_mode must be AS_OF_PRODUCTION or HISTORICAL_REPLAY")
    assignment_mode = artifact.assignment_mode
    assignments = assign_m3_regimes(
        artifact,
        feature_matrix,
        decision_time=decision_timestamp_utc,
        assignment_mode=assignment_mode,
    )
    window_assignments = assignments[
        (assignments["instrument"] == instrument)
        & (assignments["observation_timestamp"] >= _normalize_utc(evaluation_window_start_utc))
        & (assignments["observation_timestamp"] < _normalize_utc(evaluation_window_end_utc))
    ].copy()
    if window_assignments.empty:
        window_assignments = requested.rename(columns={"scope_id": "instrument", "timestamp": "observation_timestamp"}).copy()
        window_assignments["assignment_status"] = "abstained"
        window_assignments["cluster_id"] = None
        window_assignments["abstention_reason"] = "no_assignment_record_in_requested_domain"
        window_assignments["decision_timestamp"] = decision_timestamp_utc
        window_assignments["model_id"] = artifact.model_id
        window_assignments["feature_spec_set_hash"] = artifact.feature_spec_set_hash
        window_assignments["model_training_cutoff"] = artifact.training_cutoff
    else:
        feature_lookup = requested.loc[:, ["scope_id", "timestamp", *list(artifact.effective_feature_instance_ids)]].copy()
        feature_lookup = feature_lookup.rename(columns={"scope_id": "instrument", "timestamp": "observation_timestamp"})
        feature_lookup["observation_timestamp"] = pd.to_datetime(feature_lookup["observation_timestamp"], utc=True)
        window_assignments = window_assignments.merge(feature_lookup, on=["instrument", "observation_timestamp"], how="left")
    window_assignments = window_assignments.sort_values(["observation_timestamp"], kind="mergesort").reset_index(drop=True)
    requested_count = len(window_assignments)
    assigned_count = int((window_assignments["assignment_status"] == "assigned").sum())
    abstained_count = requested_count - assigned_count
    assigned_share = None if requested_count == 0 else assigned_count / requested_count
    abstained_share = None if requested_count == 0 else abstained_count / requested_count
    coverage = M4Coverage(
        requested_count=requested_count,
        assigned_count=assigned_count,
        abstained_count=abstained_count,
        assigned_share=_safe_float(assigned_share),
        abstained_share=_safe_float(abstained_share),
    )
    assignment_statuses = _build_assignment_statuses(window_assignments)
    n_clusters = len(artifact.canonical_centroids)
    cluster_occupancy = _cluster_occupancy(window_assignments, n_clusters)
    centroid_diagnostics, cent_diag_undefined = _centroid_diagnostics(window_assignments, artifact, n_clusters)
    transition_counts_map, transition_edges, transition_probs, transition_undefined = _transition_metrics(window_assignments)
    cluster_self_probabilities: list[M4TransitionProbability] = []
    for cluster_id in range(n_clusters):
        outgoing_total = sum(edge.count for edge in transition_edges if edge.source_cluster == cluster_id)
        if outgoing_total == 0:
            cluster_self_probabilities.append(
                M4TransitionProbability(source_cluster=cluster_id, target_cluster=cluster_id, probability=None, undefined_reason="zero_outgoing_total")
            )
        else:
            self_count = transition_counts_map.get((cluster_id, cluster_id), 0)
            probability = self_count / outgoing_total
            cluster_self_probabilities.append(
                M4TransitionProbability(source_cluster=cluster_id, target_cluster=cluster_id, probability=_safe_float(probability), undefined_reason=None)
            )
    total_valid_edges = sum(edge.count for edge in transition_edges)
    persistence_value = None if total_valid_edges == 0 else sum(edge.count for edge in transition_edges if edge.source_cluster == edge.target_cluster) / total_valid_edges
    persistence = M4OverallPersistence(overall_persistence=_safe_float(persistence_value), undefined_reason=None if persistence_value is not None else "no_valid_adjacent_edges")
    occupancy_entropy, entropy_undefined = _occupancy_entropy(cluster_occupancy)
    undefined_metrics = tuple(
        [
            *cent_diag_undefined,
            *transition_undefined,
            *entropy_undefined,
            *( [M4UndefinedMetric(metric_name="coverage_assigned_share", reason="requested_count_is_zero")] if requested_count == 0 else []),
            *( [M4UndefinedMetric(metric_name="coverage_abstained_share", reason="requested_count_is_zero")] if requested_count == 0 else []),
        ]
    )
    report = M4RegimeStabilityReport(
        report_version="m4v1",
        metric_version="m4v1",
        hash_prefix="sha256:",
        report_id="",
        model_id=artifact.model_id,
        instrument=instrument,
        evaluation_mode=evaluation_mode,
        decision_timestamp_utc=_normalize_utc(decision_timestamp_utc),
        evaluation_window_start_utc=_normalize_utc(evaluation_window_start_utc),
        evaluation_window_end_utc=_normalize_utc(evaluation_window_end_utc),
        model_training_cutoff_utc=_normalize_utc(model_training_cutoff_utc) if model_training_cutoff_utc is not None else artifact.training_cutoff,
        simulated_replay_availability_utc=_normalize_utc(simulated_replay_availability_utc) if simulated_replay_availability_utc is not None else artifact.simulated_replay_availability,
        effective_feature_ids=tuple(artifact.effective_feature_instance_ids),
        effective_feature_vectors=(
            tuple(tuple(float(v) for v in row) for row in requested.loc[:, list(artifact.effective_feature_instance_ids)].dropna().to_numpy().tolist())
            if not requested.empty
            else ()
        ),
        observation_timestamps=tuple(_normalize_utc(ts) for ts in requested["timestamp"].tolist()) if not requested.empty else (),
        feature_vector_available_at=tuple(_normalize_utc(ts) for ts in requested["feature_vector_available_at"].dropna().tolist()) if ("feature_vector_available_at" in requested.columns and not requested.empty) else (),
        assignment_statuses=assignment_statuses,
        coverage=coverage,
        cluster_occupancy=cluster_occupancy,
        centroid_diagnostics=centroid_diagnostics,
        cluster_self_transition_probability=tuple(cluster_self_probabilities),
        transition_counts=transition_edges,
        transition_probabilities=tuple(transition_probs),
        occupancy_entropy=occupancy_entropy,
        overall_persistence=persistence,
        provenance={
            "feature_spec_set_hash": artifact.feature_spec_set_hash,
            "parent_provenance_digest": artifact.parent_provenance_digest,
            "return_type": artifact.return_type,
            "price_type": artifact.price_type,
            "adjustment_state": artifact.adjustment_state,
            "assignment_mode": artifact.assignment_mode.value,
        },
        undefined_metrics=tuple(undefined_metrics),
    )
    object.__setattr__(report, "report_id", compute_m4_report_identity(report))
    return report

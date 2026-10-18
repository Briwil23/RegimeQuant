from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regimequant.regimes import build_m3_feature_matrix, fit_m3_regime_model
from regimequant.regimes.contracts import M3AvailabilityMode, M3ModelArtifact
from regimequant.regimes.m4 import compute_m4_regime_stability_report


def _set_historical_availability(model: object, decision_timestamp_utc: pd.Timestamp) -> object:
    decision_ts = pd.Timestamp(decision_timestamp_utc).tz_convert("UTC")
    object.__setattr__(model, "actual_creation_timestamp_utc", decision_ts)
    object.__setattr__(model, "effective_model_availability", decision_ts)
    return model


def _build_feature_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.date_range("2023-01-01", periods=120, freq="D", tz="UTC")
    for idx, ts in enumerate(dates):
        for fid, val in [("f1", 0.01 + idx * 0.002), ("f2", 0.02 + idx * 0.003)]:
            rows.append(
                {
                    "scope_type": "instrument",
                    "scope_id": "AAA",
                    "timestamp": ts,
                    "feature_name": "rolling_mean_return" if fid == "f1" else "rolling_sample_std",
                    "feature_instance_id": fid,
                    "feature_value": val,
                    "math_status": "defined",
                    "undefined_reason": None,
                    "input_availability_status": "verified_timestamp",
                    "available_at": ts,
                    "input_return_type": "simple",
                    "price_type": "close",
                    "adjustment_state": "raw",
                    "window_kind": "observation_count",
                    "window_size": 3,
                    "min_observations": 3,
                    "observation_count_used": 3,
                    "feature_definition_version": "1",
                }
            )
    frame = pd.DataFrame(rows)
    frame.attrs["provenance"] = {
        "feature_spec_set_hash": "hash-1",
        "parent_return_provenance_digest": "digest-1",
        "return_type": "simple",
        "price_type": "close",
        "adjustment_state": "raw",
        "generated_at_utc": pd.Timestamp("2023-01-01", tz="UTC"),
        "metadata_durability": "DataFrame.attrs is in-memory only and not a durable serialization contract",
    }
    return frame


def test_m4_metrics_respect_requested_domain_and_transition_boundaries() -> None:
    frame = _build_feature_frame()
    matrix = build_m3_feature_matrix(frame, requested_feature_instance_ids=["f1", "f2"])
    model = fit_m3_regime_model(
        matrix,
        instrument="AAA",
        requested_feature_instance_ids=["f1", "f2"],
        training_cutoff=pd.Timestamp("2023-02-20", tz="UTC"),
        training_interval=(pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2023-01-31", tz="UTC")),
        validation_interval=(pd.Timestamp("2023-02-01", tz="UTC"), pd.Timestamp("2023-02-21", tz="UTC")),
        test_interval=(pd.Timestamp("2023-02-21", tz="UTC"), pd.Timestamp("2023-03-13", tz="UTC")),
        training_information_cutoff=pd.Timestamp("2023-02-20", tz="UTC"),
        validation_information_cutoff=pd.Timestamp("2023-02-21", tz="UTC"),
        test_information_cutoff=pd.Timestamp("2023-03-13", tz="UTC"),
        feature_spec_set_hash="hash-1",
        parent_provenance_digest="digest-1",
        return_type="simple",
        price_type="close",
        adjustment_state="raw",
        n_clusters=2,
    )
    model = _set_historical_availability(model, pd.Timestamp("2023-02-20", tz="UTC"))
    report = compute_m4_regime_stability_report(
        matrix,
        model,
        instrument="AAA",
        evaluation_window_start_utc=pd.Timestamp("2023-01-01", tz="UTC"),
        evaluation_window_end_utc=pd.Timestamp("2023-02-01", tz="UTC"),
        decision_timestamp_utc=pd.Timestamp("2023-02-20", tz="UTC"),
        evaluation_mode="AS_OF_PRODUCTION",
    )
    assert report.coverage.requested_count == report.coverage.assigned_count + report.coverage.abstained_count
    assert report.transition_probabilities
    assert report.occupancy_entropy is not None
    assert any(item.metric_name == "occupancy_entropy" for item in report.undefined_metrics) is False


def test_m4_observed_cluster_mean_displacement_tracks_frozen_centroid() -> None:
    frame = _build_feature_frame()
    matrix = build_m3_feature_matrix(frame, requested_feature_instance_ids=["f1", "f2"])
    model = fit_m3_regime_model(
        matrix,
        instrument="AAA",
        requested_feature_instance_ids=["f1", "f2"],
        training_cutoff=pd.Timestamp("2023-02-20", tz="UTC"),
        training_interval=(pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2023-01-31", tz="UTC")),
        validation_interval=(pd.Timestamp("2023-02-01", tz="UTC"), pd.Timestamp("2023-02-21", tz="UTC")),
        test_interval=(pd.Timestamp("2023-02-21", tz="UTC"), pd.Timestamp("2023-03-13", tz="UTC")),
        training_information_cutoff=pd.Timestamp("2023-02-20", tz="UTC"),
        validation_information_cutoff=pd.Timestamp("2023-02-21", tz="UTC"),
        test_information_cutoff=pd.Timestamp("2023-03-13", tz="UTC"),
        feature_spec_set_hash="hash-1",
        parent_provenance_digest="digest-1",
        return_type="simple",
        price_type="close",
        adjustment_state="raw",
        n_clusters=2,
    )
    model = _set_historical_availability(model, pd.Timestamp("2023-02-20", tz="UTC"))
    report = compute_m4_regime_stability_report(
        matrix,
        model,
        instrument="AAA",
        evaluation_window_start_utc=pd.Timestamp("2023-01-01", tz="UTC"),
        evaluation_window_end_utc=pd.Timestamp("2023-02-15", tz="UTC"),
        decision_timestamp_utc=pd.Timestamp("2023-02-20", tz="UTC"),
        evaluation_mode="AS_OF_PRODUCTION",
    )
    for diag in report.centroid_diagnostics:
        if diag.undefined_reason is not None:
            continue
        assert diag.observed_cluster_mean_displacement is not None
        assert np.isfinite(diag.observed_cluster_mean_displacement)
        assert diag.observed_cluster_mean_displacement >= 0.0


def _build_exact_two_cluster_fixture() -> tuple[pd.DataFrame, M3ModelArtifact]:
    timestamps = pd.date_range("2023-01-01", periods=3, freq="D", tz="UTC")
    matrix = pd.DataFrame(
        {
            "scope_id": ["AAA"] * 3,
            "timestamp": timestamps,
            "x": [-1.0, 1.0, 10.0],
            "feature_vector_available_at": timestamps,
            "feature_observation_available": [True, True, True],
        }
    )
    matrix.attrs["provenance"] = {
        "feature_spec_set_hash": "feature-spec-1",
        "parent_return_provenance_digest": "parent-digest-1",
        "parent_provenance_digest": "parent-digest-1",
        "return_type": "simple",
        "price_type": "close",
        "adjustment_state": "raw",
        "generated_at_utc": timestamps[0],
        "metadata_durability": "test-fixture",
    }
    artifact = M3ModelArtifact(
        instrument="AAA",
        requested_feature_instance_ids=("x",),
        effective_feature_instance_ids=("x",),
        dropped_constant_feature_ids=(),
        feature_spec_set_hash="feature-spec-1",
        parent_provenance_digest="parent-digest-1",
        return_type="simple",
        price_type="close",
        adjustment_state="raw",
        training_interval=(timestamps[0], timestamps[1]),
        training_cutoff=timestamps[1],
        training_information_cutoff=timestamps[1],
        validation_interval=(timestamps[1], timestamps[2]),
        test_interval=(timestamps[2], timestamps[2] + pd.Timedelta(days=2)),
        kmeans_config={},
        scaler_parameters={"mean_": [0.0], "scale_": [1.0]},
        canonical_centroids=((0.0,), (10.0,)),
        training_occupancy={"0": 2, "1": 1},
        assignment_mode=M3AvailabilityMode.AS_OF_PRODUCTION,
        actual_creation_timestamp_utc=timestamps[1],
        effective_model_availability=timestamps[1],
        seed=42,
    )
    return matrix, artifact


def test_m4_requested_domain_keeps_abstained_observations_in_coverage() -> None:
    frame = _build_feature_frame()
    matrix = build_m3_feature_matrix(frame, requested_feature_instance_ids=["f1", "f2"])
    matrix.loc[(matrix["timestamp"] == pd.Timestamp("2023-02-08", tz="UTC")) & (matrix["f1"].notna()), "feature_observation_available"] = False
    model = fit_m3_regime_model(
        matrix,
        instrument="AAA",
        requested_feature_instance_ids=["f1", "f2"],
        training_cutoff=pd.Timestamp("2023-02-20", tz="UTC"),
        training_interval=(pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2023-02-01", tz="UTC")),
        validation_interval=(pd.Timestamp("2023-02-01", tz="UTC"), pd.Timestamp("2023-03-03", tz="UTC")),
        test_interval=(pd.Timestamp("2023-03-03", tz="UTC"), pd.Timestamp("2023-03-25", tz="UTC")),
        training_information_cutoff=pd.Timestamp("2023-02-20", tz="UTC"),
        validation_information_cutoff=pd.Timestamp("2023-03-03", tz="UTC"),
        test_information_cutoff=pd.Timestamp("2023-03-25", tz="UTC"),
        feature_spec_set_hash="hash-1",
        parent_provenance_digest="digest-1",
        return_type="simple",
        price_type="close",
        adjustment_state="raw",
        n_clusters=2,
    )
    model = _set_historical_availability(model, pd.Timestamp("2023-02-20", tz="UTC"))
    report = compute_m4_regime_stability_report(
        matrix,
        model,
        instrument="AAA",
        evaluation_window_start_utc=pd.Timestamp("2023-01-01", tz="UTC"),
        evaluation_window_end_utc=pd.Timestamp("2023-02-15", tz="UTC"),
        decision_timestamp_utc=pd.Timestamp("2023-02-20", tz="UTC"),
        evaluation_mode="AS_OF_PRODUCTION",
    )
    assert report.coverage.requested_count == report.coverage.assigned_count + report.coverage.abstained_count
    assert report.coverage.abstained_count >= 1


def test_m4_exact_geometry_and_transition_metrics_on_known_two_cluster_fixture() -> None:
    matrix, artifact = _build_exact_two_cluster_fixture()
    decision_ts = pd.Timestamp("2023-01-03", tz="UTC")
    report = compute_m4_regime_stability_report(
        matrix,
        artifact,
        instrument="AAA",
        evaluation_window_start_utc=pd.Timestamp("2023-01-01", tz="UTC"),
        evaluation_window_end_utc=pd.Timestamp("2023-01-05", tz="UTC"),
        decision_timestamp_utc=decision_ts,
        evaluation_mode="AS_OF_PRODUCTION",
    )

    assert report.coverage == report.coverage
    assert report.coverage.requested_count == 3
    assert report.coverage.assigned_count == 3
    assert report.coverage.abstained_count == 0

    cluster_diag = {item.cluster_id: item for item in report.centroid_diagnostics}
    assert cluster_diag[0].mean_assigned_centroid_distance == pytest.approx(1.0)
    assert cluster_diag[0].centroid_margin_mean == pytest.approx(9.0)
    assert cluster_diag[0].observed_cluster_mean_displacement == pytest.approx(0.0)
    assert cluster_diag[1].mean_assigned_centroid_distance == pytest.approx(0.0)
    assert cluster_diag[1].centroid_margin_mean == pytest.approx(10.0)
    assert cluster_diag[1].observed_cluster_mean_displacement == pytest.approx(0.0)

    assert report.occupancy_entropy == pytest.approx(0.9182958340544896)
    assert report.overall_persistence.overall_persistence == pytest.approx(0.5)
    assert report.cluster_self_transition_probability[0].probability == pytest.approx(0.5)
    assert report.cluster_self_transition_probability[1].probability is None
    transition_by_pair = {(item.source_cluster, item.target_cluster): item for item in report.transition_probabilities}
    assert transition_by_pair[(0, 0)].probability == pytest.approx(0.5)
    assert transition_by_pair[(0, 1)].probability == pytest.approx(0.5)
    assert {(edge.source_cluster, edge.target_cluster, edge.count) for edge in report.transition_counts} == {(0, 0, 1), (0, 1, 1)}
    assert sum(edge.count for edge in report.transition_counts) == 2
    assert sum(edge.count for edge in report.transition_counts) == report.coverage.assigned_count - 1


def test_m4_feature_availability_missing_and_late_are_exactly_recorded() -> None:
    matrix, artifact = _build_exact_two_cluster_fixture()
    decision_ts = pd.Timestamp("2023-01-03", tz="UTC")

    missing = matrix.copy()
    missing.loc[missing["timestamp"] == pd.Timestamp("2023-01-03", tz="UTC"), "feature_vector_available_at"] = pd.NaT
    missing_report = compute_m4_regime_stability_report(
        missing,
        artifact,
        instrument="AAA",
        evaluation_window_start_utc=pd.Timestamp("2023-01-01", tz="UTC"),
        evaluation_window_end_utc=pd.Timestamp("2023-01-05", tz="UTC"),
        decision_timestamp_utc=decision_ts,
        evaluation_mode="AS_OF_PRODUCTION",
    )
    assert missing_report.coverage.requested_count == 3
    assert missing_report.coverage.assigned_count == 2
    assert missing_report.coverage.abstained_count == 1
    assert any(
        status.assignment_status == "abstained" and status.abstention_reason == "missing_feature_vector_availability"
        for status in missing_report.assignment_statuses
    )

    late = matrix.copy()
    late.loc[late["timestamp"] == pd.Timestamp("2023-01-03", tz="UTC"), "feature_vector_available_at"] = pd.Timestamp("2023-01-04", tz="UTC")
    late_report = compute_m4_regime_stability_report(
        late,
        artifact,
        instrument="AAA",
        evaluation_window_start_utc=pd.Timestamp("2023-01-01", tz="UTC"),
        evaluation_window_end_utc=pd.Timestamp("2023-01-05", tz="UTC"),
        decision_timestamp_utc=decision_ts,
        evaluation_mode="AS_OF_PRODUCTION",
    )
    assert late_report.coverage.requested_count == 3
    assert late_report.coverage.assigned_count == 2
    assert late_report.coverage.abstained_count == 1
    assert any(
        status.assignment_status == "abstained" and status.abstention_reason == "late_feature_availability"
        for status in late_report.assignment_statuses
    )


def test_m4_single_cluster_margin_is_undefined_with_frozen_reason() -> None:
    timestamps = pd.date_range("2023-01-01", periods=3, freq="D", tz="UTC")
    matrix = pd.DataFrame(
        {
            "scope_id": ["AAA"] * 3,
            "timestamp": timestamps,
            "x": [0.5, 0.6, 0.7],
            "feature_vector_available_at": timestamps,
            "feature_observation_available": [True, True, True],
        }
    )
    matrix.attrs["provenance"] = {
        "feature_spec_set_hash": "feature-spec-1",
        "parent_return_provenance_digest": "parent-digest-1",
        "parent_provenance_digest": "parent-digest-1",
        "return_type": "simple",
        "price_type": "close",
        "adjustment_state": "raw",
        "generated_at_utc": timestamps[0],
        "metadata_durability": "test-fixture",
    }
    artifact = M3ModelArtifact(
        instrument="AAA",
        requested_feature_instance_ids=("x",),
        effective_feature_instance_ids=("x",),
        dropped_constant_feature_ids=(),
        feature_spec_set_hash="feature-spec-1",
        parent_provenance_digest="parent-digest-1",
        return_type="simple",
        price_type="close",
        adjustment_state="raw",
        training_interval=(timestamps[0], timestamps[1]),
        training_cutoff=timestamps[1],
        training_information_cutoff=timestamps[1],
        validation_interval=(timestamps[1], timestamps[2]),
        test_interval=(timestamps[2], timestamps[2] + pd.Timedelta(days=2)),
        kmeans_config={},
        scaler_parameters={"mean_": [0.0], "scale_": [1.0]},
        canonical_centroids=((0.0,),),
        training_occupancy={"0": 3},
        assignment_mode=M3AvailabilityMode.AS_OF_PRODUCTION,
        actual_creation_timestamp_utc=timestamps[1],
        effective_model_availability=timestamps[1],
        seed=42,
    )
    report = compute_m4_regime_stability_report(
        matrix,
        artifact,
        instrument="AAA",
        evaluation_window_start_utc=pd.Timestamp("2023-01-01", tz="UTC"),
        evaluation_window_end_utc=pd.Timestamp("2023-01-05", tz="UTC"),
        decision_timestamp_utc=pd.Timestamp("2023-01-03", tz="UTC"),
        evaluation_mode="AS_OF_PRODUCTION",
    )
    cluster = report.centroid_diagnostics[0]
    assert cluster.centroid_margin_mean is None
    assert cluster.undefined_reason == "k_equals_1_has_no_alternative_cluster"
    assert any(item.metric_name == "centroid_margin_mean_cluster_0" and item.reason == "k_equals_1_has_no_alternative_cluster" for item in report.undefined_metrics)

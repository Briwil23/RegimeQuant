from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pandas as pd

from regimequant.regimes import build_m3_feature_matrix, fit_m3_regime_model
from regimequant.regimes.contracts import M3AvailabilityMode, M3ModelArtifact
from regimequant.regimes.m4 import compute_m4_regime_stability_report, compute_m4_report_identity
from regimequant.regimes.m4.contracts import M4Coverage


def _set_historical_availability(model: object, decision_timestamp_utc: pd.Timestamp) -> object:
    decision_ts = pd.Timestamp(decision_timestamp_utc).tz_convert("UTC")
    object.__setattr__(model, "actual_creation_timestamp_utc", decision_ts)
    object.__setattr__(model, "effective_model_availability", decision_ts)
    return model


def _build_feature_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.date_range("2023-01-01", periods=120, freq="D", tz="UTC")
    for idx, ts in enumerate(dates):
        value1 = 0.01 + idx * 0.002
        value2 = 0.02 + idx * 0.003
        for fid, val in [("f1", value1), ("f2", value2)]:
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


def test_m4_identity_is_deterministic_and_excludes_generated_at_utc() -> None:
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
    identity_a = compute_m4_report_identity(report)
    identity_b = compute_m4_report_identity(report)
    assert identity_a == identity_b
    assert "generated_at_utc" not in json.dumps(report.provenance, sort_keys=True)
    assert identity_a.startswith("m4v1_sha256_")
    payload = {"model_id": report.model_id, "instrument": report.instrument, "report_version": report.report_version}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert digest != identity_a.replace("m4v1_sha256_", "")


def test_m4_identity_changes_when_scientific_values_change() -> None:
    matrix, artifact = _build_exact_two_cluster_fixture()
    report = compute_m4_regime_stability_report(
        matrix,
        artifact,
        instrument="AAA",
        evaluation_window_start_utc=pd.Timestamp("2023-01-01", tz="UTC"),
        evaluation_window_end_utc=pd.Timestamp("2023-01-05", tz="UTC"),
        decision_timestamp_utc=pd.Timestamp("2023-01-03", tz="UTC"),
        evaluation_mode="AS_OF_PRODUCTION",
    )
    baseline_identity = compute_m4_report_identity(report)
    modified_coverage = replace(report, coverage=M4Coverage(requested_count=3, assigned_count=2, abstained_count=1, assigned_share=2 / 3, abstained_share=1 / 3))
    modified_availability = replace(report, simulated_replay_availability_utc=pd.Timestamp("2023-01-04", tz="UTC"))

    assert compute_m4_report_identity(modified_coverage) != baseline_identity
    assert compute_m4_report_identity(modified_availability) != baseline_identity
    assert "generated_at_utc" not in json.dumps(report.provenance, sort_keys=True)


def test_m4_report_identity_is_invariant_to_input_row_order_and_generated_at_utc() -> None:
    matrix, artifact = _build_exact_two_cluster_fixture()
    base_report = compute_m4_regime_stability_report(
        matrix,
        artifact,
        instrument="AAA",
        evaluation_window_start_utc=pd.Timestamp("2023-01-01", tz="UTC"),
        evaluation_window_end_utc=pd.Timestamp("2023-01-05", tz="UTC"),
        decision_timestamp_utc=pd.Timestamp("2023-01-03", tz="UTC"),
        evaluation_mode="AS_OF_PRODUCTION",
    )
    permuted = matrix.iloc[[2, 0, 1]].copy().reset_index(drop=True)
    permuted_report = compute_m4_regime_stability_report(
        permuted,
        artifact,
        instrument="AAA",
        evaluation_window_start_utc=pd.Timestamp("2023-01-01", tz="UTC"),
        evaluation_window_end_utc=pd.Timestamp("2023-01-05", tz="UTC"),
        decision_timestamp_utc=pd.Timestamp("2023-01-03", tz="UTC"),
        evaluation_mode="AS_OF_PRODUCTION",
    )
    assert compute_m4_report_identity(base_report) == compute_m4_report_identity(permuted_report)

    provenance_variant = matrix.attrs["provenance"].copy()
    provenance_variant["generated_at_utc"] = pd.Timestamp("2025-01-01", tz="UTC")
    matrix_variant = matrix.copy()
    matrix_variant.attrs["provenance"] = provenance_variant
    generated_at_report = compute_m4_regime_stability_report(
        matrix_variant,
        artifact,
        instrument="AAA",
        evaluation_window_start_utc=pd.Timestamp("2023-01-01", tz="UTC"),
        evaluation_window_end_utc=pd.Timestamp("2023-01-05", tz="UTC"),
        decision_timestamp_utc=pd.Timestamp("2023-01-03", tz="UTC"),
        evaluation_mode="AS_OF_PRODUCTION",
    )
    assert compute_m4_report_identity(base_report) == compute_m4_report_identity(generated_at_report)


def test_m4_report_identity_senses_feature_value_and_availability_changes() -> None:
    matrix, artifact = _build_exact_two_cluster_fixture()
    base_report = compute_m4_regime_stability_report(
        matrix,
        artifact,
        instrument="AAA",
        evaluation_window_start_utc=pd.Timestamp("2023-01-01", tz="UTC"),
        evaluation_window_end_utc=pd.Timestamp("2023-01-05", tz="UTC"),
        decision_timestamp_utc=pd.Timestamp("2023-01-03", tz="UTC"),
        evaluation_mode="AS_OF_PRODUCTION",
    )

    value_shifted = matrix.copy()
    value_shifted.loc[value_shifted["timestamp"] == pd.Timestamp("2023-01-03", tz="UTC"), "x"] = 3.0
    shifted_report = compute_m4_regime_stability_report(
        value_shifted,
        artifact,
        instrument="AAA",
        evaluation_window_start_utc=pd.Timestamp("2023-01-01", tz="UTC"),
        evaluation_window_end_utc=pd.Timestamp("2023-01-05", tz="UTC"),
        decision_timestamp_utc=pd.Timestamp("2023-01-03", tz="UTC"),
        evaluation_mode="AS_OF_PRODUCTION",
    )
    assert compute_m4_report_identity(shifted_report) != compute_m4_report_identity(base_report)

    availability_shifted = matrix.copy()
    availability_shifted.loc[availability_shifted["timestamp"] == pd.Timestamp("2023-01-03", tz="UTC"), "feature_vector_available_at"] = pd.Timestamp("2023-01-04", tz="UTC")
    availability_report = compute_m4_regime_stability_report(
        availability_shifted,
        artifact,
        instrument="AAA",
        evaluation_window_start_utc=pd.Timestamp("2023-01-01", tz="UTC"),
        evaluation_window_end_utc=pd.Timestamp("2023-01-05", tz="UTC"),
        decision_timestamp_utc=pd.Timestamp("2023-01-03", tz="UTC"),
        evaluation_mode="AS_OF_PRODUCTION",
    )
    assert compute_m4_report_identity(availability_report) != compute_m4_report_identity(base_report)

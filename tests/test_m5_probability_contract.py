from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.mixture import GaussianMixture

from regimequant.data import (
    AVAILABILITY_POLICY_FIXTURE_DECLARED,
    AdjustmentState,
    AvailabilityStatus,
    MarketDataProvenance,
    PriceType,
    ReturnType,
    UniverseType,
)
from regimequant.features import compute_features_from_returns, list_m2_v1_feature_specs
from regimequant.regimes import build_m3_feature_matrix
from regimequant.regimes.m5 import (
    M5AvailabilityMode,
    M5ValidationError,
    assign_m5_regimes,
    compute_m5_descriptive_diagnostics,
    compute_m5_model_identity,
    fit_m5_regime_model,
)


def _build_feature_matrix(rows: list[dict], feature_ids: list[str]) -> pd.DataFrame:
    data = []
    for row in rows:
        item = {
            "scope_id": row["scope_id"],
            "timestamp": row["timestamp"],
            "feature_vector_available_at": row["feature_vector_available_at"],
            "feature_observation_available": row["feature_observation_available"],
        }
        for feature_id in feature_ids:
            item[feature_id] = row[feature_id]
        data.append(item)
    return pd.DataFrame(data)


def test_probability_entropy_fixture() -> None:
    probs = (0.62, 0.28, 0.10)
    entropy = -sum(value * math.log2(value) for value in probs)
    assert math.isclose(entropy, 1.2740022896699967, rel_tol=1e-12, abs_tol=1e-12)


def test_fit_and_assign_round_trip() -> None:
    timestamps = pd.date_range("2023-01-01", periods=120, freq="D", tz="UTC")
    feature_ids = ["feature_a", "feature_b"]
    rows = []
    for idx, ts in enumerate(timestamps):
        base = float(idx % 12)
        value_a = 0.0 + (base * 0.8) + (0.1 if idx % 2 else -0.1)
        value_b = 3.0 + (base * 0.5) + (0.2 if idx % 3 else -0.2)
        rows.append({
            "scope_id": "TEST",
            "timestamp": ts,
            "feature_vector_available_at": ts,
            "feature_observation_available": True,
            "feature_a": value_a,
            "feature_b": value_b,
        })
    matrix = _build_feature_matrix(rows, feature_ids)
    matrix.attrs["provenance"] = {
        "feature_spec_set_hash": "hash-abc",
        "generated_at_utc": pd.Timestamp("2023-01-01", tz="UTC"),
        "metadata_durability": "published",
    }
    artifact = fit_m5_regime_model(
        matrix,
        instrument="TEST",
        requested_feature_ids=feature_ids,
        training_interval=(timestamps[0], timestamps[40]),
        validation_interval=(timestamps[40], timestamps[80]),
        test_interval=(timestamps[80], timestamps[119]),
        training_information_cutoff=timestamps[39],
        training_cutoff=timestamps[39],
        availability_mode=M5AvailabilityMode.AS_OF_PRODUCTION,
        n_components=2,
    )
    assert artifact.model_id.startswith("m5v1_sha256_")
    assignment = assign_m5_regimes(
        artifact,
        matrix,
        decision_time=timestamps[110],
        availability_mode=M5AvailabilityMode.AS_OF_PRODUCTION,
    )
    assert set(assignment["assignment_status"]) <= {"assigned", "abstained"}
    assert len(assignment) == len(matrix)


def test_duplicate_component_rejection() -> None:
    timestamps = pd.date_range("2023-01-01", periods=40, freq="D", tz="UTC")
    rows = []
    for idx, ts in enumerate(timestamps):
        rows.append({
            "scope_id": "X",
            "timestamp": ts,
            "feature_vector_available_at": ts,
            "feature_observation_available": True,
            "feature_a": float(idx),
            "feature_b": 1.0,
        })
    matrix = _build_feature_matrix(rows, ["feature_a", "feature_b"])
    matrix.attrs["provenance"] = {
        "feature_spec_set_hash": "hash-dup",
        "generated_at_utc": pd.Timestamp("2023-01-01", tz="UTC"),
        "metadata_durability": "published",
    }
    with pytest.raises(Exception):
        fit_m5_regime_model(
            matrix,
            instrument="X",
            requested_feature_ids=["feature_a", "feature_b"],
            training_interval=(timestamps[0], timestamps[20]),
            validation_interval=(timestamps[20], timestamps[30]),
            test_interval=(timestamps[30], timestamps[40]),
            training_information_cutoff=timestamps[19],
            training_cutoff=timestamps[19],
            availability_mode=M5AvailabilityMode.AS_OF_PRODUCTION,
            n_components=3,
        )


def test_constant_feature_is_removed() -> None:
    timestamps = pd.date_range("2023-01-01", periods=60, freq="D", tz="UTC")
    rows = []
    for ts in timestamps:
        rows.append({
            "scope_id": "CONST",
            "timestamp": ts,
            "feature_vector_available_at": ts,
            "feature_observation_available": True,
            "feature_a": 1.0,
            "feature_b": float(ts.value) / 1e18,
        })
    matrix = _build_feature_matrix(rows, ["feature_a", "feature_b"])
    matrix.attrs["provenance"] = {
        "feature_spec_set_hash": "hash-constant",
        "generated_at_utc": pd.Timestamp("2023-01-01", tz="UTC"),
        "metadata_durability": "published",
    }
    artifact = fit_m5_regime_model(
        matrix,
        instrument="CONST",
        requested_feature_ids=["feature_a", "feature_b"],
        training_interval=(timestamps[0], timestamps[40]),
        validation_interval=(timestamps[40], timestamps[50]),
        test_interval=(timestamps[50], timestamps[59]),
        training_information_cutoff=timestamps[39],
        training_cutoff=timestamps[39],
        availability_mode=M5AvailabilityMode.AS_OF_PRODUCTION,
        n_components=2,
    )
    assert "feature_a" in artifact.dropped_constant_feature_ids


def test_assignment_abstains_on_missing_feature_availability() -> None:
    timestamps = pd.date_range("2023-01-01", periods=30, freq="D", tz="UTC")
    rows = []
    for idx, ts in enumerate(timestamps):
        rows.append({
            "scope_id": "TEST",
            "timestamp": ts,
            "feature_vector_available_at": ts,
            "feature_observation_available": True,
            "feature_a": float(idx),
            "feature_b": float(idx) + 1.0,
        })
    matrix = _build_feature_matrix(rows, ["feature_a", "feature_b"])
    matrix.attrs["provenance"] = {
        "feature_spec_set_hash": "hash-abc",
        "generated_at_utc": pd.Timestamp("2023-01-01", tz="UTC"),
        "metadata_durability": "published",
    }
    artifact = fit_m5_regime_model(
        matrix,
        instrument="TEST",
        requested_feature_ids=["feature_a", "feature_b"],
        training_interval=(timestamps[0], timestamps[20]),
        validation_interval=(timestamps[20], timestamps[25]),
        test_interval=(timestamps[25], timestamps[29]),
        training_information_cutoff=timestamps[19],
        training_cutoff=timestamps[19],
        availability_mode=M5AvailabilityMode.AS_OF_PRODUCTION,
        n_components=2,
    )
    matrix2 = matrix.copy()
    matrix2.loc[0, "feature_observation_available"] = False
    result = assign_m5_regimes(
        artifact,
        matrix2,
        decision_time=timestamps[29],
        availability_mode=M5AvailabilityMode.AS_OF_PRODUCTION,
    )
    assert result.iloc[0]["assignment_status"] == "abstained"


def test_fit_rejects_nonconverged_gmm(monkeypatch) -> None:
    timestamps = pd.date_range("2023-02-01", periods=90, freq="D", tz="UTC")
    rows = []
    for idx, ts in enumerate(timestamps):
        rows.append({
            "scope_id": "NONCONV",
            "timestamp": ts,
            "feature_vector_available_at": ts,
            "feature_observation_available": True,
            "feature_a": float(idx),
            "feature_b": float(idx) * 1.5 + 0.25,
        })
    matrix = _build_feature_matrix(rows, ["feature_a", "feature_b"])
    matrix.attrs["provenance"] = {
        "feature_spec_set_hash": "hash-nonconv",
        "generated_at_utc": pd.Timestamp("2023-02-01", tz="UTC"),
        "metadata_durability": "published",
    }

    def fake_fit(self, X, y=None):
        self.weights_ = np.asarray([0.5, 0.5], dtype=float)
        self.means_ = np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=float)
        self.covariances_ = np.asarray([np.eye(2), np.eye(2)], dtype=float)
        self.converged_ = False
        self.lower_bound_ = -1.0
        self.n_iter_ = 1
        return self

    monkeypatch.setattr(GaussianMixture, "fit", fake_fit)
    with pytest.raises(M5ValidationError, match="converge"):
        fit_m5_regime_model(
            matrix,
            instrument="NONCONV",
            requested_feature_ids=["feature_a", "feature_b"],
            training_interval=(timestamps[0], timestamps[60]),
            validation_interval=(timestamps[60], timestamps[75]),
            test_interval=(timestamps[75], timestamps[89]),
            training_information_cutoff=timestamps[59],
            training_cutoff=timestamps[59],
            availability_mode=M5AvailabilityMode.AS_OF_PRODUCTION,
            n_components=2,
        )


def test_descriptive_diagnostics_are_reported_without_refitting() -> None:
    timestamps = pd.date_range("2023-03-01", periods=90, freq="D", tz="UTC")
    rows = []
    for idx, ts in enumerate(timestamps):
        value_a = float(idx) + 0.3
        value_b = float(idx) * 0.8 + 0.9
        rows.append({
            "scope_id": "DIAG",
            "timestamp": ts,
            "feature_vector_available_at": ts,
            "feature_observation_available": True,
            "feature_a": value_a,
            "feature_b": value_b,
        })
    matrix = _build_feature_matrix(rows, ["feature_a", "feature_b"])
    matrix.attrs["provenance"] = {
        "feature_spec_set_hash": "hash-diag",
        "generated_at_utc": pd.Timestamp("2023-03-01", tz="UTC"),
        "metadata_durability": "published",
    }
    artifact = fit_m5_regime_model(
        matrix,
        instrument="DIAG",
        requested_feature_ids=["feature_a", "feature_b"],
        training_interval=(timestamps[0], timestamps[60]),
        validation_interval=(timestamps[60], timestamps[75]),
        test_interval=(timestamps[75], timestamps[89]),
        training_information_cutoff=timestamps[74],
        training_cutoff=timestamps[74],
        availability_mode=M5AvailabilityMode.AS_OF_PRODUCTION,
        n_components=2,
    )
    diagnostics = compute_m5_descriptive_diagnostics(
        artifact,
        matrix,
        interval=(timestamps[60], timestamps[75]),
        label="validation",
    )
    assert diagnostics["diagnostic_kind"] == "descriptive"
    assert diagnostics["eligible_sample_count"] > 0
    assert diagnostics["mean_log_likelihood"] <= 0.0
    assert diagnostics["mean_posterior_entropy"] >= 0.0
    assert diagnostics["numerical_validity"] is True


def test_m2_feature_api_integration_and_identity_stability() -> None:
    timestamps = pd.date_range("2024-01-01T10:00:00Z", periods=72, freq="5min")
    rows = []
    previous_timestamp = pd.NaT
    previous_available_at = pd.NaT
    previous_status = None
    for idx, ts in enumerate(timestamps):
        value = 0.015 + 0.002 * idx if idx % 2 == 0 else -0.012 + 0.0015 * idx
        row = {
            "instrument": "AAPL",
            "timestamp": ts,
            "available_at": ts,
            "availability_status": AvailabilityStatus.VERIFIED_TIMESTAMP.value,
            "previous_timestamp": previous_timestamp,
            "previous_available_at": previous_available_at,
            "previous_availability_status": previous_status,
            "return_type": ReturnType.SIMPLE.value,
            "return_value": value,
            "price_type": PriceType.CLOSE.value,
            "adjustment_state": AdjustmentState.RAW.value,
            "interval_elapsed_seconds": 300.0,
        }
        rows.append(row)
        previous_timestamp = ts
        previous_available_at = ts
        previous_status = AvailabilityStatus.VERIFIED_TIMESTAMP.value

    returns = pd.DataFrame(rows)
    returns.attrs["provenance"] = {
        "source_canonical_provenance": MarketDataProvenance(
            provider_id="fixture",
            requested_instruments=("AAPL",),
            retrieval_timestamp_utc=pd.Timestamp("2024-01-06T12:00:00Z"),
            requested_price_type=PriceType.CLOSE,
            requested_adjustment_state=AdjustmentState.RAW,
            price_type=PriceType.CLOSE,
            adjustment_state=AdjustmentState.RAW,
            universe_type=UniverseType.STATIC,
            universe_instruments=("AAPL",),
            availability_policy_id=AVAILABILITY_POLICY_FIXTURE_DECLARED,
            pit_membership_not_supported_flag=True,
        ),
        "return_policy_version": "1",
        "return_type": ReturnType.SIMPLE.value,
        "generated_at_utc": pd.Timestamp("2024-01-06T12:00:00Z"),
    }
    features, _ = compute_features_from_returns(returns, list_m2_v1_feature_specs())
    feature_matrix = features[features["math_status"] == "defined"].copy()
    pivoted = feature_matrix.pivot_table(
        index=["scope_id", "timestamp"],
        columns="feature_instance_id",
        values="feature_value",
        aggfunc="mean",
    ).reset_index()
    pivoted["feature_vector_available_at"] = pivoted["timestamp"]
    pivoted["feature_observation_available"] = True
    feature_columns = [
        col for col in pivoted.columns
        if col not in {"scope_id", "timestamp", "feature_vector_available_at", "feature_observation_available"}
    ]
    feature_columns = [col for col in feature_columns if pivoted[col].dropna().nunique() > 1][:3]
    assert len(feature_columns) >= 2
    pivoted = pivoted.loc[:, ["scope_id", "timestamp", "feature_vector_available_at", "feature_observation_available", *feature_columns]]
    pivoted.attrs["provenance"] = {
        "feature_spec_set_hash": features.attrs["provenance"]["feature_spec_set_hash"],
        "generated_at_utc": pd.Timestamp("2024-01-06T12:00:00Z"),
        "metadata_durability": "published",
    }
    training_end = pivoted["timestamp"].iloc[40]
    artifact = fit_m5_regime_model(
        pivoted,
        instrument="AAPL",
        requested_feature_ids=feature_columns,
        training_interval=(pivoted["timestamp"].min(), training_end),
        validation_interval=(training_end, pivoted["timestamp"].iloc[-1]),
        test_interval=(pivoted["timestamp"].iloc[-1], pivoted["timestamp"].iloc[-1]),
        training_information_cutoff=training_end,
        training_cutoff=training_end,
        availability_mode=M5AvailabilityMode.AS_OF_PRODUCTION,
        n_components=2,
    )
    assert artifact.model_id.startswith("m5v1_sha256_")
    diagnostics = compute_m5_descriptive_diagnostics(
        artifact,
        pivoted,
        interval=(training_end, pivoted["timestamp"].iloc[-1]),
        label="validation",
    )
    assert diagnostics["numerical_validity"] is True


def test_m1_m2_m3_m5_pipeline_end_to_end() -> None:
    timestamps = pd.date_range("2024-01-01T10:00:00Z", periods=96, freq="5min")
    rows: list[dict] = []
    previous_timestamp = pd.NaT
    previous_available_at = pd.NaT
    previous_status = None
    for idx, ts in enumerate(timestamps):
        drift = 0.008 + 0.0015 * (idx % 7)
        signal = 0.006 * math.sin(idx / 3.0)
        ret = drift + signal - 0.0009 * ((idx % 5) - 2)
        row = {
            "instrument": "AAPL",
            "timestamp": ts,
            "available_at": ts,
            "availability_status": AvailabilityStatus.VERIFIED_TIMESTAMP.value,
            "previous_timestamp": previous_timestamp,
            "previous_available_at": previous_available_at,
            "previous_availability_status": previous_status,
            "return_type": ReturnType.SIMPLE.value,
            "return_value": ret,
            "price_type": PriceType.CLOSE.value,
            "adjustment_state": AdjustmentState.RAW.value,
            "interval_elapsed_seconds": 300.0,
        }
        rows.append(row)
        previous_timestamp = ts
        previous_available_at = ts
        previous_status = AvailabilityStatus.VERIFIED_TIMESTAMP.value

    returns = pd.DataFrame(rows)
    returns.attrs["provenance"] = {
        "source_canonical_provenance": MarketDataProvenance(
            provider_id="fixture",
            requested_instruments=("AAPL",),
            retrieval_timestamp_utc=pd.Timestamp("2024-01-06T12:00:00Z"),
            requested_price_type=PriceType.CLOSE,
            requested_adjustment_state=AdjustmentState.RAW,
            price_type=PriceType.CLOSE,
            adjustment_state=AdjustmentState.RAW,
            universe_type=UniverseType.STATIC,
            universe_instruments=("AAPL",),
            availability_policy_id=AVAILABILITY_POLICY_FIXTURE_DECLARED,
            pit_membership_not_supported_flag=True,
        ),
        "return_policy_version": "1",
        "return_type": ReturnType.SIMPLE.value,
        "generated_at_utc": pd.Timestamp("2024-01-06T12:00:00Z"),
    }

    features, _ = compute_features_from_returns(returns, list_m2_v1_feature_specs())
    assert not features.empty
    feature_matrix = features[features["math_status"] == "defined"].copy()
    assert not feature_matrix.empty
    matrix = build_m3_feature_matrix(
        feature_matrix,
        requested_feature_instance_ids=sorted({str(value) for value in feature_matrix["feature_instance_id"].dropna().unique().tolist()}),
    )
    assert not matrix.empty
    assert matrix.attrs["provenance"]["feature_spec_set_hash"] == features.attrs["provenance"]["feature_spec_set_hash"]

    feature_columns = [
        column for column in matrix.columns
        if column not in {"scope_id", "timestamp", "feature_vector_available_at", "feature_observation_available"}
    ]
    feature_columns = [column for column in feature_columns if matrix[column].dropna().nunique() > 1]
    assert len(feature_columns) >= 2
    feature_columns = feature_columns[:3]

    training_end = matrix["timestamp"].iloc[80]
    artifact = fit_m5_regime_model(
        matrix,
        instrument="AAPL",
        requested_feature_ids=feature_columns,
        training_interval=(matrix["timestamp"].min(), training_end),
        validation_interval=(training_end, matrix["timestamp"].iloc[-1]),
        test_interval=(matrix["timestamp"].iloc[-1], matrix["timestamp"].iloc[-1]),
        training_information_cutoff=training_end,
        training_cutoff=training_end,
        availability_mode=M5AvailabilityMode.AS_OF_PRODUCTION,
        n_components=2,
    )
    assert artifact.model_id.startswith("m5v1_sha256_")
    assert artifact.feature_spec_hash == matrix.attrs["provenance"]["feature_spec_set_hash"]

    decision_time = artifact.effective_model_availability + pd.Timedelta(minutes=1)
    assignment = assign_m5_regimes(
        artifact,
        matrix,
        decision_time=decision_time,
        availability_mode=M5AvailabilityMode.AS_OF_PRODUCTION,
    )
    assert not assignment.empty
    assert set(assignment["assignment_status"]) <= {"assigned", "abstained"}
    assert assignment["probability_vector"].notna().any()
    for values in assignment["probability_vector"].dropna():
        assert math.isclose(float(sum(values)), 1.0, rel_tol=0.0, abs_tol=1e-8)
    assert assignment["model_id"].eq(artifact.model_id).all()
    assert assignment["provenance_digest"].eq(artifact.training_input_digest).all()
    late_row = matrix.copy()
    late_row.loc[late_row.index[0], "feature_vector_available_at"] = decision_time + pd.Timedelta(minutes=5)
    late_assignment = assign_m5_regimes(
        artifact,
        late_row,
        decision_time=decision_time,
        availability_mode=M5AvailabilityMode.AS_OF_PRODUCTION,
    )
    assert (late_assignment["assignment_status"] == "abstained").any()
    assert (late_assignment["abstention_reason"] == "late_feature_availability").any()
    assert assignment["model_id"].nunique() == 1


def test_scientific_identity_is_sensitive_only_to_scientific_payload() -> None:
    timestamps = pd.date_range("2024-01-01", periods=90, freq="D", tz="UTC")
    rows = []
    for idx, ts in enumerate(timestamps):
        rows.append({
            "scope_id": "ID",
            "timestamp": ts,
            "feature_vector_available_at": ts,
            "feature_observation_available": True,
            "feature_a": float(idx) + 0.4,
            "feature_b": float(idx) * 1.3 + 0.7,
        })
    matrix = pd.DataFrame(rows)
    matrix.attrs["provenance"] = {
        "feature_spec_set_hash": "hash-identity",
        "generated_at_utc": pd.Timestamp("2024-01-01", tz="UTC"),
        "metadata_durability": "published",
    }
    training_end = timestamps[45]
    artifact_a = fit_m5_regime_model(
        matrix,
        instrument="ID",
        requested_feature_ids=["feature_a", "feature_b"],
        training_interval=(timestamps[0], training_end),
        validation_interval=(training_end, timestamps[59]),
        test_interval=(timestamps[59], timestamps[89]),
        training_information_cutoff=training_end,
        training_cutoff=training_end,
        availability_mode=M5AvailabilityMode.AS_OF_PRODUCTION,
        n_components=2,
    )

    reordered = matrix.iloc[::-1].copy()
    reordered.attrs = dict(matrix.attrs)
    artifact_b = fit_m5_regime_model(
        reordered,
        instrument="ID",
        requested_feature_ids=["feature_a", "feature_b"],
        training_interval=(timestamps[0], training_end),
        validation_interval=(training_end, timestamps[59]),
        test_interval=(timestamps[59], timestamps[89]),
        training_information_cutoff=training_end,
        training_cutoff=training_end,
        availability_mode=M5AvailabilityMode.AS_OF_PRODUCTION,
        n_components=2,
    )
    assert artifact_a.training_input_digest == artifact_b.training_input_digest
    assert artifact_a.model_id == artifact_b.model_id
    assert artifact_a.model_id == compute_m5_model_identity(artifact_a)

    perturbed = matrix.copy()
    changed_ts = timestamps[42]
    perturbed.loc[perturbed["timestamp"] == changed_ts, "feature_a"] += 0.25
    artifact_c = fit_m5_regime_model(
        perturbed,
        instrument="ID",
        requested_feature_ids=["feature_a", "feature_b"],
        training_interval=(timestamps[0], training_end),
        validation_interval=(training_end, timestamps[59]),
        test_interval=(timestamps[59], timestamps[89]),
        training_information_cutoff=training_end,
        training_cutoff=training_end,
        availability_mode=M5AvailabilityMode.AS_OF_PRODUCTION,
        n_components=2,
    )
    assert artifact_c.training_input_digest != artifact_a.training_input_digest
    assert artifact_c.model_id != artifact_a.model_id

    shifted = replace(
        artifact_a,
        actual_creation_timestamp_utc=artifact_a.actual_creation_timestamp_utc + pd.Timedelta(days=365),
        effective_model_availability=(artifact_a.effective_model_availability + pd.Timedelta(days=365) if artifact_a.effective_model_availability is not None else None),
        model_id="",
    )
    assert shifted.model_id == artifact_a.model_id
    assert compute_m5_model_identity(shifted) == artifact_a.model_id

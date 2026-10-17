from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from regimequant.features import compute_features_from_returns, list_m2_v1_feature_specs
from regimequant.regimes import (
    M3AvailabilityMode,
    M3ModelArtifact,
    M3ValidationError,
    assign_m3_regimes,
    build_m3_feature_matrix,
    fit_m3_regime_model,
    validate_m3_model_artifact,
)
from tests.test_m2_feature_schema import _returns_table


def _split_plan() -> tuple[tuple[pd.Timestamp, pd.Timestamp], tuple[pd.Timestamp, pd.Timestamp], tuple[pd.Timestamp, pd.Timestamp]]:
    train_start = pd.Timestamp("2023-01-01", tz="UTC")
    train_end = train_start + pd.Timedelta(days=30)
    validation_end = train_end + pd.Timedelta(days=20)
    test_end = validation_end + pd.Timedelta(days=20)
    return (train_start, train_end), (train_end, validation_end), (validation_end, test_end)


def _base_feature_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.date_range("2023-01-01", periods=70, freq="D", tz="UTC")
    feature_ids = ["f1", "f2"]
    for idx, ts in enumerate(dates):
        for instrument in ["AAA", "BBB"]:
            value1 = 0.01 + idx * 0.002 if instrument == "AAA" else -0.01 - idx * 0.002
            value2 = 0.02 + idx * 0.003 if instrument == "AAA" else 0.04 + idx * 0.004
            for fid, val in zip(feature_ids, [value1, value2], strict=True):
                rows.append(
                    {
                        "scope_type": "instrument",
                        "scope_id": instrument,
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


def _fit_kwargs(**overrides: object) -> dict[str, object]:
    training_interval, validation_interval, test_interval = _split_plan()
    kwargs: dict[str, object] = {
        "instrument": "AAA",
        "requested_feature_instance_ids": ["f1", "f2"],
        "training_cutoff": pd.Timestamp("2023-02-20", tz="UTC"),
        "training_interval": training_interval,
        "validation_interval": validation_interval,
        "test_interval": test_interval,
        "training_information_cutoff": pd.Timestamp("2023-02-20", tz="UTC"),
        "validation_information_cutoff": validation_interval[1],
        "test_information_cutoff": test_interval[1],
        "feature_spec_set_hash": "hash-1",
        "parent_provenance_digest": "digest-1",
        "return_type": "simple",
        "price_type": "close",
        "adjustment_state": "raw",
        "n_clusters": 2,
    }
    kwargs.update(overrides)
    return kwargs


def test_public_m3_apis_are_exported() -> None:
    assert callable(build_m3_feature_matrix)
    assert callable(fit_m3_regime_model)
    assert callable(assign_m3_regimes)
    assert callable(validate_m3_model_artifact)


def test_build_m3_feature_matrix_is_wide_and_exact() -> None:
    table = _base_feature_frame()
    matrix = build_m3_feature_matrix(table, requested_feature_instance_ids=["f1", "f2"])
    assert list(matrix.columns[:2]) == ["scope_id", "timestamp"]
    assert set(matrix.columns) >= {"scope_id", "timestamp", "f1", "f2", "feature_vector_available_at"}
    assert len(matrix) == 140
    assert matrix["f1"].notna().all()


def test_fit_and_assign_model_uses_actual_creation_and_abstains_pre_creation() -> None:
    table = _base_feature_frame()
    matrix = build_m3_feature_matrix(table, requested_feature_instance_ids=["f1", "f2"])
    matrix = matrix[matrix["scope_id"] == "AAA"].copy()
    model = fit_m3_regime_model(
        matrix,
        **_fit_kwargs(),
    )
    assert model.model_id.startswith("m3v1_sha256_")
    pre_creation = assign_m3_regimes(model, matrix, decision_time=model.actual_creation_timestamp_utc - pd.Timedelta(days=1))
    assert (pre_creation["assignment_status"] == "abstained").all()
    assert pre_creation["cluster_id"].isna().all()
    assert pre_creation["abstention_reason"].notna().all()

    post_creation = assign_m3_regimes(model, matrix, decision_time=model.actual_creation_timestamp_utc)
    assert set(post_creation["assignment_status"]) <= {"assigned", "abstained"}
    assert post_creation["cluster_id"].notna().sum() >= 1
    assert post_creation["abstention_reason"].isna().sum() >= 1


def test_unknown_availability_abstains_without_losing_key() -> None:
    table = _base_feature_frame()
    bad = table.iloc[0].copy()
    bad["input_availability_status"] = "unknown_unverified"
    bad["available_at"] = None
    table2 = pd.concat([table.drop(index=0).copy(), pd.DataFrame([bad.to_dict()])], ignore_index=True)
    table2.attrs["provenance"] = table.attrs["provenance"]
    matrix = build_m3_feature_matrix(table2, requested_feature_instance_ids=["f1", "f2"])
    matrix = matrix[matrix["scope_id"] == "AAA"].copy()
    model = fit_m3_regime_model(
        matrix,
        **_fit_kwargs(),
    )
    assignments = assign_m3_regimes(model, matrix, decision_time=pd.Timestamp("2023-02-20", tz="UTC"))
    assert (assignments["assignment_status"] == "abstained").any()
    assert (assignments["cluster_id"].isna()).any()


def test_build_m3_feature_matrix_requires_m2_provenance() -> None:
    table = _base_feature_frame()
    table.attrs.pop("provenance", None)
    with pytest.raises(M3ValidationError, match="provenance"):
        build_m3_feature_matrix(table, requested_feature_instance_ids=["f1", "f2"])


def test_fit_m3_regime_model_rejects_late_training_feature() -> None:
    table = _base_feature_frame()
    cutoff = pd.Timestamp("2023-02-20", tz="UTC")
    late = table.copy()
    late.loc[(late["scope_id"] == "AAA") & (late["feature_instance_id"] == "f1"), "available_at"] = cutoff + pd.Timedelta(days=1)
    matrix = build_m3_feature_matrix(late, requested_feature_instance_ids=["f1", "f2"])
    matrix = matrix[matrix["scope_id"] == "AAA"].copy()
    with pytest.raises(M3ValidationError, match="training.*available|cutoff|availability"):
        fit_m3_regime_model(
            matrix,
            **_fit_kwargs(),
        )


def test_assignment_requires_model_availability_before_decision() -> None:
    table = _base_feature_frame()
    matrix = build_m3_feature_matrix(table, requested_feature_instance_ids=["f1", "f2"])
    matrix = matrix[matrix["scope_id"] == "AAA"].copy()
    model = fit_m3_regime_model(
        matrix,
        **_fit_kwargs(),
    )
    cutoff = pd.Timestamp("2023-02-20", tz="UTC")
    result = assign_m3_regimes(model, matrix, decision_time=cutoff - pd.Timedelta(days=1), assignment_mode="AS_OF_PRODUCTION")
    assert not result.empty
    assert (result["assignment_status"] == "abstained").all()
    assert result["cluster_id"].isna().all()
    assert (result["abstention_reason"] == "model_unavailable").all()


def test_fit_m3_regime_model_rejects_mixed_instrument_training() -> None:
    table = _base_feature_frame()
    table.loc[table["scope_id"] == "BBB", "scope_id"] = "CCC"
    matrix = build_m3_feature_matrix(table, requested_feature_instance_ids=["f1", "f2"])
    with pytest.raises(M3ValidationError, match="single instrument|mixed instrument"):
        fit_m3_regime_model(
            matrix,
            **_fit_kwargs(),
        )


def test_fit_m3_regime_model_enforces_minimum_training_observations() -> None:
    table = _base_feature_frame()
    cut = table[table["scope_id"] == "AAA"].copy()
    cut = cut[cut["timestamp"] <= pd.Timestamp("2023-01-10", tz="UTC")].copy()
    cut = cut.iloc[:20].copy()
    cut.attrs["provenance"] = table.attrs["provenance"]
    matrix = build_m3_feature_matrix(cut, requested_feature_instance_ids=["f1", "f2"])
    with pytest.raises(M3ValidationError, match="minimum|eligible|observations"):
        fit_m3_regime_model(
            matrix,
            **_fit_kwargs(),
        )


def test_assign_m3_regimes_abstains_for_cross_instrument_rows() -> None:
    table = _base_feature_frame()
    matrix = build_m3_feature_matrix(table, requested_feature_instance_ids=["f1", "f2"])
    matrix = matrix[matrix["scope_id"] == "AAA"].copy()
    model = fit_m3_regime_model(
        matrix,
        **_fit_kwargs(),
    )
    matrix = matrix.copy()
    matrix["scope_id"] = "BBB"
    assignments = assign_m3_regimes(model, matrix, decision_time=pd.Timestamp("2023-02-20", tz="UTC"))
    assert (assignments["assignment_status"] == "abstained").all()
    assert (assignments["instrument"] == "BBB").all()


def test_fit_m3_regime_model_requires_validation_interval() -> None:
    table = _base_feature_frame()
    matrix = build_m3_feature_matrix(table, requested_feature_instance_ids=["f1", "f2"])
    matrix = matrix[matrix["scope_id"] == "AAA"].copy()
    kwargs = _fit_kwargs()
    kwargs.pop("validation_interval")
    with pytest.raises(M3ValidationError, match="validation_interval.*required"):
        fit_m3_regime_model(matrix, **kwargs)


def test_fit_m3_regime_model_requires_test_interval() -> None:
    table = _base_feature_frame()
    matrix = build_m3_feature_matrix(table, requested_feature_instance_ids=["f1", "f2"])
    matrix = matrix[matrix["scope_id"] == "AAA"].copy()
    kwargs = _fit_kwargs()
    kwargs.pop("test_interval")
    with pytest.raises(M3ValidationError, match="test_interval.*required"):
        fit_m3_regime_model(matrix, **kwargs)


def test_fit_m3_regime_model_rejects_validation_below_minimum() -> None:
    table = _base_feature_frame()
    matrix = build_m3_feature_matrix(table, requested_feature_instance_ids=["f1", "f2"])
    matrix = matrix[matrix["scope_id"] == "AAA"].copy()
    training_interval, _, test_interval = _split_plan()
    validation_interval = (training_interval[1], training_interval[1] + pd.Timedelta(days=5))
    kwargs = _fit_kwargs(training_interval=training_interval, validation_interval=validation_interval, test_interval=test_interval)
    with pytest.raises(M3ValidationError, match="validation.*20|minimum"):
        fit_m3_regime_model(matrix, **kwargs)


def test_fit_m3_regime_model_rejects_test_below_minimum() -> None:
    table = _base_feature_frame()
    matrix = build_m3_feature_matrix(table, requested_feature_instance_ids=["f1", "f2"])
    matrix = matrix[matrix["scope_id"] == "AAA"].copy()
    training_interval, validation_interval, _ = _split_plan()
    test_interval = (validation_interval[1], validation_interval[1] + pd.Timedelta(days=5))
    kwargs = _fit_kwargs(training_interval=training_interval, validation_interval=validation_interval, test_interval=test_interval)
    with pytest.raises(M3ValidationError, match="test.*20|minimum"):
        fit_m3_regime_model(matrix, **kwargs)


def test_fit_m3_regime_model_accepts_exact_minimum_split() -> None:
    table = _base_feature_frame()
    matrix = build_m3_feature_matrix(table, requested_feature_instance_ids=["f1", "f2"])
    matrix = matrix[matrix["scope_id"] == "AAA"].copy()
    training_interval, validation_interval, test_interval = _split_plan()
    model = fit_m3_regime_model(
        matrix,
        **_fit_kwargs(
            training_interval=training_interval,
            validation_interval=validation_interval,
            test_interval=test_interval,
            training_information_cutoff=pd.Timestamp("2023-02-20", tz="UTC"),
            validation_information_cutoff=validation_interval[1],
            test_information_cutoff=test_interval[1],
        ),
    )
    assert model.model_id.startswith("m3v1_sha256_")
    assert model.training_interval == training_interval
    assert model.validation_interval == validation_interval
    assert model.test_interval == test_interval


def test_held_out_eligibility_excludes_late_or_unknown_availability() -> None:
    table = _base_feature_frame()
    matrix = build_m3_feature_matrix(table, requested_feature_instance_ids=["f1", "f2"])
    matrix = matrix[matrix["scope_id"] == "AAA"].copy()
    training_interval, validation_interval, test_interval = _split_plan()
    late_row = matrix[matrix["timestamp"] == validation_interval[0]].iloc[0].copy()
    late_row["feature_vector_available_at"] = validation_interval[1] + pd.Timedelta(days=1)
    unknown_row = matrix[matrix["timestamp"] == validation_interval[1]].iloc[0].copy()
    unknown_row["feature_vector_available_at"] = None
    modified = pd.concat([matrix, pd.DataFrame([late_row.to_dict(), unknown_row.to_dict()])], ignore_index=True)
    with pytest.raises(M3ValidationError, match="availability|cutoff"):
        fit_m3_regime_model(
            modified,
            **_fit_kwargs(
                training_interval=training_interval,
                validation_interval=validation_interval,
                test_interval=test_interval,
            ),
        )


def test_held_out_values_within_existing_windows_do_not_change_model_invariants() -> None:
    table = _base_feature_frame()
    matrix = build_m3_feature_matrix(table, requested_feature_instance_ids=["f1", "f2"])
    matrix = matrix[matrix["scope_id"] == "AAA"].copy()
    training_interval, validation_interval, test_interval = _split_plan()
    model = fit_m3_regime_model(matrix, **_fit_kwargs(training_interval=training_interval, validation_interval=validation_interval, test_interval=test_interval))
    baseline_scaler = dict(model.scaler_parameters)
    baseline_centroids = model.canonical_centroids
    baseline_model_id = model.model_id
    baseline_occupancy = model.training_occupancy

    modified = matrix.copy()
    validation_mask = modified["timestamp"].between(validation_interval[0], validation_interval[1], inclusive="left")
    test_mask = modified["timestamp"].between(test_interval[0], test_interval[1], inclusive="left")
    modified.loc[validation_mask, "f1"] = modified.loc[validation_mask, "f1"] + 100.0
    modified.loc[validation_mask, "f2"] = modified.loc[validation_mask, "f2"] + 100.0
    modified.loc[test_mask, "f1"] = modified.loc[test_mask, "f1"] - 100.0
    modified.loc[test_mask, "f2"] = modified.loc[test_mask, "f2"] - 100.0

    model2 = fit_m3_regime_model(modified, **_fit_kwargs(training_interval=training_interval, validation_interval=validation_interval, test_interval=test_interval))
    assert model2.scaler_parameters == baseline_scaler
    assert model2.canonical_centroids == baseline_centroids
    assert model2.training_occupancy == baseline_occupancy
    assert model2.model_id == baseline_model_id


def test_mixed_instrument_held_out_rows_cannot_pass_eligibility() -> None:
    table = _base_feature_frame()
    matrix = build_m3_feature_matrix(table, requested_feature_instance_ids=["f1", "f2"])
    instrument_matrix = matrix[matrix["scope_id"] == "AAA"].copy()
    training_interval, validation_interval, test_interval = _split_plan()
    bad_rows = instrument_matrix[instrument_matrix["timestamp"].between(validation_interval[0], validation_interval[1])].copy()
    bad_rows["scope_id"] = "BBB"
    mixed = pd.concat([instrument_matrix, bad_rows], ignore_index=True)
    with pytest.raises(M3ValidationError, match="single instrument|multiple instruments|instrument"):
        fit_m3_regime_model(
            mixed,
            **_fit_kwargs(training_interval=training_interval, validation_interval=validation_interval, test_interval=test_interval),
        )


def test_assignment_abstains_before_model_availability() -> None:
    table = _base_feature_frame()
    matrix = build_m3_feature_matrix(table, requested_feature_instance_ids=["f1", "f2"])
    matrix = matrix[matrix["scope_id"] == "AAA"].copy()
    model = fit_m3_regime_model(matrix, **_fit_kwargs())
    output = assign_m3_regimes(
        model,
        matrix,
        decision_time=model.effective_model_availability - pd.Timedelta(days=1),
        assignment_mode=model.assignment_mode,
    )
    assert not output.empty
    assert (output["assignment_status"] == "abstained").all()
    assert output["cluster_id"].isna().all()
    assert (output["abstention_reason"] == "model_unavailable").all()


def test_training_row_permutation_invariance() -> None:
    table = _base_feature_frame()
    matrix = build_m3_feature_matrix(table, requested_feature_instance_ids=["f1", "f2"])
    matrix = matrix[matrix["scope_id"] == "AAA"].copy()
    permuted = matrix.sample(frac=1.0, random_state=7).reset_index(drop=True)
    model_a = fit_m3_regime_model(matrix, **_fit_kwargs())
    model_b = fit_m3_regime_model(permuted, **_fit_kwargs())
    assert model_a.model_id == model_b.model_id
    assert model_a.scaler_parameters == model_b.scaler_parameters
    assert model_a.canonical_centroids == model_b.canonical_centroids
    assert model_a.training_occupancy == model_b.training_occupancy


def test_repeated_fit_model_identity_is_stable() -> None:
    table = _base_feature_frame()
    matrix = build_m3_feature_matrix(table, requested_feature_instance_ids=["f1", "f2"])
    matrix = matrix[matrix["scope_id"] == "AAA"].copy()
    model_a = fit_m3_regime_model(matrix, **_fit_kwargs())
    model_b = fit_m3_regime_model(matrix.copy(), **_fit_kwargs())
    assert model_a.model_id == model_b.model_id
    assert model_a.scaler_parameters == model_b.scaler_parameters


def test_assignment_tie_breaks_by_lowest_canonical_index() -> None:
    artifact = M3ModelArtifact(
        instrument="AAA",
        requested_feature_instance_ids=("f1",),
        effective_feature_instance_ids=("f1",),
        feature_spec_set_hash="hash-1",
        parent_provenance_digest="digest-1",
        return_type="simple",
        price_type="close",
        adjustment_state="raw",
        training_interval=(pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2023-01-03", tz="UTC")),
        training_cutoff=pd.Timestamp("2023-01-03", tz="UTC"),
        training_information_cutoff=pd.Timestamp("2023-01-03", tz="UTC"),
        validation_interval=(pd.Timestamp("2023-01-03", tz="UTC"), pd.Timestamp("2023-01-05", tz="UTC")),
        test_interval=(pd.Timestamp("2023-01-05", tz="UTC"), pd.Timestamp("2023-01-07", tz="UTC")),
        kmeans_config={"n_clusters": 2, "random_state": 42},
        scaler_parameters={"mean_": [0.0], "scale_": [1.0]},
        canonical_centroids=((-1.0,), (1.0,)),
        training_occupancy={"0": 1, "1": 1},
        assignment_mode=M3AvailabilityMode.AS_OF_PRODUCTION,
        actual_creation_timestamp_utc=pd.Timestamp("2023-01-04", tz="UTC"),
        effective_model_availability=pd.Timestamp("2023-01-04", tz="UTC"),
        seed=42,
    )
    frame = pd.DataFrame(
        [{
            "scope_id": "AAA",
            "timestamp": pd.Timestamp("2023-01-04", tz="UTC"),
            "f1": 0.0,
            "feature_vector_available_at": pd.Timestamp("2023-01-04", tz="UTC"),
            "feature_observation_available": True,
        }]
    )
    frame.attrs["provenance"] = {
        "feature_spec_set_hash": "hash-1",
        "parent_return_provenance_digest": "digest-1",
        "return_type": "simple",
        "price_type": "close",
        "adjustment_state": "raw",
        "generated_at_utc": pd.Timestamp("2023-01-01", tz="UTC"),
        "metadata_durability": "DataFrame.attrs is in-memory only and not a durable serialization contract",
    }
    out = assign_m3_regimes(artifact, frame, decision_time=pd.Timestamp("2023-01-04", tz="UTC"))
    assert out["assignment_status"].tolist() == ["assigned"]
    assert out["cluster_id"].tolist() == [0]


def test_production_availability_before_actual_creation_is_rejected() -> None:
    base = M3ModelArtifact(
        instrument="AAA",
        requested_feature_instance_ids=("f1",),
        effective_feature_instance_ids=("f1",),
        feature_spec_set_hash="hash-1",
        parent_provenance_digest="digest-1",
        return_type="simple",
        price_type="close",
        adjustment_state="raw",
        training_interval=(pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2023-01-03", tz="UTC")),
        training_cutoff=pd.Timestamp("2023-01-03", tz="UTC"),
        training_information_cutoff=pd.Timestamp("2023-01-03", tz="UTC"),
        validation_interval=(pd.Timestamp("2023-01-03", tz="UTC"), pd.Timestamp("2023-01-05", tz="UTC")),
        test_interval=(pd.Timestamp("2023-01-05", tz="UTC"), pd.Timestamp("2023-01-07", tz="UTC")),
        kmeans_config={"n_clusters": 2, "random_state": 42},
        scaler_parameters={"mean_": [0.0], "scale_": [1.0]},
        canonical_centroids=((-1.0,), (1.0,)),
        training_occupancy={"0": 1, "1": 1},
        assignment_mode=M3AvailabilityMode.AS_OF_PRODUCTION,
        actual_creation_timestamp_utc=pd.Timestamp("2023-01-08", tz="UTC"),
        effective_model_availability=pd.Timestamp("2023-01-07", tz="UTC"),
        seed=42,
    )
    with pytest.raises(M3ValidationError, match="actual artifact creation|training information cutoff"):
        validate_m3_model_artifact(base)


def test_production_decision_before_creation_abstains() -> None:
    base = M3ModelArtifact(
        instrument="AAA",
        requested_feature_instance_ids=("f1",),
        effective_feature_instance_ids=("f1",),
        feature_spec_set_hash="hash-1",
        parent_provenance_digest="digest-1",
        return_type="simple",
        price_type="close",
        adjustment_state="raw",
        training_interval=(pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2023-01-03", tz="UTC")),
        training_cutoff=pd.Timestamp("2023-01-03", tz="UTC"),
        training_information_cutoff=pd.Timestamp("2023-01-03", tz="UTC"),
        validation_interval=(pd.Timestamp("2023-01-03", tz="UTC"), pd.Timestamp("2023-01-05", tz="UTC")),
        test_interval=(pd.Timestamp("2023-01-05", tz="UTC"), pd.Timestamp("2023-01-07", tz="UTC")),
        kmeans_config={"n_clusters": 2, "random_state": 42},
        scaler_parameters={"mean_": [0.0], "scale_": [1.0]},
        canonical_centroids=((-1.0,), (1.0,)),
        training_occupancy={"0": 1, "1": 1},
        assignment_mode=M3AvailabilityMode.AS_OF_PRODUCTION,
        actual_creation_timestamp_utc=pd.Timestamp("2023-01-08", tz="UTC"),
        effective_model_availability=pd.Timestamp("2023-01-08", tz="UTC"),
        seed=42,
    )
    frame = pd.DataFrame([
        {
            "scope_id": "AAA",
            "timestamp": pd.Timestamp("2023-01-04", tz="UTC"),
            "f1": 0.0,
            "feature_vector_available_at": pd.Timestamp("2023-01-04", tz="UTC"),
            "feature_observation_available": True,
        }
    ])
    frame.attrs["provenance"] = {
        "feature_spec_set_hash": "hash-1",
        "parent_return_provenance_digest": "digest-1",
        "return_type": "simple",
        "price_type": "close",
        "adjustment_state": "raw",
        "generated_at_utc": pd.Timestamp("2023-01-01", tz="UTC"),
        "metadata_durability": "DataFrame.attrs is in-memory only and not a durable serialization contract",
    }
    out = assign_m3_regimes(base, frame, decision_time=pd.Timestamp("2023-01-07", tz="UTC"))
    assert (out["assignment_status"] == "abstained").all()
    assert out["cluster_id"].isna().all()
    assert out["abstention_reason"].notna().all()


def test_replay_without_simulated_availability_is_rejected() -> None:
    artifact = M3ModelArtifact(
        instrument="AAA",
        requested_feature_instance_ids=("f1",),
        effective_feature_instance_ids=("f1",),
        feature_spec_set_hash="hash-1",
        parent_provenance_digest="digest-1",
        return_type="simple",
        price_type="close",
        adjustment_state="raw",
        training_interval=(pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2023-01-03", tz="UTC")),
        training_cutoff=pd.Timestamp("2023-01-03", tz="UTC"),
        training_information_cutoff=pd.Timestamp("2023-01-03", tz="UTC"),
        validation_interval=(pd.Timestamp("2023-01-03", tz="UTC"), pd.Timestamp("2023-01-05", tz="UTC")),
        test_interval=(pd.Timestamp("2023-01-05", tz="UTC"), pd.Timestamp("2023-01-07", tz="UTC")),
        kmeans_config={"n_clusters": 2, "random_state": 42},
        scaler_parameters={"mean_": [0.0], "scale_": [1.0]},
        canonical_centroids=((-1.0,), (1.0,)),
        training_occupancy={"0": 1, "1": 1},
        assignment_mode=M3AvailabilityMode.HISTORICAL_REPLAY,
        actual_creation_timestamp_utc=pd.Timestamp("2023-01-08", tz="UTC"),
        effective_model_availability=pd.Timestamp("2023-01-08", tz="UTC"),
        seed=42,
    )
    with pytest.raises(M3ValidationError, match="simulated_replay_availability|historical replay"):
        validate_m3_model_artifact(artifact)


def test_replay_availability_before_cutoff_is_rejected() -> None:
    artifact = M3ModelArtifact(
        instrument="AAA",
        requested_feature_instance_ids=("f1",),
        effective_feature_instance_ids=("f1",),
        feature_spec_set_hash="hash-1",
        parent_provenance_digest="digest-1",
        return_type="simple",
        price_type="close",
        adjustment_state="raw",
        training_interval=(pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2023-01-03", tz="UTC")),
        training_cutoff=pd.Timestamp("2023-01-03", tz="UTC"),
        training_information_cutoff=pd.Timestamp("2023-01-03", tz="UTC"),
        validation_interval=(pd.Timestamp("2023-01-03", tz="UTC"), pd.Timestamp("2023-01-05", tz="UTC")),
        test_interval=(pd.Timestamp("2023-01-05", tz="UTC"), pd.Timestamp("2023-01-07", tz="UTC")),
        kmeans_config={"n_clusters": 2, "random_state": 42},
        scaler_parameters={"mean_": [0.0], "scale_": [1.0]},
        canonical_centroids=((-1.0,), (1.0,)),
        training_occupancy={"0": 1, "1": 1},
        assignment_mode=M3AvailabilityMode.HISTORICAL_REPLAY,
        actual_creation_timestamp_utc=pd.Timestamp("2023-01-08", tz="UTC"),
        effective_model_availability=pd.Timestamp("2023-01-08", tz="UTC"),
        simulated_replay_availability=pd.Timestamp("2023-01-02", tz="UTC"),
        seed=42,
    )
    with pytest.raises(M3ValidationError, match="training information cutoff|replay availability"):
        validate_m3_model_artifact(artifact)


def test_replay_availability_at_or_after_cutoff_is_accepted() -> None:
    artifact = M3ModelArtifact(
        instrument="AAA",
        requested_feature_instance_ids=("f1",),
        effective_feature_instance_ids=("f1",),
        feature_spec_set_hash="hash-1",
        parent_provenance_digest="digest-1",
        return_type="simple",
        price_type="close",
        adjustment_state="raw",
        training_interval=(pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2023-01-03", tz="UTC")),
        training_cutoff=pd.Timestamp("2023-01-03", tz="UTC"),
        training_information_cutoff=pd.Timestamp("2023-01-03", tz="UTC"),
        validation_interval=(pd.Timestamp("2023-01-03", tz="UTC"), pd.Timestamp("2023-01-05", tz="UTC")),
        test_interval=(pd.Timestamp("2023-01-05", tz="UTC"), pd.Timestamp("2023-01-07", tz="UTC")),
        kmeans_config={"n_clusters": 2, "random_state": 42},
        scaler_parameters={"mean_": [0.0], "scale_": [1.0]},
        canonical_centroids=((-1.0,), (1.0,)),
        training_occupancy={"0": 1, "1": 1},
        assignment_mode=M3AvailabilityMode.HISTORICAL_REPLAY,
        actual_creation_timestamp_utc=pd.Timestamp("2023-01-08", tz="UTC"),
        effective_model_availability=pd.Timestamp("2023-01-08", tz="UTC"),
        simulated_replay_availability=pd.Timestamp("2023-01-03", tz="UTC"),
        seed=42,
    )
    validate_m3_model_artifact(artifact)


def test_mode_mismatch_is_rejected() -> None:
    artifact = M3ModelArtifact(
        instrument="AAA",
        requested_feature_instance_ids=("f1",),
        effective_feature_instance_ids=("f1",),
        feature_spec_set_hash="hash-1",
        parent_provenance_digest="digest-1",
        return_type="simple",
        price_type="close",
        adjustment_state="raw",
        training_interval=(pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2023-01-03", tz="UTC")),
        training_cutoff=pd.Timestamp("2023-01-03", tz="UTC"),
        training_information_cutoff=pd.Timestamp("2023-01-03", tz="UTC"),
        validation_interval=(pd.Timestamp("2023-01-03", tz="UTC"), pd.Timestamp("2023-01-05", tz="UTC")),
        test_interval=(pd.Timestamp("2023-01-05", tz="UTC"), pd.Timestamp("2023-01-07", tz="UTC")),
        kmeans_config={"n_clusters": 2, "random_state": 42},
        scaler_parameters={"mean_": [0.0], "scale_": [1.0]},
        canonical_centroids=((-1.0,), (1.0,)),
        training_occupancy={"0": 1, "1": 1},
        assignment_mode=M3AvailabilityMode.AS_OF_PRODUCTION,
        actual_creation_timestamp_utc=pd.Timestamp("2023-01-08", tz="UTC"),
        effective_model_availability=pd.Timestamp("2023-01-08", tz="UTC"),
        seed=42,
    )
    frame = pd.DataFrame([{"scope_id": "AAA", "timestamp": pd.Timestamp("2023-01-09", tz="UTC"), "f1": 0.0, "feature_vector_available_at": pd.Timestamp("2023-01-09", tz="UTC"), "feature_observation_available": True}])
    frame.attrs["provenance"] = {
        "feature_spec_set_hash": "hash-1",
        "parent_return_provenance_digest": "digest-1",
        "return_type": "simple",
        "price_type": "close",
        "adjustment_state": "raw",
        "generated_at_utc": pd.Timestamp("2023-01-01", tz="UTC"),
        "metadata_durability": "DataFrame.attrs is in-memory only and not a durable serialization contract",
    }
    with pytest.raises(M3ValidationError, match="assignment_mode.*match|available mode"):
        assign_m3_regimes(artifact, frame, decision_time=pd.Timestamp("2023-01-09", tz="UTC"), assignment_mode=M3AvailabilityMode.HISTORICAL_REPLAY)


def test_actual_creation_timestamp_change_does_not_change_model_identity() -> None:
    base = M3ModelArtifact(
        instrument="AAA",
        requested_feature_instance_ids=("f1",),
        effective_feature_instance_ids=("f1",),
        feature_spec_set_hash="hash-1",
        parent_provenance_digest="digest-1",
        return_type="simple",
        price_type="close",
        adjustment_state="raw",
        training_interval=(pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2023-01-03", tz="UTC")),
        training_cutoff=pd.Timestamp("2023-01-03", tz="UTC"),
        training_information_cutoff=pd.Timestamp("2023-01-03", tz="UTC"),
        validation_interval=(pd.Timestamp("2023-01-03", tz="UTC"), pd.Timestamp("2023-01-05", tz="UTC")),
        test_interval=(pd.Timestamp("2023-01-05", tz="UTC"), pd.Timestamp("2023-01-07", tz="UTC")),
        kmeans_config={"n_clusters": 2, "random_state": 42},
        scaler_parameters={"mean_": [0.0], "scale_": [1.0]},
        canonical_centroids=((-1.0,), (1.0,)),
        training_occupancy={"0": 1, "1": 1},
        assignment_mode=M3AvailabilityMode.AS_OF_PRODUCTION,
        actual_creation_timestamp_utc=pd.Timestamp("2023-01-08", tz="UTC"),
        effective_model_availability=pd.Timestamp("2023-01-08", tz="UTC"),
        seed=42,
    )
    backdated = M3ModelArtifact(
        model_contract_version=base.model_contract_version,
        instrument=base.instrument,
        requested_feature_instance_ids=base.requested_feature_instance_ids,
        effective_feature_instance_ids=base.effective_feature_instance_ids,
        dropped_constant_feature_ids=base.dropped_constant_feature_ids,
        feature_spec_set_hash=base.feature_spec_set_hash,
        parent_provenance_digest=base.parent_provenance_digest,
        return_type=base.return_type,
        price_type=base.price_type,
        adjustment_state=base.adjustment_state,
        training_interval=base.training_interval,
        training_cutoff=base.training_cutoff,
        training_information_cutoff=base.training_information_cutoff,
        validation_interval=base.validation_interval,
        test_interval=base.test_interval,
        kmeans_config=base.kmeans_config,
        scaler_parameters=base.scaler_parameters,
        canonical_centroids=base.canonical_centroids,
        training_occupancy=base.training_occupancy,
        assignment_mode=base.assignment_mode,
        actual_creation_timestamp_utc=pd.Timestamp("2023-01-09", tz="UTC"),
        simulated_replay_availability=base.simulated_replay_availability,
        effective_model_availability=pd.Timestamp("2023-01-09", tz="UTC"),
        seed=base.seed,
    )
    assert base.model_id == backdated.model_id


def test_backdating_creation_cannot_certify_historical_production() -> None:
    artifact = M3ModelArtifact(
        instrument="AAA",
        requested_feature_instance_ids=("f1",),
        effective_feature_instance_ids=("f1",),
        feature_spec_set_hash="hash-1",
        parent_provenance_digest="digest-1",
        return_type="simple",
        price_type="close",
        adjustment_state="raw",
        training_interval=(pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2023-01-03", tz="UTC")),
        training_cutoff=pd.Timestamp("2023-01-03", tz="UTC"),
        training_information_cutoff=pd.Timestamp("2023-01-03", tz="UTC"),
        validation_interval=(pd.Timestamp("2023-01-03", tz="UTC"), pd.Timestamp("2023-01-05", tz="UTC")),
        test_interval=(pd.Timestamp("2023-01-05", tz="UTC"), pd.Timestamp("2023-01-07", tz="UTC")),
        kmeans_config={"n_clusters": 2, "random_state": 42},
        scaler_parameters={"mean_": [0.0], "scale_": [1.0]},
        canonical_centroids=((-1.0,), (1.0,)),
        training_occupancy={"0": 1, "1": 1},
        assignment_mode=M3AvailabilityMode.AS_OF_PRODUCTION,
        actual_creation_timestamp_utc=pd.Timestamp("2023-01-09", tz="UTC"),
        effective_model_availability=pd.Timestamp("2023-01-09", tz="UTC"),
        seed=42,
    )
    frame = pd.DataFrame([
        {
            "scope_id": "AAA",
            "timestamp": pd.Timestamp("2023-01-04", tz="UTC"),
            "f1": 0.0,
            "feature_vector_available_at": pd.Timestamp("2023-01-04", tz="UTC"),
            "feature_observation_available": True,
        }
    ])
    frame.attrs["provenance"] = {
        "feature_spec_set_hash": "hash-1",
        "parent_return_provenance_digest": "digest-1",
        "return_type": "simple",
        "price_type": "close",
        "adjustment_state": "raw",
        "generated_at_utc": pd.Timestamp("2023-01-01", tz="UTC"),
        "metadata_durability": "DataFrame.attrs is in-memory only and not a durable serialization contract",
    }
    out = assign_m3_regimes(artifact, frame, decision_time=pd.Timestamp("2023-01-04", tz="UTC"))
    assert (out["assignment_status"] == "abstained").all()
    assert out["cluster_id"].isna().all()
    assert out["abstention_reason"].notna().all()


def test_constant_feature_exclusion_and_all_constant_rejection() -> None:
    table = _base_feature_frame()
    table.loc[table["feature_instance_id"] == "f2", "feature_value"] = 5.0
    matrix = build_m3_feature_matrix(table, requested_feature_instance_ids=["f1", "f2"])
    matrix = matrix[matrix["scope_id"] == "AAA"].copy()
    model = fit_m3_regime_model(matrix, **_fit_kwargs())
    assert model.effective_feature_instance_ids == ("f1",)
    assert "f2" in model.dropped_constant_feature_ids

    all_constant = table.copy()
    all_constant["feature_value"] = 3.0
    matrix2 = build_m3_feature_matrix(all_constant, requested_feature_instance_ids=["f1", "f2"])
    matrix2 = matrix2[matrix2["scope_id"] == "AAA"].copy()
    with pytest.raises(M3ValidationError, match="constant"):
        fit_m3_regime_model(matrix2, **_fit_kwargs())


def test_validate_m3_model_artifact_rejects_duplicate_centroids_and_empty_clusters() -> None:
    base = M3ModelArtifact(
        instrument="AAA",
        requested_feature_instance_ids=("f1",),
        effective_feature_instance_ids=("f1",),
        feature_spec_set_hash="hash-1",
        parent_provenance_digest="digest-1",
        return_type="simple",
        price_type="close",
        adjustment_state="raw",
        training_interval=(pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2023-01-31", tz="UTC")),
        training_cutoff=pd.Timestamp("2023-02-20", tz="UTC"),
        training_information_cutoff=pd.Timestamp("2023-02-20", tz="UTC"),
        validation_interval=(pd.Timestamp("2023-01-31", tz="UTC"), pd.Timestamp("2023-02-20", tz="UTC")),
        test_interval=(pd.Timestamp("2023-02-20", tz="UTC"), pd.Timestamp("2023-03-12", tz="UTC")),
        kmeans_config={"n_clusters": 2, "random_state": 42},
        scaler_parameters={"mean_": [0.0], "scale_": [1.0]},
        canonical_centroids=((0.0,), (0.0,)),
        training_occupancy={"0": 1, "1": 0},
        assignment_mode=M3AvailabilityMode.AS_OF_PRODUCTION,
        actual_creation_timestamp_utc=pd.Timestamp("2023-03-13", tz="UTC"),
        effective_model_availability=pd.Timestamp("2023-02-20", tz="UTC"),
        seed=42,
    )
    with pytest.raises(M3ValidationError, match="duplicate centroid|empty cluster|training_occupancy"):
        validate_m3_model_artifact(base)


def test_m3_artifact_nested_identity_payloads_are_defensively_copied() -> None:
    config = {"n_clusters": 2, "random_state": 42}
    scaler = {"mean_": [0.0], "scale_": [1.0]}
    occupancy = {"0": 1, "1": 1}
    centroid = ((0.0,), (1.0,))
    artifact = M3ModelArtifact(
        instrument="AAA",
        requested_feature_instance_ids=("f1",),
        effective_feature_instance_ids=("f1",),
        feature_spec_set_hash="hash-1",
        parent_provenance_digest="digest-1",
        return_type="simple",
        price_type="close",
        adjustment_state="raw",
        training_interval=(pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2023-01-31", tz="UTC")),
        training_cutoff=pd.Timestamp("2023-02-20", tz="UTC"),
        training_information_cutoff=pd.Timestamp("2023-02-20", tz="UTC"),
        validation_interval=(pd.Timestamp("2023-01-31", tz="UTC"), pd.Timestamp("2023-02-20", tz="UTC")),
        test_interval=(pd.Timestamp("2023-02-20", tz="UTC"), pd.Timestamp("2023-03-12", tz="UTC")),
        kmeans_config=config,
        scaler_parameters=scaler,
        canonical_centroids=centroid,
        training_occupancy=occupancy,
        assignment_mode=M3AvailabilityMode.AS_OF_PRODUCTION,
        actual_creation_timestamp_utc=pd.Timestamp("2023-03-13", tz="UTC"),
        effective_model_availability=pd.Timestamp("2023-02-20", tz="UTC"),
        seed=42,
    )
    config["n_clusters"] = 99
    scaler["mean_"][0] = 7.5
    occupancy["1"] = 99
    assert artifact.kmeans_config["n_clusters"] == 2
    assert list(artifact.scaler_parameters["mean_"]) == [0.0]
    assert artifact.training_occupancy == {"0": 1, "1": 1}
    assert artifact.model_id == M3ModelArtifact(
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
    ).model_id


def test_artifact_identity_changes_with_output_affecting_fields() -> None:
    base = M3ModelArtifact(
        instrument="AAA",
        requested_feature_instance_ids=("f1", "f2"),
        effective_feature_instance_ids=("f1", "f2"),
        feature_spec_set_hash="hash-1",
        parent_provenance_digest="digest-1",
        return_type="simple",
        price_type="close",
        adjustment_state="raw",
        training_interval=(pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2023-01-31", tz="UTC")),
        training_cutoff=pd.Timestamp("2023-02-20", tz="UTC"),
        training_information_cutoff=pd.Timestamp("2023-02-20", tz="UTC"),
        validation_interval=(pd.Timestamp("2023-01-31", tz="UTC"), pd.Timestamp("2023-02-20", tz="UTC")),
        test_interval=(pd.Timestamp("2023-02-20", tz="UTC"), pd.Timestamp("2023-03-12", tz="UTC")),
        kmeans_config={"n_clusters": 2, "random_state": 42},
        scaler_parameters={"mean_": [0.0, 0.0], "scale_": [1.0, 1.0]},
        canonical_centroids=((0.0, 0.0), (1.0, 1.0)),
        training_occupancy={"0": 10, "1": 20},
        assignment_mode=M3AvailabilityMode.AS_OF_PRODUCTION,
        actual_creation_timestamp_utc=pd.Timestamp("2023-03-12", tz="UTC"),
        effective_model_availability=pd.Timestamp("2023-02-20", tz="UTC"),
        seed=42,
    )
    changed_k = M3ModelArtifact(
        model_contract_version=base.model_contract_version,
        instrument=base.instrument,
        requested_feature_instance_ids=base.requested_feature_instance_ids,
        effective_feature_instance_ids=base.effective_feature_instance_ids,
        dropped_constant_feature_ids=base.dropped_constant_feature_ids,
        feature_spec_set_hash=base.feature_spec_set_hash,
        parent_provenance_digest=base.parent_provenance_digest,
        return_type=base.return_type,
        price_type=base.price_type,
        adjustment_state=base.adjustment_state,
        training_interval=base.training_interval,
        training_cutoff=base.training_cutoff,
        training_information_cutoff=base.training_information_cutoff,
        validation_interval=base.validation_interval,
        test_interval=base.test_interval,
        kmeans_config={"n_clusters": 3, "random_state": 42},
        scaler_parameters=base.scaler_parameters,
        canonical_centroids=base.canonical_centroids,
        training_occupancy=base.training_occupancy,
        assignment_mode=base.assignment_mode,
        actual_creation_timestamp_utc=base.actual_creation_timestamp_utc,
        simulated_replay_availability=base.simulated_replay_availability,
        effective_model_availability=base.effective_model_availability,
        seed=base.seed,
    )
    changed_seed = M3ModelArtifact(
        model_contract_version=base.model_contract_version,
        instrument=base.instrument,
        requested_feature_instance_ids=base.requested_feature_instance_ids,
        effective_feature_instance_ids=base.effective_feature_instance_ids,
        dropped_constant_feature_ids=base.dropped_constant_feature_ids,
        feature_spec_set_hash=base.feature_spec_set_hash,
        parent_provenance_digest=base.parent_provenance_digest,
        return_type=base.return_type,
        price_type=base.price_type,
        adjustment_state=base.adjustment_state,
        training_interval=base.training_interval,
        training_cutoff=base.training_cutoff,
        training_information_cutoff=base.training_information_cutoff,
        validation_interval=base.validation_interval,
        test_interval=base.test_interval,
        kmeans_config=base.kmeans_config,
        scaler_parameters=base.scaler_parameters,
        canonical_centroids=base.canonical_centroids,
        training_occupancy=base.training_occupancy,
        assignment_mode=base.assignment_mode,
        actual_creation_timestamp_utc=base.actual_creation_timestamp_utc,
        simulated_replay_availability=base.simulated_replay_availability,
        effective_model_availability=base.effective_model_availability,
        seed=43,
    )
    changed_feature = M3ModelArtifact(
        model_contract_version=base.model_contract_version,
        instrument=base.instrument,
        requested_feature_instance_ids=base.requested_feature_instance_ids,
        effective_feature_instance_ids=("f1",),
        dropped_constant_feature_ids=base.dropped_constant_feature_ids,
        feature_spec_set_hash=base.feature_spec_set_hash,
        parent_provenance_digest=base.parent_provenance_digest,
        return_type=base.return_type,
        price_type=base.price_type,
        adjustment_state=base.adjustment_state,
        training_interval=base.training_interval,
        training_cutoff=base.training_cutoff,
        training_information_cutoff=base.training_information_cutoff,
        validation_interval=base.validation_interval,
        test_interval=base.test_interval,
        kmeans_config=base.kmeans_config,
        scaler_parameters=base.scaler_parameters,
        canonical_centroids=base.canonical_centroids,
        training_occupancy=base.training_occupancy,
        assignment_mode=base.assignment_mode,
        actual_creation_timestamp_utc=base.actual_creation_timestamp_utc,
        simulated_replay_availability=base.simulated_replay_availability,
        effective_model_availability=base.effective_model_availability,
        seed=base.seed,
    )
    changed_cutoff = M3ModelArtifact(
        model_contract_version=base.model_contract_version,
        instrument=base.instrument,
        requested_feature_instance_ids=base.requested_feature_instance_ids,
        effective_feature_instance_ids=base.effective_feature_instance_ids,
        dropped_constant_feature_ids=base.dropped_constant_feature_ids,
        feature_spec_set_hash=base.feature_spec_set_hash,
        parent_provenance_digest=base.parent_provenance_digest,
        return_type=base.return_type,
        price_type=base.price_type,
        adjustment_state=base.adjustment_state,
        training_interval=base.training_interval,
        training_cutoff=pd.Timestamp("2023-02-21", tz="UTC"),
        training_information_cutoff=base.training_information_cutoff,
        validation_interval=base.validation_interval,
        test_interval=base.test_interval,
        kmeans_config=base.kmeans_config,
        scaler_parameters=base.scaler_parameters,
        canonical_centroids=base.canonical_centroids,
        training_occupancy=base.training_occupancy,
        assignment_mode=base.assignment_mode,
        actual_creation_timestamp_utc=base.actual_creation_timestamp_utc,
        simulated_replay_availability=base.simulated_replay_availability,
        effective_model_availability=base.effective_model_availability,
        seed=base.seed,
    )
    changed_provenance = M3ModelArtifact(
        model_contract_version=base.model_contract_version,
        instrument=base.instrument,
        requested_feature_instance_ids=base.requested_feature_instance_ids,
        effective_feature_instance_ids=base.effective_feature_instance_ids,
        dropped_constant_feature_ids=base.dropped_constant_feature_ids,
        feature_spec_set_hash=base.feature_spec_set_hash,
        parent_provenance_digest="digest-2",
        return_type=base.return_type,
        price_type=base.price_type,
        adjustment_state=base.adjustment_state,
        training_interval=base.training_interval,
        training_cutoff=base.training_cutoff,
        training_information_cutoff=base.training_information_cutoff,
        validation_interval=base.validation_interval,
        test_interval=base.test_interval,
        kmeans_config=base.kmeans_config,
        scaler_parameters=base.scaler_parameters,
        canonical_centroids=base.canonical_centroids,
        training_occupancy=base.training_occupancy,
        assignment_mode=base.assignment_mode,
        actual_creation_timestamp_utc=base.actual_creation_timestamp_utc,
        simulated_replay_availability=base.simulated_replay_availability,
        effective_model_availability=base.effective_model_availability,
        seed=base.seed,
    )
    changed_creation = M3ModelArtifact(
        model_contract_version=base.model_contract_version,
        instrument=base.instrument,
        requested_feature_instance_ids=base.requested_feature_instance_ids,
        effective_feature_instance_ids=base.effective_feature_instance_ids,
        dropped_constant_feature_ids=base.dropped_constant_feature_ids,
        feature_spec_set_hash=base.feature_spec_set_hash,
        parent_provenance_digest=base.parent_provenance_digest,
        return_type=base.return_type,
        price_type=base.price_type,
        adjustment_state=base.adjustment_state,
        training_interval=base.training_interval,
        training_cutoff=base.training_cutoff,
        training_information_cutoff=base.training_information_cutoff,
        validation_interval=base.validation_interval,
        test_interval=base.test_interval,
        kmeans_config=base.kmeans_config,
        scaler_parameters=base.scaler_parameters,
        canonical_centroids=base.canonical_centroids,
        training_occupancy=base.training_occupancy,
        assignment_mode=base.assignment_mode,
        actual_creation_timestamp_utc=pd.Timestamp("2023-03-13", tz="UTC"),
        simulated_replay_availability=base.simulated_replay_availability,
        effective_model_availability=base.effective_model_availability,
        seed=base.seed,
    )
    assert base.model_id == changed_creation.model_id
    assert base.model_id != changed_k.model_id
    assert base.model_id != changed_seed.model_id
    assert base.model_id != changed_feature.model_id
    assert base.model_id != changed_provenance.model_id
    assert base.model_id != changed_cutoff.model_id


def test_assignment_schema_and_empty_output_dtypes() -> None:
    expected_columns = [
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
    ]
    matrix = build_m3_feature_matrix(_base_feature_frame(), requested_feature_instance_ids=["f1", "f2"])
    matrix = matrix[matrix["scope_id"] == "AAA"].copy()
    model = fit_m3_regime_model(matrix, **_fit_kwargs())
    result = assign_m3_regimes(model, matrix, decision_time=pd.Timestamp("2023-02-20", tz="UTC"))
    assert list(result.columns) == expected_columns
    assert result["cluster_id"].dtype == "Int64"
    assert result["observation_timestamp"].dtype.kind == "M"
    assert result["decision_timestamp"].dtype.kind == "M"


def test_build_feature_matrix_does_not_mutate_input_and_accepts_valid_provenance() -> None:
    table = _base_feature_frame()
    original = table.copy(deep=True)
    transformed = build_m3_feature_matrix(table, requested_feature_instance_ids=["f1", "f2"])
    assert table.equals(original)
    assert transformed.attrs["provenance"]["feature_spec_set_hash"] == "hash-1"
    bad = table.copy()
    bad.attrs.pop("provenance", None)
    with pytest.raises(M3ValidationError, match="provenance"):
        build_m3_feature_matrix(bad, requested_feature_instance_ids=["f1", "f2"])
    mixed_return = table.copy()
    mixed_return.loc[mixed_return["feature_instance_id"] == "f2", "input_return_type"] = "log"
    with pytest.raises(M3ValidationError, match="single input_return_type"):
        build_m3_feature_matrix(mixed_return, requested_feature_instance_ids=["f1", "f2"])


def test_build_m3_feature_matrix_accepts_genuine_frozen_m2_feature_frame() -> None:
    returns = _returns_table()
    m2_features, _ = compute_features_from_returns(returns, list_m2_v1_feature_specs())
    requested = sorted({str(value) for value in m2_features["feature_instance_id"].dropna().unique().tolist()})

    matrix = build_m3_feature_matrix(m2_features, requested_feature_instance_ids=requested)
    assert not matrix.empty
    provenance = matrix.attrs["provenance"]
    assert provenance["feature_spec_set_hash"] == m2_features.attrs["provenance"]["feature_spec_set_hash"]
    assert provenance["return_type"] == "simple"
    assert provenance["price_type"] == "close"
    assert provenance["adjustment_state"] == "raw"
    assert provenance["parent_return_provenance_digest"] == m2_features.attrs["provenance"]["parent_return_provenance_digest"]


def test_build_m3_feature_matrix_rejects_mixed_m2_row_metadata() -> None:
    returns = _returns_table()
    m2_features, _ = compute_features_from_returns(returns, list_m2_v1_feature_specs())
    bad = m2_features.copy()
    bad.loc[bad["feature_instance_id"] == bad["feature_instance_id"].iloc[0], "input_return_type"] = "log"
    with pytest.raises(M3ValidationError, match="single input_return_type|return_type"):
        build_m3_feature_matrix(bad, requested_feature_instance_ids=sorted({str(value) for value in bad["feature_instance_id"].dropna().unique().tolist()}))


def _valid_matrix_and_model() -> tuple[pd.DataFrame, M3ModelArtifact]:
    table = _base_feature_frame()
    matrix = build_m3_feature_matrix(table[table["scope_id"] == "AAA"].copy(), requested_feature_instance_ids=["f1", "f2"])
    model = fit_m3_regime_model(matrix, **_fit_kwargs())
    return matrix, model


def test_assignment_accepts_valid_matching_provenance() -> None:
    matrix, model = _valid_matrix_and_model()
    result = assign_m3_regimes(model, matrix, decision_time=pd.Timestamp("2023-02-20", tz="UTC"))
    assert not result.empty
    assert set(result.columns) == {
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
    }


@pytest.mark.parametrize(
    "mutator",
    [
        lambda frame: frame.attrs.clear(),
        lambda frame: frame.attrs.pop("provenance", None),
        lambda frame: frame.attrs.__setitem__("provenance", None),
        lambda frame: frame.attrs.__setitem__("provenance", {}),
        lambda frame: frame.attrs.__setitem__("provenance", {"feature_spec_set_hash": "hash-1"}),
        lambda frame: frame.attrs.__setitem__("provenance", ["not", "a", "dict"]),
    ],
)
def test_assignment_rejects_missing_or_malformed_provenance(mutator) -> None:
    matrix, model = _valid_matrix_and_model()
    bad = matrix.copy()
    mutator(bad)
    with pytest.raises(M3ValidationError, match="provenance"):
        assign_m3_regimes(model, bad, decision_time=pd.Timestamp("2023-02-20", tz="UTC"))


def test_assignment_empty_matrix_with_valid_provenance_returns_typed_empty_output() -> None:
    matrix, model = _valid_matrix_and_model()
    empty = matrix.iloc[0:0].copy()
    empty.attrs["provenance"] = matrix.attrs["provenance"]
    result = assign_m3_regimes(model, empty, decision_time=pd.Timestamp("2023-02-20", tz="UTC"))
    assert result.empty
    assert list(result.columns) == [
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
    ]
    assert result["cluster_id"].dtype == "Int64"


def test_assignment_rejects_mismatched_artifact_and_matrix_metadata() -> None:
    table = _base_feature_frame()
    matrix = build_m3_feature_matrix(table[table["scope_id"] == "AAA"].copy(), requested_feature_instance_ids=["f1", "f2"])
    model = fit_m3_regime_model(matrix, **_fit_kwargs())
    decision_time = pd.Timestamp("2023-02-20", tz="UTC")

    mismatches = [
        ("feature_spec_set_hash", {"feature_spec_set_hash": "hash-999"}),
        ("parent_provenance_digest", {"parent_provenance_digest": "digest-999"}),
        ("effective_feature_instance_ids", {"effective_feature_instance_ids": ("f9",)}),
        ("return_type", {"return_type": "log"}),
        ("price_type", {"price_type": "VWAP"}),
        ("adjustment_state", {"adjustment_state": "split"}),
    ]
    for _, payload in mismatches:
        artifact = replace(model, **payload)
        with pytest.raises(M3ValidationError):
            assign_m3_regimes(artifact, matrix, decision_time=decision_time)

    bad_return = table[table["scope_id"] == "AAA"].copy()
    bad_return.loc[:, "input_return_type"] = "log"
    bad_return.attrs["provenance"] = {**table.attrs["provenance"], "return_type": "simple"}
    with pytest.raises(M3ValidationError):
        build_m3_feature_matrix(bad_return, requested_feature_instance_ids=["f1", "f2"])


def test_assignment_accepts_valid_constant_feature_exclusion() -> None:
    table = _base_feature_frame()
    table.loc[(table["scope_id"] == "AAA") & (table["feature_instance_id"] == "f2"), "feature_value"] = 5.0
    matrix = build_m3_feature_matrix(table[table["scope_id"] == "AAA"].copy(), requested_feature_instance_ids=["f1", "f2"])
    model = fit_m3_regime_model(matrix, **_fit_kwargs())
    assert model.effective_feature_instance_ids == ("f1",)
    result = assign_m3_regimes(model, matrix, decision_time=pd.Timestamp("2023-02-20", tz="UTC"))
    assert result["assignment_status"].isin(["assigned", "abstained"]).all()
    assert "f2" in model.dropped_constant_feature_ids


def test_rejects_invalid_artifact_before_assignment() -> None:
    artifact = M3ModelArtifact(
        instrument="AAA",
        requested_feature_instance_ids=("f1",),
        effective_feature_instance_ids=("f1",),
        feature_spec_set_hash="hash-1",
        parent_provenance_digest="digest-1",
        return_type="simple",
        price_type="close",
        adjustment_state="raw",
        training_interval=(pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2023-01-31", tz="UTC")),
        training_cutoff=pd.Timestamp("2023-02-20", tz="UTC"),
        training_information_cutoff=pd.Timestamp("2023-02-20", tz="UTC"),
        validation_interval=(pd.Timestamp("2023-01-31", tz="UTC"), pd.Timestamp("2023-02-20", tz="UTC")),
        test_interval=(pd.Timestamp("2023-02-20", tz="UTC"), pd.Timestamp("2023-03-12", tz="UTC")),
        kmeans_config={"n_clusters": 2, "random_state": 42},
        scaler_parameters={"mean_": [1.0], "scale_": [0.0]},
        canonical_centroids=((0.0,), (0.0,)),
        training_occupancy={"0": 1, "1": 0},
        assignment_mode=M3AvailabilityMode.AS_OF_PRODUCTION,
        actual_creation_timestamp_utc=pd.Timestamp("2023-03-13", tz="UTC"),
        effective_model_availability=pd.Timestamp("2023-02-20", tz="UTC"),
        seed=42,
    )
    with pytest.raises(M3ValidationError, match="duplicate centroid|empty cluster|training_occupancy|scale"):
        validate_m3_model_artifact(artifact)

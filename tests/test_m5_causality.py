from __future__ import annotations

import pandas as pd
import pytest

from regimequant.regimes.m5 import M5AvailabilityMode, assign_m5_regimes, fit_m5_regime_model


def _make_matrix() -> pd.DataFrame:
    timestamps = pd.date_range("2023-01-01", periods=61, freq="D", tz="UTC")
    rows = []
    for idx, ts in enumerate(timestamps):
        rows.append({
            "scope_id": "CAUSAL",
            "timestamp": ts,
            "feature_vector_available_at": ts,
            "feature_observation_available": True,
            "feature_a": float(idx),
            "feature_b": float(idx) * 2.0,
        })
    matrix = pd.DataFrame(rows)
    matrix.attrs["provenance"] = {
        "feature_spec_set_hash": "hash-causal",
        "generated_at_utc": pd.Timestamp("2023-01-01", tz="UTC"),
        "metadata_durability": "published",
    }
    return matrix


def test_production_and_replay_modes_are_separate() -> None:
    matrix = _make_matrix()
    timestamps = matrix["timestamp"]
    artifact = fit_m5_regime_model(
        matrix,
        instrument="CAUSAL",
        requested_feature_ids=["feature_a", "feature_b"],
        training_interval=(timestamps[0], timestamps[40]),
        validation_interval=(timestamps[40], timestamps[50]),
        test_interval=(timestamps[50], timestamps[60]),
        training_information_cutoff=timestamps[39],
        training_cutoff=timestamps[39],
        availability_mode=M5AvailabilityMode.AS_OF_PRODUCTION,
        n_components=2,
    )
    with pytest.raises(Exception):
        assign_m5_regimes(
            artifact,
            matrix,
            decision_time=timestamps[10],
            availability_mode=M5AvailabilityMode.HISTORICAL_REPLAY,
        )


def test_late_feature_availability_abstains() -> None:
    matrix = _make_matrix()
    timestamps = matrix["timestamp"]
    artifact = fit_m5_regime_model(
        matrix,
        instrument="CAUSAL",
        requested_feature_ids=["feature_a", "feature_b"],
        training_interval=(timestamps[0], timestamps[40]),
        validation_interval=(timestamps[40], timestamps[50]),
        test_interval=(timestamps[50], timestamps[60]),
        training_information_cutoff=timestamps[39],
        training_cutoff=timestamps[39],
        availability_mode=M5AvailabilityMode.AS_OF_PRODUCTION,
        n_components=2,
    )
    matrix2 = matrix.copy()
    matrix2.loc[0, "feature_vector_available_at"] = timestamps[4]
    result = assign_m5_regimes(
        artifact,
        matrix2,
        decision_time=timestamps[3],
        availability_mode=M5AvailabilityMode.AS_OF_PRODUCTION,
    )
    assert result.iloc[0]["assignment_status"] == "abstained"
    assert result.iloc[0]["abstention_reason"] == "late_feature_availability"

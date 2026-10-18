from __future__ import annotations

import hashlib
import json

import pandas as pd

from regimequant.regimes.m5 import M5AvailabilityMode, fit_m5_regime_model


def _make_matrix() -> pd.DataFrame:
    timestamps = pd.date_range("2023-01-01", periods=80, freq="D", tz="UTC")
    rows = []
    for idx, ts in enumerate(timestamps):
        rows.append({
            "scope_id": "REPRO",
            "timestamp": ts,
            "feature_vector_available_at": ts,
            "feature_observation_available": True,
            "feature_a": float(idx) + 0.4,
            "feature_b": float(idx) * 1.2 + 0.9,
        })
    matrix = pd.DataFrame(rows)
    matrix.attrs["provenance"] = {
        "feature_spec_set_hash": "hash-repro",
        "generated_at_utc": pd.Timestamp("2023-01-01", tz="UTC"),
        "metadata_durability": "published",
    }
    return matrix


def test_training_input_digest_is_row_order_invariant() -> None:
    matrix = _make_matrix()
    training_end = matrix["timestamp"].iloc[39]
    model_a = fit_m5_regime_model(
        matrix,
        instrument="REPRO",
        requested_feature_ids=["feature_a", "feature_b"],
        training_interval=(matrix["timestamp"].min(), training_end),
        validation_interval=(training_end, matrix["timestamp"].iloc[59]),
        test_interval=(matrix["timestamp"].iloc[59], matrix["timestamp"].max()),
        training_information_cutoff=training_end,
        training_cutoff=training_end,
        availability_mode=M5AvailabilityMode.AS_OF_PRODUCTION,
        n_components=2,
    )
    reversed_matrix = matrix.iloc[::-1].copy()
    reversed_matrix.attrs = dict(matrix.attrs)
    model_b = fit_m5_regime_model(
        reversed_matrix,
        instrument="REPRO",
        requested_feature_ids=["feature_a", "feature_b"],
        training_interval=(reversed_matrix["timestamp"].min(), training_end),
        validation_interval=(training_end, matrix["timestamp"].iloc[59]),
        test_interval=(matrix["timestamp"].iloc[59], matrix["timestamp"].max()),
        training_information_cutoff=training_end,
        training_cutoff=training_end,
        availability_mode=M5AvailabilityMode.AS_OF_PRODUCTION,
        n_components=2,
    )
    assert model_a.training_input_digest == model_b.training_input_digest
    assert model_a.model_id == model_b.model_id

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .contracts import (
    M3ValidationError,
    REQUIRED_M2_FEATURE_COLUMNS,
    validate_m2_feature_provenance,
    validate_m3_feature_row,
)


def build_m3_feature_matrix(
    feature_frame: pd.DataFrame,
    *,
    requested_feature_instance_ids: Iterable[str] | None = None,
    instrument_column: str = "scope_id",
    timestamp_column: str = "timestamp",
    feature_id_column: str = "feature_instance_id",
    feature_value_column: str = "feature_value",
    math_status_column: str = "math_status",
    availability_column: str = "input_availability_status",
    available_at_column: str = "available_at",
) -> pd.DataFrame:
    """Build an M3 feature matrix with one row per instrument/timestamp."""

    if feature_frame is None:
        raise M3ValidationError("feature_frame must be a pandas DataFrame")
    if not isinstance(feature_frame, pd.DataFrame):
        raise M3ValidationError("feature_frame must be a pandas DataFrame")

    frame = feature_frame.copy()
    provenance = validate_m2_feature_provenance(frame)

    required = REQUIRED_M2_FEATURE_COLUMNS
    missing = sorted(required - set(frame.columns))
    if missing:
        raise M3ValidationError(f"missing required M2 feature columns: {missing!r}")

    frame = frame.reset_index(drop=True)
    key_columns = ["scope_id", "timestamp", "feature_instance_id"]
    duplicates = frame.duplicated(subset=key_columns, keep=False)
    if duplicates.any():
        raise M3ValidationError("duplicate M2 feature keys are not allowed before M3 model fitting")

    for row in frame.to_dict("records"):
        validate_m3_feature_row(row)

    unique_return_types = sorted({str(v) for v in frame["input_return_type"].astype(str).tolist()})
    if len(unique_return_types) != 1:
        raise M3ValidationError("M3 requires a single input_return_type across the feature frame")
    unique_price = sorted({str(v) for v in frame["price_type"].astype(str).tolist()})
    if len(unique_price) != 1:
        raise M3ValidationError("M3 requires a single price_type across the feature frame")
    unique_adjustment = sorted({str(v) for v in frame["adjustment_state"].astype(str).tolist()})
    if len(unique_adjustment) != 1:
        raise M3ValidationError("M3 requires a single adjustment_state across the feature frame")

    derived_provenance = dict(provenance)
    derived_provenance["return_type"] = unique_return_types[0]
    derived_provenance["price_type"] = unique_price[0]
    derived_provenance["adjustment_state"] = unique_adjustment[0]

    for key, value in {"return_type": unique_return_types[0], "price_type": unique_price[0], "adjustment_state": unique_adjustment[0]}.items():
        if provenance.get(key) is not None and str(provenance.get(key)) != str(value):
            raise M3ValidationError(f"attrs.provenance.{key} does not match the feature rows")

    requested = tuple(dict.fromkeys(str(value) for value in (requested_feature_instance_ids or sorted(frame[feature_id_column].dropna().unique().tolist()))))
    filtered = frame[frame[feature_id_column].isin(requested)].copy()
    result = pd.DataFrame(columns=[instrument_column, timestamp_column, *requested, "feature_vector_available_at", "feature_observation_available"])
    result.attrs["provenance"] = derived_provenance
    if filtered.empty:
        return result

    pivot = filtered.pivot_table(
        index=[instrument_column, timestamp_column],
        columns=feature_id_column,
        values=feature_value_column,
        aggfunc="first",
    ).reset_index()

    for feature in requested:
        if feature not in pivot.columns:
            pivot[feature] = np.nan

    pivot = pivot.loc[:, [instrument_column, timestamp_column, *requested]].copy()
    availability = filtered.groupby([instrument_column, timestamp_column])[available_at_column].max()
    reindex_index = pivot.set_index([instrument_column, timestamp_column]).index
    pivot["feature_vector_available_at"] = availability.reindex(reindex_index).to_numpy()
    pivot["feature_observation_available"] = pivot.loc[:, requested].notna().all(axis=1)
    pivot.attrs["provenance"] = derived_provenance
    return pivot.reset_index(drop=True)

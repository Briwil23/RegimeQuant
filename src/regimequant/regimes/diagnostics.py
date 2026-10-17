from __future__ import annotations

from typing import Any

import pandas as pd

from .contracts import M3ValidationError


def summarize_m3_diagnostics(model_artifact: Any) -> dict[str, Any]:
    if hasattr(model_artifact, "training_occupancy"):
        return {
            "model_id": model_artifact.model_id,
            "effective_feature_count": len(model_artifact.effective_feature_instance_ids),
            "dropped_constant_feature_ids": list(model_artifact.dropped_constant_feature_ids),
            "training_occupancy": dict(model_artifact.training_occupancy),
        }
    raise M3ValidationError("model_artifact is not a recognized M3 artifact")


def validate_assignment_status(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    required = {"cluster_id", "assignment_status", "abstention_reason"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise M3ValidationError(f"assignment frame missing required columns: {missing!r}")

    assigned = frame[frame["assignment_status"] == "assigned"].copy()
    if not assigned.empty:
        if assigned["cluster_id"].isna().any():
            raise M3ValidationError("assigned rows cannot carry null cluster_id")
        if assigned["abstention_reason"].notna().any():
            raise M3ValidationError("assigned rows cannot carry a non-null abstention_reason")

    abstained = frame[frame["assignment_status"] == "abstained"].copy()
    if not abstained.empty:
        if abstained["cluster_id"].notna().any():
            raise M3ValidationError("abstained rows cannot carry a non-null cluster_id")
        if abstained["abstention_reason"].isna().any():
            raise M3ValidationError("abstained rows require a non-null abstention_reason")
    return frame.copy()

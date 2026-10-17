"""RegimeQuant M3 regime-discovery public API."""

from .assignment import assign_m3_regimes
from .contracts import (
    M3AssignmentStatus,
    M3AvailabilityMode,
    M3Error,
    M3ModelArtifact,
    M3ModelUnavailableError,
    M3ValidationError,
)
from .diagnostics import summarize_m3_diagnostics, validate_assignment_status
from .matrix import build_m3_feature_matrix
from .model import fit_m3_regime_model, validate_m3_model_artifact
from .splits import build_m3_split_plan

__all__ = [
    "M3AssignmentStatus",
    "M3AvailabilityMode",
    "M3Error",
    "M3ModelArtifact",
    "M3ModelUnavailableError",
    "M3ValidationError",
    "assign_m3_regimes",
    "build_m3_feature_matrix",
    "build_m3_split_plan",
    "fit_m3_regime_model",
    "summarize_m3_diagnostics",
    "validate_assignment_status",
    "validate_m3_model_artifact",
]

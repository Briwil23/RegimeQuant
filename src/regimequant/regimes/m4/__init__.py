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
from .metrics import compute_m4_regime_stability_report, compute_requested_domain
from .report import compute_m4_regime_stability_report as _compute_report
from .synthetic import build_synthetic_m4_fixture
from .validation import validate_m4_report

__all__ = [
    "M4AssignmentStatus",
    "M4CentroidDiagnostics",
    "M4ClusterOccupancy",
    "M4Coverage",
    "M4OverallPersistence",
    "M4RegimeStabilityReport",
    "M4TransitionEdge",
    "M4TransitionProbability",
    "M4UndefinedMetric",
    "build_synthetic_m4_fixture",
    "compute_m4_regime_stability_report",
    "compute_m4_report_identity",
    "compute_requested_domain",
    "validate_m4_report",
]

# Keep the canonical primary symbol aligned with the report-generation API.
compute_m4_regime_stability_report = _compute_report

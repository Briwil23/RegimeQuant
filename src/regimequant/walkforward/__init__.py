"""M10 walk-forward infrastructure.

This Phase 1 package defines immutable policy, structural contracts, and
canonical identity logic for the owner-ratified M10 research-only workflow.
It deliberately avoids model fitting and execution logic.
"""

from .contracts import (
    AccountingStatus,
    ExecutionStatus,
    ExperimentPlan,
    ExperimentResult,
    FoldCertification,
    FoldEvidenceArtifact,
    FoldInput,
    FoldInterval,
    FoldOutput,
    IdentityStatus,
    M10Policy,
    MetricStatus,
    NoEntryArtifact,
    ScientificStatus,
    ValidationResult,
)
from .identity import (
    build_experiment_result_digest,
    build_fold_input_digest,
    build_no_entry_digest,
    canonical_json,
    compute_identity_digest,
    normalize_timestamp_utc,
)
from .manifest import M10_RATIFIED_POLICY, build_experiment_plan, validate_experiment_plan

__all__ = [
    "AccountingStatus",
    "ExecutionStatus",
    "ExperimentPlan",
    "ExperimentResult",
    "FoldCertification",
    "FoldEvidenceArtifact",
    "FoldInput",
    "FoldInterval",
    "FoldOutput",
    "IdentityStatus",
    "M10Policy",
    "MetricStatus",
    "NoEntryArtifact",
    "ScientificStatus",
    "ValidationResult",
    "M10_RATIFIED_POLICY",
    "build_experiment_plan",
    "validate_experiment_plan",
    "build_experiment_result_digest",
    "build_fold_input_digest",
    "build_no_entry_digest",
    "canonical_json",
    "compute_identity_digest",
    "normalize_timestamp_utc",
]

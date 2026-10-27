from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Literal

ScientificStatus = Literal[
    "VALID_RESEARCH_EXECUTION",
    "VALID_NO_TRADE_ABSTENTION",
    "INSUFFICIENT_HISTORY",
    "MISSING_CAUSAL_EVIDENCE",
    "INVALID_SCIENTIFIC_IDENTITY",
    "UNCERTIFIABLE_ACCOUNTING",
    "UNSUPPORTED_EXECUTION",
]

MetricStatus = Literal[
    "METRICS_VALID",
    "METRICS_INVALID",
    "METRICS_MISSING",
    "METRICS_INSUFFICIENT",
    "METRICS_UNSUPPORTED",
]

ExecutionStatus = Literal["FILLED", "NO_TRADE", "ABSTAINED", "UNCERTIFIABLE"]
AccountingStatus = Literal["CERTIFIED", "UNCERTIFIABLE", "MISSING"]
IdentityStatus = Literal["VERIFIED", "FAILED", "MISSING"]


@dataclass(frozen=True, slots=True)
class M10Policy:
    initial_training_sessions: int = 120
    validation_sessions: int = 20
    outer_test_sessions: int = 20
    stride_sessions: int = 20
    final_holdout_sessions: int = 20
    formation_min_aligned_levels: int = 253
    discovery_freshness_limit_completed_sessions: int = 20
    discovery_refresh_sessions: int = 20
    by_q_threshold: float = 0.05
    empty_signal_policy: str = "valid_abstention_only"
    research_mode: Literal["research-only"] = "research-only"

    def __post_init__(self) -> None:
        if self.initial_training_sessions != 120:
            raise ValueError("initial_training_sessions must equal 120")
        if self.validation_sessions != 20:
            raise ValueError("validation_sessions must equal 20")
        if self.outer_test_sessions != 20:
            raise ValueError("outer_test_sessions must equal 20")
        if self.stride_sessions != 20:
            raise ValueError("stride_sessions must equal 20")
        if self.final_holdout_sessions != 20:
            raise ValueError("final_holdout_sessions must equal 20")
        if self.formation_min_aligned_levels != 253:
            raise ValueError("formation_min_aligned_levels must equal 253")
        if self.discovery_freshness_limit_completed_sessions != 20:
            raise ValueError("discovery_freshness_limit_completed_sessions must equal 20")
        if self.discovery_refresh_sessions != 20:
            raise ValueError("discovery_refresh_sessions must equal 20")
        if not 0.0 <= float(self.by_q_threshold) <= 1.0:
            raise ValueError("by_q_threshold must be in [0, 1]")
        if float(self.by_q_threshold) != 0.05:
            raise ValueError("by_q_threshold must equal 0.05")
        if self.empty_signal_policy != "valid_abstention_only":
            raise ValueError("empty_signal_policy must be 'valid_abstention_only'")
        if self.research_mode != "research-only":
            raise ValueError("research_mode must be 'research-only'")


@dataclass(frozen=True, slots=True)
class FoldInterval:
    train_start: int
    train_end: int
    validation_start: int
    validation_end: int
    test_start: int
    test_end: int
    holdout_start: int
    holdout_end: int

    def __post_init__(self) -> None:
        if self.train_start < 0:
            raise ValueError("train_start must be non-negative")
        ordered = [
            ("train_start", self.train_start),
            ("train_end", self.train_end),
            ("validation_start", self.validation_start),
            ("validation_end", self.validation_end),
            ("test_start", self.test_start),
            ("test_end", self.test_end),
            ("holdout_start", self.holdout_start),
            ("holdout_end", self.holdout_end),
        ]
        for name, value in ordered:
            if not isinstance(value, int):
                raise ValueError(f"{name} must be an int")
        if self.train_end < self.train_start:
            raise ValueError("train_end cannot precede train_start")
        if self.validation_start < self.train_end:
            raise ValueError("validation_start cannot precede train_end")
        if self.validation_end < self.validation_start:
            raise ValueError("validation_end cannot precede validation_start")
        if self.test_start < self.validation_end:
            raise ValueError("test_start cannot precede validation_end")
        if self.test_end < self.test_start:
            raise ValueError("test_end cannot precede test_start")
        if self.holdout_start < self.test_end:
            raise ValueError("holdout_start cannot precede test_end")
        if self.holdout_end < self.holdout_start:
            raise ValueError("holdout_end cannot precede holdout_start")


@dataclass(frozen=True, slots=True)
class FoldEvidenceArtifact:
    family_provenance_digest: str
    calendar_provenance_digest: str
    model_provenance_digest: str
    validation_digest: str
    evidence_digest: str
    provenance_state: Literal["UNVERIFIED", "CALLER_SUPPLIED", "MISSING"] = "UNVERIFIED"
    family_provenance_state: Literal["UNVERIFIED", "CALLER_SUPPLIED", "MISSING"] = "UNVERIFIED"

    def __post_init__(self) -> None:
        for name in [
            "family_provenance_digest",
            "calendar_provenance_digest",
            "model_provenance_digest",
            "validation_digest",
            "evidence_digest",
        ]:
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.provenance_state not in {"UNVERIFIED", "CALLER_SUPPLIED", "MISSING"}:
            raise ValueError("provenance_state must be one of UNVERIFIED, CALLER_SUPPLIED, MISSING")
        if self.family_provenance_state not in {"UNVERIFIED", "CALLER_SUPPLIED", "MISSING"}:
            raise ValueError("family_provenance_state must be one of UNVERIFIED, CALLER_SUPPLIED, MISSING")


@dataclass(frozen=True, slots=True)
class FoldInput:
    schema: str
    version: str
    plan_id: str
    fold_id: str
    interval: FoldInterval
    dataset_identity: str
    feature_spec_set_hash: str
    calendar_identity: str
    evidence: FoldEvidenceArtifact
    model_policy_identity: str
    formation_policy_identity: str
    family_policy_identity: str
    input_digest: str | None = None

    def __post_init__(self) -> None:
        if self.schema != "regimequant.walkforward.fold_input":
            raise ValueError("schema must be 'regimequant.walkforward.fold_input'")
        if self.version != "m10.v1":
            raise ValueError("version must be 'm10.v1'")
        for field_name in [
            "plan_id",
            "fold_id",
            "dataset_identity",
            "feature_spec_set_hash",
            "calendar_identity",
            "model_policy_identity",
            "formation_policy_identity",
            "family_policy_identity",
        ]:
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class ValidationResult:
    valid: bool
    scientific_status: ScientificStatus
    metric_status: MetricStatus
    reason_code: str
    details: str | None = None
    authoritative: bool = False
    status_role: Literal["STRUCTURAL_ONLY"] = "STRUCTURAL_ONLY"

    def __post_init__(self) -> None:
        if not isinstance(self.reason_code, str) or not self.reason_code.strip():
            raise ValueError("reason_code must be a non-empty string")
        if self.authoritative is not False:
            raise ValueError("ValidationResult must remain non-authoritative in Phase 1")
        if self.status_role != "STRUCTURAL_ONLY":
            raise ValueError("ValidationResult status_role must be STRUCTURAL_ONLY")


@dataclass(frozen=True, slots=True)
class FoldCertification:
    fold_id: str
    scientific_status: ScientificStatus
    metric_status: MetricStatus
    execution_status: ExecutionStatus
    accounting_status: AccountingStatus
    identity_status: IdentityStatus
    failure_codes: tuple[str, ...] = field(default_factory=tuple)
    authoritative: bool = False
    certification_role: Literal["STRUCTURAL_ONLY"] = "STRUCTURAL_ONLY"

    def __post_init__(self) -> None:
        if not self.fold_id.strip():
            raise ValueError("fold_id must be non-empty")
        if self.authoritative is not False:
            raise ValueError("FoldCertification must remain non-authoritative in Phase 1")
        if self.certification_role != "STRUCTURAL_ONLY":
            raise ValueError("FoldCertification certification_role must be STRUCTURAL_ONLY")


@dataclass(frozen=True, slots=True)
class FoldOutput:
    schema: str
    version: str
    plan_id: str
    fold_id: str
    model_digest: str
    family_digest: str
    no_trade_digest: str
    accounting_digest: str
    decision_digest: str
    output_digest: str | None = None

    def __post_init__(self) -> None:
        if self.schema != "regimequant.walkforward.fold_output":
            raise ValueError("schema must be 'regimequant.walkforward.fold_output'")
        if self.version != "m10.v1":
            raise ValueError("version must be 'm10.v1'")
        for field_name in ["plan_id", "fold_id", "model_digest", "family_digest", "no_trade_digest", "accounting_digest", "decision_digest"]:
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    schema: str
    version: str
    experiment_id: str
    plan_id: str
    fold_inputs: tuple[FoldInput, ...]
    fold_outputs: tuple[FoldOutput, ...]
    scientific_status: ScientificStatus
    accounting_status: AccountingStatus
    metric_status: MetricStatus
    result_digest: str | None = None

    def __post_init__(self) -> None:
        if self.schema != "regimequant.walkforward.experiment_result":
            raise ValueError("schema must be 'regimequant.walkforward.experiment_result'")
        if self.version != "m10.v1":
            raise ValueError("version must be 'm10.v1'")
        if not self.experiment_id.strip():
            raise ValueError("experiment_id must be non-empty")
        if not self.plan_id.strip():
            raise ValueError("plan_id must be non-empty")


@dataclass(frozen=True, slots=True)
class ExperimentPlan:
    schema: str = "regimequant.walkforward.experiment_plan"
    experiment_id: str = ""
    experiment_version: str = "m10.v1"
    regime_primary: str = "M5"
    regime_comparator: str | None = "M3"
    calendar_id: str = ""
    calendar_version: str = ""
    dataset_identity: str = ""
    feature_spec_set_hash: str = ""
    model_policy_identity: str = ""
    formation_policy_identity: str = ""
    family_policy_identity: str = ""
    final_holdout_sessions: int = 20
    created_at: datetime | None = None
    folds: tuple[FoldInput, ...] = ()
    plan_digest: str | None = None
    structural_only: bool = True
    scientific_certification_status: Literal["STRUCTURAL_ONLY"] = "STRUCTURAL_ONLY"

    def __post_init__(self) -> None:
        if self.schema != "regimequant.walkforward.experiment_plan":
            raise ValueError("schema must be 'regimequant.walkforward.experiment_plan'")
        if self.experiment_version != "m10.v1":
            raise ValueError("experiment_version must be 'm10.v1'")
        if self.regime_primary != "M5":
            raise ValueError("regime_primary must be 'M5'")
        if self.regime_comparator not in (None, "M3"):
            raise ValueError("regime_comparator must be None or 'M3'")
        if self.final_holdout_sessions != 20:
            raise ValueError("final_holdout_sessions must equal 20")
        if self.created_at is None or self.created_at.tzinfo is None:
            raise ValueError("created_at must be a timezone-aware UTC datetime")
        if self.structural_only is not True:
            raise ValueError("ExperimentPlan must remain STRUCTURAL_ONLY in Phase 1")
        if self.scientific_certification_status != "STRUCTURAL_ONLY":
            raise ValueError("scientific_certification_status must be STRUCTURAL_ONLY")
        for field_name in [
            "experiment_id",
            "calendar_id",
            "calendar_version",
            "dataset_identity",
            "feature_spec_set_hash",
            "model_policy_identity",
            "formation_policy_identity",
            "family_policy_identity",
        ]:
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class NoEntryArtifact:
    schema: str
    version: str
    experiment_id: str
    fold_id: str
    fold_input_identity: str
    decision_session: int
    boundary_b: int
    reason_code: str
    evidence_dependency_identities: tuple[str, ...] = field(default_factory=tuple)
    invoked_m9: bool = False
    no_entry_digest: str | None = None
    structural_only: bool = True

    def __post_init__(self) -> None:
        if self.schema != "regimequant.walkforward.no_entry":
            raise ValueError("schema must be 'regimequant.walkforward.no_entry'")
        if self.version != "m10.v1":
            raise ValueError("version must be 'm10.v1'")
        if not self.experiment_id.strip():
            raise ValueError("experiment_id must be non-empty")
        if not self.fold_id.strip():
            raise ValueError("fold_id must be non-empty")
        if not self.fold_input_identity.strip():
            raise ValueError("fold_input_identity must be non-empty")
        if not self.reason_code.strip():
            raise ValueError("reason_code must be non-empty")
        if self.invoked_m9 is not False:
            raise ValueError("invoked_m9 must be False in Phase 1")
        if self.structural_only is not True:
            raise ValueError("NoEntryArtifact must remain STRUCTURAL_ONLY in Phase 1")

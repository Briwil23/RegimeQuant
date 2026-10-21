from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import pandas as pd


class AuthorityConflictError(ValueError):
    """Raised when admissible evidence is irreconcilably contradictory."""


class MembershipStatus(str, Enum):
    LISTED = "listed"
    DELISTED = "delisted"
    INFERRED = "inferred"
    UNKNOWN = "unknown"


class EvidenceStatus(str, Enum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    UNKNOWN = "unknown"


class CoverageStatus(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"


class EligibilityStatus(str, Enum):
    ELIGIBLE = "eligible"
    INELIGIBLE = "ineligible"
    INFERRED = "inferred"


class ExclusionReason(str, Enum):
    MISSING_VERIFIED_MEMBERSHIP = "missing_verified_membership"
    UNKNOWN_UNVERIFIED_EVIDENCE = "unknown_unverified_evidence"
    UNSUPPORTED_LIQUIDITY_METADATA = "unsupported_liquidity_metadata"
    FUTURE_EFFECTIVE_EVENT = "future_effective_event"
    CONFLICTING_AUTHORITY = "conflicting_authority"
    INCOMPLETE_CANDIDATE_COVERAGE = "incomplete_candidate_coverage"
    RESEARCH_INFERRED = "research_inferred"
    INSUFFICIENT_VERIFIED_COVERAGE = "insufficient_verified_coverage"


def _as_utc(value: Any, *, field_name: str) -> pd.Timestamp:
    if value is None or value is pd.NaT or pd.isna(value):
        raise ValueError(f"{field_name} is required")
    ts = pd.Timestamp(value)
    if ts.tzinfo is None or ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


@dataclass(frozen=True, slots=True)
class SecurityAliasRecord:
    canonical_security_id: str
    symbol: str
    listing_venue: str
    valid_from: pd.Timestamp
    valid_to: pd.Timestamp | None = None
    verified_at: pd.Timestamp | None = None
    evidence_provenance: str = ""
    source_authority: str = "exchange"
    evidence_digest: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "canonical_security_id", str(self.canonical_security_id).strip())
        object.__setattr__(self, "symbol", str(self.symbol).strip())
        object.__setattr__(self, "listing_venue", str(self.listing_venue).strip())
        object.__setattr__(self, "source_authority", str(self.source_authority).strip())
        object.__setattr__(self, "evidence_provenance", str(self.evidence_provenance).strip())
        object.__setattr__(self, "evidence_digest", str(self.evidence_digest).strip())
        object.__setattr__(self, "valid_from", _as_utc(self.valid_from, field_name="valid_from"))
        if self.valid_to is not None:
            object.__setattr__(self, "valid_to", _as_utc(self.valid_to, field_name="valid_to"))
        if self.verified_at is not None:
            object.__setattr__(self, "verified_at", _as_utc(self.verified_at, field_name="verified_at"))
        if not self.canonical_security_id:
            raise ValueError("canonical_security_id must not be empty")
        if not self.symbol:
            raise ValueError("symbol must not be empty")
        if self.valid_to is not None and self.valid_to < self.valid_from:
            raise ValueError("valid_to must be >= valid_from")


@dataclass(frozen=True, slots=True)
class CorporateActionContinuity:
    predecessor_security_id: str
    successor_security_id: str
    event_kind: str
    effective_at: pd.Timestamp
    verified_at: pd.Timestamp
    source_authority: str
    evidence_provenance: str = ""
    evidence_digest: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "predecessor_security_id", str(self.predecessor_security_id).strip())
        object.__setattr__(self, "successor_security_id", str(self.successor_security_id).strip())
        object.__setattr__(self, "event_kind", str(self.event_kind).strip())
        object.__setattr__(self, "source_authority", str(self.source_authority).strip())
        object.__setattr__(self, "evidence_provenance", str(self.evidence_provenance).strip())
        object.__setattr__(self, "evidence_digest", str(self.evidence_digest).strip())
        object.__setattr__(self, "effective_at", _as_utc(self.effective_at, field_name="effective_at"))
        object.__setattr__(self, "verified_at", _as_utc(self.verified_at, field_name="verified_at"))
        if self.predecessor_security_id == self.successor_security_id:
            raise ValueError("predecessor and successor security_id cannot be identical")
        if not self.event_kind:
            raise ValueError("event_kind must not be empty")


@dataclass(frozen=True, slots=True)
class SecurityIdentityEvidence:
    security_id: str
    ticker: str
    aliases: tuple[str, ...] = ()
    authority: str = "exchange"
    verified_at: pd.Timestamp = field(default_factory=lambda: pd.Timestamp(datetime.now(timezone.utc)))
    source: str = "exchange"
    evidence_digest: str = ""
    status: str = "verified"
    ambiguous: bool = False
    issuer_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "security_id", str(self.security_id).strip())
        object.__setattr__(self, "ticker", str(self.ticker).strip())
        object.__setattr__(self, "issuer_id", None if self.issuer_id is None else str(self.issuer_id).strip())
        if not self.security_id:
            raise ValueError("security_id must not be empty")
        if not self.ticker:
            raise ValueError("ticker must not be empty")
        if self.status not in {EvidenceStatus.VERIFIED.value, EvidenceStatus.UNVERIFIED.value, EvidenceStatus.UNKNOWN.value}:
            raise ValueError(f"unsupported identity status: {self.status!r}")
        if self.ambiguous and self.status == EvidenceStatus.VERIFIED.value:
            raise ValueError("verified identity evidence cannot be ambiguous")
        object.__setattr__(self, "aliases", tuple(sorted({str(alias).strip() for alias in self.aliases if str(alias).strip()})))
        object.__setattr__(self, "authority", str(self.authority).strip())
        object.__setattr__(self, "source", str(self.source).strip())
        object.__setattr__(self, "evidence_digest", str(self.evidence_digest).strip())
        object.__setattr__(self, "verified_at", _as_utc(self.verified_at, field_name="verified_at"))


@dataclass(frozen=True, slots=True)
class MembershipEvent:
    event_id: str
    security_id: str
    ticker: str
    effective_at: pd.Timestamp
    verified_at: pd.Timestamp
    source: str
    authority: str
    status: str = MembershipStatus.LISTED.value
    supersedes: tuple[str, ...] = ()
    is_inferred: bool = False
    evidence_digest: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", str(self.event_id).strip())
        object.__setattr__(self, "security_id", str(self.security_id).strip())
        object.__setattr__(self, "ticker", str(self.ticker).strip())
        object.__setattr__(self, "source", str(self.source).strip())
        object.__setattr__(self, "authority", str(self.authority).strip())
        object.__setattr__(self, "effective_at", _as_utc(self.effective_at, field_name="effective_at"))
        object.__setattr__(self, "verified_at", _as_utc(self.verified_at, field_name="verified_at"))
        object.__setattr__(self, "status", str(self.status).strip())
        object.__setattr__(self, "supersedes", tuple(str(value) for value in self.supersedes))
        if not self.event_id:
            raise ValueError("event_id must not be empty")
        if not self.security_id:
            raise ValueError("security_id must not be empty")
        if not self.ticker:
            raise ValueError("ticker must not be empty")


@dataclass(frozen=True, slots=True)
class UniverseCandidate:
    security_id: str
    ticker: str
    as_of: pd.Timestamp
    status: str = "candidate"
    is_delisted: bool = False
    source: str = "exchange"
    coverage_complete: bool = True
    liquidity_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "security_id", str(self.security_id).strip())
        object.__setattr__(self, "ticker", str(self.ticker).strip())
        object.__setattr__(self, "as_of", _as_utc(self.as_of, field_name="as_of"))
        object.__setattr__(self, "status", str(self.status).strip())
        object.__setattr__(self, "source", str(self.source).strip())
        object.__setattr__(self, "liquidity_metadata", dict(self.liquidity_metadata))


@dataclass(frozen=True, slots=True)
class EligibilityDecision:
    security_id: str
    decision_timestamp: pd.Timestamp
    eligibility_status: str
    evidence_available: bool
    snapshot_identity: str
    provenance_digest: str
    coverage_digest: str
    exclusion_reasons: tuple[str, ...] = ()
    mode: str = "production"

    coverage_status: str = CoverageStatus.COMPLETE.value
    research_override: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "security_id", str(self.security_id).strip())
        object.__setattr__(self, "decision_timestamp", _as_utc(self.decision_timestamp, field_name="decision_timestamp"))
        object.__setattr__(self, "eligibility_status", str(self.eligibility_status).strip())
        object.__setattr__(self, "snapshot_identity", str(self.snapshot_identity).strip())
        object.__setattr__(self, "provenance_digest", str(self.provenance_digest).strip())
        object.__setattr__(self, "coverage_digest", str(self.coverage_digest).strip())
        object.__setattr__(self, "exclusion_reasons", tuple(str(item) for item in self.exclusion_reasons))
        object.__setattr__(self, "mode", str(self.mode).strip())
        object.__setattr__(self, "coverage_status", str(self.coverage_status).strip())
        object.__setattr__(self, "research_override", bool(self.research_override))


@dataclass(frozen=True, slots=True)
class UniverseSnapshot:
    candidate_universe_identity: str
    coverage_evidence: tuple[UniverseCandidate, ...]
    admissible_membership_versions: tuple[str, ...]
    eligibility_policy: dict[str, Any]
    decisions: tuple[EligibilityDecision, ...]
    source_provenance_digest: str
    snapshot_scientific_identity: str
    scientific_payload: dict[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_universe_identity", str(self.candidate_universe_identity).strip())
        object.__setattr__(self, "coverage_evidence", tuple(self.coverage_evidence))
        object.__setattr__(self, "admissible_membership_versions", tuple(str(item) for item in self.admissible_membership_versions))
        object.__setattr__(self, "eligibility_policy", dict(self.eligibility_policy))
        object.__setattr__(self, "decisions", tuple(self.decisions))
        object.__setattr__(self, "source_provenance_digest", str(self.source_provenance_digest).strip())
        object.__setattr__(self, "snapshot_scientific_identity", str(self.snapshot_scientific_identity).strip())
        object.__setattr__(self, "scientific_payload", dict(self.scientific_payload))

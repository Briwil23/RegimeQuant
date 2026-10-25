from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import pandas as pd

SourceMode = Literal["production", "replay", "evaluation"]


class M8SignalEnum(str, Enum):
    SHORT_SPREAD_CANDIDATE = "SHORT_SPREAD_CANDIDATE"
    LONG_SPREAD_CANDIDATE = "LONG_SPREAD_CANDIDATE"
    NEUTRAL = "NEUTRAL"
    ABSTAIN = "ABSTAIN"


class M8AbstainReason(str, Enum):
    PREDECLARATION_UNPROVEN = "ABSTAIN_PREDECLARATION_UNPROVEN"
    FAMILY_INCOMPLETE = "ABSTAIN_FAMILY_INCOMPLETE"
    BY_INCONSISTENT_EVIDENCE = "ABSTAIN_BY_INCONSISTENT_EVIDENCE"
    BY_NOT_SIGNIFICANT = "ABSTAIN_BY_NOT_SIGNIFICANT"
    MEMBER_INVALID = "ABSTAIN_MEMBER_INVALID"
    DISCOVERY_STALE = "ABSTAIN_DISCOVERY_STALE"
    EVIDENCE_LATE = "ABSTAIN_EVIDENCE_LATE"
    CALENDAR_UNVERIFIED = "ABSTAIN_CALENDAR_UNVERIFIED"
    REVOCATION_COVERAGE_UNKNOWN = "ABSTAIN_REVOCATION_COVERAGE_UNKNOWN"
    REVOKED_EFFECTIVE = "ABSTAIN_REVOKED_EFFECTIVE"
    MODE_MISMATCH = "ABSTAIN_MODE_MISMATCH"
    IDENTITY_MISMATCH = "ABSTAIN_IDENTITY_MISMATCH"
    X_NEAR_CONSTANT = "ABSTAIN_X_NEAR_CONSTANT"
    ILL_CONDITIONED = "ABSTAIN_ILL_CONDITIONED"
    SPREAD_VAR_TOO_SMALL = "ABSTAIN_SPREAD_VAR_TOO_SMALL"
    Z_NONFINITE = "ABSTAIN_Z_NONFINITE"
    FORMATION_TOO_SHORT = "ABSTAIN_FORMATION_TOO_SHORT"
    DUPLICATE_OR_CONFLICTING_OBS = "ABSTAIN_DUPLICATE_OR_CONFLICTING_OBS"
    MISSING_REQUIRED_SESSION = "ABSTAIN_MISSING_REQUIRED_SESSION"
    LEAKAGE_FORMATION_EVAL_OVERLAP = "ABSTAIN_LEAKAGE_FORMATION_EVAL_OVERLAP"


def _as_utc(value: object) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


@dataclass(frozen=True, slots=True)
class M8FamilyMember:
    hypothesis_id: str
    canonical_pair_id: str
    canonical_security_id_left: str
    canonical_security_id_right: str
    valid_for_discovery: bool
    invalid_reason: str | None = None
    raw_p_value: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "hypothesis_id", str(self.hypothesis_id).strip())
        object.__setattr__(self, "canonical_pair_id", str(self.canonical_pair_id).strip())
        object.__setattr__(self, "canonical_security_id_left", str(self.canonical_security_id_left).strip())
        object.__setattr__(self, "canonical_security_id_right", str(self.canonical_security_id_right).strip())
        object.__setattr__(self, "invalid_reason", None if self.invalid_reason is None else str(self.invalid_reason).strip())
        if not self.hypothesis_id or not self.canonical_pair_id:
            raise ValueError("hypothesis_id and canonical_pair_id are required")
        if self.canonical_security_id_left >= self.canonical_security_id_right:
            raise ValueError("canonical_security_id_left must be lexically smaller than canonical_security_id_right")
        if self.valid_for_discovery:
            if self.raw_p_value is None:
                raise ValueError("valid discovery members require raw_p_value")
            p = float(self.raw_p_value)
            if not 0.0 <= p <= 1.0:
                raise ValueError("raw_p_value must lie in [0, 1]")
        else:
            if self.invalid_reason is None:
                raise ValueError("invalid members require invalid_reason")


@dataclass(frozen=True, slots=True)
class M8DiscoveryFamilyEvidence:
    family_definition_version: str
    family_id: str
    universe_snapshot_identity: str
    discovery_window_start: pd.Timestamp
    discovery_window_end: pd.Timestamp
    discovery_observation_session: pd.Timestamp
    discovery_available_at: pd.Timestamp
    discovery_decision_cutoff: pd.Timestamp
    dataset_identity: str
    statistical_config_digest: str
    declared_membership_ordered: tuple[M8FamilyMember, ...]
    supplied_by_adjusted_values: tuple[float, ...]
    predeclared_committed_at: pd.Timestamp
    outcomes_available_at: pd.Timestamp
    exchange_calendar_id: str
    exchange_calendar_version: str
    exchange_session_sequence_digest: str
    provenance_mode: SourceMode
    availability_verified: bool
    identity_verified: bool
    continuity_verified: bool
    evidence_provenance_digest: str
    family_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "family_definition_version", str(self.family_definition_version).strip())
        object.__setattr__(self, "family_id", str(self.family_id).strip())
        object.__setattr__(self, "universe_snapshot_identity", str(self.universe_snapshot_identity).strip())
        object.__setattr__(self, "dataset_identity", str(self.dataset_identity).strip())
        object.__setattr__(self, "statistical_config_digest", str(self.statistical_config_digest).strip())
        object.__setattr__(self, "exchange_calendar_id", str(self.exchange_calendar_id).strip())
        object.__setattr__(self, "exchange_calendar_version", str(self.exchange_calendar_version).strip())
        object.__setattr__(self, "exchange_session_sequence_digest", str(self.exchange_session_sequence_digest).strip())
        object.__setattr__(self, "evidence_provenance_digest", str(self.evidence_provenance_digest).strip())
        object.__setattr__(self, "family_digest", str(self.family_digest).strip())

        start = _as_utc(self.discovery_window_start)
        end = _as_utc(self.discovery_window_end)
        obs = _as_utc(self.discovery_observation_session)
        available = _as_utc(self.discovery_available_at)
        cutoff = _as_utc(self.discovery_decision_cutoff)
        predeclared = _as_utc(self.predeclared_committed_at)
        outcomes_at = _as_utc(self.outcomes_available_at)

        object.__setattr__(self, "discovery_window_start", start)
        object.__setattr__(self, "discovery_window_end", end)
        object.__setattr__(self, "discovery_observation_session", obs)
        object.__setattr__(self, "discovery_available_at", available)
        object.__setattr__(self, "discovery_decision_cutoff", cutoff)
        object.__setattr__(self, "predeclared_committed_at", predeclared)
        object.__setattr__(self, "outcomes_available_at", outcomes_at)

        if not self.declared_membership_ordered:
            raise ValueError("declared_membership_ordered must not be empty")
        ordered_ids = [member.canonical_pair_id for member in self.declared_membership_ordered]
        if ordered_ids != sorted(ordered_ids):
            raise ValueError("declared_membership_ordered must be sorted by canonical_pair_id")
        if len(set(ordered_ids)) != len(ordered_ids):
            raise ValueError("declared_membership_ordered must not contain duplicate canonical_pair_id")
        if len(self.supplied_by_adjusted_values) != len(self.declared_membership_ordered):
            raise ValueError("supplied_by_adjusted_values must match declared family size")
        for value in self.supplied_by_adjusted_values:
            adjusted = float(value)
            if not 0.0 <= adjusted <= 1.0:
                raise ValueError("supplied_by_adjusted_values must lie in [0, 1]")
        if start > end:
            raise ValueError("discovery window must satisfy start <= end")
        if not (start <= obs <= end):
            raise ValueError("discovery_observation_session must be inside the discovery window")
        if available > cutoff:
            raise ValueError("discovery evidence must be available by discovery cutoff")
        if predeclared > outcomes_at:
            raise ValueError("predeclaration must be committed at or before outcome availability")
        if predeclared > cutoff:
            raise ValueError("predeclaration must be committed at or before discovery cutoff")
        if not self.availability_verified:
            raise ValueError("availability_verified must be true for family evidence")
        if not self.identity_verified:
            raise ValueError("identity_verified must be true for family evidence")
        if not self.continuity_verified:
            raise ValueError("continuity_verified must be true for family evidence")


@dataclass(frozen=True, slots=True)
class M8RevocationRecord:
    canonical_pair_id: str
    authority_id: str
    effective_at_session: pd.Timestamp
    known_at: pd.Timestamp
    coverage_complete: bool
    reason_code: str
    revocation_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "canonical_pair_id", str(self.canonical_pair_id).strip())
        object.__setattr__(self, "authority_id", str(self.authority_id).strip())
        object.__setattr__(self, "reason_code", str(self.reason_code).strip())
        object.__setattr__(self, "revocation_digest", str(self.revocation_digest).strip())
        object.__setattr__(self, "effective_at_session", _as_utc(self.effective_at_session))
        object.__setattr__(self, "known_at", _as_utc(self.known_at))
        if not self.canonical_pair_id or not self.revocation_digest:
            raise ValueError("canonical_pair_id and revocation_digest are required")


@dataclass(frozen=True, slots=True)
class M8FormationModel:
    canonical_pair_id: str
    x_security_id: str
    y_security_id: str
    formation_start: pd.Timestamp
    formation_end: pd.Timestamp
    fitting_cutoff: pd.Timestamp
    alpha: float
    beta: float
    spread_mean: float
    spread_std: float
    observation_count: int


@dataclass(frozen=True, slots=True)
class M8SignalObservation:
    canonical_pair_id: str
    eval_observation_time: pd.Timestamp
    eval_available_at: pd.Timestamp
    m8_decision_cutoff: pd.Timestamp
    signal: M8SignalEnum
    z_score: float | None
    abstain_reason: M8AbstainReason | None
    source_mode: SourceMode
    scientific_identity: str
    evidence_digest: str
    decision_digest: str

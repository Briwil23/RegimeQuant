from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable, Sequence

from regimequant.data.models import MarketDataProvenance
from regimequant.features.definitions import FeatureSpec, feature_instance_id
from regimequant.walkforward.contracts import FoldInterval
from regimequant.walkforward.identity import compute_identity_digest


class EvidenceOrigin(str, Enum):
    DIRECT_SOURCE = "DIRECT_SOURCE"
    DETERMINISTICALLY_DERIVED = "DETERMINISTICALLY_DERIVED"
    CALLER_ASSERTED = "CALLER_ASSERTED"
    UNSUPPORTED = "UNSUPPORTED"


class AuthorityLevel(str, Enum):
    STRUCTURAL_ONLY = "STRUCTURAL_ONLY"
    FEATURE_RECORD_BOUND = "FEATURE_RECORD_BOUND"
    SOURCE_DEPENDENCY_BOUND = "SOURCE_DEPENDENCY_BOUND"
    UNSUPPORTED = "UNSUPPORTED"


class SourceAuthorityFailureCode(str, Enum):
    MISSING_SOURCE_RECORD = "MISSING_SOURCE_RECORD"
    AMBIGUOUS_SOURCE_IDENTITY = "AMBIGUOUS_SOURCE_IDENTITY"
    INSTRUMENT_MISMATCH = "INSTRUMENT_MISMATCH"
    FEATURE_SPEC_MISMATCH = "FEATURE_SPEC_MISMATCH"
    AVAILABILITY_MISMATCH = "AVAILABILITY_MISMATCH"
    FEATURE_VALUE_MISMATCH = "FEATURE_VALUE_MISMATCH"
    PROVENANCE_MISSING = "PROVENANCE_MISSING"
    DATASET_IDENTITY_MISMATCH = "DATASET_IDENTITY_MISMATCH"
    INCOMPLETE_LINEAGE = "INCOMPLETE_LINEAGE"
    UNSUPPORTED_HISTORICAL_ARTIFACT = "UNSUPPORTED_HISTORICAL_ARTIFACT"


@dataclass(frozen=True, slots=True)
class SourceAuthorityFailure:
    code: SourceAuthorityFailureCode
    message: str
    missing_evidence: tuple[str, ...] = ()
    evidence_identity: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.code, SourceAuthorityFailureCode):
            raise TypeError("code must be a SourceAuthorityFailureCode")
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("message must be a non-empty string")
        object.__setattr__(self, "missing_evidence", tuple(str(item) for item in self.missing_evidence))


def _as_utc(value: datetime | str | None, *, field_name: str, allow_none: bool = False) -> datetime | None:
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{field_name} is required")
    if isinstance(value, datetime):
        timestamp = value
    elif isinstance(value, str):
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        raise TypeError(f"{field_name} must be datetime or ISO-8601 string")
    if timestamp.tzinfo is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return timestamp.astimezone(timezone.utc)


def _require_non_empty(value: Any, field_name: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError(f"{field_name} must be a non-empty string")
    return text


def _canonical_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key in sorted(payload):
        value = payload[key]
        if value is None:
            normalized[key] = None
        elif isinstance(value, (str, int, float, bool)):
            if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
                raise ValueError(f"non-finite value is not allowed in {key}")
            normalized[key] = value
        elif isinstance(value, datetime):
            normalized[key] = _as_utc(value, field_name=key).isoformat().replace("+00:00", "Z")
        elif isinstance(value, tuple):
            normalized[key] = [_canonical_payload({"item": item})["item"] for item in value]
        elif isinstance(value, list):
            normalized[key] = [_canonical_payload({"item": item})["item"] for item in value]
        elif isinstance(value, dict):
            normalized[key] = _canonical_payload(value)
        else:
            normalized[key] = str(value)
    return normalized


def _material_digest(payload: dict[str, Any], *, domain: str) -> str:
    material = _canonical_payload(payload)
    return compute_identity_digest(material, domain=domain)


def _identity_material_without_digest(payload: dict[str, Any], *, extra_exclusions: set[str] | None = None) -> dict[str, Any]:
    excluded = {"evidence_digest"}
    if extra_exclusions:
        excluded |= set(extra_exclusions)
    return {key: value for key, value in _canonical_payload(payload).items() if key not in excluded}


def recompute_local_evidence_digest(payload: dict[str, Any], *, domain: str, extra_exclusions: set[str] | None = None) -> str:
    if not isinstance(payload, dict):
        raise TypeError("payload must be a dict")
    material = _identity_material_without_digest(payload, extra_exclusions=extra_exclusions)
    if not material:
        raise ValueError("payload must contain at least one non-digest identity field")
    return compute_identity_digest(material, domain=domain)


def validate_local_evidence_digest(
    payload: dict[str, Any],
    *,
    domain: str,
    supplied_digest: str | None,
    extra_exclusions: set[str] | None = None,
) -> str:
    expected = recompute_local_evidence_digest(payload, domain=domain, extra_exclusions=extra_exclusions)
    if supplied_digest is not None and supplied_digest != expected:
        raise ValueError(
            f"supplied local evidence digest mismatch for domain {domain!r}: "
            f"expected={expected!r} supplied={supplied_digest!r}"
        )
    return expected


@dataclass(frozen=True, slots=True)
class SourceRecordEvidence:
    source_dataset_identity: str
    instrument: str
    source_observation_id: str | None
    observation_timestamp_utc: datetime
    availability_timestamp_utc: datetime | None = None
    availability_status: str = "verified_timestamp"
    price_type: str = "close"
    adjustment_state: str = "raw"
    source_provider_id: str | None = None
    source_artifact_id: str | None = None
    source_artifact_hash: str | None = None
    origin: EvidenceOrigin = EvidenceOrigin.DIRECT_SOURCE
    authority_level: AuthorityLevel = AuthorityLevel.STRUCTURAL_ONLY
    evidence_digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_dataset_identity", _require_non_empty(self.source_dataset_identity, "source_dataset_identity"))
        object.__setattr__(self, "instrument", _require_non_empty(self.instrument, "instrument"))
        object.__setattr__(self, "availability_status", _require_non_empty(self.availability_status, "availability_status"))
        object.__setattr__(self, "price_type", _require_non_empty(self.price_type, "price_type"))
        object.__setattr__(self, "adjustment_state", _require_non_empty(self.adjustment_state, "adjustment_state"))
        object.__setattr__(self, "observation_timestamp_utc", _as_utc(self.observation_timestamp_utc, field_name="observation_timestamp_utc"))
        object.__setattr__(self, "availability_timestamp_utc", _as_utc(self.availability_timestamp_utc, field_name="availability_timestamp_utc", allow_none=True))
        if self.origin not in set(EvidenceOrigin):
            raise ValueError("origin must be a valid EvidenceOrigin")
        if self.authority_level not in {AuthorityLevel.STRUCTURAL_ONLY, AuthorityLevel.UNSUPPORTED}:
            raise ValueError("evidence objects may not silently promote authority level")
        if self.source_provider_id is not None:
            object.__setattr__(self, "source_provider_id", _require_non_empty(self.source_provider_id, "source_provider_id"))
        if self.source_artifact_id is not None:
            object.__setattr__(self, "source_artifact_id", _require_non_empty(self.source_artifact_id, "source_artifact_id"))
        if self.source_artifact_hash is not None:
            object.__setattr__(self, "source_artifact_hash", _require_non_empty(self.source_artifact_hash, "source_artifact_hash"))
        if self.source_observation_id is not None:
            object.__setattr__(self, "source_observation_id", _require_non_empty(self.source_observation_id, "source_observation_id"))
        payload = {
            "source_dataset_identity": self.source_dataset_identity,
            "instrument": self.instrument,
            "source_observation_id": self.source_observation_id,
            "observation_timestamp_utc": self.observation_timestamp_utc,
            "availability_timestamp_utc": self.availability_timestamp_utc,
            "availability_status": self.availability_status,
            "price_type": self.price_type,
            "adjustment_state": self.adjustment_state,
            "source_provider_id": self.source_provider_id,
            "source_artifact_id": self.source_artifact_id,
            "source_artifact_hash": self.source_artifact_hash,
            "origin": self.origin.value,
            "authority_level": self.authority_level.value,
        }
        object.__setattr__(self, "evidence_digest", _material_digest(payload, domain="regimequant.walkforward.phase3b.source_record_evidence"))

    @property
    def evidence_key(self) -> tuple[str, str, str | None]:
        return (self.source_dataset_identity, self.instrument, self.source_observation_id)


@dataclass(frozen=True, slots=True)
class FeatureSpecEvidence:
    feature_name: str
    feature_definition_version: str = "1"
    input_return_type: str = "simple"
    output_transformation_mode: str = "native"
    window_kind: str = "observation_count"
    window_size: int = 3
    min_observations: int = 3
    include_current: bool = True
    require_complete_window: bool = False
    allow_partial_valid: bool = True
    ddof: int | None = None
    downside_threshold: float = 0.0
    short_window_size: int | None = None
    long_window_size: int | None = None
    source_spec: FeatureSpec | None = None
    origin: EvidenceOrigin = EvidenceOrigin.DIRECT_SOURCE
    authority_level: AuthorityLevel = AuthorityLevel.STRUCTURAL_ONLY
    evidence_digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "feature_name", _require_non_empty(self.feature_name, "feature_name"))
        object.__setattr__(self, "feature_definition_version", _require_non_empty(self.feature_definition_version, "feature_definition_version"))
        object.__setattr__(self, "input_return_type", _require_non_empty(self.input_return_type, "input_return_type"))
        object.__setattr__(self, "output_transformation_mode", _require_non_empty(self.output_transformation_mode, "output_transformation_mode"))
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.min_observations <= 0:
            raise ValueError("min_observations must be positive")
        if self.origin not in set(EvidenceOrigin):
            raise ValueError("origin must be a valid EvidenceOrigin")
        if self.authority_level not in {AuthorityLevel.STRUCTURAL_ONLY, AuthorityLevel.UNSUPPORTED}:
            raise ValueError("feature-spec evidence may not promote authority level")
        if self.source_spec is not None:
            if not isinstance(self.source_spec, FeatureSpec):
                raise TypeError("source_spec must be a FeatureSpec")
            resolved = {
                "feature_instance_id": feature_instance_id(self.source_spec),
                "feature_name": self.source_spec.feature_name,
                "feature_definition_version": self.source_spec.feature_definition_version,
                "input_return_type": self.source_spec.input_return_type.value,
                "output_transformation_mode": self.source_spec.output_transformation_mode,
                "window_kind": self.source_spec.window_kind,
                "window_size": self.source_spec.window_size,
                "min_observations": self.source_spec.min_observations,
                "include_current": self.source_spec.include_current,
                "require_complete_window": self.source_spec.require_complete_window,
                "allow_partial_valid": self.source_spec.allow_partial_valid,
                "ddof": self.source_spec.ddof,
                "downside_threshold": self.source_spec.downside_threshold,
                "short_window_size": self.source_spec.short_window_size,
                "long_window_size": self.source_spec.long_window_size,
            }
            material = {
                "feature_name": self.source_spec.feature_name,
                "feature_definition_version": self.source_spec.feature_definition_version,
                "input_return_type": self.source_spec.input_return_type.value,
                "output_transformation_mode": self.source_spec.output_transformation_mode,
                "window_kind": self.source_spec.window_kind,
                "window_size": self.source_spec.window_size,
                "min_observations": self.source_spec.min_observations,
                "include_current": self.source_spec.include_current,
                "require_complete_window": self.source_spec.require_complete_window,
                "allow_partial_valid": self.source_spec.allow_partial_valid,
                "ddof": self.source_spec.ddof,
                "downside_threshold": self.source_spec.downside_threshold,
                "short_window_size": self.source_spec.short_window_size,
                "long_window_size": self.source_spec.long_window_size,
                "feature_instance_id": feature_instance_id(self.source_spec),
                "origin": self.origin.value,
                "authority_level": self.authority_level.value,
            }
        else:
            resolved = {
                "feature_name": self.feature_name,
                "feature_definition_version": self.feature_definition_version,
                "input_return_type": self.input_return_type,
                "output_transformation_mode": self.output_transformation_mode,
                "window_kind": self.window_kind,
                "window_size": self.window_size,
                "min_observations": self.min_observations,
                "include_current": self.include_current,
                "require_complete_window": self.require_complete_window,
                "allow_partial_valid": self.allow_partial_valid,
                "ddof": self.ddof,
                "downside_threshold": self.downside_threshold,
                "short_window_size": self.short_window_size,
                "long_window_size": self.long_window_size,
            }
            material = {
                **resolved,
                "origin": self.origin.value,
                "authority_level": self.authority_level.value,
            }
        object.__setattr__(self, "evidence_digest", _material_digest(material, domain="regimequant.walkforward.phase3b.feature_spec_evidence"))

    @property
    def feature_instance_id_value(self) -> str:
        if self.source_spec is not None:
            return feature_instance_id(self.source_spec)
        return _material_digest({
            "feature_name": self.feature_name,
            "feature_definition_version": self.feature_definition_version,
            "input_return_type": self.input_return_type,
            "output_transformation_mode": self.output_transformation_mode,
            "window_kind": self.window_kind,
            "window_size": self.window_size,
            "min_observations": self.min_observations,
            "include_current": self.include_current,
            "require_complete_window": self.require_complete_window,
            "allow_partial_valid": self.allow_partial_valid,
            "ddof": self.ddof,
            "downside_threshold": self.downside_threshold,
            "short_window_size": self.short_window_size,
            "long_window_size": self.long_window_size,
        }, domain="regimequant.walkforward.phase3b.feature_spec_identity")


@dataclass(frozen=True, slots=True)
class OriginalProvenanceEvidence:
    provider_id: str
    provider_version: str | None = None
    source_artifact_id: str | None = None
    source_artifact_hash: str | None = None
    retrieval_timestamp_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    availability_policy_id: str = "fixture_declared_availability"
    source_availability_status_distribution: tuple[tuple[str, int], ...] = ()
    availability_status_distribution: tuple[tuple[str, int], ...] = ()
    universe_instruments: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    original_provenance: MarketDataProvenance | None = None
    origin: EvidenceOrigin = EvidenceOrigin.DIRECT_SOURCE
    authority_level: AuthorityLevel = AuthorityLevel.STRUCTURAL_ONLY
    evidence_digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider_id", _require_non_empty(self.provider_id, "provider_id"))
        object.__setattr__(self, "availability_policy_id", _require_non_empty(self.availability_policy_id, "availability_policy_id"))
        object.__setattr__(self, "retrieval_timestamp_utc", _as_utc(self.retrieval_timestamp_utc, field_name="retrieval_timestamp_utc"))
        if self.original_provenance is not None:
            if not isinstance(self.original_provenance, MarketDataProvenance):
                raise TypeError("original_provenance must be a MarketDataProvenance")
        if self.origin not in set(EvidenceOrigin):
            raise ValueError("origin must be a valid EvidenceOrigin")
        if self.authority_level not in {AuthorityLevel.STRUCTURAL_ONLY, AuthorityLevel.UNSUPPORTED}:
            raise ValueError("provenance evidence may not promote authority level")
        payload = {
            "provider_id": self.provider_id,
            "provider_version": self.provider_version,
            "source_artifact_id": self.source_artifact_id,
            "source_artifact_hash": self.source_artifact_hash,
            "retrieval_timestamp_utc": self.retrieval_timestamp_utc,
            "availability_policy_id": self.availability_policy_id,
            "source_availability_status_distribution": list(self.source_availability_status_distribution),
            "availability_status_distribution": list(self.availability_status_distribution),
            "universe_instruments": list(self.universe_instruments),
            "limitations": list(self.limitations),
            "origin": self.origin.value,
            "authority_level": self.authority_level.value,
        }
        if self.original_provenance is not None:
            payload["original_provenance"] = asdict(self.original_provenance)
        object.__setattr__(self, "evidence_digest", _material_digest(payload, domain="regimequant.walkforward.phase3b.original_provenance_evidence"))


@dataclass(frozen=True, slots=True)
class FeatureRecordEvidence:
    scope_id: str
    timestamp_utc: datetime
    feature_name: str
    feature_instance_id: str
    feature_value: float
    math_status: str = "defined"
    undefined_reason: str | None = None
    input_availability_status: str = "verified_timestamp"
    available_at_utc: datetime | None = None
    input_return_type: str = "simple"
    price_type: str = "close"
    adjustment_state: str = "raw"
    window_kind: str = "observation_count"
    window_size: int = 3
    min_observations: int = 3
    observation_count_used: int = 3
    feature_definition_version: str = "1"
    feature_spec_set_hash: str | None = None
    parent_return_provenance_digest: str | None = None
    source_observation_ids: tuple[str, ...] = ()
    origin: EvidenceOrigin = EvidenceOrigin.DIRECT_SOURCE
    authority_level: AuthorityLevel = AuthorityLevel.STRUCTURAL_ONLY
    evidence_digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope_id", _require_non_empty(self.scope_id, "scope_id"))
        object.__setattr__(self, "timestamp_utc", _as_utc(self.timestamp_utc, field_name="timestamp_utc"))
        object.__setattr__(self, "feature_name", _require_non_empty(self.feature_name, "feature_name"))
        object.__setattr__(self, "feature_instance_id", _require_non_empty(self.feature_instance_id, "feature_instance_id"))
        if not isinstance(self.feature_value, (int, float)):
            raise TypeError("feature_value must be numeric")
        if self.origin not in set(EvidenceOrigin):
            raise ValueError("origin must be a valid EvidenceOrigin")
        if self.authority_level not in {AuthorityLevel.STRUCTURAL_ONLY, AuthorityLevel.UNSUPPORTED}:
            raise ValueError("feature record evidence may not promote authority level")
        object.__setattr__(self, "source_observation_ids", tuple(str(item) for item in self.source_observation_ids))
        object.__setattr__(self, "available_at_utc", _as_utc(self.available_at_utc, field_name="available_at_utc", allow_none=True))
        payload = {
            "scope_id": self.scope_id,
            "timestamp_utc": self.timestamp_utc,
            "feature_name": self.feature_name,
            "feature_instance_id": self.feature_instance_id,
            "feature_value": float(self.feature_value),
            "math_status": self.math_status,
            "undefined_reason": self.undefined_reason,
            "input_availability_status": self.input_availability_status,
            "available_at_utc": self.available_at_utc,
            "input_return_type": self.input_return_type,
            "price_type": self.price_type,
            "adjustment_state": self.adjustment_state,
            "window_kind": self.window_kind,
            "window_size": self.window_size,
            "min_observations": self.min_observations,
            "observation_count_used": self.observation_count_used,
            "feature_definition_version": self.feature_definition_version,
            "feature_spec_set_hash": self.feature_spec_set_hash,
            "parent_return_provenance_digest": self.parent_return_provenance_digest,
            "source_observation_ids": list(self.source_observation_ids),
            "origin": self.origin.value,
            "authority_level": self.authority_level.value,
        }
        object.__setattr__(self, "evidence_digest", _material_digest(payload, domain="regimequant.walkforward.phase3b.feature_record_evidence"))


@dataclass(frozen=True, slots=True)
class FoldStageEvidence:
    source_dataset_identity: str
    session_calendar_id: str
    session_calendar_version: str
    session_sequence_digest: str
    fold_id: str
    stage: str
    fold_interval: FoldInterval
    stage_information_cutoff_utc: datetime
    boundary_session_index: int
    projection_digest: str | None = None
    dataset_identity_digest: str | None = None
    origin: EvidenceOrigin = EvidenceOrigin.DIRECT_SOURCE
    authority_level: AuthorityLevel = AuthorityLevel.STRUCTURAL_ONLY
    evidence_digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_dataset_identity", _require_non_empty(self.source_dataset_identity, "source_dataset_identity"))
        object.__setattr__(self, "session_calendar_id", _require_non_empty(self.session_calendar_id, "session_calendar_id"))
        object.__setattr__(self, "session_calendar_version", _require_non_empty(self.session_calendar_version, "session_calendar_version"))
        object.__setattr__(self, "session_sequence_digest", _require_non_empty(self.session_sequence_digest, "session_sequence_digest"))
        object.__setattr__(self, "fold_id", _require_non_empty(self.fold_id, "fold_id"))
        object.__setattr__(self, "stage", _require_non_empty(self.stage, "stage"))
        if not isinstance(self.fold_interval, FoldInterval):
            raise TypeError("fold_interval must be a FoldInterval")
        object.__setattr__(self, "stage_information_cutoff_utc", _as_utc(self.stage_information_cutoff_utc, field_name="stage_information_cutoff_utc"))
        if not isinstance(self.boundary_session_index, int):
            raise TypeError("boundary_session_index must be an int")
        if self.origin not in set(EvidenceOrigin):
            raise ValueError("origin must be a valid EvidenceOrigin")
        if self.authority_level not in {AuthorityLevel.STRUCTURAL_ONLY, AuthorityLevel.UNSUPPORTED}:
            raise ValueError("fold/stage evidence may not promote authority level")
        payload = {
            "source_dataset_identity": self.source_dataset_identity,
            "session_calendar_id": self.session_calendar_id,
            "session_calendar_version": self.session_calendar_version,
            "session_sequence_digest": self.session_sequence_digest,
            "fold_id": self.fold_id,
            "stage": self.stage,
            "fold_interval": {
                "train_start": self.fold_interval.train_start,
                "train_end": self.fold_interval.train_end,
                "validation_start": self.fold_interval.validation_start,
                "validation_end": self.fold_interval.validation_end,
                "test_start": self.fold_interval.test_start,
                "test_end": self.fold_interval.test_end,
                "holdout_start": self.fold_interval.holdout_start,
                "holdout_end": self.fold_interval.holdout_end,
            },
            "stage_information_cutoff_utc": self.stage_information_cutoff_utc,
            "boundary_session_index": self.boundary_session_index,
            "projection_digest": self.projection_digest,
            "dataset_identity_digest": self.dataset_identity_digest,
            "origin": self.origin.value,
            "authority_level": self.authority_level.value,
        }
        object.__setattr__(self, "evidence_digest", _material_digest(payload, domain="regimequant.walkforward.phase3b.fold_stage_evidence"))


def authority_level_from_evidence(
    *,
    evidence: SourceRecordEvidence | FeatureSpecEvidence | OriginalProvenanceEvidence | FeatureRecordEvidence | FoldStageEvidence | None,
    require_reconciliation_proof: bool = False,
) -> AuthorityLevel:
    if evidence is None:
        return AuthorityLevel.UNSUPPORTED
    if getattr(evidence, "authority_level", None) in {AuthorityLevel.UNSUPPORTED}: 
        return AuthorityLevel.UNSUPPORTED
    if require_reconciliation_proof:
        return AuthorityLevel.STRUCTURAL_ONLY
    return AuthorityLevel.STRUCTURAL_ONLY


def require_unique_evidence_keys(records: Sequence[SourceRecordEvidence]) -> None:
    seen: set[tuple[str, str, str | None]] = set()
    duplicates: list[tuple[str, str, str | None]] = []
    for record in records:
        key = record.evidence_key
        if key in seen:
            duplicates.append(key)
        else:
            seen.add(key)
    if duplicates:
        raise ValueError(f"duplicate source evidence keys are not allowed: {duplicates!r}")


def require_unique_feature_keys(records: Sequence[FeatureRecordEvidence]) -> None:
    seen: set[tuple[str, datetime, str, str]] = set()
    duplicates: list[tuple[str, datetime, str, str]] = []
    for record in records:
        key = (record.scope_id, record.timestamp_utc, record.feature_name, record.feature_instance_id)
        if key in seen:
            duplicates.append(key)
        else:
            seen.add(key)
    if duplicates:
        raise ValueError(f"duplicate feature evidence keys are not allowed: {duplicates!r}")


def evidence_failures_for_historical_artifact() -> tuple[SourceAuthorityFailure, ...]:
    return (
        SourceAuthorityFailure(
            code=SourceAuthorityFailureCode.UNSUPPORTED_HISTORICAL_ARTIFACT,
            message="historical artifact cannot be re-authorized without retained upstream evidence",
            missing_evidence=("source_record_identity", "feature_record_identity", "original_provenance"),
        ),
    )

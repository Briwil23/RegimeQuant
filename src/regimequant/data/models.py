"""Semantic models for RegimeQuant M1 market data."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from numbers import Integral
from typing import Iterable

import pandas as pd


class PriceType(str, Enum):
    CLOSE = "close"


class AdjustmentState(str, Enum):
    RAW = "raw"
    ADJUSTED = "adjusted"


class AvailabilityStatus(str, Enum):
    VERIFIED_TIMESTAMP = "verified_timestamp"
    UNKNOWN_UNVERIFIED = "unknown_unverified"
    UNDEFINED = "undefined"


class UniverseType(str, Enum):
    STATIC = "static"


class ReturnType(str, Enum):
    SIMPLE = "simple"
    LOG = "log"


AVAILABILITY_POLICY_FIXTURE_DECLARED = "fixture_declared_availability"
AVAILABILITY_POLICY_EXTERNAL_VERIFIED = "external_verified_availability"
AVAILABILITY_POLICY_UNKNOWN_UNVERIFIED = "unknown_unverified_availability"

ALLOWED_AVAILABILITY_POLICY_IDS = {
    AVAILABILITY_POLICY_FIXTURE_DECLARED,
    AVAILABILITY_POLICY_EXTERNAL_VERIFIED,
    AVAILABILITY_POLICY_UNKNOWN_UNVERIFIED,
}


def _coerce_enum(enum_cls: type[Enum], value: object, field_name: str) -> Enum:
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(value)  # type: ignore[arg-type]
    except Exception as exc:  # pragma: no cover - defensive, exercised in tests
        raise ValueError(f"invalid {field_name}: {value!r}") from exc


def _ensure_utc_timestamp(value: object, field_name: str, *, allow_none: bool = False) -> pd.Timestamp | None:
    if value is None or value is pd.NaT or pd.isna(value):
        if allow_none:
            return None
        raise ValueError(f"{field_name} is required")

    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None or timestamp.tz is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return timestamp.tz_convert("UTC")


def _normalize_tuple_of_strings(value: Iterable[object] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(sorted({str(item) for item in value}))


def _normalize_availability_policy_id(value: object) -> str:
    policy_id = str(value).strip()
    if not policy_id:
        raise ValueError("availability_policy_id must not be empty")
    if policy_id not in ALLOWED_AVAILABILITY_POLICY_IDS:
        raise ValueError(
            f"availability_policy_id must be one of {sorted(ALLOWED_AVAILABILITY_POLICY_IDS)!r}"
        )
    return policy_id


def _normalize_distribution(
    value: Iterable[tuple[AvailabilityStatus | str, int]] | None,
    field_name: str,
) -> tuple[tuple[AvailabilityStatus, int], ...]:
    if value is None:
        return ()

    normalized: list[tuple[AvailabilityStatus, int]] = []
    for status, count in value:
        status_enum = _coerce_enum(AvailabilityStatus, status, field_name)
        if not isinstance(count, Integral) or isinstance(count, bool):
            raise ValueError(f"{field_name} counts must be integers")
        count_value = int(count)
        if count_value < 0:
            raise ValueError(f"{field_name} counts must be non-negative")
        if status_enum == AvailabilityStatus.UNDEFINED:
            raise ValueError(f"{field_name} cannot include undefined status")
        normalized.append((status_enum, count_value))
    sort_rank = {
        AvailabilityStatus.VERIFIED_TIMESTAMP: 0,
        AvailabilityStatus.UNKNOWN_UNVERIFIED: 1,
        AvailabilityStatus.UNDEFINED: 2,
    }
    normalized.sort(key=lambda item: sort_rank[item[0]])
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class MarketDataProvenance:
    provider_id: str
    provider_version: str | None = None
    source_artifact_id: str | None = None
    source_artifact_hash: str | None = None
    requested_instruments: tuple[str, ...] = ()
    requested_start_timestamp: pd.Timestamp | None = None
    requested_end_timestamp: pd.Timestamp | None = None
    requested_price_type: PriceType = PriceType.CLOSE
    requested_adjustment_state: AdjustmentState = AdjustmentState.RAW
    retrieval_timestamp_utc: pd.Timestamp = field(
        default_factory=lambda: pd.Timestamp(datetime.now(timezone.utc))
    )
    retrieval_environment: str = "fixture"
    price_type: PriceType = PriceType.CLOSE
    adjustment_state: AdjustmentState = AdjustmentState.RAW
    timestamp_timezone_policy: str = "utc"
    availability_policy_id: str = AVAILABILITY_POLICY_FIXTURE_DECLARED
    source_availability_status_distribution: tuple[tuple[AvailabilityStatus, int], ...] = ()
    availability_status_distribution: tuple[tuple[AvailabilityStatus, int], ...] = ()
    universe_type: UniverseType = UniverseType.STATIC
    universe_instruments: tuple[str, ...] = ()
    universe_definition: str | None = None
    survivorship_limitation_flag: bool = False
    survivorship_limitation_code: str | None = None
    canonical_schema_version: str = "1"
    returns_schema_version: str = "1"
    regimequant_version: str = "0.0.0"
    adjusted_vintage_unverified_flag: bool = False
    pit_membership_not_supported_flag: bool = True
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider_id", str(self.provider_id))
        if not self.provider_id:
            raise ValueError("provider_id must not be empty")

        object.__setattr__(self, "provider_version", None if self.provider_version is None else str(self.provider_version))
        object.__setattr__(self, "source_artifact_id", None if self.source_artifact_id is None else str(self.source_artifact_id))
        object.__setattr__(self, "source_artifact_hash", None if self.source_artifact_hash is None else str(self.source_artifact_hash))

        requested_instruments = _normalize_tuple_of_strings(self.requested_instruments)
        universe_instruments = _normalize_tuple_of_strings(self.universe_instruments)
        limitations = _normalize_tuple_of_strings(self.limitations)

        object.__setattr__(self, "requested_instruments", requested_instruments)
        object.__setattr__(self, "universe_instruments", universe_instruments)
        object.__setattr__(self, "limitations", limitations)
        object.__setattr__(
            self,
            "availability_policy_id",
            _normalize_availability_policy_id(self.availability_policy_id),
        )

        requested_start = _ensure_utc_timestamp(self.requested_start_timestamp, "requested_start_timestamp", allow_none=True)
        requested_end = _ensure_utc_timestamp(self.requested_end_timestamp, "requested_end_timestamp", allow_none=True)
        retrieval_timestamp = _ensure_utc_timestamp(self.retrieval_timestamp_utc, "retrieval_timestamp_utc")
        object.__setattr__(self, "requested_start_timestamp", requested_start)
        object.__setattr__(self, "requested_end_timestamp", requested_end)
        object.__setattr__(self, "retrieval_timestamp_utc", retrieval_timestamp)

        object.__setattr__(self, "requested_price_type", _coerce_enum(PriceType, self.requested_price_type, "requested_price_type"))
        object.__setattr__(self, "requested_adjustment_state", _coerce_enum(AdjustmentState, self.requested_adjustment_state, "requested_adjustment_state"))
        object.__setattr__(self, "price_type", _coerce_enum(PriceType, self.price_type, "price_type"))
        object.__setattr__(self, "adjustment_state", _coerce_enum(AdjustmentState, self.adjustment_state, "adjustment_state"))
        object.__setattr__(self, "universe_type", _coerce_enum(UniverseType, self.universe_type, "universe_type"))
        object.__setattr__(
            self,
            "source_availability_status_distribution",
            _normalize_distribution(
                self.source_availability_status_distribution,
                "source_availability_status_distribution",
            ),
        )
        object.__setattr__(
            self,
            "availability_status_distribution",
            _normalize_distribution(
                self.availability_status_distribution,
                "availability_status_distribution",
            ),
        )

        if self.requested_price_type != self.price_type:
            raise ValueError("requested_price_type and price_type must match")
        if self.requested_adjustment_state != self.adjustment_state:
            raise ValueError("requested_adjustment_state and adjustment_state must match")
        if self.universe_type != UniverseType.STATIC:
            raise ValueError("M1 supports only UniverseType.STATIC")
        if (
            self.availability_policy_id == AVAILABILITY_POLICY_UNKNOWN_UNVERIFIED
            and (
                any(status == AvailabilityStatus.VERIFIED_TIMESTAMP for status, _ in self.availability_status_distribution)
                or any(status == AvailabilityStatus.VERIFIED_TIMESTAMP for status, _ in self.source_availability_status_distribution)
            )
        ):
            raise ValueError("unknown-unverified availability policy cannot claim verified availability observations")
        if self.adjustment_state == AdjustmentState.ADJUSTED and not self.adjusted_vintage_unverified_flag:
            raise ValueError("adjusted historical data must preserve unverified vintage limitation")
        if self.adjustment_state == AdjustmentState.RAW and self.adjusted_vintage_unverified_flag:
            raise ValueError("raw data cannot claim adjusted vintage limitation")
        if self.survivorship_limitation_flag and not self.survivorship_limitation_code:
            raise ValueError("survivorship_limitation_code is required when survivorship_limitation_flag is true")
        if not self.pit_membership_not_supported_flag:
            raise ValueError("M1 must not claim point-in-time membership support")
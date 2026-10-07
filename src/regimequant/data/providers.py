"""Provider abstractions and deterministic fixture provider for M1."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timezone
from typing import Mapping, Protocol, Sequence

import pandas as pd

from .errors import MarketDataProviderError
from .models import (
    AVAILABILITY_POLICY_FIXTURE_DECLARED,
    AdjustmentState,
    AvailabilityStatus,
    MarketDataProvenance,
    PriceType,
    UniverseType,
)


class MarketDataProvider(Protocol):
    def fetch_raw(
        self,
        *,
        instruments: Sequence[str] | None = None,
        start_timestamp: pd.Timestamp | str | None = None,
        end_timestamp: pd.Timestamp | str | None = None,
        price_type: PriceType = PriceType.CLOSE,
        adjustment_state: AdjustmentState = AdjustmentState.RAW,
    ) -> tuple[pd.DataFrame, MarketDataProvenance]:
        """Return normalized raw observations and provenance."""


def _to_utc_timestamp(
    value: object,
    field_name: str,
    *,
    allow_none: bool = False,
    source_timezone: str | None = None,
) -> pd.Timestamp | None:
    if value is None or value is pd.NaT or pd.isna(value):
        if allow_none:
            return None
        raise MarketDataProviderError(f"{field_name} is required")

    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None or timestamp.tz is None:
        if source_timezone is None:
            raise MarketDataProviderError(f"{field_name} is naive and no source timezone policy was declared")
        timestamp = timestamp.tz_localize(source_timezone)
    return timestamp.tz_convert("UTC")


def _normalize_column_map(column_map: Mapping[str, str] | None) -> dict[str, str]:
    if column_map is None:
        return {}
    return {str(source): str(target) for source, target in column_map.items()}


@dataclass(frozen=True, slots=True)
class FixtureMarketDataProvider:
    """Deterministic fixture provider that normalizes source rows for tests."""

    source_table: pd.DataFrame
    provider_id: str = "fixture"
    provider_version: str | None = None
    source_artifact_id: str | None = "fixture"
    source_artifact_hash: str | None = None
    retrieval_timestamp_utc: pd.Timestamp = field(
        default_factory=lambda: pd.Timestamp.now(tz=timezone.utc)
    )
    retrieval_environment: str = "fixture"
    source_timezone: str | None = "UTC"
    column_map: Mapping[str, str] | None = None
    universe_instruments: Sequence[str] = ()
    universe_definition: str | None = "static fixture universe"
    survivorship_limitation_flag: bool = False
    survivorship_limitation_code: str | None = None

    def fetch_raw(
        self,
        *,
        instruments: Sequence[str] | None = None,
        start_timestamp: pd.Timestamp | str | None = None,
        end_timestamp: pd.Timestamp | str | None = None,
        price_type: PriceType = PriceType.CLOSE,
        adjustment_state: AdjustmentState = AdjustmentState.RAW,
    ) -> tuple[pd.DataFrame, MarketDataProvenance]:
        try:
            source = self.source_table.copy(deep=True)
            rename_map = _normalize_column_map(self.column_map)
            if rename_map:
                source = source.rename(columns=rename_map)

            required_columns = {
                "instrument",
                "timestamp",
                "available_at",
                "availability_status",
                "price",
                "price_type",
                "adjustment_state",
                "currency",
            }
            missing = required_columns.difference(source.columns)
            if missing:
                raise MarketDataProviderError(f"fixture source table missing required columns: {sorted(missing)!r}")

            if "source_observation_id" not in source.columns:
                source["source_observation_id"] = None

            instruments_set = None if instruments is None else {str(instrument) for instrument in instruments}
            if instruments_set is not None:
                source = source[source["instrument"].astype(str).isin(instruments_set)]

            source["timestamp"] = [
                _to_utc_timestamp(value, "timestamp", source_timezone=self.source_timezone)
                for value in source["timestamp"]
            ]
            source["available_at"] = [
                _to_utc_timestamp(value, "available_at", allow_none=True, source_timezone=self.source_timezone)
                for value in source["available_at"]
            ]

            if start_timestamp is not None:
                start_utc = _to_utc_timestamp(start_timestamp, "start_timestamp", source_timezone=self.source_timezone)
                source = source[source["timestamp"] >= start_utc]
            if end_timestamp is not None:
                end_utc = _to_utc_timestamp(end_timestamp, "end_timestamp", source_timezone=self.source_timezone)
                source = source[source["timestamp"] <= end_utc]

            source["price_type"] = source["price_type"].map(lambda value: PriceType(value).value)
            source["adjustment_state"] = source["adjustment_state"].map(lambda value: AdjustmentState(value).value)
            source["availability_status"] = source["availability_status"].map(lambda value: AvailabilityStatus(value).value)
            source["instrument"] = source["instrument"].astype(str)
            source["currency"] = source["currency"].where(~source["currency"].isna(), None)
            source["source_observation_id"] = source["source_observation_id"].where(~source["source_observation_id"].isna(), None)
            source["price"] = source["price"].astype(float)

            requested_instruments = tuple(str(instrument) for instrument in (instruments or ()))
            provenance = MarketDataProvenance(
                provider_id=self.provider_id,
                provider_version=self.provider_version,
                source_artifact_id=self.source_artifact_id,
                source_artifact_hash=self.source_artifact_hash,
                requested_instruments=requested_instruments,
                requested_start_timestamp=start_timestamp,
                requested_end_timestamp=end_timestamp,
                requested_price_type=price_type,
                requested_adjustment_state=adjustment_state,
                retrieval_timestamp_utc=self.retrieval_timestamp_utc,
                retrieval_environment=self.retrieval_environment,
                price_type=price_type,
                adjustment_state=adjustment_state,
                timestamp_timezone_policy=self.source_timezone or "explicit-naive-source-localization",
                availability_policy_id=AVAILABILITY_POLICY_FIXTURE_DECLARED,
                source_availability_status_distribution=tuple(
                    (
                        AvailabilityStatus(status),
                        int(count),
                    )
                    for status, count in source["availability_status"].value_counts(sort=False).items()
                ),
                availability_status_distribution=(),
                universe_type=UniverseType.STATIC,
                universe_instruments=tuple(str(instrument) for instrument in self.universe_instruments),
                universe_definition=self.universe_definition,
                survivorship_limitation_flag=self.survivorship_limitation_flag,
                survivorship_limitation_code=self.survivorship_limitation_code,
                canonical_schema_version="1",
                returns_schema_version="1",
                regimequant_version="0.0.0",
                adjusted_vintage_unverified_flag=adjustment_state == AdjustmentState.ADJUSTED,
                pit_membership_not_supported_flag=True,
                limitations=("fixture-provider",),
            )

            return source.reset_index(drop=True), provenance
        except MarketDataProviderError:
            raise
        except Exception as exc:  # pragma: no cover - defensive wrapper
            raise MarketDataProviderError(str(exc)) from exc
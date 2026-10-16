"""Input and provenance validation for RegimeQuant M2 feature computation."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import math
from typing import Any, Iterable

import pandas as pd

from regimequant.data.models import AvailabilityStatus, MarketDataProvenance, ReturnType
from regimequant.data.returns import RETURN_COLUMNS

from .definitions import FeatureSpec, canonical_json_dumps, feature_instance_id
from .models import FeatureValidationError


_ALLOWED_INPUT_AVAILABILITY = {
    AvailabilityStatus.VERIFIED_TIMESTAMP.value,
    AvailabilityStatus.UNKNOWN_UNVERIFIED.value,
    AvailabilityStatus.UNDEFINED.value,
}


def _coerce_utc_timestamp(value: object, *, allow_none: bool) -> pd.Timestamp | None:
    if value is None or value is pd.NaT or pd.isna(value):
        if allow_none:
            return None
        raise FeatureValidationError("timestamp field must not be null")
    ts = pd.Timestamp(value)
    if ts.tzinfo is None or ts.tz is None:
        raise FeatureValidationError("timestamp fields must be timezone-aware")
    return ts.tz_convert("UTC")


def _validate_return_row(row: pd.Series) -> None:
    status = str(row["availability_status"])
    if status not in _ALLOWED_INPUT_AVAILABILITY:
        raise FeatureValidationError(f"invalid availability_status: {status!r}")

    value = row["return_value"]
    available_at = row["available_at"]

    if status == AvailabilityStatus.UNDEFINED.value:
        if not pd.isna(value):
            raise FeatureValidationError("UNDEFINED source returns must carry NaN return_value")
        if available_at is not None and not pd.isna(available_at):
            raise FeatureValidationError("UNDEFINED source returns must carry null available_at")
        return

    if value is None or pd.isna(value):
        raise FeatureValidationError("DEFINED source returns must carry numeric return_value")
    value_float = float(value)
    if not math.isfinite(value_float):
        raise FeatureValidationError("source return_value must be finite")

    if status == AvailabilityStatus.VERIFIED_TIMESTAMP.value:
        if available_at is None or pd.isna(available_at):
            raise FeatureValidationError("VERIFIED source returns must carry available_at")
    else:
        if available_at is not None and not pd.isna(available_at):
            raise FeatureValidationError("UNKNOWN source returns must carry null available_at")


def _validate_provenance(frame: pd.DataFrame) -> dict[str, Any]:
    provenance = frame.attrs.get("provenance")
    if not isinstance(provenance, dict):
        raise FeatureValidationError("returns table attrs.provenance must be a dictionary")

    source_canonical_provenance = provenance.get("source_canonical_provenance")
    if not isinstance(source_canonical_provenance, MarketDataProvenance):
        raise FeatureValidationError(
            "attrs.provenance.source_canonical_provenance must be MarketDataProvenance"
        )

    return_policy_version = provenance.get("return_policy_version")
    if not isinstance(return_policy_version, str):
        raise FeatureValidationError("attrs.provenance.return_policy_version must be a string")

    return_type = provenance.get("return_type")
    if return_type not in {ReturnType.SIMPLE.value, ReturnType.LOG.value}:
        raise FeatureValidationError("attrs.provenance.return_type must be simple or log")

    generated_at_utc = provenance.get("generated_at_utc")
    if generated_at_utc is None:
        raise FeatureValidationError("attrs.provenance.generated_at_utc is required")

    return provenance


def build_parent_provenance_digest(frame: pd.DataFrame, provenance: dict[str, Any]) -> str:
    """Build deterministic digest over parent M1 return provenance."""

    source_provenance = provenance["source_canonical_provenance"]
    payload = {
        "source_canonical_provenance": asdict(source_provenance),
        "return_policy_version": provenance["return_policy_version"],
        "return_type": provenance["return_type"],
        "price_type": sorted({str(v) for v in frame["price_type"].astype(str).tolist()}),
        "adjustment_state": sorted(
            {str(v) for v in frame["adjustment_state"].astype(str).tolist()}
        ),
    }
    serialized = canonical_json_dumps(payload)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def build_feature_spec_set_hash(specs: Iterable[FeatureSpec]) -> str:
    """Build deterministic hash over the validated feature spec identity set."""

    ids = sorted(feature_instance_id(spec) for spec in specs)
    serialized = canonical_json_dumps({"feature_instance_ids": ids})
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def validate_returns_input(
    returns_table: pd.DataFrame,
    specs: Iterable[FeatureSpec],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate M1 returns table and provenance compatibility for M2 processing."""

    if not isinstance(returns_table, pd.DataFrame):
        raise FeatureValidationError("returns_table must be a pandas DataFrame")

    frame = returns_table.copy(deep=True)

    required = set(RETURN_COLUMNS)
    missing = required.difference(frame.columns)
    if missing:
        raise FeatureValidationError(f"missing required return columns: {sorted(missing)!r}")

    extras = set(frame.columns).difference(required)
    if extras:
        raise FeatureValidationError(f"unexpected return columns: {sorted(extras)!r}")

    frame = frame.loc[:, RETURN_COLUMNS].copy()

    frame["timestamp"] = [
        _coerce_utc_timestamp(value, allow_none=False) for value in frame["timestamp"]
    ]
    frame["available_at"] = [
        _coerce_utc_timestamp(value, allow_none=True) for value in frame["available_at"]
    ]
    frame["previous_timestamp"] = [
        _coerce_utc_timestamp(value, allow_none=True) for value in frame["previous_timestamp"]
    ]
    frame["previous_available_at"] = [
        _coerce_utc_timestamp(value, allow_none=True)
        for value in frame["previous_available_at"]
    ]

    for _, row in frame.iterrows():
        _validate_return_row(row)

    key_columns = ["instrument", "timestamp", "return_type", "price_type", "adjustment_state"]
    if frame.duplicated(subset=key_columns).any():
        raise FeatureValidationError("duplicate return keys detected")

    return_types = sorted({str(v) for v in frame["return_type"].astype(str).tolist()})
    if len(return_types) != 1:
        raise FeatureValidationError("M2 requires one input return_type per computation")

    price_types = sorted({str(v) for v in frame["price_type"].astype(str).tolist()})
    if len(price_types) != 1:
        raise FeatureValidationError("M2 requires one price_type per computation")

    adjustment_states = sorted(
        {str(v) for v in frame["adjustment_state"].astype(str).tolist()}
    )
    if len(adjustment_states) != 1:
        raise FeatureValidationError("M2 requires one adjustment_state per computation")

    provenance = _validate_provenance(frame)
    if provenance["return_type"] != return_types[0]:
        raise FeatureValidationError("attrs.provenance.return_type does not match return rows")

    spec_return_types = sorted({spec.input_return_type.value for spec in specs})
    if len(spec_return_types) > 1:
        raise FeatureValidationError("all feature specs must use the same input return_type")
    if spec_return_types and spec_return_types[0] != return_types[0]:
        raise FeatureValidationError("feature spec input_return_type mismatch with return rows")

    parent_digest = build_parent_provenance_digest(frame, provenance)
    context = {
        "return_type": return_types[0],
        "price_type": price_types[0],
        "adjustment_state": adjustment_states[0],
        "parent_provenance_digest": parent_digest,
    }
    return frame, context

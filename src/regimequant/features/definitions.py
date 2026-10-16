"""Feature specifications and deterministic identity generation for M2."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import math
from typing import Any, Iterable, Mapping

import pandas as pd

from regimequant.data.models import ReturnType

from .models import FeatureIdentityError, FeatureValidationError, ScopeType


FEATURE_ROLLING_MEAN = "rolling_mean_return"
FEATURE_CUMULATIVE_RETURN = "cumulative_return_simple_equivalent"
FEATURE_LAGGED_MOMENTUM = "lagged_momentum_simple_equivalent"
FEATURE_ROLLING_STD = "rolling_sample_std"
FEATURE_DOWNSIDE_DEVIATION = "downside_deviation"
FEATURE_REALIZED_VARIANCE = "realized_variance"
FEATURE_VOLATILITY_RATIO = "short_long_volatility_ratio"
FEATURE_DOWNSIDE_FREQUENCY = "downside_observation_frequency"

CORE_FEATURE_NAMES = {
    FEATURE_ROLLING_MEAN,
    FEATURE_CUMULATIVE_RETURN,
    FEATURE_LAGGED_MOMENTUM,
    FEATURE_ROLLING_STD,
    FEATURE_DOWNSIDE_DEVIATION,
    FEATURE_REALIZED_VARIANCE,
    FEATURE_VOLATILITY_RATIO,
    FEATURE_DOWNSIDE_FREQUENCY,
}


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Immutable M2 feature specification."""

    feature_name: str
    feature_definition_version: str = "1"
    input_return_type: ReturnType = ReturnType.SIMPLE
    output_transformation_mode: str = "native"
    scope_type: ScopeType = ScopeType.INSTRUMENT
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
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.feature_name not in CORE_FEATURE_NAMES:
            raise FeatureValidationError(f"unsupported feature_name: {self.feature_name!r}")
        if self.scope_type != ScopeType.INSTRUMENT:
            raise FeatureValidationError("M2 v1 supports instrument scope only")
        if self.window_kind != "observation_count":
            raise FeatureValidationError("M2 v1 supports observation_count windows only")
        if self.window_size <= 0:
            raise FeatureValidationError("window_size must be positive")
        if self.min_observations <= 0:
            raise FeatureValidationError("min_observations must be positive")
        if self.min_observations > self.window_size:
            raise FeatureValidationError("min_observations must be <= window_size")
        if self.ddof is not None and self.ddof < 0:
            raise FeatureValidationError("ddof must be non-negative when provided")
        if self.feature_name == FEATURE_VOLATILITY_RATIO:
            if self.short_window_size is None or self.long_window_size is None:
                raise FeatureValidationError(
                    "volatility ratio requires short_window_size and long_window_size"
                )
            if self.short_window_size <= 0 or self.long_window_size <= 0:
                raise FeatureValidationError("volatility windows must be positive")
            if self.short_window_size >= self.long_window_size:
                raise FeatureValidationError("short_window_size must be < long_window_size")


def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, ReturnType):
        return value.value
    if isinstance(value, ScopeType):
        return value.value
    if isinstance(value, datetime):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FeatureValidationError("non-finite numeric value in identity payload")
        return value
    if isinstance(value, Mapping):
        return {
            str(k): _normalize_json_value(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_json_value(v) for v in value]
    return str(value)


def canonical_json_dumps(payload: Mapping[str, Any]) -> str:
    """Serialize payload deterministically for identity and digest hashes."""

    normalized = _normalize_json_value(payload)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def feature_spec_identity_payload(spec: FeatureSpec) -> dict[str, Any]:
    """Return canonical identity payload for a feature specification."""

    payload = {
        "feature_name": spec.feature_name,
        "feature_definition_version": spec.feature_definition_version,
        "input_return_type": spec.input_return_type.value,
        "output_transformation_mode": spec.output_transformation_mode,
        "scope_type": spec.scope_type.value,
        "window_kind": spec.window_kind,
        "window_size": spec.window_size,
        "min_observations": spec.min_observations,
        "include_current": spec.include_current,
        "require_complete_window": spec.require_complete_window,
        "allow_partial_valid": spec.allow_partial_valid,
        "ddof": spec.ddof,
        "downside_threshold": spec.downside_threshold,
        "short_window_size": spec.short_window_size,
        "long_window_size": spec.long_window_size,
        "parameters": dict(spec.parameters),
    }
    return payload


def feature_instance_id(spec: FeatureSpec) -> str:
    """Build deterministic feature identity using canonical JSON + SHA-256."""

    serialized = canonical_json_dumps(feature_spec_identity_payload(spec))
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return f"m2v1_sha256_{digest}"


def validate_feature_specs(specs: Iterable[FeatureSpec]) -> list[FeatureSpec]:
    """Validate specs and enforce deterministic identity uniqueness."""

    spec_list = list(specs)
    seen_payload_by_id: dict[str, str] = {}
    for spec in spec_list:
        identity = feature_instance_id(spec)
        payload = canonical_json_dumps(feature_spec_identity_payload(spec))
        previous_payload = seen_payload_by_id.get(identity)
        if previous_payload is None:
            seen_payload_by_id[identity] = payload
            continue
        if previous_payload != payload:
            raise FeatureIdentityError("feature identity collision detected")
        raise FeatureValidationError(f"duplicate feature specification: {spec.feature_name}")
    return spec_list


def list_m2_v1_feature_specs() -> list[FeatureSpec]:
    """Return the approved default M2 v1 core feature specifications."""

    return [
        FeatureSpec(
            feature_name=FEATURE_ROLLING_MEAN,
            output_transformation_mode="native",
            window_size=3,
            min_observations=3,
            include_current=True,
            allow_partial_valid=True,
        ),
        FeatureSpec(
            feature_name=FEATURE_CUMULATIVE_RETURN,
            output_transformation_mode="simple_equivalent",
            window_size=3,
            min_observations=3,
            include_current=True,
            require_complete_window=True,
            allow_partial_valid=False,
        ),
        FeatureSpec(
            feature_name=FEATURE_LAGGED_MOMENTUM,
            output_transformation_mode="simple_equivalent",
            window_size=2,
            min_observations=2,
            include_current=False,
            require_complete_window=True,
            allow_partial_valid=False,
        ),
        FeatureSpec(
            feature_name=FEATURE_ROLLING_STD,
            output_transformation_mode="native",
            window_size=3,
            min_observations=3,
            ddof=1,
            include_current=True,
            allow_partial_valid=True,
        ),
        FeatureSpec(
            feature_name=FEATURE_DOWNSIDE_DEVIATION,
            output_transformation_mode="native",
            window_size=3,
            min_observations=3,
            downside_threshold=0.0,
            include_current=True,
            allow_partial_valid=True,
        ),
        FeatureSpec(
            feature_name=FEATURE_REALIZED_VARIANCE,
            output_transformation_mode="native",
            window_size=3,
            min_observations=3,
            include_current=True,
            allow_partial_valid=True,
        ),
        FeatureSpec(
            feature_name=FEATURE_VOLATILITY_RATIO,
            output_transformation_mode="native",
            window_size=3,
            min_observations=2,
            ddof=1,
            include_current=True,
            allow_partial_valid=True,
            short_window_size=2,
            long_window_size=3,
        ),
        FeatureSpec(
            feature_name=FEATURE_DOWNSIDE_FREQUENCY,
            output_transformation_mode="native",
            window_size=3,
            min_observations=3,
            downside_threshold=0.0,
            include_current=True,
            allow_partial_valid=True,
        ),
    ]

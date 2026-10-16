"""Models and schema contracts for RegimeQuant M2 features."""

from __future__ import annotations

from enum import Enum

import pandas as pd


class FeatureError(Exception):
    """Base error for M2 feature-engine failures."""


class FeatureValidationError(FeatureError):
    """Raised when feature inputs or specifications violate the contract."""


class FeatureComputationError(FeatureError):
    """Raised when feature computation fails unexpectedly."""


class FeatureIdentityError(FeatureError):
    """Raised when feature identity generation or uniqueness fails."""


class ScopeType(str, Enum):
    INSTRUMENT = "instrument"


class MathStatus(str, Enum):
    DEFINED = "defined"
    UNDEFINED = "undefined"


class UndefinedReason(str, Enum):
    INSUFFICIENT_OBSERVATIONS = "insufficient_observations"
    INPUT_UNDEFINED_RETURN = "input_undefined_return"
    ZERO_DENOMINATOR = "zero_denominator"
    NONFINITE_INPUT = "nonfinite_input"


FEATURE_COLUMNS = [
    "scope_type",
    "scope_id",
    "timestamp",
    "feature_name",
    "feature_instance_id",
    "feature_value",
    "math_status",
    "undefined_reason",
    "input_availability_status",
    "available_at",
    "input_return_type",
    "price_type",
    "adjustment_state",
    "window_kind",
    "window_size",
    "min_observations",
    "observation_count_used",
    "feature_definition_version",
]

PRIMARY_KEY_COLUMNS = [
    "scope_type",
    "scope_id",
    "timestamp",
    "feature_instance_id",
]

SORT_COLUMNS = [
    "scope_type",
    "scope_id",
    "timestamp",
    "feature_name",
    "feature_definition_version",
    "feature_instance_id",
]


def empty_feature_frame() -> pd.DataFrame:
    """Return an empty feature frame with stable schema and dtypes."""

    return pd.DataFrame(
        {
            "scope_type": pd.Series(dtype="string"),
            "scope_id": pd.Series(dtype="string"),
            "timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
            "feature_name": pd.Series(dtype="string"),
            "feature_instance_id": pd.Series(dtype="string"),
            "feature_value": pd.Series(dtype="float64"),
            "math_status": pd.Series(dtype="string"),
            "undefined_reason": pd.Series(dtype="string"),
            "input_availability_status": pd.Series(dtype="string"),
            "available_at": pd.Series(dtype="datetime64[ns, UTC]"),
            "input_return_type": pd.Series(dtype="string"),
            "price_type": pd.Series(dtype="string"),
            "adjustment_state": pd.Series(dtype="string"),
            "window_kind": pd.Series(dtype="string"),
            "window_size": pd.Series(dtype="Int64"),
            "min_observations": pd.Series(dtype="Int64"),
            "observation_count_used": pd.Series(dtype="Int64"),
            "feature_definition_version": pd.Series(dtype="string"),
        }
    ).loc[:, FEATURE_COLUMNS]


def validate_feature_state_row(row: dict[str, object]) -> None:
    """Validate legal combinations for value/math/availability state fields."""

    math_status = str(row["math_status"])
    undefined_reason = row.get("undefined_reason")
    feature_value = row.get("feature_value")
    input_availability_status = str(row["input_availability_status"])
    available_at = row.get("available_at")

    if math_status not in {MathStatus.DEFINED.value, MathStatus.UNDEFINED.value}:
        raise FeatureValidationError(f"invalid math_status: {math_status!r}")
    if input_availability_status not in {"verified_timestamp", "unknown_unverified"}:
        raise FeatureValidationError(
            f"invalid input_availability_status: {input_availability_status!r}"
        )

    if math_status == MathStatus.DEFINED.value:
        if undefined_reason is not None and not pd.isna(undefined_reason):
            raise FeatureValidationError("DEFINED rows must not carry undefined_reason")
        if feature_value is None or pd.isna(feature_value):
            raise FeatureValidationError("DEFINED rows must carry finite feature_value")
        if input_availability_status == "verified_timestamp" and (
            available_at is None or pd.isna(available_at)
        ):
            raise FeatureValidationError(
                "DEFINED verified rows must carry non-null available_at"
            )
    else:
        if undefined_reason is None or pd.isna(undefined_reason):
            raise FeatureValidationError("UNDEFINED rows must carry undefined_reason")
        if feature_value is None or not pd.isna(feature_value):
            raise FeatureValidationError("UNDEFINED rows must carry NaN feature_value")
        if available_at is not None and not pd.isna(available_at):
            raise FeatureValidationError("UNDEFINED rows must carry null available_at")

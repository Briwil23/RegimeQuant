"""Public compute entry points for the RegimeQuant M2 feature engine."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

import pandas as pd

from .definitions import FeatureSpec, validate_feature_specs
from .models import (
    FEATURE_COLUMNS,
    PRIMARY_KEY_COLUMNS,
    SORT_COLUMNS,
    FeatureValidationError,
    empty_feature_frame,
    validate_feature_state_row,
)
from .rolling import compute_feature_rows_for_instrument
from .validation import build_feature_spec_set_hash, validate_returns_input


def compute_features_from_returns(
    returns_table: pd.DataFrame,
    feature_specs: Iterable[FeatureSpec],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Compute approved M2 v1 instrument features from validated M1 return tables."""

    specs = validate_feature_specs(feature_specs)
    if not specs:
        diagnostics = {"input_row_count": int(len(returns_table)), "output_row_count": 0}
        frame = empty_feature_frame()
        frame.attrs["provenance"] = {
            "parent_return_provenance_digest": None,
            "feature_spec_set_hash": None,
            "generated_at_utc": pd.Timestamp(datetime.now(timezone.utc)),
            "metadata_durability": "DataFrame.attrs is in-memory only and not a durable serialization contract",
        }
        frame.attrs["diagnostics"] = diagnostics
        return frame, diagnostics

    validated, context = validate_returns_input(returns_table, specs)

    rows: list[dict[str, object]] = []
    grouped = validated.sort_values(["instrument", "timestamp"], kind="mergesort").groupby(
        "instrument", sort=False, dropna=False
    )

    for _, instrument_frame in grouped:
        instrument_frame = instrument_frame.reset_index(drop=True)
        for spec in specs:
            rows.extend(
                compute_feature_rows_for_instrument(
                    instrument_frame,
                    spec,
                    price_type=context["price_type"],
                    adjustment_state=context["adjustment_state"],
                )
            )

    if rows:
        output = pd.DataFrame(rows).loc[:, FEATURE_COLUMNS]
    else:
        output = empty_feature_frame()

    if output.duplicated(subset=PRIMARY_KEY_COLUMNS).any():
        raise FeatureValidationError("duplicate output feature keys detected")

    for row in output.to_dict("records"):
        validate_feature_state_row(row)

    output = output.sort_values(SORT_COLUMNS, kind="mergesort").reset_index(drop=True)

    diagnostics = {
        "input_row_count": int(len(validated)),
        "output_row_count": int(len(output)),
    }
    output.attrs["provenance"] = {
        "parent_return_provenance_digest": context["parent_provenance_digest"],
        "feature_spec_set_hash": build_feature_spec_set_hash(specs),
        "generated_at_utc": pd.Timestamp(datetime.now(timezone.utc)),
        "metadata_durability": "DataFrame.attrs is in-memory only and not a durable serialization contract",
    }
    output.attrs["diagnostics"] = diagnostics
    return output, diagnostics

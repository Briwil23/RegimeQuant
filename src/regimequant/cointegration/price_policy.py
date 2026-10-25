from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from regimequant.cointegration.contracts import PricePolicyResult
from regimequant.cointegration.validation import ValidationError, validate_causal_availability


class PricePolicyError(ValueError):
    """Raised when raw close data fails the frozen M7 price contract."""


@dataclass(frozen=True, slots=True)
class DailyPricePolicy:
    required_min_prices: int = 253
    required_min_differences: int = 252
    duplicate_policy: str = "reject"
    require_business_day_calendar: bool = True

    @staticmethod
    def validate_series(
        prices: pd.Series,
        ticker: str,
        *,
        required_min_prices: int = 253,
        required_min_differences: int = 252,
        cutoff: object | None = None,
        availability_evidence: list[dict[str, object]] | tuple[dict[str, object], ...] | None = None,
        identity_evidence: list[dict[str, object]] | tuple[dict[str, object], ...] | None = None,
        alias_records: list[dict[str, object]] | tuple[dict[str, object], ...] | None = None,
        continuity_records: list[dict[str, object]] | tuple[dict[str, object], ...] | None = None,
    ) -> PricePolicyResult:
        if not isinstance(prices, pd.Series):
            raise PricePolicyError("prices must be a pandas Series")
        series = prices.copy().sort_index()
        if series.empty:
            raise PricePolicyError("prices must not be empty")
        if series.index.has_duplicates:
            raise PricePolicyError("duplicate timestamps are not admissible")

        if not np.isfinite(series.to_numpy(dtype=float)).all():
            raise PricePolicyError("prices must be finite")
        if (series <= 0).any():
            raise PricePolicyError("prices must be strictly positive")

        normalized = series.copy()
        normalized.index = pd.DatetimeIndex(normalized.index).tz_localize("UTC") if pd.DatetimeIndex(normalized.index).tz is None else pd.DatetimeIndex(normalized.index).tz_convert("UTC")
        if isinstance(normalized.index, pd.DatetimeIndex):
            expected = pd.bdate_range(normalized.index.min(), normalized.index.max())
            if len(expected) != len(normalized.index):
                missing = expected.difference(normalized.index)
                if len(missing) > 0:
                    raise PricePolicyError(f"missing required sessions: {len(missing)}")

        if len(normalized) < required_min_prices:
            raise PricePolicyError("at least 253 aligned prices are required")

        diff = normalized.pct_change().dropna()
        if len(diff) < required_min_differences:
            raise PricePolicyError("at least 252 first differences are required")

        if cutoff is not None:
            try:
                validate_causal_availability(
                    cutoff=cutoff,
                    price_series=normalized,
                    availability_evidence=availability_evidence,
                    identity_evidence=identity_evidence,
                    alias_records=alias_records,
                    continuity_records=continuity_records,
                )
            except ValidationError as exc:
                raise PricePolicyError(str(exc)) from exc

        return PricePolicyResult(
            valid=True,
            ticker=str(ticker).strip(),
            aligned_index=normalized.index,
            price_series=normalized,
            first_difference_count=len(diff),
            reasons=(),
        )

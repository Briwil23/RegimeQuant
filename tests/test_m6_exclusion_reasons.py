from __future__ import annotations

import pandas as pd

from regimequant.universe import (
    ExclusionReason,
    UniverseCandidate,
    determine_exclusion_reasons,
)


def _ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value).tz_convert("UTC")


def test_reason_codes_are_deterministic() -> None:
    reasons = determine_exclusion_reasons(
        observed_member=False,
        verified=False,
        unsupported_liquidity_metadata=True,
        future_effective_event=True,
    )
    assert ExclusionReason.MISSING_VERIFIED_MEMBERSHIP in reasons
    assert ExclusionReason.UNSUPPORTED_LIQUIDITY_METADATA in reasons
    assert ExclusionReason.FUTURE_EFFECTIVE_EVENT in reasons

    candidate = UniverseCandidate(
        security_id="sec-Z",
        ticker="Z",
        as_of=_ts("2024-01-06T00:00:00Z"),
        status="candidate",
        is_delisted=False,
        source="exchange",
        coverage_complete=False,
    )
    assert candidate.coverage_complete is False

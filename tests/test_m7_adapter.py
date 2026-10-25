from __future__ import annotations

import pandas as pd

from regimequant.cointegration.adapter import IdentityAdapterError, VerifiedIdentityAdapter


def test_verified_identity_adapter_accepts_m6_evidence() -> None:
    adapter = VerifiedIdentityAdapter(
        security_id="sec-1",
        ticker="AAPL",
        issuer_id="issuer-1",
        alias_records=[{"canonical_security_id": "sec-1", "symbol": "AAPL", "listing_venue": "NASDAQ", "valid_from": "2020-01-01", "valid_to": "2025-01-01"}],
        continuity_records=[{"predecessor_security_id": "sec-0", "successor_security_id": "sec-1", "event_kind": "listing", "effective_at": "2020-01-02", "verified_at": "2020-01-03", "source_authority": "exchange"}],
        evidence_records=[{"security_id": "sec-1", "ticker": "AAPL", "aliases": ("AAPL",), "authority": "exchange", "verified_at": "2020-01-02", "source": "exchange", "status": "verified", "ambiguous": False, "issuer_id": "issuer-1"}],
    )
    assert adapter.security_id == "sec-1"
    assert adapter.ticker == "AAPL"
    assert adapter.is_admissible_at(pd.Timestamp("2020-01-03", tz="UTC"))


def test_verified_identity_adapter_rejects_same_ticker_different_issuer() -> None:
    try:
        VerifiedIdentityAdapter(
            security_id="sec-1",
            ticker="MSFT",
            issuer_id="issuer-1",
            alias_records=[],
            continuity_records=[],
            evidence_records=[
                {"security_id": "sec-2", "ticker": "MSFT", "aliases": ("MSFT",), "authority": "exchange", "verified_at": "2020-01-02", "source": "exchange", "status": "verified", "ambiguous": False, "issuer_id": "issuer-2"}
            ],
        )
    except IdentityAdapterError:
        return
    raise AssertionError("same ticker with different issuer should be rejected")

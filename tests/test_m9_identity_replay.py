from __future__ import annotations

from decimal import Decimal

from regimequant.backtesting.m9_identity import accounting_identity, deterministic_hash, event_stream_identity, fill_ledger_identity, policy_identity


def test_identities_are_deterministic() -> None:
    payload = {"beta": Decimal("1.2"), "pair": "pair_A_B"}
    h1 = deterministic_hash(payload)
    h2 = deterministic_hash(payload)
    assert h1 == h2


def test_identity_changes_when_material_payload_changes() -> None:
    p1 = policy_identity({"budget": "2pct", "beta": Decimal("1.1")})
    p2 = policy_identity({"budget": "2pct", "beta": Decimal("1.2")})
    assert p1 != p2

    assert event_stream_identity([{"e": 1}]) != fill_ledger_identity([{"e": 1}])
    assert accounting_identity({"cash": "100"}) == accounting_identity({"cash": "100"})

from __future__ import annotations

from regimequant.signals.m8_identity import compute_signal_identity


def test_identity_is_deterministic_and_value_bound() -> None:
    payload = {
        "pair_id": "pair_0b8f0fbd6ed5f2be",
        "formation": {"sessions": ["2024-01-02", "2024-01-03"], "values": [100.0, 101.0]},
        "eval": {"session": "2024-01-04", "x": 102.0, "y": 123.4},
        "threshold": 2.0,
        "source_mode": "production",
    }
    a = compute_signal_identity(payload)
    b = compute_signal_identity(dict(payload))
    c = compute_signal_identity({**payload, "eval": {"session": "2024-01-04", "x": 102.0, "y": 123.5}})
    assert a == b
    assert a != c

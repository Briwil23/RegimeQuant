from __future__ import annotations

from typing import Any

from regimequant.cointegration.identity import compute_scientific_hash


def compute_family_digest(payload: dict[str, Any]) -> str:
    return compute_scientific_hash(payload)


def compute_eligibility_digest(payload: dict[str, Any]) -> str:
    return compute_scientific_hash(payload)


def compute_signal_identity(payload: dict[str, Any]) -> str:
    return compute_scientific_hash(payload)

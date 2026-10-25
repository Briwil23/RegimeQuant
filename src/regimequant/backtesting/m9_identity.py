from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from typing import Any


def _normalize(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {key: _normalize(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    return value


def deterministic_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(_normalize(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def policy_identity(policy_payload: dict[str, Any]) -> str:
    return deterministic_hash(policy_payload)


def event_stream_identity(events_payload: list[dict[str, Any]]) -> str:
    return deterministic_hash({"events": events_payload})


def fill_ledger_identity(fills_payload: list[dict[str, Any]]) -> str:
    return deterministic_hash({"fills": fills_payload})


def accounting_identity(accounting_payload: dict[str, Any]) -> str:
    return deterministic_hash({"accounting": accounting_payload})


def performance_identity(report_payload: dict[str, Any]) -> str:
    return deterministic_hash({"performance": report_payload})

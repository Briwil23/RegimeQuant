from __future__ import annotations

import hashlib
import json
import math
from typing import Any


def _canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _canonicalize(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(v) for v in value]
    if isinstance(value, set):
        return [_canonicalize(v) for v in sorted(value, key=lambda item: str(item))]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _serialize_value(value: Any) -> float | int | str | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("non-finite numeric values are not permitted in a scientific payload")
        return float(value)
    return str(value)


_NON_MATERIAL_HASH_KEYS = {
    "output_id",
    "result_id",
    "run_id",
    "runtime_metadata",
    "execution_metadata",
    "metadata",
    "provenance",
    "analysis_id",
}


def _strip_non_material_hash_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _strip_non_material_hash_metadata(item)
            for key, item in value.items()
            if str(key).lower() not in _NON_MATERIAL_HASH_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_strip_non_material_hash_metadata(item) for item in value]
    return value


def compute_canonical_pair_id(instrument_a: str, instrument_b: str, start: str | None = None, end: str | None = None, *, t_value: str | None = None) -> str:
    del start, end, t_value
    left, right = sorted((str(instrument_a).strip(), str(instrument_b).strip()))
    payload = {"instrument_a": left, "instrument_b": right}
    digest = hashlib.sha256(json.dumps(_canonicalize(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return f"pair_{digest[:16]}"


def compute_dataset_identity(
    *,
    security_ids: list[str] | tuple[str, ...],
    session_ids: list[str] | tuple[str, ...],
    aligned_prices: list[float] | tuple[float, ...],
    availability_evidence: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    identity_evidence: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    continuity_records: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    provenance: dict[str, Any] | None = None,
    pricing_policy_version: str = "m7v1",
) -> str:
    del provenance
    canonical_securities = sorted({str(item).strip() for item in security_ids if str(item).strip()})
    canonical_sessions = sorted({str(item).strip() for item in session_ids if str(item).strip()})
    serialized_prices = [
        {"session": str(session), "price": _serialize_value(value)}
        for session, value in sorted(zip(session_ids, aligned_prices), key=lambda item: str(item[0]))
    ]
    payload = {
        "security_ids": canonical_securities,
        "session_ids": canonical_sessions,
        "aligned_prices": serialized_prices,
        "availability_evidence": [dict(record) for record in availability_evidence],
        "identity_evidence": [dict(record) for record in identity_evidence],
        "continuity_records": [dict(record) for record in continuity_records],
        "pricing_policy_version": str(pricing_policy_version).strip(),
    }
    canonical = json.dumps(_canonicalize(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def compute_research_instance_id(
    research_id: str,
    instrument_a: str,
    instrument_b: str,
    *,
    dataset_identity: str | None = None,
    cutoff: str | None = None,
    snapshot_id: str | None = None,
    statistical_config: dict[str, Any] | None = None,
    t_value: str | None = None,
) -> str:
    payload = {
        "research_id": str(research_id).strip(),
        "instrument_a": str(instrument_a).strip(),
        "instrument_b": str(instrument_b).strip(),
        "dataset_identity": dataset_identity,
        "cutoff": cutoff,
        "snapshot_id": snapshot_id,
        "t_value": t_value,
        "statistical_config": statistical_config or {},
    }
    digest = hashlib.sha256(json.dumps(_canonicalize(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return f"research_{digest[:24]}"


def compute_scientific_hash(payload: dict[str, Any]) -> str:
    cleaned = _strip_non_material_hash_metadata(payload)
    canonical = json.dumps(_canonicalize(cleaned), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()

from __future__ import annotations

import hashlib
import json

from .contracts import M4RegimeStabilityReport


def _canonicalize(value):
    if isinstance(value, dict):
        return {str(k): _canonicalize(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def compute_m4_report_identity(report: M4RegimeStabilityReport) -> str:
    payload = report.to_dict()
    payload.pop("report_id", None)
    payload.pop("hash_prefix", None)
    canonical_json = json.dumps(_canonicalize(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    return f"m4v1_sha256_{digest}"

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


def _canonicalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, set):
        return [_canonicalize(item) for item in sorted(value, key=lambda item: str(item))]
    if isinstance(value, np.ndarray):
        return [_canonicalize(item) for item in value.tolist()]
    if isinstance(value, pd.Timestamp):
        return value.tz_convert("UTC").isoformat()
    if isinstance(value, pd.Timedelta):
        return value.isoformat()
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def canonical_json_dumps(payload: Mapping[str, Any] | Sequence[Any] | Any) -> str:
    return json.dumps(_canonicalize(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def compute_digest(payload: Mapping[str, Any] | Sequence[Any] | Any) -> str:
    return hashlib.sha256(canonical_json_dumps(payload).encode("utf-8")).hexdigest()

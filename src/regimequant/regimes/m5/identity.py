from __future__ import annotations

import hashlib
import json
from typing import Any

from .contracts import M5ModelArtifact, _scientific_identity_payload, canonical_json_dumps


def compute_m5_model_identity(artifact: M5ModelArtifact) -> str:
    payload = _scientific_identity_payload(artifact)
    canonical_json = canonical_json_dumps(payload)
    digest = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    return f"m5v1_sha256_{digest}"

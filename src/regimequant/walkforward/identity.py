from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

_DIGEST_KEYS = {"digest", "input_digest", "output_digest", "result_digest", "no_entry_digest", "manifest_digest"}


def canonical_decimal_string(value: Decimal) -> str:
    if not isinstance(value, Decimal):
        raise TypeError("value must be a Decimal")
    if value.is_nan() or value.is_infinite():
        raise ValueError("NaN and Infinity are not allowed in canonical identity payloads")
    normalized = value.normalize()
    text = format(normalized, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text == "-0":
        text = "0"
    return text


def normalize_timestamp_utc(value: datetime) -> str:
    if not isinstance(value, datetime):
        raise TypeError("value must be a datetime")
    if value.tzinfo is None:
        raise ValueError("Timestamps must be timezone-aware UTC")
    utc_value = value.astimezone(timezone.utc)
    return utc_value.strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize_identity_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return canonical_decimal_string(value)
    if isinstance(value, datetime):
        return normalize_timestamp_utc(value)
    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
            raise ValueError("NaN and Infinity are not allowed in canonical identity payloads")
        return value
    if isinstance(value, dict):
        return {str(key): _normalize_identity_value(val) for key, val in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalize_identity_value(item) for item in value]
    if isinstance(value, set):
        return [_normalize_identity_value(item) for item in sorted(value, key=lambda item: str(item))]
    raise TypeError(f"Unsupported type in identity payload: {type(value)!r}")


def canonical_json(value: Any) -> str:
    normalized = _normalize_identity_value(value)
    return json.dumps(normalized, separators=(",", ":"), sort_keys=True, ensure_ascii=False)


def _require_required_payload_fields(payload: dict[str, Any], required: set[str]) -> None:
    for field_name in sorted(required):
        if field_name not in payload:
            raise ValueError(f"Missing required identity field: {field_name}")
        value = payload[field_name]
        if isinstance(value, str) and not value.strip():
            raise ValueError(f"Missing required identity field: {field_name}")
        if value is None:
            raise ValueError(f"Missing required identity field: {field_name}")


def compute_identity_digest(payload: dict[str, Any], *, domain: str, exclude_keys: set[str] | None = None) -> str:
    if exclude_keys is None:
        exclude_keys = set()
    if not isinstance(payload, dict):
        raise TypeError("payload must be a dict")
    if not domain or not str(domain).strip():
        raise ValueError("domain must be a non-empty string")
    if not payload:
        raise ValueError("payload must not be empty")
    serialized = {key: value for key, value in payload.items() if key not in exclude_keys and key not in _DIGEST_KEYS}
    if not serialized:
        raise ValueError("payload must contain at least one non-digest field")
    canonical = canonical_json(serialized)
    return hashlib.sha256(f"{domain}\x1f{canonical}".encode("utf-8")).hexdigest()


def hash_identity_payload(payload: dict[str, Any], *, domain: str, exclude_keys: set[str] | None = None) -> str:
    return compute_identity_digest(payload, domain=domain, exclude_keys=exclude_keys)


def build_fold_input_digest(
    *,
    plan_id: str,
    fold_id: str,
    interval: Any,
    dataset_identity: str,
    feature_spec_set_hash: str,
    calendar_identity: str,
    evidence: Any,
    model_policy_identity: str,
    formation_policy_identity: str,
    family_policy_identity: str,
) -> str:
    if not plan_id or not str(plan_id).strip():
        raise ValueError("plan_id is required")
    if not fold_id or not str(fold_id).strip():
        raise ValueError("fold_id is required")
    if not dataset_identity or not str(dataset_identity).strip():
        raise ValueError("dataset_identity is required")
    if not feature_spec_set_hash or not str(feature_spec_set_hash).strip():
        raise ValueError("feature_spec_set_hash is required")
    if not calendar_identity or not str(calendar_identity).strip():
        raise ValueError("calendar_identity is required")
    if not model_policy_identity or not str(model_policy_identity).strip():
        raise ValueError("model_policy_identity is required")
    if not formation_policy_identity or not str(formation_policy_identity).strip():
        raise ValueError("formation_policy_identity is required")
    if not family_policy_identity or not str(family_policy_identity).strip():
        raise ValueError("family_policy_identity is required")
    payload = {
        "plan_id": plan_id,
        "fold_id": fold_id,
        "interval": {
            "train_start": interval.train_start,
            "train_end": interval.train_end,
            "validation_start": interval.validation_start,
            "validation_end": interval.validation_end,
            "test_start": interval.test_start,
            "test_end": interval.test_end,
            "holdout_start": interval.holdout_start,
            "holdout_end": interval.holdout_end,
        },
        "dataset_identity": dataset_identity,
        "feature_spec_set_hash": feature_spec_set_hash,
        "calendar_identity": calendar_identity,
        "evidence": {
            "family_provenance_digest": evidence.family_provenance_digest,
            "calendar_provenance_digest": evidence.calendar_provenance_digest,
            "model_provenance_digest": evidence.model_provenance_digest,
            "validation_digest": evidence.validation_digest,
            "evidence_digest": evidence.evidence_digest,
            "provenance_state": evidence.provenance_state,
            "family_provenance_state": evidence.family_provenance_state,
        },
        "model_policy_identity": model_policy_identity,
        "formation_policy_identity": formation_policy_identity,
        "family_policy_identity": family_policy_identity,
    }
    return compute_identity_digest(payload, domain="regimequant.walkforward.fold_input")


def build_experiment_result_digest(
    *,
    experiment_id: str,
    plan_id: str,
    fold_input_identities: tuple[str, ...],
    fold_output_identities: tuple[str, ...],
    scientific_status: str,
    accounting_status: str,
    metric_status: str,
) -> str:
    if not experiment_id or not str(experiment_id).strip():
        raise ValueError("experiment_id is required")
    if not plan_id or not str(plan_id).strip():
        raise ValueError("plan_id is required")
    if not isinstance(fold_input_identities, tuple) or not fold_input_identities:
        raise ValueError("fold_input_identities must be a non-empty tuple")
    if not isinstance(fold_output_identities, tuple) or not fold_output_identities:
        raise ValueError("fold_output_identities must be a non-empty tuple")
    payload = {
        "experiment_id": experiment_id,
        "plan_id": plan_id,
        "fold_input_identities": list(fold_input_identities),
        "fold_output_identities": list(fold_output_identities),
        "scientific_status": scientific_status,
        "accounting_status": accounting_status,
        "metric_status": metric_status,
    }
    return compute_identity_digest(payload, domain="regimequant.walkforward.experiment_result")


def build_no_entry_digest(
    *,
    experiment_id: str,
    fold_id: str,
    fold_input_identity: str,
    decision_session: int,
    boundary_b: int,
    reason_code: str,
    evidence_dependency_identities: tuple[str, ...],
    invoked_m9: bool,
) -> str:
    if not experiment_id or not str(experiment_id).strip():
        raise ValueError("experiment_id is required")
    if not fold_id or not str(fold_id).strip():
        raise ValueError("fold_id is required")
    if not fold_input_identity or not str(fold_input_identity).strip():
        raise ValueError("fold_input_identity is required")
    if not reason_code or not str(reason_code).strip():
        raise ValueError("reason_code is required")
    if invoked_m9 is not False:
        raise ValueError("No-entry identities must set invoked_m9=False")
    if not isinstance(evidence_dependency_identities, tuple) or not evidence_dependency_identities:
        raise ValueError("evidence_dependency_identities must be a non-empty tuple")
    payload = {
        "experiment_id": experiment_id,
        "fold_id": fold_id,
        "fold_input_identity": fold_input_identity,
        "decision_session": int(decision_session),
        "boundary_b": int(boundary_b),
        "reason_code": reason_code,
        "evidence_dependency_identities": list(evidence_dependency_identities),
        "invoked_m9": False,
    }
    return compute_identity_digest(payload, domain="regimequant.walkforward.no_entry")


def _require_not_missing(value: Any, label: str) -> None:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"Missing required identity field: {label}")


def _schema_value_for(value: Any, label: str) -> str:
    if value is None:
        raise ValueError(f"Missing required identity field: {label}")
    return str(value)

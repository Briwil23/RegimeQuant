from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from regimequant.cointegration.contracts import DecisionEnvelope


class ValidationError(ValueError):
    """Raised when production and replay evidence violate the fail-closed contract."""


def _as_utc(value: object, *, field_name: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None or ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def validate_causal_availability(
    *,
    cutoff: object,
    price_series: pd.Series,
    availability_evidence: list[dict[str, object]] | tuple[dict[str, object], ...] | None = None,
    identity_evidence: list[dict[str, object]] | tuple[dict[str, object], ...] | None = None,
    alias_records: list[dict[str, object]] | tuple[dict[str, object], ...] | None = None,
    continuity_records: list[dict[str, object]] | tuple[dict[str, object], ...] | None = None,
) -> dict[str, object]:
    if not isinstance(price_series, pd.Series):
        raise ValidationError("price_series must be a pandas Series")
    cutoff_utc = _as_utc(cutoff, field_name="cutoff")

    required = availability_evidence or ()
    for timestamp, _value in price_series.items():
        observation_ts = _as_utc(timestamp, field_name="observation timestamp")
        if observation_ts > cutoff_utc:
            raise ValidationError("observation timestamp exceeds cutoff")

        match = next((record for record in required if _as_utc(record.get("timestamp", observation_ts), field_name="availability timestamp") == observation_ts), None)
        if match is None:
            raise ValidationError(f"missing availability evidence for {observation_ts}")
        available_at = _as_utc(match.get("available_at", observation_ts), field_name="available_at")
        if available_at > cutoff_utc:
            raise ValidationError("availability evidence is later than cutoff")

        identity_verified_at = match.get("identity_verified_at")
        if identity_verified_at is not None:
            verified_at = _as_utc(identity_verified_at, field_name="identity_verified_at")
            if verified_at > cutoff_utc:
                raise ValidationError("identity verification is later than cutoff")

        if alias_records:
            mapping_valid = False
            for alias in alias_records:
                valid_from = _as_utc(alias.get("valid_from"), field_name="valid_from")
                valid_to = None if alias.get("valid_to") is None else _as_utc(alias.get("valid_to"), field_name="valid_to")
                if valid_from <= observation_ts and (valid_to is None or observation_ts <= valid_to):
                    mapping_valid = True
                    break
            if not mapping_valid:
                raise ValidationError("alias mapping is not valid at observation")

        if continuity_records:
            valid_continuity = False
            for continuity in continuity_records:
                effective_at = _as_utc(continuity.get("effective_at"), field_name="effective_at")
                verified_at = _as_utc(continuity.get("verified_at"), field_name="verified_at")
                if effective_at <= observation_ts and verified_at <= cutoff_utc:
                    valid_continuity = True
                    break
            if not valid_continuity:
                raise ValidationError("corporate-action continuity is not valid for observation")

    if identity_evidence:
        for record in identity_evidence:
            verified_at = _as_utc(record.get("verified_at"), field_name="identity evidence verified_at")
            if verified_at > cutoff_utc:
                raise ValidationError("identity evidence is later than cutoff")

    return {
        "cutoff": cutoff_utc,
        "availability_verified": True,
        "price_count": len(price_series),
    }


def validate_decision_evidence(envelope: DecisionEnvelope) -> dict[str, object]:
    if envelope.mode == "production":
        if not envelope.availability_verified:
            raise ValidationError("production decisions require verified availability")
        if "fixture" in envelope.source_digest.lower() or "replay" in envelope.source_digest.lower():
            raise ValidationError("production must reject fixture-only or replay-only evidence")
        if envelope.replay_reference is not None:
            raise ValidationError("production decisions must not carry replay-only provenance")
    elif envelope.mode == "replay":
        if not envelope.availability_verified:
            raise ValidationError("replay decisions require verified availability")
        if "production" in envelope.source_digest.lower() and envelope.replay_reference is None:
            raise ValidationError("replay evidence cannot claim production-only origin")
    elif envelope.mode == "evaluation":
        if not envelope.availability_verified:
            raise ValidationError("evaluation evidence requires verified availability")

    return {
        "mode": envelope.mode,
        "cutoff": envelope.cutoff,
        "availability_verified": envelope.availability_verified,
        "source_digest": envelope.source_digest,
        "evidence_digest": envelope.evidence_digest,
        "decision_digest": envelope.decision_digest,
    }

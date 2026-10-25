from __future__ import annotations

import pytest
import pandas as pd

from regimequant.cointegration.validation import DecisionEnvelope, ValidationError, validate_causal_availability, validate_decision_evidence


def test_validate_decision_evidence_separates_production_and_replay() -> None:
    cutoff = pd.Timestamp("2024-01-02", tz="UTC")
    env = DecisionEnvelope(
        mode="production",
        cutoff=cutoff,
        availability_verified=True,
        source_digest="prod-digest",
        evidence_digest="evidence-digest",
        decision_digest="decision-digest",
    )
    result = validate_decision_evidence(env)
    assert result["mode"] == "production"

    replay_env = DecisionEnvelope(
        mode="replay",
        cutoff=cutoff,
        availability_verified=True,
        source_digest="replay-digest",
        evidence_digest="evidence-digest",
        decision_digest="decision-digest",
    )
    result = validate_decision_evidence(replay_env)
    assert result["mode"] == "replay"

    try:
        validate_decision_evidence(DecisionEnvelope(mode="production", cutoff=cutoff, availability_verified=False, source_digest="prod-digest", evidence_digest="evidence-digest", decision_digest="decision-digest"))
    except ValidationError:
        return
    raise AssertionError("production evidence must require verified availability")


def test_validate_causal_availability_rejects_after_cutoff_and_accepts_exact_cutoff() -> None:
    cutoff = pd.Timestamp("2024-01-05", tz="UTC")
    valid_series = pd.Series(
        [100.0, 101.0, 102.0],
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"], utc=True),
    )
    evidence = [
        {"timestamp": ts, "available_at": ts, "identity_verified_at": ts}
        for ts in valid_series.index
    ]
    result = validate_causal_availability(cutoff=cutoff, price_series=valid_series, availability_evidence=evidence)
    assert result["availability_verified"] is True

    after_cutoff = pd.Series([100.0], index=[pd.Timestamp("2024-01-06", tz="UTC")])
    with pytest.raises(ValidationError, match="observation timestamp exceeds cutoff"):
        validate_causal_availability(
            cutoff=cutoff,
            price_series=after_cutoff,
            availability_evidence=[{"timestamp": after_cutoff.index[0], "available_at": after_cutoff.index[0], "identity_verified_at": after_cutoff.index[0]}],
        )

    exact_cutoff = pd.Series([100.0], index=[pd.Timestamp("2024-01-05", tz="UTC")])
    exact_evidence = [{"timestamp": exact_cutoff.index[0], "available_at": exact_cutoff.index[0], "identity_verified_at": exact_cutoff.index[0]}]
    assert validate_causal_availability(cutoff=cutoff, price_series=exact_cutoff, availability_evidence=exact_evidence)["availability_verified"] is True


def test_validate_causal_availability_rejects_invalid_mapping_and_continuity() -> None:
    cutoff = pd.Timestamp("2024-01-10", tz="UTC")
    series = pd.Series([100.0], index=[pd.Timestamp("2024-01-02", tz="UTC")])
    availability = [{"timestamp": series.index[0], "available_at": series.index[0], "identity_verified_at": series.index[0]}]

    with pytest.raises(ValidationError, match="alias mapping is not valid at observation"):
        validate_causal_availability(
            cutoff=cutoff,
            price_series=series,
            availability_evidence=availability,
            alias_records=[{"canonical_security_id": "sec-1", "symbol": "AAPL", "listing_venue": "NASDAQ", "valid_from": "2024-01-03", "valid_to": "2024-01-04"}],
        )

    with pytest.raises(ValidationError, match="corporate-action continuity is not valid for observation"):
        validate_causal_availability(
            cutoff=cutoff,
            price_series=series,
            availability_evidence=availability,
            continuity_records=[{"predecessor_security_id": "sec-0", "successor_security_id": "sec-1", "event_kind": "listing", "effective_at": "2024-01-03", "verified_at": "2024-01-04", "source_authority": "exchange"}],
        )


def test_validate_decision_evidence_rejects_fixture_and_replay_provenance() -> None:
    cutoff = pd.Timestamp("2024-01-02", tz="UTC")
    base = dict(mode="production", cutoff=cutoff, availability_verified=True, source_digest="prod-digest", evidence_digest="evidence-digest", decision_digest="decision-digest")

    with pytest.raises(ValidationError, match="fixture-only|replay-only"):
        validate_decision_evidence(DecisionEnvelope(**{**base, "source_digest": "fixture-digest"}))

    with pytest.raises(ValidationError, match="production decisions must not carry replay-only provenance"):
        validate_decision_evidence(DecisionEnvelope(**{**base, "replay_reference": "replay-1"}))

    with pytest.raises(ValidationError, match="replay evidence cannot claim production-only origin"):
        validate_decision_evidence(DecisionEnvelope(mode="replay", cutoff=cutoff, availability_verified=True, source_digest="production-digest", evidence_digest="evidence-digest", decision_digest="decision-digest"))

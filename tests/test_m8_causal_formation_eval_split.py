from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regimequant.cointegration.contracts import CointegrationCandidate, DecisionEnvelope
from regimequant.cointegration.identity import compute_canonical_pair_id
from regimequant.signals.m8_calendar import TradingCalendarEvidence
from regimequant.signals.m8_contracts import M8AbstainReason, M8DiscoveryFamilyEvidence, M8FamilyMember, M8SignalEnum
from regimequant.signals.m8_formation import M8FormationError, fit_formation_model
from regimequant.signals.m8_identity import compute_family_digest
from regimequant.signals.m8_signal import generate_stateless_signal


def test_formation_must_strictly_precede_evaluation() -> None:
    index = pd.date_range("2024-01-02", periods=300, freq="B", tz="UTC")
    x = pd.Series(np.linspace(100.0, 200.0, len(index)), index=index, name="MSFT")
    y = pd.Series(5.0 + 1.2 * x.to_numpy(), index=index, name="AAPL")

    with pytest.raises(M8FormationError):
        fit_formation_model(
            x,
            y,
            formation_start=index[0],
            formation_end=index[-1],
            eval_observation_time=index[-1],
        )


def _family(mode: str = "production", *, discovery_observation_session: pd.Timestamp | None = None) -> M8DiscoveryFamilyEvidence:
    canonical = compute_canonical_pair_id("AAPL", "MSFT")
    member = M8FamilyMember("h1", canonical, "AAPL", "MSFT", True, raw_p_value=0.001)
    if discovery_observation_session is None:
        discovery_observation_session = pd.Timestamp("2024-01-05", tz="UTC")
    discovery_window_start = discovery_observation_session - pd.Timedelta(days=3)
    discovery_window_end = discovery_observation_session + pd.Timedelta(days=3)
    discovery_decision_cutoff = discovery_observation_session + pd.Timedelta(days=1)
    payload = {
        "family_definition_version": "v1",
        "family_id": "fam-1",
        "universe_snapshot_identity": "u1",
        "discovery_window_start": str(discovery_window_start),
        "discovery_window_end": str(discovery_window_end),
        "discovery_observation_session": str(discovery_observation_session),
        "discovery_decision_cutoff": str(discovery_decision_cutoff),
        "dataset_identity": "d1",
        "statistical_config_digest": "s1",
        "exchange_calendar_id": "XNYS",
        "exchange_calendar_version": "1",
        "exchange_session_sequence_digest": "c1",
        "provenance_mode": mode,
        "members": [
            (
                member.hypothesis_id,
                member.canonical_pair_id,
                member.valid_for_discovery,
                member.raw_p_value,
                member.invalid_reason,
            )
        ],
    }
    return M8DiscoveryFamilyEvidence(
        family_definition_version="v1",
        family_id="fam-1",
        universe_snapshot_identity="u1",
        discovery_window_start=discovery_window_start,
        discovery_window_end=discovery_window_end,
        discovery_observation_session=discovery_observation_session,
        discovery_available_at=discovery_decision_cutoff,
        discovery_decision_cutoff=discovery_decision_cutoff,
        dataset_identity="d1",
        statistical_config_digest="s1",
        declared_membership_ordered=(member,),
        supplied_by_adjusted_values=(0.001,),
        predeclared_committed_at=discovery_observation_session - pd.Timedelta(days=1),
        outcomes_available_at=discovery_decision_cutoff,
        exchange_calendar_id="XNYS",
        exchange_calendar_version="1",
        exchange_session_sequence_digest="c1",
        provenance_mode=mode,
        availability_verified=True,
        identity_verified=True,
        continuity_verified=True,
        evidence_provenance_digest="p1",
        family_digest=compute_family_digest(payload),
    )


def test_late_eval_availability_abstains_before_signal_emission() -> None:
    index = pd.date_range("2024-01-02", periods=300, freq="B", tz="UTC")
    x = pd.Series(np.linspace(100.0, 200.0, len(index)), index=index, name="MSFT")
    y = pd.Series(5.0 + 1.2 * x.to_numpy(), index=index, name="AAPL")
    candidate = CointegrationCandidate("AAPL", "MSFT", "pair_AAPL_MSFT", source_mode="production")
    cutoff = pd.Timestamp("2025-03-03", tz="UTC")
    envelope = DecisionEnvelope("production", cutoff, True, "prod", "e", "d")
    calendar = TradingCalendarEvidence("XNYS", "1", tuple(index), "cd")

    observation = generate_stateless_signal(
        candidate=candidate,
        family_evidence=_family(discovery_observation_session=index[252]),
        decision_envelope=envelope,
        calendar=calendar,
        revocations=(),
        x_series=x,
        y_series=y,
        formation_start=index[0],
        formation_end=index[252],
        formation_available_at=index[252],
        eval_observation_time=index[253],
        eval_available_at=cutoff + pd.Timedelta(days=1),
        eval_x=float(x.iloc[253]),
        eval_y=float(y.iloc[253]),
    )
    assert observation.signal == M8SignalEnum.ABSTAIN
    assert observation.abstain_reason == M8AbstainReason.PREDECLARATION_UNPROVEN


def test_late_formation_correction_abstains_before_fit() -> None:
    index = pd.date_range("2024-01-02", periods=300, freq="B", tz="UTC")
    x = pd.Series(np.linspace(100.0, 200.0, len(index)), index=index, name="MSFT")
    y = pd.Series(5.0 + 1.2 * x.to_numpy(), index=index, name="AAPL")
    candidate = CointegrationCandidate("AAPL", "MSFT", "pair_AAPL_MSFT", source_mode="production")
    cutoff = pd.Timestamp("2025-03-03", tz="UTC")
    envelope = DecisionEnvelope("production", cutoff, True, "prod", "e", "d")
    calendar = TradingCalendarEvidence("XNYS", "1", tuple(index), "cd")

    observation = generate_stateless_signal(
        candidate=candidate,
        family_evidence=_family(discovery_observation_session=index[252]),
        decision_envelope=envelope,
        calendar=calendar,
        revocations=(),
        x_series=x,
        y_series=y,
        formation_start=index[0],
        formation_end=index[252],
        formation_available_at=index[253],
        eval_observation_time=index[253],
        eval_available_at=index[253],
        eval_x=float(x.iloc[253]),
        eval_y=float(y.iloc[253]),
    )
    assert observation.signal == M8SignalEnum.ABSTAIN
    assert observation.abstain_reason == M8AbstainReason.PREDECLARATION_UNPROVEN


def test_forged_production_envelope_abstains_fail_closed() -> None:
    index = pd.date_range("2024-01-02", periods=300, freq="B", tz="UTC")
    x = pd.Series(np.linspace(100.0, 200.0, len(index)), index=index, name="MSFT")
    y = pd.Series(5.0 + 1.2 * x.to_numpy(), index=index, name="AAPL")
    candidate = CointegrationCandidate("AAPL", "MSFT", "pair_AAPL_MSFT", source_mode="production")
    cutoff = pd.Timestamp("2025-03-03", tz="UTC")
    envelope = DecisionEnvelope("production", cutoff, True, "forged_prod", "forged_evidence", "forged_decision")
    calendar = TradingCalendarEvidence("XNYS", "1", tuple(index), "cd")

    observation = generate_stateless_signal(
        candidate=candidate,
        family_evidence=_family("production", discovery_observation_session=index[252]),
        decision_envelope=envelope,
        calendar=calendar,
        revocations=(),
        x_series=x,
        y_series=y,
        formation_start=index[0],
        formation_end=index[252],
        formation_available_at=index[252],
        eval_observation_time=index[253],
        eval_available_at=index[253],
        eval_x=float(x.iloc[253]),
        eval_y=float(y.iloc[253]),
    )
    assert observation.signal == M8SignalEnum.ABSTAIN
    assert observation.abstain_reason == M8AbstainReason.PREDECLARATION_UNPROVEN


def test_fabricated_calendar_sessions_cannot_authorize_production() -> None:
    index = pd.date_range("2024-01-02", periods=300, freq="B", tz="UTC")
    x = pd.Series(np.linspace(100.0, 200.0, len(index)), index=index, name="MSFT")
    y = pd.Series(5.0 + 1.2 * x.to_numpy(), index=index, name="AAPL")
    candidate = CointegrationCandidate("AAPL", "MSFT", "pair_AAPL_MSFT", source_mode="production")
    cutoff = pd.Timestamp("2025-03-03", tz="UTC")
    envelope = DecisionEnvelope("production", cutoff, True, "prod", "e", "d")
    fabricated_sessions = tuple(pd.date_range("2030-01-01", periods=300, freq="B", tz="UTC"))
    fabricated_calendar = TradingCalendarEvidence("XNYS", "1", fabricated_sessions, "fake_cd")

    observation = generate_stateless_signal(
        candidate=candidate,
        family_evidence=_family("production", discovery_observation_session=index[252]),
        decision_envelope=envelope,
        calendar=fabricated_calendar,
        revocations=(),
        x_series=x,
        y_series=y,
        formation_start=index[0],
        formation_end=index[252],
        formation_available_at=index[252],
        eval_observation_time=index[253],
        eval_available_at=index[253],
        eval_x=float(x.iloc[253]),
        eval_y=float(y.iloc[253]),
    )
    assert observation.signal == M8SignalEnum.ABSTAIN
    assert observation.abstain_reason == M8AbstainReason.PREDECLARATION_UNPROVEN


def test_valid_looking_but_unsupported_digests_cannot_authorize_production() -> None:
    index = pd.date_range("2024-01-02", periods=300, freq="B", tz="UTC")
    x = pd.Series(np.linspace(100.0, 200.0, len(index)), index=index, name="MSFT")
    y = pd.Series(5.0 + 1.2 * x.to_numpy(), index=index, name="AAPL")
    candidate = CointegrationCandidate("AAPL", "MSFT", "pair_AAPL_MSFT", source_mode="production")
    cutoff = pd.Timestamp("2025-03-03", tz="UTC")
    envelope = DecisionEnvelope(
        "production",
        cutoff,
        True,
        "m7_prod_sha256_00000000000000000000000000000000",
        "m7_evidence_sha256_11111111111111111111111111111111",
        "m7_decision_sha256_22222222222222222222222222222222",
    )
    calendar = TradingCalendarEvidence("XNYS", "1", tuple(index), "calendar_digest_sha256_3333333333333333")

    observation = generate_stateless_signal(
        candidate=candidate,
        family_evidence=_family("production", discovery_observation_session=index[252]),
        decision_envelope=envelope,
        calendar=calendar,
        revocations=(),
        x_series=x,
        y_series=y,
        formation_start=index[0],
        formation_end=index[252],
        formation_available_at=index[252],
        eval_observation_time=index[253],
        eval_available_at=index[253],
        eval_x=float(x.iloc[253]),
        eval_y=float(y.iloc[253]),
    )
    assert observation.signal == M8SignalEnum.ABSTAIN
    assert observation.abstain_reason == M8AbstainReason.PREDECLARATION_UNPROVEN


def test_replay_path_remains_usable_under_verified_internal_consistency() -> None:
    index = pd.date_range("2024-01-02", periods=300, freq="B", tz="UTC")
    x = pd.Series(np.linspace(100.0, 200.0, len(index)), index=index, name="MSFT")
    trend = 5.0 + 1.2 * x.to_numpy()
    noise = 0.5 * np.sin(np.linspace(0.0, 12.0, len(index)))
    y = pd.Series(trend + noise, index=index, name="AAPL")
    candidate = CointegrationCandidate("AAPL", "MSFT", "pair_AAPL_MSFT", source_mode="replay")
    cutoff = pd.Timestamp("2025-03-03", tz="UTC")
    envelope = DecisionEnvelope("replay", cutoff, True, "replay_source", "replay_evidence", "replay_decision", replay_reference="r1")
    calendar = TradingCalendarEvidence("XNYS", "1", tuple(index), "cd")

    observation = generate_stateless_signal(
        candidate=candidate,
        family_evidence=_family("replay", discovery_observation_session=index[252]),
        decision_envelope=envelope,
        calendar=calendar,
        revocations=(),
        x_series=x,
        y_series=y,
        formation_start=index[0],
        formation_end=index[252],
        formation_available_at=index[252],
        eval_observation_time=index[253],
        eval_available_at=index[253],
        eval_x=float(x.iloc[253]),
        eval_y=float(y.iloc[253]),
    )
    assert observation.signal in {
        M8SignalEnum.SHORT_SPREAD_CANDIDATE,
        M8SignalEnum.LONG_SPREAD_CANDIDATE,
        M8SignalEnum.NEUTRAL,
    }
    assert observation.abstain_reason is None

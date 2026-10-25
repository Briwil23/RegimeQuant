from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pandas as pd

from regimequant.backtesting.m9_engine import CorporateActionEvent, PairSignalInput, ResearchInput, SecurityObservation, SessionPrice, run_research_backtest
from regimequant.signals.m8_contracts import M8SignalEnum, M8SignalObservation


def _signal(pair: str, signal: M8SignalEnum, sid: str) -> M8SignalObservation:
    ts = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    return M8SignalObservation(
        canonical_pair_id=pair,
        eval_observation_time=pd.Timestamp(ts),
        eval_available_at=pd.Timestamp(ts),
        m8_decision_cutoff=pd.Timestamp(ts),
        signal=signal,
        z_score=2.5,
        abstain_reason=None,
        source_mode="replay",
        scientific_identity=sid,
        evidence_digest="ev",
        decision_digest="dec",
    )


def _base_sessions(start: int = 10) -> list[SessionPrice]:
    ts = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    return [SessionPrice(idx, Decimal("100") + Decimal(idx), Decimal("50") + Decimal(idx), ts) for idx in range(start, start + 12)]


def _pair_input(pair_id: str, y: str, x: str, sig: M8SignalEnum, beta: str, sid: str) -> PairSignalInput:
    ts = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    return PairSignalInput(
        signal_observation=_signal(pair_id, sig, sid),
        beta=Decimal(beta),
        y_security_id=y,
        x_security_id=x,
        pair_id=pair_id,
        authorization_time=ts,
        submission_time=ts,
        signal_identity=sid,
    )


def _canonical_observations(
    sessions: list[SessionPrice],
    quotes_by_security: dict[str, list[Decimal]],
    *,
    base: datetime,
) -> list[SecurityObservation]:
    observations: list[SecurityObservation] = []
    for idx, session in enumerate(sessions):
        for security_id in sorted(quotes_by_security):
            observations.append(
                SecurityObservation(
                    session.session_index,
                    security_id,
                    quotes_by_security[security_id][idx],
                    base,
                    base,
                    f"{security_id}:{session.session_index}",
                )
            )
    return observations


def test_two_pairs_same_session_and_overlap_security() -> None:
    sessions = _base_sessions()
    obs = _canonical_observations(
        sessions,
        {
            "A": [Decimal("110") + Decimal(i) for i in range(12)],
            "B": [Decimal("60") + Decimal(i) for i in range(12)],
            "C": [Decimal("90") + Decimal(i) for i in range(12)],
        },
        base=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
    )
    inp = ResearchInput(
        signal_observation=_signal("pair_A_B", M8SignalEnum.LONG_SPREAD_CANDIDATE, "s0"),
        beta=Decimal("1.0"),
        y_security_id="A",
        x_security_id="B",
        pair_id="pair_A_B",
        sessions=sessions,
        security_observations=obs,
        authorization_time=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
        submission_time=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
        pair_signals=[
            _pair_input("pair_A_B", "A", "B", M8SignalEnum.LONG_SPREAD_CANDIDATE, "1.0", "s1"),
            _pair_input("pair_A_C", "A", "C", M8SignalEnum.SHORT_SPREAD_CANDIDATE, "0.8", "s2"),
        ],
    )
    result = run_research_backtest(inp)
    assert result.report["research_only"] is True
    assert len(result.fills) >= 2


def test_combined_gross_cap_breach_rejects_pair() -> None:
    sessions = _base_sessions(50)
    obs = _canonical_observations(
        sessions,
        {
            "X": [Decimal("500") for _ in range(12)],
            "Y": [Decimal("500") for _ in range(12)],
            "Z": [Decimal("500") for _ in range(12)],
            "W": [Decimal("500") for _ in range(12)],
            "V": [Decimal("500") for _ in range(12)],
            "U": [Decimal("500") for _ in range(12)],
            "T": [Decimal("500") for _ in range(12)],
            "S": [Decimal("500") for _ in range(12)],
            "R": [Decimal("500") for _ in range(12)],
            "Q": [Decimal("500") for _ in range(12)],
                "P": [Decimal("500") for _ in range(12)],
                "O": [Decimal("500") for _ in range(12)],
        },
        base=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
    )
    inp = ResearchInput(
        signal_observation=_signal("pair_X_Y", M8SignalEnum.LONG_SPREAD_CANDIDATE, "sx"),
        beta=Decimal("1.0"),
        y_security_id="X",
        x_security_id="Y",
        pair_id="pair_X_Y",
        sessions=sessions,
        security_observations=obs,
        authorization_time=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
        submission_time=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
        initial_equity=Decimal("100000"),
        pair_signals=[
            _pair_input("pair_X_Y", "X", "Y", M8SignalEnum.LONG_SPREAD_CANDIDATE, "1.0", "sx1"),
            _pair_input("pair_X_Z", "X", "Z", M8SignalEnum.LONG_SPREAD_CANDIDATE, "1.0", "sx2"),
            _pair_input("pair_X_W", "X", "W", M8SignalEnum.LONG_SPREAD_CANDIDATE, "1.0", "sx3"),
            _pair_input("pair_X_V", "X", "V", M8SignalEnum.LONG_SPREAD_CANDIDATE, "1.0", "sx4"),
            _pair_input("pair_X_U", "X", "U", M8SignalEnum.LONG_SPREAD_CANDIDATE, "1.0", "sx5"),
            _pair_input("pair_X_T", "X", "T", M8SignalEnum.LONG_SPREAD_CANDIDATE, "1.0", "sx6"),
            _pair_input("pair_X_S", "X", "S", M8SignalEnum.LONG_SPREAD_CANDIDATE, "1.0", "sx7"),
            _pair_input("pair_X_R", "X", "R", M8SignalEnum.LONG_SPREAD_CANDIDATE, "1.0", "sx8"),
            _pair_input("pair_X_Q", "X", "Q", M8SignalEnum.LONG_SPREAD_CANDIDATE, "1.0", "sx9"),
                _pair_input("pair_X_P", "X", "P", M8SignalEnum.LONG_SPREAD_CANDIDATE, "1.0", "sx10"),
                _pair_input("pair_X_O", "X", "O", M8SignalEnum.LONG_SPREAD_CANDIDATE, "1.0", "sx11"),
        ],
    )
    result = run_research_backtest(inp)
    statuses = [event for event in result.events if event.get("event") == "entry_fill"]
    assert any(str(item.get("status")) != "FILLED" for item in statuses)


def test_forced_exit_retry_exhaustion_marks_uncertifiable() -> None:
    ts = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    sessions = [
        SessionPrice(10, Decimal("100"), Decimal("50"), ts),
        SessionPrice(11, Decimal("101"), Decimal("49"), ts),
        SessionPrice(12, Decimal("102"), Decimal("48"), ts),
        SessionPrice(13, Decimal("103"), Decimal("47"), ts),
        SessionPrice(14, Decimal("104"), Decimal("46"), ts),
        SessionPrice(15, Decimal("105"), Decimal("45"), ts),
        SessionPrice(16, None, None, ts),
        SessionPrice(17, None, None, ts),
        SessionPrice(18, None, None, ts),
        SessionPrice(19, None, None, ts),
    ]
    inp = ResearchInput(
        signal_observation=_signal("pair_A_B", M8SignalEnum.LONG_SPREAD_CANDIDATE, "f1"),
        beta=Decimal("1.0"),
        y_security_id="A",
        x_security_id="B",
        pair_id="pair_A_B",
        sessions=sessions,
        authorization_time=ts,
        submission_time=ts,
        forced_exit_retry_limit=2,
        pair_signals=[_pair_input("pair_A_B", "A", "B", M8SignalEnum.LONG_SPREAD_CANDIDATE, "1.0", "f1")],
    )
    result = run_research_backtest(inp)
    assert result.report["certified"] is False


def test_corporate_actions_split_and_dividend() -> None:
    sessions = _base_sessions(100)
    ts = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    actions = [
        CorporateActionEvent(session_index=103, security_id="A", action_type="split", value=Decimal("2"), evidence_complete=True),
        CorporateActionEvent(session_index=104, security_id="A", action_type="cash_dividend", value=Decimal("1.5"), evidence_complete=True),
    ]
    inp = ResearchInput(
        signal_observation=_signal("pair_A_B", M8SignalEnum.LONG_SPREAD_CANDIDATE, "c1"),
        beta=Decimal("1.0"),
        y_security_id="A",
        x_security_id="B",
        pair_id="pair_A_B",
        sessions=sessions,
        authorization_time=ts,
        submission_time=ts,
        pair_signals=[_pair_input("pair_A_B", "A", "B", M8SignalEnum.LONG_SPREAD_CANDIDATE, "1.0", "c1")],
        corporate_actions=actions,
    )
    result = run_research_backtest(inp)
    assert any(event.get("event") == "corp_action_split" for event in result.events)
    assert any(event.get("event") == "corp_action_dividend" for event in result.events)


def test_incomplete_corporate_action_evidence_uncertifiable() -> None:
    sessions = _base_sessions(120)
    ts = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    actions = [
        CorporateActionEvent(session_index=123, security_id="A", action_type="split", value=Decimal("2"), evidence_complete=False),
    ]
    inp = ResearchInput(
        signal_observation=_signal("pair_A_B", M8SignalEnum.LONG_SPREAD_CANDIDATE, "c2"),
        beta=Decimal("1.0"),
        y_security_id="A",
        x_security_id="B",
        pair_id="pair_A_B",
        sessions=sessions,
        authorization_time=ts,
        submission_time=ts,
        pair_signals=[_pair_input("pair_A_B", "A", "B", M8SignalEnum.LONG_SPREAD_CANDIDATE, "1.0", "c2")],
        corporate_actions=actions,
    )
    result = run_research_backtest(inp)
    assert result.report["certified"] is False


def test_identity_mutation_sensitivity() -> None:
    sessions = _base_sessions(140)
    ts = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    inp1 = ResearchInput(
        signal_observation=_signal("pair_A_B", M8SignalEnum.LONG_SPREAD_CANDIDATE, "m1"),
        beta=Decimal("1.0"),
        y_security_id="A",
        x_security_id="B",
        pair_id="pair_A_B",
        sessions=sessions,
        authorization_time=ts,
        submission_time=ts,
        pair_signals=[_pair_input("pair_A_B", "A", "B", M8SignalEnum.LONG_SPREAD_CANDIDATE, "1.0", "m1")],
    )
    inp2 = ResearchInput(
        signal_observation=_signal("pair_A_B", M8SignalEnum.LONG_SPREAD_CANDIDATE, "m2"),
        beta=Decimal("1.0"),
        y_security_id="A",
        x_security_id="B",
        pair_id="pair_A_B",
        sessions=sessions,
        authorization_time=ts,
        submission_time=ts,
        pair_signals=[_pair_input("pair_A_B", "A", "B", M8SignalEnum.LONG_SPREAD_CANDIDATE, "1.0", "m2")],
    )
    r1 = run_research_backtest(inp1)
    r2 = run_research_backtest(inp2)
    assert r1.report["identities"]["m8_signal_identity"] != r2.report["identities"]["m8_signal_identity"]

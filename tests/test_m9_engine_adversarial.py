from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pandas as pd

from regimequant.backtesting.m9_engine import (
    CorporateActionEvent,
    PairSignalInput,
    ResearchInput,
    SecurityObservation,
    SessionPrice,
    run_research_backtest,
)
from regimequant.signals.m8_contracts import M8SignalEnum, M8SignalObservation


def _signal(mode: str = "replay", available_at: datetime | None = None) -> M8SignalObservation:
    ts = available_at or datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    return M8SignalObservation(
        canonical_pair_id="pair_A_B",
        eval_observation_time=pd.Timestamp(ts),
        eval_available_at=pd.Timestamp(ts),
        m8_decision_cutoff=pd.Timestamp(ts),
        signal=M8SignalEnum.LONG_SPREAD_CANDIDATE,
        z_score=2.5,
        abstain_reason=None,
        source_mode=mode,
        scientific_identity="sig_123",
        evidence_digest="ev_123",
        decision_digest="dec_123",
    )


def test_production_mode_bypass_attempt_rejected() -> None:
    base = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    result = run_research_backtest(
        ResearchInput(
            signal_observation=_signal(mode="production", available_at=base),
            beta=Decimal("1.2"),
            y_security_id="A",
            x_security_id="B",
            pair_id="pair_A_B",
            sessions=[SessionPrice(10, Decimal("100"), Decimal("50"), base), SessionPrice(11, Decimal("101"), Decimal("49"), base)],
            authorization_time=base,
            submission_time=base,
        )
    )
    assert result.report["certified"] is False


def test_late_evidence_rejection() -> None:
    base = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    late = base + timedelta(minutes=5)
    result = run_research_backtest(
        ResearchInput(
            signal_observation=_signal(available_at=late),
            beta=Decimal("1.2"),
            y_security_id="A",
            x_security_id="B",
            pair_id="pair_A_B",
            sessions=[SessionPrice(10, Decimal("100"), Decimal("50"), base), SessionPrice(11, Decimal("101"), Decimal("49"), base)],
            authorization_time=base,
            submission_time=base,
        )
    )
    assert result.report["uncertifiable_reason"] == "LATE_EVIDENCE"


def test_same_session_fill_rejected_via_chronology() -> None:
    base = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    result = run_research_backtest(
        ResearchInput(
            signal_observation=_signal(available_at=base),
            beta=Decimal("1.2"),
            y_security_id="A",
            x_security_id="B",
            pair_id="pair_A_B",
            sessions=[SessionPrice(10, Decimal("100"), Decimal("50"), base)],
            authorization_time=base,
            submission_time=base,
        )
    )
    assert result.report["certified"] is False


def test_missing_one_leg_close_cancels_fill() -> None:
    base = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    result = run_research_backtest(
        ResearchInput(
            signal_observation=_signal(available_at=base),
            beta=Decimal("1.2"),
            y_security_id="A",
            x_security_id="B",
            pair_id="pair_A_B",
            sessions=[SessionPrice(10, Decimal("100"), Decimal("50"), base), SessionPrice(11, None, Decimal("49"), base)],
            authorization_time=base,
            submission_time=base,
        )
    )
    assert result.report["certified"] is False


def test_missing_delisting_terminal_price_uncertifiable() -> None:
    base = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    sessions = [
        SessionPrice(10, Decimal("100"), Decimal("50"), base),
        SessionPrice(11, Decimal("101"), Decimal("49"), base),
        SessionPrice(12, Decimal("102"), Decimal("48"), base),
        SessionPrice(13, Decimal("103"), Decimal("47"), base),
        SessionPrice(14, Decimal("104"), Decimal("46"), base),
        SessionPrice(15, Decimal("105"), Decimal("45"), base),
        SessionPrice(16, None, None, base),
        SessionPrice(17, None, None, base),
    ]
    result = run_research_backtest(
        ResearchInput(
            signal_observation=_signal(available_at=base),
            beta=Decimal("1.2"),
            y_security_id="A",
            x_security_id="B",
            pair_id="pair_A_B",
            sessions=sessions,
            authorization_time=base,
            submission_time=base,
        )
    )
    assert result.report["certified"] is False


def test_identity_binding_present() -> None:
    base = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    sessions = [SessionPrice(10, Decimal("100"), Decimal("50"), base), SessionPrice(11, Decimal("101"), Decimal("49"), base), SessionPrice(12, Decimal("102"), Decimal("48"), base), SessionPrice(13, Decimal("103"), Decimal("47"), base), SessionPrice(14, Decimal("104"), Decimal("46"), base), SessionPrice(15, Decimal("105"), Decimal("45"), base), SessionPrice(16, Decimal("106"), Decimal("44"), base), SessionPrice(17, Decimal("107"), Decimal("43"), base)]
    result = run_research_backtest(
        ResearchInput(
            signal_observation=_signal(available_at=base),
            beta=Decimal("1.2"),
            y_security_id="A",
            x_security_id="B",
            pair_id="pair_A_B",
            sessions=sessions,
            authorization_time=base,
            submission_time=base,
        )
    )
    ids = result.report["identities"]
    assert "policy_identity" in ids
    assert "events_identity" in ids
    assert "fills_identity" in ids
    assert "accounting_identity" in ids


def _pair_signal(
    pair_id: str,
    y: str,
    x: str,
    sid: str,
    signal: M8SignalEnum,
    ts: datetime,
    decision_session: int,
) -> PairSignalInput:
    return PairSignalInput(
        signal_observation=M8SignalObservation(
            canonical_pair_id=pair_id,
            eval_observation_time=pd.Timestamp(ts),
            eval_available_at=pd.Timestamp(ts),
            m8_decision_cutoff=pd.Timestamp(ts),
            signal=signal,
            z_score=2.0,
            abstain_reason=None,
            source_mode="replay",
            scientific_identity=sid,
            evidence_digest="ev",
            decision_digest="dec",
        ),
        beta=Decimal("1.0"),
        y_security_id=y,
        x_security_id=x,
        pair_id=pair_id,
        authorization_time=ts,
        submission_time=ts,
        signal_identity=sid,
        decision_session_index=decision_session,
        decision_timestamp=ts,
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


def test_two_pairs_without_canonical_observations_reject() -> None:
    base = datetime(2026, 1, 20, 12, 0, tzinfo=timezone.utc)
    sessions = [SessionPrice(idx, Decimal("100"), Decimal("50"), base) for idx in range(1, 8)]
    out = run_research_backtest(
        ResearchInput(
            signal_observation=_signal(available_at=base),
            beta=Decimal("1.0"),
            y_security_id="A",
            x_security_id="B",
            pair_id="pair_A_B",
            sessions=sessions,
            authorization_time=base,
            submission_time=base,
            pair_signals=[
                _pair_signal("pair_A_B", "A", "B", "s1", M8SignalEnum.LONG_SPREAD_CANDIDATE, base, 1),
                _pair_signal("pair_C_D", "C", "D", "s2", M8SignalEnum.SHORT_SPREAD_CANDIDATE, base, 1),
            ],
        )
    )
    assert out.report["certified"] is False
    assert out.report["uncertifiable_reason"] == "MISSING_CANONICAL_SECURITY_OBSERVATIONS"
    assert len(out.fills) == 0


def test_disjoint_pairs_with_different_prices_and_shared_security_price_consistency() -> None:
    base = datetime(2026, 2, 1, 12, 0, tzinfo=timezone.utc)
    sessions = [SessionPrice(idx, Decimal("100"), Decimal("90"), base) for idx in range(1, 9)]
    obs = _canonical_observations(
        sessions,
        {
            "A": [Decimal("100") + Decimal(idx) for idx in range(1, 9)],
            "B": [Decimal("90") + Decimal(idx) for idx in range(1, 9)],
            "C": [Decimal("70") + Decimal(idx) for idx in range(1, 9)],
            "D": [Decimal("60") - Decimal(idx) for idx in range(1, 9)],
        },
        base=base,
    )
    inp = ResearchInput(
        signal_observation=_signal(available_at=base),
        beta=Decimal("1.0"),
        y_security_id="A",
        x_security_id="B",
        pair_id="pair_A_B",
        sessions=sessions,
        authorization_time=base,
        submission_time=base,
        security_observations=obs,
        pair_signals=[
            _pair_signal("pair_A_B", "A", "B", "sid1", M8SignalEnum.LONG_SPREAD_CANDIDATE, base, 1),
            _pair_signal("pair_C_D", "C", "D", "sid2", M8SignalEnum.SHORT_SPREAD_CANDIDATE, base, 1),
        ],
    )
    out = run_research_backtest(inp)
    assert out.report["certified"] is True
    pair_ab_entry = next(fill for fill in out.fills if fill["pair_id"] == "pair_A_B" and fill["session"] == 2)
    pair_cd_entry = next(fill for fill in out.fills if fill["pair_id"] == "pair_C_D" and fill["session"] == 2)
    prices_ab = {leg["security"]: Decimal(leg["price"]) for leg in pair_ab_entry["legs"]}
    prices_cd = {leg["security"]: Decimal(leg["price"]) for leg in pair_cd_entry["legs"]}
    assert prices_ab == {"A": Decimal("102"), "B": Decimal("92")}
    assert prices_cd == {"C": Decimal("72"), "D": Decimal("58")}


def test_conflicting_duplicate_observations_reject() -> None:
    base = datetime(2026, 2, 15, 12, 0, tzinfo=timezone.utc)
    sessions = [SessionPrice(idx, Decimal("100"), Decimal("90"), base) for idx in range(1, 8)]
    obs = _canonical_observations(
        sessions,
        {
            "A": [Decimal("100") + Decimal(idx) for idx in range(1, 8)],
            "B": [Decimal("90") + Decimal(idx) for idx in range(1, 8)],
            "C": [Decimal("70") + Decimal(idx) for idx in range(1, 8)],
            "D": [Decimal("60") - Decimal(idx) for idx in range(1, 8)],
        },
        base=base,
    )
    obs.append(SecurityObservation(2, "A", Decimal("999"), base, base, "A2_conflict"))

    out = run_research_backtest(
        ResearchInput(
            signal_observation=_signal(available_at=base),
            beta=Decimal("1.0"),
            y_security_id="A",
            x_security_id="B",
            pair_id="pair_A_B",
            sessions=sessions,
            authorization_time=base,
            submission_time=base,
            security_observations=obs,
            pair_signals=[
                _pair_signal("pair_A_B", "A", "B", "sid1", M8SignalEnum.LONG_SPREAD_CANDIDATE, base, 1),
                _pair_signal("pair_C_D", "C", "D", "sid2", M8SignalEnum.SHORT_SPREAD_CANDIDATE, base, 1),
            ],
        )
    )
    assert out.report["certified"] is False
    assert out.report["uncertifiable_reason"] == "CONFLICTING_SECURITY_OBSERVATION"
    assert len(out.fills) == 0


def test_single_pair_legacy_sessionprice_compatibility() -> None:
    base = datetime(2026, 2, 20, 12, 0, tzinfo=timezone.utc)
    sessions = [SessionPrice(idx, Decimal("100") + Decimal(idx), Decimal("50") - Decimal(idx), base) for idx in range(1, 9)]
    out = run_research_backtest(
        ResearchInput(
            signal_observation=_signal(available_at=base),
            beta=Decimal("1.0"),
            y_security_id="A",
            x_security_id="B",
            pair_id="pair_A_B",
            sessions=sessions,
            authorization_time=base,
            submission_time=base,
        )
    )
    assert out.report["research_only"] is True
    assert len(out.fills) == 2


def test_staggered_signal_decision_sessions_and_submission_after_close_rejected() -> None:
    base = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)
    sessions = [
        SessionPrice(10, Decimal("100"), Decimal("50"), base, close_event_time=base),
        SessionPrice(11, Decimal("101"), Decimal("49"), base + timedelta(minutes=1), close_event_time=base + timedelta(minutes=1)),
        SessionPrice(12, Decimal("102"), Decimal("48"), base + timedelta(minutes=2), close_event_time=base + timedelta(minutes=2)),
    ]
    obs = _canonical_observations(
        sessions,
        {
            "A": [Decimal("100"), Decimal("101"), Decimal("102")],
            "B": [Decimal("50"), Decimal("49"), Decimal("48")],
            "C": [Decimal("200"), Decimal("201"), Decimal("202")],
            "D": [Decimal("80"), Decimal("81"), Decimal("82")],
        },
        base=base,
    )
    late_submit = base + timedelta(minutes=3)
    inp = ResearchInput(
        signal_observation=_signal(available_at=base),
        beta=Decimal("1.0"),
        y_security_id="A",
        x_security_id="B",
        pair_id="pair_A_B",
        sessions=sessions,
        security_observations=obs,
        authorization_time=base,
        submission_time=base,
        pair_signals=[
            _pair_signal("pair_A_B", "A", "B", "sid_a", M8SignalEnum.LONG_SPREAD_CANDIDATE, base, 10),
            PairSignalInput(
                signal_observation=_signal(available_at=base),
                beta=Decimal("1.0"),
                y_security_id="C",
                x_security_id="D",
                pair_id="pair_C_D",
                authorization_time=base,
                submission_time=late_submit,
                signal_identity="sid_b",
                decision_session_index=11,
                decision_timestamp=base,
            ),
        ],
    )
    out = run_research_backtest(inp)
    rejects = [e for e in out.events if e.get("event") == "authorize_reject"]
    assert any(e.get("reason") == "SUBMISSION_AFTER_SELECTED_CLOSE" for e in rejects)


def test_late_close_availability_rejected_and_missing_selected_close_cancels() -> None:
    base = datetime(2026, 4, 1, 12, 0, tzinfo=timezone.utc)
    sessions = [
        SessionPrice(20, Decimal("100"), Decimal("50"), base, close_event_time=base),
        SessionPrice(21, Decimal("101"), Decimal("49"), base, close_event_time=base + timedelta(minutes=1)),
    ]
    obs = [
        SecurityObservation(21, "A", Decimal("101"), base + timedelta(minutes=1), base + timedelta(minutes=2), "late_a"),
    ]
    inp = ResearchInput(
        signal_observation=_signal(available_at=base),
        beta=Decimal("1.0"),
        y_security_id="A",
        x_security_id="B",
        pair_id="pair_A_B",
        sessions=sessions,
        authorization_time=base,
        submission_time=base,
        security_observations=obs,
        pair_signals=[_pair_signal("pair_A_B", "A", "B", "sid_a", M8SignalEnum.LONG_SPREAD_CANDIDATE, base, 20)],
    )
    out = run_research_backtest(inp)
    reject_reasons = [e.get("reason") for e in out.events if e.get("event") == "authorize_reject"]
    assert "MISSING_SELECTED_CLOSE" in reject_reasons or "LATE_CLOSE_AVAILABILITY" in reject_reasons


def test_identity_mutation_price_calendar_corp_action_certification() -> None:
    base = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    sessions = [SessionPrice(idx, Decimal("100") + Decimal(idx - 30), Decimal("50") - Decimal(idx - 30), base) for idx in range(30, 38)]
    r1 = run_research_backtest(
        ResearchInput(
            signal_observation=_signal(available_at=base),
            beta=Decimal("1.0"),
            y_security_id="A",
            x_security_id="B",
            pair_id="pair_A_B",
            sessions=sessions,
            authorization_time=base,
            submission_time=base,
            session_calendar_identity="cal_v1",
            corporate_actions=[CorporateActionEvent(30, "A", "split", Decimal("2"), True)],
        )
    )
    r2 = run_research_backtest(
        ResearchInput(
            signal_observation=_signal(available_at=base),
            beta=Decimal("1.0"),
            y_security_id="A",
            x_security_id="B",
            pair_id="pair_A_B",
            sessions=[SessionPrice(idx, Decimal("101") + Decimal(idx - 30), Decimal("50") - Decimal(idx - 30), base) for idx in range(30, 38)],
            authorization_time=base,
            submission_time=base,
            session_calendar_identity="cal_v2",
            corporate_actions=[CorporateActionEvent(30, "A", "spin_off", Decimal("1"), True)],
        )
    )
    assert r1.report["certified"] is True
    assert r2.report["certified"] is False
    assert r1.report["identities"]["price_evidence_identity"] != r2.report["identities"]["price_evidence_identity"]
    assert r1.report["identities"]["calendar_identity"] != r2.report["identities"]["calendar_identity"]
    assert r1.report["identities"]["corporate_action_identity"] != r2.report["identities"]["corporate_action_identity"]
    assert r1.report["identities"]["certification_identity"] != r2.report["identities"]["certification_identity"]


def test_decision_timestamp_mutation_changes_decision_provenance_identity() -> None:
    base = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
    sessions = [SessionPrice(idx, Decimal("100") + Decimal(idx), Decimal("60") + Decimal(idx), base) for idx in range(10, 18)]
    p1 = _pair_signal("pair_A_B", "A", "B", "sid_a", M8SignalEnum.LONG_SPREAD_CANDIDATE, base, 10)
    p2 = replace(p1, decision_timestamp=base + timedelta(seconds=1))

    out1 = run_research_backtest(
        ResearchInput(
            signal_observation=_signal(available_at=base),
            beta=Decimal("1.0"),
            y_security_id="A",
            x_security_id="B",
            pair_id="pair_A_B",
            sessions=sessions,
            authorization_time=base,
            submission_time=base,
            pair_signals=[p1],
        )
    )
    out2 = run_research_backtest(
        ResearchInput(
            signal_observation=_signal(available_at=base),
            beta=Decimal("1.0"),
            y_security_id="A",
            x_security_id="B",
            pair_id="pair_A_B",
            sessions=sessions,
            authorization_time=base,
            submission_time=base,
            pair_signals=[p2],
        )
    )
    assert out1.report["identities"]["decision_provenance_identity"] != out2.report["identities"]["decision_provenance_identity"]


def test_decision_session_mutation_changes_decision_provenance_identity() -> None:
    base = datetime(2026, 8, 2, 12, 0, tzinfo=timezone.utc)
    sessions = [SessionPrice(idx, Decimal("100") + Decimal(idx), Decimal("60") + Decimal(idx), base) for idx in range(10, 18)]
    p1 = _pair_signal("pair_A_B", "A", "B", "sid_a", M8SignalEnum.LONG_SPREAD_CANDIDATE, base, 10)
    p2 = replace(p1, decision_session_index=11)

    out1 = run_research_backtest(
        ResearchInput(
            signal_observation=_signal(available_at=base),
            beta=Decimal("1.0"),
            y_security_id="A",
            x_security_id="B",
            pair_id="pair_A_B",
            sessions=sessions,
            authorization_time=base,
            submission_time=base,
            pair_signals=[p1],
        )
    )
    out2 = run_research_backtest(
        ResearchInput(
            signal_observation=_signal(available_at=base),
            beta=Decimal("1.0"),
            y_security_id="A",
            x_security_id="B",
            pair_id="pair_A_B",
            sessions=sessions,
            authorization_time=base,
            submission_time=base,
            pair_signals=[p2],
        )
    )
    assert out1.report["identities"]["decision_provenance_identity"] != out2.report["identities"]["decision_provenance_identity"]


def test_no_fill_run_still_binds_decision_identity() -> None:
    base = datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc)
    sessions = [SessionPrice(10, Decimal("100"), Decimal("50"), base)]
    out = run_research_backtest(
        ResearchInput(
            signal_observation=_signal(available_at=base),
            beta=Decimal("1.0"),
            y_security_id="A",
            x_security_id="B",
            pair_id="pair_A_B",
            sessions=sessions,
            authorization_time=base,
            submission_time=base,
            pair_signals=[_pair_signal("pair_A_B", "A", "B", "sid_a", M8SignalEnum.LONG_SPREAD_CANDIDATE, base, 10)],
        )
    )
    assert out.report["certified"] is False
    assert out.report["identities"]["decision_provenance_identity"]


def test_missing_decision_evidence_rejects_deterministically() -> None:
    base = datetime(2026, 8, 4, 12, 0, tzinfo=timezone.utc)
    sessions = [SessionPrice(idx, Decimal("100") + Decimal(idx), Decimal("50") + Decimal(idx), base) for idx in range(10, 18)]
    bad = PairSignalInput(
        signal_observation=_signal(available_at=base),
        beta=Decimal("1.0"),
        y_security_id="A",
        x_security_id="B",
        pair_id="pair_A_B",
        authorization_time=base,
        submission_time=base,
        signal_identity="",
        decision_session_index=10,
        decision_timestamp=base,
    )
    out = run_research_backtest(
        ResearchInput(
            signal_observation=_signal(available_at=base),
            beta=Decimal("1.0"),
            y_security_id="A",
            x_security_id="B",
            pair_id="pair_A_B",
            sessions=sessions,
            authorization_time=base,
            submission_time=base,
            pair_signals=[bad],
        )
    )
    assert out.report["certified"] is False
    assert out.report["uncertifiable_reason"] == "MISSING_DECISION_PROVENANCE"


def test_unsupported_corporate_action_fails_closed() -> None:
    base = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
    sessions = [SessionPrice(40, Decimal("100"), Decimal("50"), base), SessionPrice(41, Decimal("101"), Decimal("49"), base)]
    out = run_research_backtest(
        ResearchInput(
            signal_observation=_signal(available_at=base),
            beta=Decimal("1.0"),
            y_security_id="A",
            x_security_id="B",
            pair_id="pair_A_B",
            sessions=sessions,
            authorization_time=base,
            submission_time=base,
            corporate_actions=[CorporateActionEvent(40, "A", "spin_off", Decimal("1"), True)],
        )
    )
    assert out.report["certified"] is False
    assert out.report["uncertifiable_reason"] == "UNSUPPORTED_CORPORATE_ACTION"


def test_input_order_invariance_for_pair_signals() -> None:
    base = datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc)
    sessions = [SessionPrice(idx, Decimal("100") + Decimal(idx), Decimal("50") + Decimal(idx), base) for idx in range(10, 18)]
    p1 = _pair_signal("pair_A_B", "A", "B", "s_a", M8SignalEnum.LONG_SPREAD_CANDIDATE, base, 10)
    p2 = _pair_signal("pair_C_D", "C", "D", "s_b", M8SignalEnum.SHORT_SPREAD_CANDIDATE, base, 10)
    obs = _canonical_observations(
        sessions,
        {
            "A": [Decimal("100") + Decimal(idx) for idx in range(10, 18)],
            "B": [Decimal("50") + Decimal(idx) for idx in range(10, 18)],
            "C": [Decimal("200") + Decimal(idx) for idx in range(10, 18)],
            "D": [Decimal("80") + Decimal(idx) for idx in range(10, 18)],
        },
        base=base,
    )

    out1 = run_research_backtest(
        ResearchInput(
            signal_observation=_signal(available_at=base),
            beta=Decimal("1.0"),
            y_security_id="A",
            x_security_id="B",
            pair_id="pair_A_B",
            sessions=sessions,
            authorization_time=base,
            submission_time=base,
            security_observations=obs,
            pair_signals=[p1, p2],
        )
    )
    out2 = run_research_backtest(
        ResearchInput(
            signal_observation=_signal(available_at=base),
            beta=Decimal("1.0"),
            y_security_id="A",
            x_security_id="B",
            pair_id="pair_A_B",
            sessions=sessions,
            authorization_time=base,
            submission_time=base,
            security_observations=obs,
            pair_signals=[p2, p1],
        )
    )

    assert out1.report["identities"]["events_identity"] == out2.report["identities"]["events_identity"]
    assert out1.report["identities"]["fills_identity"] == out2.report["identities"]["fills_identity"]
    assert out1.report["identities"]["price_evidence_identity"] == out2.report["identities"]["price_evidence_identity"]
    assert out1.report["identities"]["decision_provenance_identity"] == out2.report["identities"]["decision_provenance_identity"]

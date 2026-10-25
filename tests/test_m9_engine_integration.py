from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pandas as pd

from regimequant.backtesting.m9_engine import ResearchInput, SessionPrice, run_research_backtest
from regimequant.signals.m8_contracts import M8SignalEnum, M8SignalObservation


def _signal(signal: M8SignalEnum, available_at: datetime) -> M8SignalObservation:
    return M8SignalObservation(
        canonical_pair_id="pair_A_B",
        eval_observation_time=pd.Timestamp(available_at),
        eval_available_at=pd.Timestamp(available_at),
        m8_decision_cutoff=pd.Timestamp(available_at),
        signal=signal,
        z_score=2.5,
        abstain_reason=None,
        source_mode="replay",
        scientific_identity="sig_123",
        evidence_digest="ev_123",
        decision_digest="dec_123",
    )


def test_complete_long_spread_lifecycle_public_engine() -> None:
    base = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    sessions = [
        SessionPrice(10, Decimal("100"), Decimal("50"), base),
        SessionPrice(11, Decimal("101"), Decimal("49"), base),
        SessionPrice(12, Decimal("102"), Decimal("48"), base),
        SessionPrice(13, Decimal("103"), Decimal("47"), base),
        SessionPrice(14, Decimal("104"), Decimal("46"), base),
        SessionPrice(15, Decimal("105"), Decimal("45"), base),
        SessionPrice(16, Decimal("106"), Decimal("44"), base),
        SessionPrice(17, Decimal("107"), Decimal("43"), base),
    ]
    result = run_research_backtest(
        ResearchInput(
            signal_observation=_signal(M8SignalEnum.LONG_SPREAD_CANDIDATE, base),
            beta=Decimal("1.2"),
            y_security_id="A",
            x_security_id="B",
            pair_id="pair_A_B",
            sessions=sessions,
            authorization_time=base,
            submission_time=base,
        )
    )
    assert result.report["research_only"] is True
    assert len(result.fills) == 2
    assert result.fills[0]["session"] == 11
    assert result.fills[1]["session"] == 17


def test_complete_short_spread_lifecycle_public_engine() -> None:
    base = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    sessions = [SessionPrice(idx, Decimal("100") + Decimal(idx), Decimal("60") + Decimal(idx), base) for idx in range(20, 28)]
    result = run_research_backtest(
        ResearchInput(
            signal_observation=_signal(M8SignalEnum.SHORT_SPREAD_CANDIDATE, base),
            beta=Decimal("0.9"),
            y_security_id="A",
            x_security_id="B",
            pair_id="pair_A_B",
            sessions=sessions,
            authorization_time=base,
            submission_time=base,
        )
    )
    assert len(result.fills) == 2
    assert result.report["assumption_driven"] is True


def test_negative_beta_pair_is_supported() -> None:
    base = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    sessions = [SessionPrice(idx, Decimal("100"), Decimal("80"), base) for idx in range(30, 38)]
    result = run_research_backtest(
        ResearchInput(
            signal_observation=_signal(M8SignalEnum.SHORT_SPREAD_CANDIDATE, base),
            beta=Decimal("-0.8"),
            y_security_id="A",
            x_security_id="B",
            pair_id="pair_A_B",
            sessions=sessions,
            authorization_time=base,
            submission_time=base,
        )
    )
    assert len(result.fills) == 2


def test_stale_age_3_vs_4_behavior() -> None:
    base = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    sessions_ok = [
        SessionPrice(40, Decimal("100"), Decimal("50"), base),
        SessionPrice(41, Decimal("101"), Decimal("49"), base),
        SessionPrice(42, None, None, base),
        SessionPrice(43, None, None, base),
        SessionPrice(44, None, None, base),
        SessionPrice(45, Decimal("105"), Decimal("45"), base),
        SessionPrice(46, Decimal("106"), Decimal("44"), base),
        SessionPrice(47, Decimal("107"), Decimal("43"), base),
    ]
    ok_result = run_research_backtest(
        ResearchInput(
            signal_observation=_signal(M8SignalEnum.LONG_SPREAD_CANDIDATE, base),
            beta=Decimal("1.0"),
            y_security_id="A",
            x_security_id="B",
            pair_id="pair_A_B",
            sessions=sessions_ok,
            authorization_time=base,
            submission_time=base,
        )
    )
    assert ok_result.report["report_identity"]

    sessions_bad = list(sessions_ok)
    sessions_bad[5] = SessionPrice(45, None, None, base)
    bad_result = run_research_backtest(
        ResearchInput(
            signal_observation=_signal(M8SignalEnum.LONG_SPREAD_CANDIDATE, base),
            beta=Decimal("1.0"),
            y_security_id="A",
            x_security_id="B",
            pair_id="pair_A_B",
            sessions=sessions_bad,
            authorization_time=base,
            submission_time=base,
        )
    )
    assert bad_result.report["certified"] is False

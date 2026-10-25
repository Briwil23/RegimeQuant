from __future__ import annotations

from regimequant.backtesting.m9_valuation import certification_status, delisting_valuation_status, valuation_from_stale_age


def test_stale_valuation_threshold() -> None:
    ok = valuation_from_stale_age(sessions_since_mark=3, max_stale_sessions=3)
    assert ok.stale_flag
    assert not ok.uncertifiable

    breach = valuation_from_stale_age(sessions_since_mark=4, max_stale_sessions=3)
    assert breach.uncertifiable


def test_delisting_without_terminal_price_is_uncertifiable() -> None:
    status = delisting_valuation_status(has_terminal_executable_price=False)
    assert status.uncertifiable


def test_missing_corporate_action_evidence_blocks_certification() -> None:
    certified, reason = certification_status(
        prices_complete=True,
        provenance_complete=True,
        corporate_action_evidence_complete=False,
    )
    assert not certified
    assert reason == "MISSING_CORPORATE_ACTION_EVIDENCE"

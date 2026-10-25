from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ValuationStatus:
    stale_flag: bool
    uncertifiable: bool
    reason: str | None


def valuation_from_stale_age(*, sessions_since_mark: int, max_stale_sessions: int = 3) -> ValuationStatus:
    if sessions_since_mark < 0:
        raise ValueError("sessions_since_mark must be non-negative")
    if sessions_since_mark <= max_stale_sessions:
        return ValuationStatus(stale_flag=sessions_since_mark > 0, uncertifiable=False, reason=None)
    return ValuationStatus(stale_flag=True, uncertifiable=True, reason="STALE_BREACH")


def delisting_valuation_status(*, has_terminal_executable_price: bool) -> ValuationStatus:
    if has_terminal_executable_price:
        return ValuationStatus(stale_flag=False, uncertifiable=False, reason=None)
    return ValuationStatus(stale_flag=True, uncertifiable=True, reason="DELIST_NO_TERMINAL_PRICE")


def certification_status(*, prices_complete: bool, provenance_complete: bool, corporate_action_evidence_complete: bool) -> tuple[bool, str | None]:
    if not prices_complete:
        return False, "MISSING_REQUIRED_PRICES"
    if not provenance_complete:
        return False, "MISSING_PROVENANCE"
    if not corporate_action_evidence_complete:
        return False, "MISSING_CORPORATE_ACTION_EVIDENCE"
    return True, None

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class M8CalendarError(ValueError):
    """Raised when trading calendar evidence is missing or inconsistent."""


def _as_utc(value: object) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


@dataclass(frozen=True, slots=True)
class TradingCalendarEvidence:
    calendar_id: str
    calendar_version: str
    sessions: tuple[pd.Timestamp, ...]
    session_sequence_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "calendar_id", str(self.calendar_id).strip())
        object.__setattr__(self, "calendar_version", str(self.calendar_version).strip())
        object.__setattr__(self, "session_sequence_digest", str(self.session_sequence_digest).strip())
        normalized = tuple(_as_utc(value) for value in self.sessions)
        object.__setattr__(self, "sessions", normalized)
        if len(self.sessions) == 0:
            raise M8CalendarError("sessions must not be empty")
        if len(set(self.sessions)) != len(self.sessions):
            raise M8CalendarError("sessions contain duplicates")
        if list(self.sessions) != sorted(self.sessions):
            raise M8CalendarError("sessions must be strictly increasing")
        if not self.calendar_id or not self.calendar_version or not self.session_sequence_digest:
            raise M8CalendarError("calendar evidence requires id, version, and digest")

    def session_index(self, session: object) -> int:
        target = _as_utc(session)
        try:
            return self.sessions.index(target)
        except ValueError as exc:
            raise M8CalendarError("session is not present in declared calendar evidence") from exc

    def completed_sessions_after(self, left_session: object, right_session: object) -> int:
        left = self.session_index(left_session)
        right = self.session_index(right_session)
        if right <= left:
            return 0
        return right - left


def discovery_is_expired(
    *,
    calendar: TradingCalendarEvidence,
    discovery_observation_session: object,
    decision_observation_session: object,
    max_completed_sessions: int = 20,
) -> bool:
    completed = calendar.completed_sessions_after(discovery_observation_session, decision_observation_session)
    return completed > max_completed_sessions

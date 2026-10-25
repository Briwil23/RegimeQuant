from __future__ import annotations

from regimequant.cointegration.contracts import DecisionEnvelope
from regimequant.cointegration.validation import ValidationError, validate_decision_evidence


class M8ValidationError(ValueError):
    """Raised when causal evidence cannot support an M8 signal decision."""


def validate_m8_decision_envelope(envelope: DecisionEnvelope) -> dict[str, object]:
    try:
        return validate_decision_evidence(envelope)
    except ValidationError as exc:
        raise M8ValidationError(str(exc)) from exc


def require_available_by_cutoff(*, available_at, decision_cutoff, field_name: str) -> None:
    if available_at > decision_cutoff:
        raise M8ValidationError(f"{field_name} is available after decision cutoff")

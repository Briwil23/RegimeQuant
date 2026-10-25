from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd


class CointegrationContractError(ValueError):
    """Base class for M7 cointegration contract violations."""


@dataclass(frozen=True, slots=True)
class PricePolicyResult:
    valid: bool
    ticker: str
    aligned_index: pd.Index
    price_series: pd.Series
    first_difference_count: int
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "ticker", str(self.ticker).strip())
        object.__setattr__(self, "reasons", tuple(str(reason) for reason in self.reasons))


@dataclass(frozen=True, slots=True)
class EngleGrangerResult:
    pair_id: str
    statistic: float
    p_value: float
    adf_statistic: float | None = None
    adf_pvalue: float | None = None
    valid: bool = True
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "pair_id", str(self.pair_id).strip())
        object.__setattr__(self, "warnings", tuple(str(value) for value in self.warnings))


@dataclass(frozen=True, slots=True)
class DecisionEnvelope:
    mode: str
    cutoff: pd.Timestamp
    availability_verified: bool
    source_digest: str
    evidence_digest: str
    decision_digest: str
    replay_reference: str | None = None
    validation_cutoff: pd.Timestamp | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", str(self.mode).strip().lower())
        object.__setattr__(self, "cutoff", pd.Timestamp(self.cutoff).tz_convert("UTC"))
        if self.validation_cutoff is not None:
            object.__setattr__(self, "validation_cutoff", pd.Timestamp(self.validation_cutoff).tz_convert("UTC"))
        object.__setattr__(self, "source_digest", str(self.source_digest).strip())
        object.__setattr__(self, "evidence_digest", str(self.evidence_digest).strip())
        object.__setattr__(self, "decision_digest", str(self.decision_digest).strip())
        if self.mode not in {"production", "replay", "evaluation"}:
            raise ValueError("mode must be production, replay, or evaluation")


@dataclass(frozen=True, slots=True)
class ScientificHashPayload:
    pair_id: str
    research_id: str
    model_version: str = "m7v1"
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "pair_id", str(self.pair_id).strip())
        object.__setattr__(self, "research_id", str(self.research_id).strip())
        object.__setattr__(self, "payload", dict(self.payload))


@dataclass(frozen=True, slots=True)
class CointegrationCandidate:
    instrument_a: str
    instrument_b: str
    pair_id: str
    status: str = "candidate"
    p_value: float | None = None
    adjusted_p_value: float | None = None
    score: float | None = None
    source_mode: str = "production"

    def __post_init__(self) -> None:
        object.__setattr__(self, "instrument_a", str(self.instrument_a).strip())
        object.__setattr__(self, "instrument_b", str(self.instrument_b).strip())
        object.__setattr__(self, "pair_id", str(self.pair_id).strip())
        object.__setattr__(self, "status", str(self.status).strip())
        object.__setattr__(self, "source_mode", str(self.source_mode).strip())

from __future__ import annotations

import pandas as pd

from regimequant.signals.m8_contracts import M8AbstainReason, M8SignalEnum
from regimequant.signals.m8_signal import evaluate_stateless_signal


def test_signal_threshold_boundaries_and_abstain() -> None:
    kwargs = {
        "canonical_pair_id": "pair_0b8f0fbd6ed5f2be",
        "eval_observation_time": pd.Timestamp("2024-01-20", tz="UTC"),
        "eval_available_at": pd.Timestamp("2024-01-20", tz="UTC"),
        "m8_decision_cutoff": pd.Timestamp("2024-01-20", tz="UTC"),
        "spread_mean": 0.0,
        "spread_std": 1.0,
        "source_mode": "production",
        "scientific_identity": "id1",
        "evidence_digest": "e1",
        "decision_digest": "d1",
    }
    short_obs = evaluate_stateless_signal(spread_eval=2.0, **kwargs)
    long_obs = evaluate_stateless_signal(spread_eval=-2.0, **kwargs)
    neutral_obs = evaluate_stateless_signal(spread_eval=1.5, **kwargs)
    abstain_obs = evaluate_stateless_signal(spread_eval=1.0, invalid_reason=M8AbstainReason.EVIDENCE_LATE, **kwargs)

    assert short_obs.signal == M8SignalEnum.SHORT_SPREAD_CANDIDATE
    assert long_obs.signal == M8SignalEnum.LONG_SPREAD_CANDIDATE
    assert neutral_obs.signal == M8SignalEnum.NEUTRAL
    assert abstain_obs.signal == M8SignalEnum.ABSTAIN
    assert abstain_obs.abstain_reason == M8AbstainReason.EVIDENCE_LATE

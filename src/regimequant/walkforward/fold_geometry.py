from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from regimequant.walkforward.contracts import FoldInterval, M10Policy


class FoldGeometryStatus(str, Enum):
    ADMISSIBLE = "ADMISSIBLE"
    NO_ADMISSIBLE_FOLD = "NO_ADMISSIBLE_FOLD"


@dataclass(frozen=True, slots=True)
class DeterministicFold:
    fold_index: int
    fold_id: str
    interval: FoldInterval


def _validate_total_sessions(total_sessions: int) -> None:
    if not isinstance(total_sessions, int):
        raise TypeError("total_sessions must be an int")
    if total_sessions < 0:
        raise ValueError("total_sessions must be non-negative")


def classify_fold_geometry(*, total_sessions: int, policy: M10Policy) -> FoldGeometryStatus:
    _validate_total_sessions(total_sessions)
    if total_sessions < (
        policy.initial_training_sessions
        + policy.validation_sessions
        + policy.outer_test_sessions
        + policy.final_holdout_sessions
    ):
        return FoldGeometryStatus.NO_ADMISSIBLE_FOLD
    return FoldGeometryStatus.ADMISSIBLE


def build_expanding_fold_intervals(*, total_sessions: int, policy: M10Policy) -> tuple[DeterministicFold, ...]:
    _validate_total_sessions(total_sessions)

    holdout_start = total_sessions - policy.final_holdout_sessions
    holdout_end = total_sessions
    if holdout_start < 0:
        return ()

    folds: list[DeterministicFold] = []
    k = 0
    while True:
        train_end = policy.initial_training_sessions + policy.stride_sessions * k
        validation_start = train_end
        validation_end = validation_start + policy.validation_sessions
        test_start = validation_end
        test_end = test_start + policy.outer_test_sessions

        # Fold k admissible iff 160 + 20k <= N - 20 under frozen policy values.
        if test_end > holdout_start:
            break

        interval = FoldInterval(
            train_start=0,
            train_end=train_end,
            validation_start=validation_start,
            validation_end=validation_end,
            test_start=test_start,
            test_end=test_end,
            holdout_start=holdout_start,
            holdout_end=holdout_end,
        )
        fold_id = derive_fold_id(fold_index=k, interval=interval)
        folds.append(DeterministicFold(fold_index=k, fold_id=fold_id, interval=interval))
        k += 1

    return tuple(folds)


def derive_fold_id(*, fold_index: int, interval: FoldInterval) -> str:
    if fold_index < 0:
        raise ValueError("fold_index must be non-negative")
    return (
        f"fold_{fold_index:04d}"
        f"_tr_0_{interval.train_end}"
        f"_va_{interval.validation_start}_{interval.validation_end}"
        f"_te_{interval.test_start}_{interval.test_end}"
    )

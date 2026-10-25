from __future__ import annotations

from decimal import Decimal

from regimequant.backtesting.m9_identity import performance_identity


def build_research_report(
    *,
    equity_start: Decimal,
    equity_end: Decimal,
    uncertifiable: bool,
    uncertifiable_reason: str | None,
    assumptions: dict[str, str],
    identities: dict[str, str],
) -> dict[str, object]:
    pnl = equity_end - equity_start
    report = {
        "research_only": True,
        "assumption_driven": True,
        "equity_start": str(equity_start),
        "equity_end": str(equity_end),
        "pnl": str(pnl),
        "certified": not uncertifiable,
        "uncertifiable_reason": uncertifiable_reason,
        "assumptions": assumptions,
        "identities": identities,
    }
    report["report_identity"] = performance_identity(report)
    return report

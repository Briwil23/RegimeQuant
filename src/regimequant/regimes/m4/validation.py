from __future__ import annotations

from typing import Any

from .contracts import M4RegimeStabilityReport


def validate_m4_report(report: Any) -> M4RegimeStabilityReport:
    if not isinstance(report, M4RegimeStabilityReport):
        raise TypeError("report must be a M4RegimeStabilityReport")
    if report.report_version != "m4v1":
        raise ValueError("report_version must be 'm4v1'")
    if report.metric_version != "m4v1":
        raise ValueError("metric_version must be 'm4v1'")
    if not report.model_id:
        raise ValueError("model_id is required")
    if not report.instrument:
        raise ValueError("instrument is required")
    if report.evaluation_window_start_utc is None or report.evaluation_window_end_utc is None:
        raise ValueError("evaluation_window_start_utc and evaluation_window_end_utc are required")
    if report.evaluation_window_start_utc >= report.evaluation_window_end_utc:
        raise ValueError("evaluation_window must be strictly ordered")
    return report

from __future__ import annotations

from typing import Any


def build_synthetic_m4_fixture() -> dict[str, Any]:
    payload = {
        "synthetic": True,
        "report_version": "m4v1",
        "model_id": "synthetic_m4_fixture__AAA__2023-01-03T00:00:00Z",
        "instrument": "AAA",
        "evaluation_mode": "HISTORICAL_REPLAY",
        "coverage": {"requested_count": 3, "assigned_count": 3, "abstained_count": 0},
        "cluster_occupancy": [
            {"cluster_id": 0, "assigned_count": 2, "assigned_share": 0.6666666666666666},
            {"cluster_id": 1, "assigned_count": 1, "assigned_share": 0.3333333333333333},
        ],
    }
    return payload

from __future__ import annotations

import pytest

from regimequant.regimes.m4 import build_synthetic_m4_fixture


def test_synthetic_fixture_is_marked_non_production() -> None:
    fixture = build_synthetic_m4_fixture()
    assert fixture["model_id"].startswith("synthetic_")
    assert not fixture["model_id"].startswith("m3v1_sha256_")
    assert fixture["synthetic"] is True


def test_synthetic_fixture_contract_has_exact_nonproduction_payload() -> None:
    fixture = build_synthetic_m4_fixture()
    assert fixture["report_version"] == "m4v1"
    assert fixture["instrument"] == "AAA"
    assert fixture["evaluation_mode"] == "HISTORICAL_REPLAY"
    assert fixture["coverage"] == {"requested_count": 3, "assigned_count": 3, "abstained_count": 0}
    assert fixture["cluster_occupancy"][0]["assigned_share"] == pytest.approx(2 / 3)
    assert fixture["cluster_occupancy"][1]["assigned_share"] == pytest.approx(1 / 3)
    assert fixture["model_id"].startswith("synthetic_")
    assert not fixture["model_id"].startswith("m3v1_sha256_")

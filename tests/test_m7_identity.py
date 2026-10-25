from __future__ import annotations

from regimequant.cointegration.identity import (
    compute_canonical_pair_id,
    compute_dataset_identity,
    compute_research_instance_id,
    compute_scientific_hash,
)


def test_pair_and_research_identity_are_stable_and_deterministic() -> None:
    pair_id_a = compute_canonical_pair_id("AAPL", "MSFT", "2020-01-01", "2025-01-01")
    pair_id_b = compute_canonical_pair_id("MSFT", "AAPL", "2020-01-01", "2025-01-01")
    assert pair_id_a == pair_id_b
    research_id = compute_research_instance_id("demo-research", "AAPL", "MSFT")
    assert research_id.startswith("research_")
    digest = compute_scientific_hash({"pair_id": pair_id_a, "research_id": research_id})
    assert len(digest) == 64


def test_pair_id_excludes_time_window_material_inputs_are_same() -> None:
    pair_default = compute_canonical_pair_id("AAPL", "MSFT")
    pair_windowed = compute_canonical_pair_id("AAPL", "MSFT", "2020-01-01", "2025-01-01")
    pair_reversed = compute_canonical_pair_id("MSFT", "AAPL", "2019-01-01", "2026-01-01")
    assert pair_default == pair_windowed == pair_reversed


def test_research_instance_id_changes_with_cutoff_and_material_configuration() -> None:
    base = compute_research_instance_id(
        "demo-research",
        "AAPL",
        "MSFT",
        dataset_identity="dataset-v1",
        cutoff="2024-01-02",
        snapshot_id="snapshot-v1",
        statistical_config={"maxlag": 12, "trend": "c", "autolag": "AIC"},
    )
    changed_cutoff = compute_research_instance_id(
        "demo-research",
        "AAPL",
        "MSFT",
        dataset_identity="dataset-v1",
        cutoff="2024-01-03",
        snapshot_id="snapshot-v1",
        statistical_config={"maxlag": 12, "trend": "c", "autolag": "AIC"},
    )
    changed_config = compute_research_instance_id(
        "demo-research",
        "AAPL",
        "MSFT",
        dataset_identity="dataset-v1",
        cutoff="2024-01-02",
        snapshot_id="snapshot-v1",
        statistical_config={"maxlag": 13, "trend": "c", "autolag": "AIC"},
    )
    assert base != changed_cutoff
    assert base != changed_config


def test_dataset_identity_changes_with_price_or_session_and_is_order_invariant() -> None:
    base = compute_dataset_identity(
        security_ids=("AAPL", "MSFT"),
        session_ids=("2024-01-02", "2024-01-03"),
        aligned_prices=(100.0, 110.0),
    )
    changed_price = compute_dataset_identity(
        security_ids=("AAPL", "MSFT"),
        session_ids=("2024-01-02", "2024-01-03"),
        aligned_prices=(100.0, 111.0),
    )
    changed_session = compute_dataset_identity(
        security_ids=("AAPL", "MSFT"),
        session_ids=("2024-01-02", "2024-01-04"),
        aligned_prices=(100.0, 110.0),
    )
    reordered = compute_dataset_identity(
        security_ids=("MSFT", "AAPL"),
        session_ids=("2024-01-03", "2024-01-02"),
        aligned_prices=(110.0, 100.0),
    )
    assert base == reordered
    assert base != changed_price
    assert base != changed_session


def test_dataset_identity_excludes_runtime_metadata() -> None:
    base = compute_dataset_identity(
        security_ids=("AAPL", "MSFT"),
        session_ids=("2024-01-02", "2024-01-03"),
        aligned_prices=(100.0, 110.0),
        provenance={"run_id": "run-1", "runtime_metadata": {"pid": 123}},
    )
    altered = compute_dataset_identity(
        security_ids=("AAPL", "MSFT"),
        session_ids=("2024-01-02", "2024-01-03"),
        aligned_prices=(100.0, 110.0),
        provenance={"run_id": "run-2", "runtime_metadata": {"pid": 456}},
    )
    assert base == altered


def test_scientific_hash_excludes_output_and_runtime_metadata() -> None:
    payload = {"pair_id": "pair_abc", "research_id": "research_abc", "data": {"price": 100.0}}
    with_output_id = {**payload, "output_id": "out-123"}
    with_runtime_metadata = {**payload, "runtime_metadata": {"pid": 456}}
    assert compute_scientific_hash(payload) == compute_scientific_hash(with_output_id)
    assert compute_scientific_hash(payload) == compute_scientific_hash(with_runtime_metadata)

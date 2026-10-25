from __future__ import annotations

from pathlib import Path


def test_m7_scope_is_limited_to_authorized_paths() -> None:
    expected = {
        Path("src/regimequant/cointegration/__init__.py"),
        Path("src/regimequant/cointegration/contracts.py"),
        Path("src/regimequant/cointegration/adapter.py"),
        Path("src/regimequant/cointegration/price_policy.py"),
        Path("src/regimequant/cointegration/pair_selection.py"),
        Path("src/regimequant/cointegration/integration_screening.py"),
        Path("src/regimequant/cointegration/engle_granger.py"),
        Path("src/regimequant/cointegration/multiple_testing.py"),
        Path("src/regimequant/cointegration/validation.py"),
        Path("src/regimequant/cointegration/identity.py"),
    }
    observed = {path for path in Path("src/regimequant/cointegration").rglob("*.py") if path.is_file()}
    assert observed == expected

from __future__ import annotations

from pathlib import Path


def test_m8_scope_is_limited_to_authorized_signals_files() -> None:
    expected = {
        Path("src/regimequant/signals/m8_contracts.py"),
        Path("src/regimequant/signals/m8_family_evidence.py"),
        Path("src/regimequant/signals/m8_eligibility.py"),
        Path("src/regimequant/signals/m8_calendar.py"),
        Path("src/regimequant/signals/m8_formation.py"),
        Path("src/regimequant/signals/m8_numerics.py"),
        Path("src/regimequant/signals/m8_signal.py"),
        Path("src/regimequant/signals/m8_identity.py"),
        Path("src/regimequant/signals/m8_validation.py"),
    }
    observed = {path for path in Path("src/regimequant/signals").glob("m8_*.py") if path.is_file()}
    assert observed == expected

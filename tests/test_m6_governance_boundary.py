from __future__ import annotations

from pathlib import Path


def test_authorized_universe_package_and_no_init_py() -> None:
    universe_root = Path("src/regimequant/universe")
    assert universe_root.is_dir()
    expected_files = {
        Path("src/regimequant/universe/__init__.py"),
        Path("src/regimequant/universe/contracts.py"),
        Path("src/regimequant/universe/membership.py"),
        Path("src/regimequant/universe/eligibility.py"),
        Path("src/regimequant/universe/snapshot.py"),
        Path("src/regimequant/universe/hash.py"),
        Path("src/regimequant/universe/validation.py"),
    }
    observed = {path for path in universe_root.glob("*.py")}
    assert observed == expected_files
    assert not Path("src/regimequant/universe/init.py").exists()


def test_authorized_universe_files_are_only_python_sources() -> None:
    package = Path("src/regimequant/universe")
    observed = sorted(path.name for path in package.glob("*.py"))
    assert observed == [
        "__init__.py",
        "contracts.py",
        "eligibility.py",
        "hash.py",
        "membership.py",
        "snapshot.py",
        "validation.py",
    ]

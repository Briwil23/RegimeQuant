from __future__ import annotations

from pathlib import Path


FUTURE_PACKAGES = [
    "data",
    "features",
    "regimes",
    "statarb",
    "signals",
    "backtesting",
    "validation",
    "research",
]


FORBIDDEN_IMPLEMENTATION_MARKERS = {
    "def ",
    "class ",
    "import sklearn",
    "from sklearn",
    "import statsmodels",
    "from statsmodels",
    "import pandas",
    "from pandas",
    "import numpy",
    "from numpy",
}


def test_future_packages_are_placeholders_only() -> None:
    root = Path("src/regimequant")
    for pkg in FUTURE_PACKAGES:
        package_path = root / pkg
        assert package_path.is_dir()
        python_files = sorted(package_path.glob("*.py"))
        assert [p.name for p in python_files] == ["__init__.py"]


def test_placeholder_packages_contain_no_runtime_implementation() -> None:
    source_root = Path("src/regimequant")
    for package in FUTURE_PACKAGES:
        text = (source_root / package / "__init__.py").read_text(encoding="utf-8").lower()
        for marker in FORBIDDEN_IMPLEMENTATION_MARKERS:
            assert marker not in text

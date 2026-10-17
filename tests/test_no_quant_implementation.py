from __future__ import annotations

import ast
from pathlib import Path


APPROVED_M3_REGIME_FILES = {
    Path("src/regimequant/regimes/__init__.py"),
    Path("src/regimequant/regimes/contracts.py"),
    Path("src/regimequant/regimes/matrix.py"),
    Path("src/regimequant/regimes/model.py"),
    Path("src/regimequant/regimes/assignment.py"),
    Path("src/regimequant/regimes/diagnostics.py"),
    Path("src/regimequant/regimes/splits.py"),
}


IMPLEMENTED_M1_DATA_FILES = {
    Path("src/regimequant/data/__init__.py"),
    Path("src/regimequant/data/errors.py"),
    Path("src/regimequant/data/models.py"),
    Path("src/regimequant/data/providers.py"),
    Path("src/regimequant/data/canonicalize.py"),
    Path("src/regimequant/data/returns.py"),
}

IMPLEMENTED_M2_FEATURE_FILES = {
    Path("src/regimequant/features/__init__.py"),
    Path("src/regimequant/features/definitions.py"),
    Path("src/regimequant/features/engine.py"),
    Path("src/regimequant/features/models.py"),
    Path("src/regimequant/features/rolling.py"),
    Path("src/regimequant/features/validation.py"),
}


FUTURE_PACKAGES = [
    "statarb",
    "signals",
    "backtesting",
    "validation",
    "research",
]


FORBIDDEN_IMPORT_PREFIXES = {
    "dash",
    "plotly",
    "requests",
    "httpx",
    "yfinance",
    "statsmodels",
    "sklearn",
    "networkx",
}

FORBIDDEN_EXPORTED_APIS = {
    "compute_signals",
    "generate_signals",
    "run_backtest",
    "backtest",
    "fit_cointegration",
}

FORBIDDEN_CALLABLE_NAMES = {
    "compute_signals",
    "generate_signals",
    "run_backtest",
    "fit_cointegration",
    "fit_pca",
    "fit_gmm",
    "fit_hmm",
    "optimize_portfolio",
}


def _all_python_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)


def _read_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _iter_import_names(module: ast.Module) -> list[str]:
    names: list[str] = []
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.append(node.module)
    return names


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
        assert "def " not in text
        assert "class " not in text
        assert "import pandas" not in text
        assert "import numpy" not in text


def test_authorized_m1_data_files_only_are_present() -> None:
    data_root = Path("src/regimequant/data")
    observed = {path for path in _all_python_files(data_root)}
    assert observed == IMPLEMENTED_M1_DATA_FILES


def test_authorized_m3_regime_files_only_are_present() -> None:
    regime_root = Path("src/regimequant/regimes")
    observed = {path for path in _all_python_files(regime_root)}
    assert observed == APPROVED_M3_REGIME_FILES
    assert "regimes" not in FUTURE_PACKAGES


def test_authorized_m2_feature_files_only_are_present() -> None:
    features_root = Path("src/regimequant/features")
    observed = {path for path in _all_python_files(features_root)}
    assert observed == IMPLEMENTED_M2_FEATURE_FILES


def test_authorized_m1_data_files_have_no_forbidden_imports() -> None:
    source_root = Path("src/regimequant")
    for file_path in IMPLEMENTED_M1_DATA_FILES:
        module = _read_ast(source_root / file_path.relative_to(Path("src/regimequant")))
        import_names = _iter_import_names(module)
        for import_name in import_names:
            for forbidden_prefix in FORBIDDEN_IMPORT_PREFIXES:
                assert not import_name.startswith(forbidden_prefix)


def test_no_forbidden_exports_in_m1_data_api() -> None:
    data_init = Path("src/regimequant/data/__init__.py")
    module = _read_ast(data_init)
    exported_names: set[str] = set()
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__" and isinstance(node.value, (ast.List, ast.Tuple)):
                    for element in node.value.elts:
                        if isinstance(element, ast.Constant) and isinstance(element.value, str):
                            exported_names.add(element.value)
    assert not (exported_names & FORBIDDEN_EXPORTED_APIS)


def test_no_forbidden_callable_definitions_in_m1_data_files() -> None:
    source_root = Path("src/regimequant")
    for file_path in IMPLEMENTED_M1_DATA_FILES:
        module = _read_ast(source_root / file_path.relative_to(Path("src/regimequant")))
        callables = {
            node.name
            for node in ast.walk(module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
        assert not (callables & FORBIDDEN_CALLABLE_NAMES)
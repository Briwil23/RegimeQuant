from __future__ import annotations

from pathlib import Path


def test_research_constitution_exists() -> None:
    assert Path("docs/RESEARCH_CONSTITUTION.md").is_file()


def test_readme_declares_m0_foundation() -> None:
    text = Path("README.md").read_text(encoding="utf-8")
    assert "M0 FOUNDATION" in text


def test_no_dashboard_package_in_m0() -> None:
    assert not Path("src/regimequant/dashboard").exists()

from __future__ import annotations

import regimequant


def test_package_imports() -> None:
    assert regimequant.__version__ == "0.0.0"


def test_core_research_question_present() -> None:
    expected = (
        "Can observable market structure and statistical regimes be identified using "
        "information available at the time, and can those regimes improve the "
        "robustness of systematic trading decisions out of sample?"
    )
    assert regimequant.RESEARCH_QUESTION == expected

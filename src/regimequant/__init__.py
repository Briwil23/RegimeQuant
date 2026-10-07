"""RegimeQuant foundational package.

M0 provides repository and research-governance foundations only.
No quantitative research models or trading logic are implemented here.
"""

__all__ = ["__version__", "RESEARCH_QUESTION", "MILESTONE_STATUS"]

__version__ = "0.0.0"

RESEARCH_QUESTION = (
    "Can observable market structure and statistical regimes be identified using "
    "information available at the time, and can those regimes improve the "
    "robustness of systematic trading decisions out of sample?"
)

MILESTONE_STATUS = "M0 FOUNDATION"

"""M7 cointegration contract implementation."""

from .adapter import IdentityAdapterError, VerifiedIdentityAdapter
from .contracts import (
    CointegrationCandidate,
    DecisionEnvelope,
    EngleGrangerResult,
    PricePolicyResult,
    ScientificHashPayload,
)
from .engle_granger import EngleGrangerError, canonical_engle_granger_test
from .identity import compute_canonical_pair_id, compute_research_instance_id, compute_scientific_hash
from .integration_screening import IntegrationScreeningError, screen_pair_for_i1
from .multiple_testing import BYAdjustmentError, by_fdr_adjustment
from .pair_selection import deterministic_pair_selection, select_canonical_pair
from .price_policy import DailyPricePolicy, PricePolicyError
from .validation import ValidationError, validate_decision_evidence

__all__ = [
    "BYAdjustmentError",
    "CointegrationCandidate",
    "DailyPricePolicy",
    "DecisionEnvelope",
    "EngleGrangerError",
    "EngleGrangerResult",
    "IdentityAdapterError",
    "IntegrationScreeningError",
    "PricePolicyError",
    "PricePolicyResult",
    "ScientificHashPayload",
    "ValidationError",
    "VerifiedIdentityAdapter",
    "by_fdr_adjustment",
    "canonical_engle_granger_test",
    "compute_canonical_pair_id",
    "compute_research_instance_id",
    "compute_scientific_hash",
    "deterministic_pair_selection",
    "screen_pair_for_i1",
    "select_canonical_pair",
    "validate_decision_evidence",
]

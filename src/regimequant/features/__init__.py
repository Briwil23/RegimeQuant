"""RegimeQuant M2 statistical feature engine public API."""

from .definitions import FeatureSpec, list_m2_v1_feature_specs, validate_feature_specs
from .engine import compute_features_from_returns
from .models import (
	FeatureComputationError,
	FeatureError,
	FeatureIdentityError,
	FeatureValidationError,
)

__all__ = [
	"FeatureComputationError",
	"FeatureError",
	"FeatureIdentityError",
	"FeatureSpec",
	"FeatureValidationError",
	"compute_features_from_returns",
	"list_m2_v1_feature_specs",
	"validate_feature_specs",
]

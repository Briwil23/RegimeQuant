from __future__ import annotations

import pytest

from regimequant.data import ReturnType
from regimequant.features.definitions import (
    FEATURE_ROLLING_MEAN,
    FeatureSpec,
    feature_instance_id,
    validate_feature_specs,
)
from regimequant.features.models import FeatureValidationError


def test_feature_instance_id_is_deterministic() -> None:
    spec = FeatureSpec(
        feature_name=FEATURE_ROLLING_MEAN,
        input_return_type=ReturnType.SIMPLE,
        window_size=5,
        min_observations=3,
    )
    assert feature_instance_id(spec) == feature_instance_id(spec)


def test_feature_instance_id_changes_when_output_affecting_parameter_changes() -> None:
    base = FeatureSpec(
        feature_name=FEATURE_ROLLING_MEAN,
        input_return_type=ReturnType.SIMPLE,
        window_size=5,
        min_observations=3,
    )
    changed = FeatureSpec(
        feature_name=FEATURE_ROLLING_MEAN,
        input_return_type=ReturnType.SIMPLE,
        window_size=6,
        min_observations=3,
    )
    assert feature_instance_id(base) != feature_instance_id(changed)


def test_duplicate_feature_spec_is_rejected() -> None:
    spec = FeatureSpec(feature_name=FEATURE_ROLLING_MEAN, window_size=3, min_observations=3)
    with pytest.raises(FeatureValidationError):
        validate_feature_specs([spec, spec])

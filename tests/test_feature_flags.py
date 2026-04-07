from __future__ import annotations

import pytest

from weft.feature_flags import FeatureFlags


def test_feature_flags_supports_rich18() -> None:
    ff = FeatureFlags.from_dict({"candidate_bank": "rich18", "multi_rounds": 2})
    assert ff.candidate_bank == "rich18"
    assert ff.multi_rounds == 2


def test_feature_flags_rejects_unknown_bank() -> None:
    with pytest.raises(ValueError):
        FeatureFlags.from_dict({"candidate_bank": "unknown"})


def test_feature_flags_phase4_controls() -> None:
    ff = FeatureFlags.from_dict(
        {
            "search_mode": "beam",
            "beam_width": 6,
            "mcmc_steps": 32,
            "stochastic_restarts": 2,
            "early_exit_patience": 3,
        }
    )
    assert ff.search_mode == "beam"
    assert ff.beam_width == 6
    assert ff.mcmc_steps == 32
    assert ff.stochastic_restarts == 2
    assert ff.early_exit_patience == 3


def test_feature_flags_maxcompute_fit_passes() -> None:
    ff = FeatureFlags.from_dict({"maxcompute_fit_passes": 5})
    assert ff.maxcompute_fit_passes == 5


def test_feature_flags_technique_levels_validate() -> None:
    ff = FeatureFlags.from_dict({"hierarchical_tiling_level": "mid", "decode_refinement_level": "max"})
    assert ff.hierarchical_tiling_level == "mid"
    assert ff.decode_refinement_level == "max"
    with pytest.raises(ValueError):
        FeatureFlags.from_dict({"stroke_objective_level": "bad"})

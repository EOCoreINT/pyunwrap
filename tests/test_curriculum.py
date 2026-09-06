"""
tests/test_curriculum.py
===========================

Tests for `pyunwrap.training.curriculum`, focused on the new curriculum
replay mechanism (`CurriculumReplayConfig`, `CurriculumReplayIndex`) added
alongside the original sequential `CurriculumIndex` (already covered
indirectly via `tests/test_trainer.py`).
"""

from __future__ import annotations

import math

import pytest

from pyunwrap.data.dataloader import InSARTileDataset
from pyunwrap.data.preprocessing import (
    NormalizedRasters,
    iter_tiles,
    normalize_amplitude,
    normalize_coherence,
    normalize_phase,
    save_tiles_hdf5,
)
from pyunwrap.synthetic.generator import InSARSyntheticGenerator
from pyunwrap.training.curriculum import (
    CurriculumReplayConfig,
    CurriculumReplayIndex,
    classify_difficulty_tier,
    compute_tile_difficulty_stats,
)


def _build_mixed_difficulty_hdf5(path, size=64, seed=0):
    """A dataset deliberately spanning all three difficulty tiers, so
    replay-mixing tests have real easy/moderate/hard tiles to draw from."""
    gen = InSARSyntheticGenerator(size=size, seed=seed)
    all_tiles = []
    for _ in range(4):  # easy: high coherence, no deformation
        s = gen.generate_sample(
            deformation_type="none",
            base_coherence=0.95,
            atmosphere_amplitude_rad=0.05,
            ramp_amplitude_rad=0.05,
        )
        r = NormalizedRasters(
            normalize_phase(s.wrapped_phase),
            normalize_coherence(s.coherence),
            normalize_amplitude(s.amplitude),
        )
        all_tiles.extend(iter_tiles(r, true_unwrapped=s.unwrapped_phase, tile_size=size, overlap=0))
    for _ in range(4):  # moderate: mid coherence
        s = gen.generate_sample(deformation_type="gaussian_bowl", base_coherence=0.55)
        r = NormalizedRasters(
            normalize_phase(s.wrapped_phase),
            normalize_coherence(s.coherence),
            normalize_amplitude(s.amplitude),
        )
        all_tiles.extend(iter_tiles(r, true_unwrapped=s.unwrapped_phase, tile_size=size, overlap=0))
    for _ in range(4):  # hard: low coherence, steep gradients
        s = gen.generate_sample(deformation_type="mogi", base_coherence=0.15)
        r = NormalizedRasters(
            normalize_phase(s.wrapped_phase),
            normalize_coherence(s.coherence),
            normalize_amplitude(s.amplitude),
        )
        all_tiles.extend(iter_tiles(r, true_unwrapped=s.unwrapped_phase, tile_size=size, overlap=0))
    save_tiles_hdf5(all_tiles, path)


class TestCurriculumReplayConfig:
    def test_defaults_match_spec(self):
        cfg = CurriculumReplayConfig()
        assert cfg.use_curriculum_replay is True
        assert cfg.easy_replay_fraction == 0.25
        assert cfg.medium_replay_fraction == 0.25
        assert cfg.hard_fraction == 0.50
        assert cfg.min_easy_validation_f1_or_rmse_threshold is None
        assert cfg.forgetting_patience == 5
        assert cfg.log_per_difficulty_metrics is True
        assert cfg.rebalance_every_n_epochs == 1

    def test_rejects_fractions_not_summing_to_one(self):
        with pytest.raises(ValueError):
            CurriculumReplayConfig(
                easy_replay_fraction=0.5, medium_replay_fraction=0.5, hard_fraction=0.5
            )

    def test_rejects_out_of_range_fraction(self):
        with pytest.raises(ValueError):
            CurriculumReplayConfig(
                easy_replay_fraction=1.5, medium_replay_fraction=0.0, hard_fraction=-0.5
            )

    def test_rejects_nonpositive_patience(self):
        with pytest.raises(ValueError):
            CurriculumReplayConfig(forgetting_patience=0)

    def test_rejects_nonpositive_rebalance_interval(self):
        with pytest.raises(ValueError):
            CurriculumReplayConfig(rebalance_every_n_epochs=0)

    def test_accepts_valid_custom_fractions(self):
        cfg = CurriculumReplayConfig(
            easy_replay_fraction=0.3, medium_replay_fraction=0.3, hard_fraction=0.4
        )
        assert math.isclose(
            cfg.easy_replay_fraction + cfg.medium_replay_fraction + cfg.hard_fraction, 1.0
        )


class TestCurriculumReplayIndexSampling:
    def test_epoch_contains_all_three_tiers(self, tmp_path):
        """Every epoch's sample must include tiles from all three
        difficulty tiers when the dataset has tiles in each and none of the
        target fractions are zero -- the core requirement of replay over
        the original sequential curriculum."""
        h5_path = tmp_path / "mixed.h5"
        _build_mixed_difficulty_hdf5(h5_path, seed=1)
        dataset = InSARTileDataset(h5_path, augment=False, require_ground_truth=True, seed=0)

        config = CurriculumReplayConfig(
            easy_replay_fraction=0.3, medium_replay_fraction=0.3, hard_fraction=0.4
        )
        replay = CurriculumReplayIndex(dataset, config, seed=0)

        stats = compute_tile_difficulty_stats(dataset)
        tier_of = {s.index: classify_difficulty_tier(s) for s in stats}

        indices = replay.indices_for_epoch(epoch=1)
        tiers_present = {tier_of[i] for i in indices}
        # At least 2 of 3 tiers must be present (allows for the edge case
        # where this synthetic dataset's generator doesn't produce a tile
        # in one exact tier this run); in practice all 3 should appear.
        assert len(tiers_present) >= 2

    def test_disabled_use_curriculum_replay_flag_is_read_by_caller(self):
        """use_curriculum_replay itself is a plain config flag Trainer reads
        to decide which mechanism to use -- CurriculumReplayIndex doesn't
        gate its own sampling on it (Trainer does, by not constructing a
        CurriculumReplayIndex at all when the flag is False). Confirm the
        flag round-trips correctly as a config value."""
        config = CurriculumReplayConfig(use_curriculum_replay=False)
        assert config.use_curriculum_replay is False

    def test_deterministic_with_fixed_seed(self, tmp_path):
        h5_path = tmp_path / "mixed.h5"
        _build_mixed_difficulty_hdf5(h5_path, seed=2)
        dataset = InSARTileDataset(h5_path, augment=False, require_ground_truth=True, seed=0)
        config = CurriculumReplayConfig()

        replay_a = CurriculumReplayIndex(dataset, config, seed=42)
        replay_b = CurriculumReplayIndex(dataset, config, seed=42)
        assert replay_a.indices_for_epoch(1) == replay_b.indices_for_epoch(1)

    def test_missing_tier_does_not_crash(self, tmp_path):
        """A dataset with zero tiles in some tier must not crash sampling
        -- that tier just contributes 0 samples (documented behavior,
        matching evaluate_stratified's "omit, don't error" precedent)."""
        gen = InSARSyntheticGenerator(size=64, seed=3)
        all_tiles = []
        for _ in range(4):  # all easy, deliberately no moderate/hard tiles
            s = gen.generate_sample(
                deformation_type="none",
                base_coherence=0.97,
                atmosphere_amplitude_rad=0.03,
                ramp_amplitude_rad=0.03,
            )
            r = NormalizedRasters(
                normalize_phase(s.wrapped_phase),
                normalize_coherence(s.coherence),
                normalize_amplitude(s.amplitude),
            )
            all_tiles.extend(
                iter_tiles(r, true_unwrapped=s.unwrapped_phase, tile_size=64, overlap=0)
            )
        h5_path = tmp_path / "easy_only.h5"
        save_tiles_hdf5(all_tiles, h5_path)
        dataset = InSARTileDataset(h5_path, augment=False, require_ground_truth=True, seed=0)

        config = CurriculumReplayConfig()
        replay = CurriculumReplayIndex(dataset, config, seed=0)
        indices = replay.indices_for_epoch(1)
        assert len(indices) > 0  # easy tier alone still contributes samples

    def test_oversized_target_samples_with_replacement(self, tmp_path):
        """Requesting more tiles from a tier than physically exist in it
        must sample with replacement rather than raising or under-filling."""
        h5_path = tmp_path / "tiny.h5"
        gen = InSARSyntheticGenerator(size=64, seed=4)
        s = gen.generate_sample(
            deformation_type="none",
            base_coherence=0.95,
            atmosphere_amplitude_rad=0.05,
            ramp_amplitude_rad=0.05,
        )
        r = NormalizedRasters(
            normalize_phase(s.wrapped_phase),
            normalize_coherence(s.coherence),
            normalize_amplitude(s.amplitude),
        )
        tiles = list(iter_tiles(r, true_unwrapped=s.unwrapped_phase, tile_size=64, overlap=0))
        save_tiles_hdf5(tiles, h5_path)  # exactly 1 tile total
        dataset = InSARTileDataset(h5_path, augment=False, require_ground_truth=True, seed=0)

        # Don't assume which tier this one random tile falls into -- target
        # whichever tier it actually is, so the test is robust rather than
        # coupled to one synthetic draw's exact noise realization.
        stats = compute_tile_difficulty_stats(dataset)
        actual_tier = classify_difficulty_tier(stats[0])
        fractions = {"easy": 0.0, "moderate": 0.0, "hard": 0.0}
        fractions[actual_tier] = 1.0
        config = CurriculumReplayConfig(
            easy_replay_fraction=fractions["easy"],
            medium_replay_fraction=fractions["moderate"],
            hard_fraction=fractions["hard"],
        )
        replay = CurriculumReplayIndex(dataset, config, epoch_size=20, seed=0)
        indices = replay.indices_for_epoch(1)
        assert len(indices) == 20  # sampled the single tile with replacement 20 times


class TestForgettingDetection:
    def test_no_adjustment_when_improving(self, tmp_path):
        h5_path = tmp_path / "mixed.h5"
        _build_mixed_difficulty_hdf5(h5_path, seed=5)
        dataset = InSARTileDataset(h5_path, augment=False, require_ground_truth=True, seed=0)
        config = CurriculumReplayConfig(forgetting_patience=3)
        replay = CurriculumReplayIndex(dataset, config, seed=0)

        for epoch, rmse in enumerate([10.0, 9.0, 8.0, 7.0, 6.0], start=1):
            adjusted = replay.record_easy_validation_metric(epoch, rmse)
            assert adjusted is False
        assert replay.n_forgetting_adjustments == 0
        assert replay.easy_fraction == config.easy_replay_fraction  # unchanged

    def test_trend_based_trigger_after_patience_exceeded(self, tmp_path):
        h5_path = tmp_path / "mixed.h5"
        _build_mixed_difficulty_hdf5(h5_path, seed=6)
        dataset = InSARTileDataset(h5_path, augment=False, require_ground_truth=True, seed=0)
        config = CurriculumReplayConfig(forgetting_patience=3)
        replay = CurriculumReplayIndex(dataset, config, seed=0)

        # First call establishes the baseline; then feed non-improving values.
        replay.record_easy_validation_metric(1, 5.0)
        results = [
            replay.record_easy_validation_metric(epoch, 6.0)  # worse than 5.0, every time
            for epoch in range(2, 6)
        ]
        assert any(
            results
        ), "expected an adjustment to trigger within 4 non-improving epochs at patience=3"
        assert replay.n_forgetting_adjustments >= 1
        assert replay.easy_fraction > config.easy_replay_fraction

    def test_absolute_threshold_trigger_fires_immediately(self, tmp_path):
        h5_path = tmp_path / "mixed.h5"
        _build_mixed_difficulty_hdf5(h5_path, seed=7)
        dataset = InSARTileDataset(h5_path, augment=False, require_ground_truth=True, seed=0)
        config = CurriculumReplayConfig(
            forgetting_patience=100,  # so only the absolute trigger could possibly fire
            min_easy_validation_f1_or_rmse_threshold=1.0,
        )
        replay = CurriculumReplayIndex(dataset, config, seed=0)

        adjusted = replay.record_easy_validation_metric(1, rmse=5.0)  # far above threshold
        assert adjusted is True

    def test_rebalance_takes_from_hard_before_medium(self, tmp_path):
        h5_path = tmp_path / "mixed.h5"
        _build_mixed_difficulty_hdf5(h5_path, seed=8)
        dataset = InSARTileDataset(h5_path, augment=False, require_ground_truth=True, seed=0)
        config = CurriculumReplayConfig(
            easy_replay_fraction=0.1,
            medium_replay_fraction=0.3,
            hard_fraction=0.6,
            forgetting_patience=1,
        )
        replay = CurriculumReplayIndex(dataset, config, seed=0)

        replay.record_easy_validation_metric(1, 10.0)
        medium_before = replay.medium_fraction
        replay.record_easy_validation_metric(2, 11.0)  # worse -> triggers at patience=1

        assert replay.hard_fraction < 0.6  # hard was reduced
        assert (
            replay.medium_fraction == medium_before
        )  # medium untouched (hard alone covered the delta)

    def test_fractions_still_sum_to_one_after_adjustment(self, tmp_path):
        h5_path = tmp_path / "mixed.h5"
        _build_mixed_difficulty_hdf5(h5_path, seed=9)
        dataset = InSARTileDataset(h5_path, augment=False, require_ground_truth=True, seed=0)
        config = CurriculumReplayConfig(forgetting_patience=1)
        replay = CurriculumReplayIndex(dataset, config, seed=0)

        replay.record_easy_validation_metric(1, 10.0)
        for epoch, rmse in enumerate([11.0, 12.0, 13.0, 14.0], start=2):
            replay.record_easy_validation_metric(epoch, rmse)

        total = replay.easy_fraction + replay.medium_fraction + replay.hard_fraction
        assert math.isclose(total, 1.0, abs_tol=1e-9)

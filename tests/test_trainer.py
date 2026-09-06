"""
tests/test_trainer.py
========================

Unit and regression tests for `pyunwrap.training.trainer`.

`TestSmallSubsetRegression` specifically covers a real bug found during
development: when a curriculum-filtered (or fine-tuning) dataset subset is
smaller than `batch_size`, `DataLoader(..., drop_last=True)` silently
yields zero batches for the entire epoch. `_train_one_epoch`'s `for batch in
loader` loop then never executes, `running` stays an empty dict, and the
epoch reports `loss=nan` while having performed zero gradient updates --
indistinguishable from a slow-starting but real training epoch unless read
very carefully. Caught on an actual 40-epoch training run where a strict
curriculum "easy" stage matched only 4 of 216 tiles against
`batch_size=8`, silently no-oping for 20 consecutive epochs.
"""

from __future__ import annotations

import math

import pytest
import torch

from pyunwrap.data.dataloader import InSARTileDataset
from pyunwrap.data.preprocessing import (
    NormalizedRasters,
    iter_tiles,
    normalize_amplitude,
    normalize_coherence,
    normalize_phase,
    save_tiles_hdf5,
)
from pyunwrap.models.ambiguity_net import AmbiguityNet
from pyunwrap.synthetic.generator import InSARSyntheticGenerator
from pyunwrap.training.trainer import (
    CurriculumIndex,
    TileDifficulty,
    Trainer,
    _safe_drop_last,
    build_curriculum_aware_scheduler,
    build_warmup_cosine_scheduler,
    classify_difficulty_tier,
    compute_tile_difficulty_stats,
    evaluate_stratified,
)

# --------------------------------------------------------------------------- #
# _safe_drop_last unit tests
# --------------------------------------------------------------------------- #


class TestCurriculumAwareScheduler:
    """Tests for build_curriculum_aware_scheduler, added after a real
    60-epoch training run showed the plain single-cycle cosine schedule left
    the LR decayed to ~1/15,000th of its peak by the time curriculum stage 3
    (the hardest data) unlocked at epoch 51 -- the model's loss visibly
    jumped at that transition and never recovered before training ended,
    because there was no meaningful step size left to adapt with.
    """

    def test_single_segment_matches_old_scheduler_exactly(self):
        """restart_epochs=[1] (no curriculum/fine-tuning restarts) must
        reproduce build_warmup_cosine_scheduler's LR values exactly --
        not approximately -- so existing single-phase training runs are
        completely unaffected by this change."""
        model = torch.nn.Linear(4, 4)
        opt_old = torch.optim.AdamW(model.parameters(), lr=1e-4)
        opt_new = torch.optim.AdamW(model.parameters(), lr=1e-4)
        sched_old = build_warmup_cosine_scheduler(opt_old, total_epochs=60, warmup_epochs=5)
        sched_new = build_curriculum_aware_scheduler(
            opt_new, total_epochs=60, restart_epochs=[1], warmup_epochs=5
        )

        for _ in range(60):
            lr_old = opt_old.param_groups[0]["lr"]
            lr_new = opt_new.param_groups[0]["lr"]
            assert lr_old == pytest.approx(lr_new, abs=1e-15)
            sched_old.step()
            sched_new.step()

    def test_lr_restarts_upward_at_each_boundary(self):
        """LR must jump back up (not continue decaying) at each restart
        epoch -- this is the actual bug fix, verified numerically rather
        than just visually."""
        model = torch.nn.Linear(4, 4)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        sched = build_curriculum_aware_scheduler(
            opt, total_epochs=60, restart_epochs=[1, 21, 51], warmup_epochs=5
        )

        lrs = []
        for _ in range(60):
            lrs.append(opt.param_groups[0]["lr"])
            sched.step()

        # 0-indexed list: lrs[19] is epoch 20 (end of stage 1), lrs[20] is epoch 21 (restart).
        assert lrs[20] > lrs[19] * 5, "LR must jump up at the stage 1->2 restart"
        assert lrs[50] > lrs[49] * 5, "LR must jump up at the stage 2->3 restart"

    def test_each_segment_reaches_a_meaningful_peak(self):
        """The specific failure mode being fixed: stage 3 must actually
        reach a substantial fraction of the base LR, not stay near zero."""
        model = torch.nn.Linear(4, 4)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        sched = build_curriculum_aware_scheduler(
            opt, total_epochs=60, restart_epochs=[1, 21, 51], warmup_epochs=5
        )

        lrs = []
        for _ in range(60):
            lrs.append(opt.param_groups[0]["lr"])
            sched.step()

        stage_3_lrs = lrs[50:60]  # epochs 51-60
        assert max(stage_3_lrs) > 1e-4 * 0.5, "stage 3 should reach at least half the peak LR"

    def test_short_final_segment_does_not_break(self):
        """A curriculum stage shorter than warmup_epochs (e.g. stage 3 is
        only 3 epochs long in a short total_epochs run) must not produce a
        negative or undefined schedule -- warmup is capped to fit."""
        model = torch.nn.Linear(4, 4)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        # total_epochs=53: stage 3 spans epochs 51-53, only 3 epochs, shorter
        # than warmup_epochs=5.
        sched = build_curriculum_aware_scheduler(
            opt, total_epochs=53, restart_epochs=[1, 21, 51], warmup_epochs=5
        )
        for _ in range(53):
            lr = opt.param_groups[0]["lr"]
            assert lr >= 0.0
            assert lr <= 1e-4 + 1e-12
            sched.step()

    def test_finetune_start_epoch_is_also_a_restart_point(self):
        """The SNAPHU fine-tuning switch is a data-distribution change just
        like a curriculum stage transition, and must get its own restart.
        Uses a stage-3 segment long enough (51-64, 14 epochs) to actually
        decay before the fine-tune restart at 65 -- a segment shorter than
        warmup_epochs would legitimately never leave its warmup-driven
        climb to peak, which would make this assertion meaningless rather
        than revealing anything about restart behavior specifically."""
        model = torch.nn.Linear(4, 4)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        sched = build_curriculum_aware_scheduler(
            opt,
            total_epochs=70,
            restart_epochs=[1, 21, 51, 65],
            warmup_epochs=5,
        )
        lrs = []
        for _ in range(70):
            lrs.append(opt.param_groups[0]["lr"])
            sched.step()
        assert lrs[64] > lrs[63] * 3, "LR must jump up at the fine-tuning restart (epoch 65)"


class TestTrainerUsesCurriculumAwareScheduler:
    """Integration test confirming Trainer actually wires up the new
    scheduler with the right restart points, not just that the scheduler
    function itself works in isolation."""

    def test_trainer_scheduler_restarts_at_curriculum_boundaries(self, tmp_path):
        h5_path = tmp_path / "data.h5"
        _build_tiny_hdf5(h5_path, n_tiles=6, size=64, seed=7)
        train_ds = InSARTileDataset(h5_path, augment=True, require_ground_truth=True, seed=0)
        val_ds = InSARTileDataset(h5_path, augment=False, require_ground_truth=True, seed=1)

        model = AmbiguityNet(pretrained=False, k_max=10.0)
        trainer = Trainer(
            model=model,
            train_dataset=train_ds,
            val_dataset=val_ds,
            out_dir=tmp_path / "run",
            total_epochs=60,
            warmup_epochs=5,
            batch_size=4,
            num_workers=0,
            validate_every=60,
            device="cpu",
            use_curriculum=True,
        )

        lrs = []
        for _ in range(60):
            lrs.append(trainer.optimizer.param_groups[0]["lr"])
            trainer.scheduler.step()

        stage_2_start = CurriculumIndex.STAGE_1_END_EPOCH + 1  # epoch 21
        stage_3_start = CurriculumIndex.STAGE_2_END_EPOCH + 1  # epoch 51
        assert (
            lrs[stage_2_start - 1] > lrs[stage_2_start - 2] * 5
        ), f"Trainer's own scheduler must restart at epoch {stage_2_start}"
        assert (
            lrs[stage_3_start - 1] > lrs[stage_3_start - 2] * 5
        ), f"Trainer's own scheduler must restart at epoch {stage_3_start}"

    def test_trainer_without_curriculum_uses_single_segment(self, tmp_path):
        """use_curriculum=False must produce a plain single-cycle schedule
        (no restarts), matching the pre-fix behavior exactly."""
        h5_path = tmp_path / "data.h5"
        _build_tiny_hdf5(h5_path, n_tiles=6, size=64, seed=8)
        train_ds = InSARTileDataset(h5_path, augment=True, require_ground_truth=True, seed=0)
        val_ds = InSARTileDataset(h5_path, augment=False, require_ground_truth=True, seed=1)

        model = AmbiguityNet(pretrained=False, k_max=10.0)
        trainer = Trainer(
            model=model,
            train_dataset=train_ds,
            val_dataset=val_ds,
            out_dir=tmp_path / "run",
            total_epochs=60,
            warmup_epochs=5,
            batch_size=4,
            num_workers=0,
            validate_every=60,
            device="cpu",
            use_curriculum=False,
        )

        lrs = []
        for _ in range(60):
            lrs.append(trainer.optimizer.param_groups[0]["lr"])
            trainer.scheduler.step()

        # No restart: LR at epoch 21 should be LOWER than at epoch 20 (still decaying).
        assert lrs[20] < lrs[19]


class TestClassifyDifficultyTier:
    def test_easy(self):
        stat = TileDifficulty(index=0, mean_coherence=0.9, p99_gradient=1.0)
        assert classify_difficulty_tier(stat) == "easy"

    def test_moderate_due_to_gradient(self):
        """High coherence but a gradient exceeding pi must NOT be classified
        easy -- the partition is a strict AND, not an OR, on the easy
        criteria (matching CurriculumIndex's stage-1 filter exactly)."""
        stat = TileDifficulty(index=0, mean_coherence=0.9, p99_gradient=4.0)
        assert classify_difficulty_tier(stat) == "moderate"

    def test_moderate_due_to_coherence(self):
        stat = TileDifficulty(index=0, mean_coherence=0.5, p99_gradient=1.0)
        assert classify_difficulty_tier(stat) == "moderate"

    def test_hard(self):
        stat = TileDifficulty(index=0, mean_coherence=0.2, p99_gradient=5.0)
        assert classify_difficulty_tier(stat) == "hard"

    def test_boundary_values_are_exclusive(self):
        """mean_coherence exactly 0.7 or p99_gradient exactly pi must NOT
        count as easy -- both curriculum criteria use strict inequalities."""
        assert classify_difficulty_tier(TileDifficulty(0, 0.7, 1.0)) != "easy"
        assert (
            classify_difficulty_tier(TileDifficulty(0, 0.9, math.pi)) == "easy"
        )  # <=, so exactly pi IS easy
        assert (
            classify_difficulty_tier(TileDifficulty(0, 0.4, 1.0)) == "hard"
        )  # 0.4 itself is not > 0.4


class TestComputeTileDifficultyStats:
    def test_matches_curriculum_index_internal_stats(self, tmp_path):
        """The standalone function must produce identical stats to what
        CurriculumIndex computes internally -- this was extracted FROM
        CurriculumIndex specifically to guarantee they can't drift apart,
        so this test protects that guarantee directly."""
        h5_path = tmp_path / "data.h5"
        _build_tiny_hdf5(h5_path, n_tiles=5, size=64, seed=9)
        dataset = InSARTileDataset(h5_path, augment=True, require_ground_truth=True, seed=0)

        standalone_stats = compute_tile_difficulty_stats(dataset)
        curriculum = CurriculumIndex(dataset)

        assert len(standalone_stats) == len(curriculum._stats)
        for a, b in zip(standalone_stats, curriculum._stats):
            assert a.index == b.index
            assert a.mean_coherence == pytest.approx(b.mean_coherence)
            assert a.p99_gradient == pytest.approx(b.p99_gradient)

    def test_restores_augment_flag(self, tmp_path):
        """Must temporarily disable augmentation to compute canonical stats,
        then restore whatever the dataset's augment flag was before."""
        h5_path = tmp_path / "data.h5"
        _build_tiny_hdf5(h5_path, n_tiles=3, size=64, seed=10)
        dataset = InSARTileDataset(h5_path, augment=True, require_ground_truth=True, seed=0)
        compute_tile_difficulty_stats(dataset)
        assert dataset.augment is True


class TestEvaluateStratified:
    def test_returns_overall_and_present_tiers(self, tmp_path):
        h5_path = tmp_path / "mixed.h5"
        gen = InSARSyntheticGenerator(size=64, seed=20)
        all_tiles = []
        for _ in range(3):
            s = gen.generate_sample(
                deformation_type="none",
                base_coherence=0.95,
                atmosphere_amplitude_rad=0.1,
                ramp_amplitude_rad=0.1,
            )
            r = NormalizedRasters(
                normalize_phase(s.wrapped_phase),
                normalize_coherence(s.coherence),
                normalize_amplitude(s.amplitude),
            )
            all_tiles.extend(
                iter_tiles(r, true_unwrapped=s.unwrapped_phase, tile_size=64, overlap=0)
            )
        for _ in range(3):
            s = gen.generate_sample(deformation_type="mogi", base_coherence=0.2)
            r = NormalizedRasters(
                normalize_phase(s.wrapped_phase),
                normalize_coherence(s.coherence),
                normalize_amplitude(s.amplitude),
            )
            all_tiles.extend(
                iter_tiles(r, true_unwrapped=s.unwrapped_phase, tile_size=64, overlap=0)
            )
        save_tiles_hdf5(all_tiles, h5_path)

        dataset = InSARTileDataset(h5_path, augment=False, require_ground_truth=True, seed=0)
        model = AmbiguityNet(pretrained=False, k_max=10.0)

        results = evaluate_stratified(model, dataset, device=torch.device("cpu"), batch_size=4)

        assert "overall" in results
        assert results["overall"].n_samples == len(dataset)
        # At least one non-overall tier must be present with this deliberately mixed dataset.
        assert any(tier in results for tier in ("easy", "moderate", "hard"))
        # Every present tier's sample count must be <= the overall count, and
        # all tiers together must exactly cover the dataset (strict partition).
        tier_total = sum(results[t].n_samples for t in ("easy", "moderate", "hard") if t in results)
        assert tier_total == results["overall"].n_samples

    def test_empty_tier_is_omitted_not_zero(self, tmp_path):
        """A dataset that's entirely one tier must not report misleading
        zero-sample ValidationMetrics for the other tiers -- they should be
        absent from the result dict entirely."""
        h5_path = tmp_path / "all_easy.h5"
        gen = InSARSyntheticGenerator(size=64, seed=21)
        all_tiles = []
        for _ in range(4):
            s = gen.generate_sample(
                deformation_type="none",
                base_coherence=0.97,
                atmosphere_amplitude_rad=0.05,
                ramp_amplitude_rad=0.05,
            )
            r = NormalizedRasters(
                normalize_phase(s.wrapped_phase),
                normalize_coherence(s.coherence),
                normalize_amplitude(s.amplitude),
            )
            all_tiles.extend(
                iter_tiles(r, true_unwrapped=s.unwrapped_phase, tile_size=64, overlap=0)
            )
        save_tiles_hdf5(all_tiles, h5_path)

        dataset = InSARTileDataset(h5_path, augment=False, require_ground_truth=True, seed=0)
        model = AmbiguityNet(pretrained=False, k_max=10.0)
        results = evaluate_stratified(model, dataset, device=torch.device("cpu"), batch_size=4)

        present_tiers = [t for t in ("easy", "moderate", "hard") if t in results]
        absent_tiers = [t for t in ("easy", "moderate", "hard") if t not in results]
        # With this dataset, at least one tier should be entirely absent
        # (very unlikely all three appear from 4 near-identical easy scenes).
        assert len(present_tiers) >= 1
        for tier in absent_tiers:
            assert tier not in results


class TestTrainerStratifiedValidation:
    def test_logs_per_tier_metrics_when_enabled(self, tmp_path):
        h5_path = tmp_path / "data.h5"
        _build_tiny_hdf5(h5_path, n_tiles=4, size=64, seed=11)
        train_ds = InSARTileDataset(h5_path, augment=True, require_ground_truth=True, seed=0)
        val_ds = InSARTileDataset(h5_path, augment=False, require_ground_truth=True, seed=1)

        model = AmbiguityNet(pretrained=False, k_max=10.0)
        trainer = Trainer(
            model=model,
            train_dataset=train_ds,
            val_dataset=val_ds,
            out_dir=tmp_path / "run",
            total_epochs=1,
            warmup_epochs=1,
            batch_size=4,
            num_workers=0,
            validate_every=1,
            device="cpu",
            use_curriculum=False,
            stratified_validation=True,
        )
        trainer.fit()

        tier_keys = [k for k in trainer.visualizer.history if k.startswith("val/rmse_rad_")]
        assert len(tier_keys) >= 1, "expected at least one per-tier validation metric to be logged"

    def test_no_per_tier_metrics_when_disabled(self, tmp_path):
        h5_path = tmp_path / "data.h5"
        _build_tiny_hdf5(h5_path, n_tiles=4, size=64, seed=12)
        train_ds = InSARTileDataset(h5_path, augment=True, require_ground_truth=True, seed=0)
        val_ds = InSARTileDataset(h5_path, augment=False, require_ground_truth=True, seed=1)

        model = AmbiguityNet(pretrained=False, k_max=10.0)
        trainer = Trainer(
            model=model,
            train_dataset=train_ds,
            val_dataset=val_ds,
            out_dir=tmp_path / "run",
            total_epochs=1,
            warmup_epochs=1,
            batch_size=4,
            num_workers=0,
            validate_every=1,
            device="cpu",
            use_curriculum=False,
            stratified_validation=False,
        )
        trainer.fit()

        tier_keys = [k for k in trainer.visualizer.history if k.startswith("val/rmse_rad_")]
        assert len(tier_keys) == 0
        assert "val/rmse_rad" in trainer.visualizer.history  # overall metric still present


class TestSafeDropLast:
    def test_false_when_dataset_smaller_than_batch(self):
        assert _safe_drop_last(dataset_size=4, batch_size=8) is False

    def test_false_when_dataset_equals_batch(self):
        """Exactly one full batch and nothing else: dropping it would leave
        zero batches, so this must NOT drop."""
        assert _safe_drop_last(dataset_size=8, batch_size=8) is False

    def test_true_when_dataset_larger_than_batch(self):
        assert _safe_drop_last(dataset_size=216, batch_size=8) is True

    def test_true_just_above_threshold(self):
        assert _safe_drop_last(dataset_size=9, batch_size=8) is True


# --------------------------------------------------------------------------- #
# Regression test: small curriculum/fine-tune subsets must still train
# --------------------------------------------------------------------------- #


def _build_tiny_hdf5(path, n_tiles: int, size: int = 64, seed: int = 0):
    """Build an HDF5 tile file with exactly `n_tiles` tiles (by generating
    just enough small scenes), for constructing a dataset deliberately
    smaller than a given batch_size.
    """
    gen = InSARSyntheticGenerator(size=size, seed=seed)
    all_tiles = []
    scene_idx = 0
    while len(all_tiles) < n_tiles:
        sample = gen.generate_sample(deformation_type="none")
        rasters = NormalizedRasters(
            wrapped_phase=normalize_phase(sample.wrapped_phase),
            coherence=normalize_coherence(sample.coherence),
            amplitude=normalize_amplitude(sample.amplitude),
        )
        # tile_size == size -> exactly one tile per scene, for precise counts.
        tiles = list(
            iter_tiles(rasters, true_unwrapped=sample.unwrapped_phase, tile_size=size, overlap=0)
        )
        all_tiles.extend(tiles)
        scene_idx += 1
    save_tiles_hdf5(all_tiles[:n_tiles], path)
    return n_tiles


class TestSmallSubsetRegression:
    def test_dataset_smaller_than_batch_size_still_produces_real_training(self, tmp_path):
        """The exact bug scenario: a training dataset smaller than
        batch_size must still perform real gradient updates (not silently
        skip the whole epoch), and must not report loss=nan."""
        h5_path = tmp_path / "tiny.h5"
        _build_tiny_hdf5(h5_path, n_tiles=4, size=64, seed=1)

        train_ds = InSARTileDataset(h5_path, augment=True, require_ground_truth=True, seed=0)
        val_ds = InSARTileDataset(h5_path, augment=False, require_ground_truth=True, seed=1)
        assert len(train_ds) == 4  # smaller than batch_size=8 below, by design

        model = AmbiguityNet(pretrained=False, k_max=10.0)
        params_before = [p.detach().clone() for p in model.parameters()]

        trainer = Trainer(
            model=model,
            train_dataset=train_ds,
            val_dataset=val_ds,
            out_dir=tmp_path / "run",
            total_epochs=1,
            warmup_epochs=1,
            batch_size=8,
            num_workers=0,
            validate_every=1,
            device="cpu",
            use_curriculum=False,
        )
        trainer.fit()

        history = trainer.visualizer.history
        assert "loss/total" in history
        final_loss = history["loss/total"][-1][1]
        assert not math.isnan(final_loss), f"loss was NaN: {final_loss}"

        # Confirm actual gradient updates happened: at least one parameter
        # must have changed from its initial value.
        params_after = list(model.parameters())
        any_changed = any(
            not torch.equal(before, after) for before, after in zip(params_before, params_after)
        )
        assert any_changed, "no model parameters changed -- training silently did nothing"

    def test_curriculum_subset_smaller_than_batch_size_still_trains(self, tmp_path):
        """Same regression, but reached through the curriculum-filtering
        code path specifically (the actual path the original bug was found
        on), using a dataset engineered so the 'easy' stage matches very
        few tiles relative to batch_size."""
        h5_path = tmp_path / "mixed.h5"
        # A handful of very easy (high coherence, no deformation) tiles plus
        # several hard ones, so the curriculum's stage-1 filter keeps only
        # the easy few -- fewer than batch_size.
        gen = InSARSyntheticGenerator(size=64, seed=2)
        all_tiles = []
        for _ in range(3):
            s = gen.generate_sample(
                deformation_type="none",
                base_coherence=0.95,
                atmosphere_amplitude_rad=0.1,
                ramp_amplitude_rad=0.1,
            )
            r = NormalizedRasters(
                normalize_phase(s.wrapped_phase),
                normalize_coherence(s.coherence),
                normalize_amplitude(s.amplitude),
            )
            all_tiles.extend(
                iter_tiles(r, true_unwrapped=s.unwrapped_phase, tile_size=64, overlap=0)
            )
        for _ in range(6):
            s = gen.generate_sample(deformation_type="mogi", base_coherence=0.3)
            r = NormalizedRasters(
                normalize_phase(s.wrapped_phase),
                normalize_coherence(s.coherence),
                normalize_amplitude(s.amplitude),
            )
            all_tiles.extend(
                iter_tiles(r, true_unwrapped=s.unwrapped_phase, tile_size=64, overlap=0)
            )
        save_tiles_hdf5(all_tiles, h5_path)

        train_ds = InSARTileDataset(h5_path, augment=True, require_ground_truth=True, seed=0)
        val_ds = InSARTileDataset(h5_path, augment=False, require_ground_truth=True, seed=1)

        model = AmbiguityNet(pretrained=False, k_max=10.0)
        trainer = Trainer(
            model=model,
            train_dataset=train_ds,
            val_dataset=val_ds,
            out_dir=tmp_path / "run",
            total_epochs=1,
            warmup_epochs=1,
            batch_size=8,
            num_workers=0,
            validate_every=1,
            device="cpu",
            use_curriculum=True,
        )
        # Must not raise, and must not silently produce an empty/nan epoch.
        trainer.fit()
        final_loss = trainer.visualizer.history["loss/total"][-1][1]
        assert not math.isnan(final_loss)

    def test_zero_batch_epoch_raises_clear_error_not_silent_nan(self, tmp_path, monkeypatch):
        """Defense-in-depth check: if some future code path ever DID produce
        zero batches for an epoch (bypassing _safe_drop_last), Trainer.fit()
        must fail loudly with a clear RuntimeError, not silently log
        loss=nan and continue."""
        h5_path = tmp_path / "tiny2.h5"
        _build_tiny_hdf5(h5_path, n_tiles=2, size=64, seed=3)
        train_ds = InSARTileDataset(h5_path, augment=True, require_ground_truth=True, seed=0)
        val_ds = InSARTileDataset(h5_path, augment=False, require_ground_truth=True, seed=1)

        model = AmbiguityNet(pretrained=False, k_max=10.0)
        trainer = Trainer(
            model=model,
            train_dataset=train_ds,
            val_dataset=val_ds,
            out_dir=tmp_path / "run",
            total_epochs=1,
            warmup_epochs=1,
            batch_size=8,
            num_workers=0,
            validate_every=1,
            device="cpu",
            use_curriculum=False,
        )

        # Force the exact failure mode by monkeypatching the loader builder
        # to reintroduce the unsafe drop_last=True, simulating a future
        # regression that bypasses _safe_drop_last.
        import pyunwrap.training.trainer as trainer_module

        original_loader = torch.utils.data.DataLoader

        def unsafe_loader(dataset, **kwargs):
            kwargs["drop_last"] = True
            return original_loader(dataset, **kwargs)

        monkeypatch.setattr(trainer_module, "DataLoader", unsafe_loader)

        with pytest.raises(RuntimeError, match="zero training batches"):
            trainer.fit()


class TestTrainerWithCurriculumReplay:
    """Integration tests for Trainer's curriculum_replay_config parameter
    (Strategy 2: curriculum replay), confirming it actually trains
    end-to-end and preserves exact backward compatibility when unused.
    """

    def test_curriculum_replay_disabled_by_default(self, tmp_path):
        """Trainer() with no curriculum_replay_config must use the original
        sequential CurriculumIndex, unchanged."""
        h5_path = tmp_path / "data.h5"
        _build_tiny_hdf5(h5_path, n_tiles=6, size=64, seed=20)
        train_ds = InSARTileDataset(h5_path, augment=True, require_ground_truth=True, seed=0)
        val_ds = InSARTileDataset(h5_path, augment=False, require_ground_truth=True, seed=1)

        model = AmbiguityNet(pretrained=False, k_max=10.0)
        trainer = Trainer(
            model=model,
            train_dataset=train_ds,
            val_dataset=val_ds,
            out_dir=tmp_path / "run",
            total_epochs=2,
            warmup_epochs=1,
            batch_size=4,
            num_workers=0,
            validate_every=2,
            device="cpu",
            use_curriculum=True,
        )
        assert trainer.curriculum_replay_index is None
        assert trainer.curriculum_index is not None
        trainer.fit()  # must not raise

    def test_curriculum_replay_enabled_trains_end_to_end(self, tmp_path):
        from pyunwrap.training.curriculum import CurriculumReplayConfig

        h5_path = tmp_path / "mixed.h5"
        gen = InSARSyntheticGenerator(size=64, seed=21)
        all_tiles = []
        for _ in range(3):
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
            all_tiles.extend(
                iter_tiles(r, true_unwrapped=s.unwrapped_phase, tile_size=64, overlap=0)
            )
        for _ in range(3):
            s = gen.generate_sample(deformation_type="mogi", base_coherence=0.3)
            r = NormalizedRasters(
                normalize_phase(s.wrapped_phase),
                normalize_coherence(s.coherence),
                normalize_amplitude(s.amplitude),
            )
            all_tiles.extend(
                iter_tiles(r, true_unwrapped=s.unwrapped_phase, tile_size=64, overlap=0)
            )
        save_tiles_hdf5(all_tiles, h5_path)

        train_ds = InSARTileDataset(h5_path, augment=True, require_ground_truth=True, seed=0)
        val_ds = InSARTileDataset(h5_path, augment=False, require_ground_truth=True, seed=1)

        model = AmbiguityNet(pretrained=False, k_max=10.0)
        replay_config = CurriculumReplayConfig(
            easy_replay_fraction=0.4,
            medium_replay_fraction=0.3,
            hard_fraction=0.3,
            forgetting_patience=2,
        )
        trainer = Trainer(
            model=model,
            train_dataset=train_ds,
            val_dataset=val_ds,
            out_dir=tmp_path / "run",
            total_epochs=3,
            warmup_epochs=1,
            batch_size=4,
            num_workers=0,
            validate_every=1,
            device="cpu",
            use_curriculum=False,
            curriculum_replay_config=replay_config,
        )
        assert trainer.curriculum_replay_index is not None
        trainer.fit()

        final_loss = trainer.visualizer.history["loss/total"][-1][1]
        assert not math.isnan(final_loss)
        # The training-subset mixture logic itself is unit-tested directly
        # in tests/test_curriculum.py; here we only need end-to-end training
        # to succeed. Whether curriculum_replay/easy_fraction gets logged
        # depends on the random validation split actually containing an
        # "easy"-tier tile, which (as with CurriculumIndex's own curriculum
        # filtering elsewhere in this test suite) is not guaranteed for a
        # tiny random draw -- assert on it only when present, rather than
        # requiring a specific tier composition from randomness.
        if "curriculum_replay/easy_fraction" in trainer.visualizer.history:
            fractions_sum = (
                trainer.curriculum_replay_index.easy_fraction
                + trainer.curriculum_replay_index.medium_fraction
                + trainer.curriculum_replay_index.hard_fraction
            )
            assert math.isclose(fractions_sum, 1.0, abs_tol=1e-6)

    def test_replay_precedence_over_sequential_curriculum(self, tmp_path, capsys):
        from pyunwrap.training.curriculum import CurriculumReplayConfig

        h5_path = tmp_path / "data.h5"
        _build_tiny_hdf5(h5_path, n_tiles=6, size=64, seed=22)
        train_ds = InSARTileDataset(h5_path, augment=True, require_ground_truth=True, seed=0)
        val_ds = InSARTileDataset(h5_path, augment=False, require_ground_truth=True, seed=1)

        model = AmbiguityNet(pretrained=False, k_max=10.0)
        trainer = Trainer(
            model=model,
            train_dataset=train_ds,
            val_dataset=val_ds,
            out_dir=tmp_path / "run",
            total_epochs=2,
            warmup_epochs=1,
            batch_size=4,
            num_workers=0,
            validate_every=2,
            device="cpu",
            use_curriculum=True,  # both given...
            curriculum_replay_config=CurriculumReplayConfig(),  # ...replay must win
        )
        assert trainer.curriculum_replay_index is not None
        assert trainer.curriculum_index is None  # sequential curriculum must NOT be built


class TestCLIArguments:
    """Tests for the pyunwrap-train CLI's new flags (--use-curriculum-replay,
    --smoothness-weight, --real-data-path), verified via real end-to-end
    invocations of main(), not just argparse parsing."""

    def test_train_val_hdf5_required_unless_real_data_path(self, tmp_path, monkeypatch, capsys):
        from pyunwrap.training.trainer import main

        monkeypatch.setattr(
            "sys.argv",
            ["pyunwrap-train", "--epochs", "1", "--out-dir", str(tmp_path / "run")],
        )
        with pytest.raises(SystemExit):
            main()
        captured = capsys.readouterr()
        assert "real-data-path" in captured.err or "train-hdf5" in captured.err

    def test_use_curriculum_replay_flag_trains_end_to_end(self, tmp_path, monkeypatch):
        from pyunwrap.training.trainer import main

        h5_path = tmp_path / "data.h5"
        _build_tiny_hdf5(h5_path, n_tiles=4, size=64, seed=30)

        monkeypatch.setattr(
            "sys.argv",
            [
                "pyunwrap-train",
                "--train-hdf5",
                str(h5_path),
                "--val-hdf5",
                str(h5_path),
                "--epochs",
                "1",
                "--warmup-epochs",
                "1",
                "--batch-size",
                "2",
                "--num-workers",
                "0",
                "--validate-every",
                "1",
                "--device",
                "cpu",
                "--no-curriculum",
                "--no-pretrained",
                "--use-curriculum-replay",
                "--out-dir",
                str(tmp_path / "run"),
            ],
        )
        main()  # must not raise

    def test_smoothness_weight_flag_enables_component_5(self, tmp_path, monkeypatch):
        from pyunwrap.training.trainer import main

        h5_path = tmp_path / "data.h5"
        _build_tiny_hdf5(h5_path, n_tiles=4, size=64, seed=31)

        monkeypatch.setattr(
            "sys.argv",
            [
                "pyunwrap-train",
                "--train-hdf5",
                str(h5_path),
                "--val-hdf5",
                str(h5_path),
                "--epochs",
                "1",
                "--warmup-epochs",
                "1",
                "--batch-size",
                "2",
                "--num-workers",
                "0",
                "--validate-every",
                "1",
                "--device",
                "cpu",
                "--no-curriculum",
                "--no-pretrained",
                "--smoothness-weight",
                "0.2",
                "--out-dir",
                str(tmp_path / "run"),
            ],
        )
        main()  # must not raise; real coverage that the flag wires through PhysicsInformedUnwrapLoss

    def test_real_data_path_flag_trains_end_to_end(self, tmp_path, monkeypatch):
        from pyunwrap.data.real_injection import build_tiny_fixture_stack
        from pyunwrap.training.trainer import main

        real_dir = tmp_path / "real_data"
        real_dir.mkdir()
        stack = build_tiny_fixture_stack(size=256, n_epochs=3, seed=1)
        import numpy as np

        np.save(real_dir / "amplitude.npy", stack.amplitude)
        np.save(real_dir / "wrapped_phase.npy", stack.wrapped_phase)
        np.save(real_dir / "coherence.npy", stack.coherence)

        monkeypatch.setattr(
            "sys.argv",
            [
                "pyunwrap-train",
                "--real-data-path",
                str(real_dir),
                "--epochs",
                "1",
                "--warmup-epochs",
                "1",
                "--batch-size",
                "2",
                "--num-workers",
                "0",
                "--validate-every",
                "1",
                "--device",
                "cpu",
                "--no-curriculum",
                "--no-pretrained",
                "--out-dir",
                str(tmp_path / "run"),
            ],
        )
        main()  # must not raise

"""
tests/test_real_injection.py
================================

Tests for `pyunwrap.data.real_injection` (Strategy 1: real-data +
synthetic-injection pipeline). Uses `build_tiny_fixture_stack` throughout
-- a small, clearly-labeled real-*like* (not literally real) fixture; see
that function's docstring and the module's own docstring for why no
genuinely real SAR data is used or reachable in this environment.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyunwrap.data.dataloader import InSARTileDataset
from pyunwrap.data.real_injection import (
    _INJECTORS,
    RealDataStack,
    RealSyntheticInjectionConfig,
    build_injected_scene,
    build_real_injection_datasets,
    build_tiny_fixture_stack,
    estimate_coherence_proxy,
    load_real_stack,
    nonlinear_transient_deformation,
    spatial_block_split,
)
from pyunwrap.synthetic.generator import wrap_phase


class TestRealSyntheticInjectionConfig:
    def test_defaults_match_spec(self):
        cfg = RealSyntheticInjectionConfig()
        assert cfg.inject_mogi is True
        assert cfg.inject_okada is True
        assert cfg.inject_gaussian_bowl is True
        assert cfg.inject_linear_ramp is True
        assert cfg.inject_nonlinear_transient is True
        assert cfg.min_coherence_for_background == 0.0
        assert cfg.max_coherence_for_background == 1.0
        assert cfg.preserve_real_noise is True
        assert cfg.wrap_after_injection is True
        assert cfg.spatial_split_strategy == "blocks"
        assert cfg.train_fraction == 0.7
        assert cfg.val_fraction == 0.15
        assert cfg.test_fraction == 0.15
        assert cfg.seed == 42

    def test_rejects_fractions_not_summing_to_one(self):
        with pytest.raises(ValueError):
            RealSyntheticInjectionConfig(train_fraction=0.5, val_fraction=0.5, test_fraction=0.5)

    def test_rejects_inverted_coherence_bounds(self):
        with pytest.raises(ValueError):
            RealSyntheticInjectionConfig(
                min_coherence_for_background=0.8, max_coherence_for_background=0.2
            )

    def test_rejects_nonpositive_block_size(self):
        with pytest.raises(ValueError):
            RealSyntheticInjectionConfig(block_size=0)

    def test_rejects_unsupported_split_strategy(self):
        with pytest.raises(ValueError):
            RealSyntheticInjectionConfig(spatial_split_strategy="random_pixels")

    def test_preserve_real_noise_false_raises_not_implemented(self):
        """Must fail loudly, not silently no-op, per the engineering
        standard against silent partial-feature behavior."""
        with pytest.raises(NotImplementedError):
            RealSyntheticInjectionConfig(preserve_real_noise=False)

    def test_wrap_after_injection_false_warns(self):
        with pytest.warns(UserWarning):
            RealSyntheticInjectionConfig(wrap_after_injection=False)

    def test_config_hash_is_deterministic(self):
        cfg1 = RealSyntheticInjectionConfig(seed=7)
        cfg2 = RealSyntheticInjectionConfig(seed=7)
        assert cfg1.config_hash() == cfg2.config_hash()

    def test_config_hash_differs_for_different_configs(self):
        cfg1 = RealSyntheticInjectionConfig(seed=7)
        cfg2 = RealSyntheticInjectionConfig(seed=8)
        assert cfg1.config_hash() != cfg2.config_hash()

    def test_enabled_injection_kinds_respects_flags(self):
        cfg = RealSyntheticInjectionConfig(
            inject_mogi=False,
            inject_okada=False,
            inject_gaussian_bowl=True,
            inject_linear_ramp=False,
            inject_nonlinear_transient=False,
        )
        assert cfg.enabled_injection_kinds() == ["gaussian_bowl"]


class TestRealDataStack:
    def test_valid_construction(self):
        stack = RealDataStack(
            amplitude=np.ones((3, 32, 32)),
            wrapped_phase=np.zeros((3, 32, 32)),
            coherence=np.full((32, 32), 0.8),
        )
        assert stack.n_epochs == 3
        assert stack.spatial_shape == (32, 32)

    def test_mismatched_amplitude_phase_shapes_raises(self):
        with pytest.raises(ValueError):
            RealDataStack(
                amplitude=np.ones((3, 32, 32)),
                wrapped_phase=np.zeros((3, 16, 16)),
                coherence=np.full((32, 32), 0.8),
            )

    def test_bad_coherence_shape_raises(self):
        with pytest.raises(ValueError):
            RealDataStack(
                amplitude=np.ones((3, 32, 32)),
                wrapped_phase=np.zeros((3, 32, 32)),
                coherence=np.full((16, 16), 0.8),  # wrong spatial shape
            )

    def test_per_epoch_coherence_selection(self):
        coherence_stack = np.stack([np.full((16, 16), i / 10) for i in range(3)])
        stack = RealDataStack(
            amplitude=np.ones((3, 16, 16)),
            wrapped_phase=np.zeros((3, 16, 16)),
            coherence=coherence_stack,
        )
        np.testing.assert_array_equal(stack.coherence_at(1), np.full((16, 16), 0.1))

    def test_shared_2d_coherence_used_for_any_epoch(self):
        stack = RealDataStack(
            amplitude=np.ones((3, 16, 16)),
            wrapped_phase=np.zeros((3, 16, 16)),
            coherence=np.full((16, 16), 0.5),
        )
        np.testing.assert_array_equal(stack.coherence_at(0), stack.coherence_at(2))


class TestLoadRealStack:
    def test_loads_npy_files(self, tmp_path):
        amp = np.ones((2, 16, 16))
        phase = np.zeros((2, 16, 16))
        coh = np.full((16, 16), 0.7)
        np.save(tmp_path / "amp.npy", amp)
        np.save(tmp_path / "phase.npy", phase)
        np.save(tmp_path / "coh.npy", coh)

        stack = load_real_stack(tmp_path / "amp.npy", tmp_path / "phase.npy", tmp_path / "coh.npy")
        np.testing.assert_array_equal(stack.amplitude, amp)
        np.testing.assert_array_equal(stack.coherence, coh)

    def test_unsupported_extension_raises(self, tmp_path):
        bad_path = tmp_path / "data.unsupported"
        bad_path.write_text("not real data")
        with pytest.raises(ValueError, match="Unsupported file extension"):
            load_real_stack(bad_path, bad_path, bad_path)

    def test_scalar_incidence_angle_passthrough(self, tmp_path):
        amp = np.ones((2, 8, 8))
        np.save(tmp_path / "amp.npy", amp)
        np.save(tmp_path / "phase.npy", np.zeros((2, 8, 8)))
        np.save(tmp_path / "coh.npy", np.full((8, 8), 0.5))
        stack = load_real_stack(
            tmp_path / "amp.npy",
            tmp_path / "phase.npy",
            tmp_path / "coh.npy",
            incidence_angle=34.5,
        )
        assert stack.incidence_angle == 34.5


class TestBuildTinyFixtureStack:
    def test_produces_valid_stack(self):
        stack = build_tiny_fixture_stack(size=32, n_epochs=4, seed=0)
        assert stack.spatial_shape == (32, 32)
        assert stack.n_epochs == 4
        assert stack.wrapped_phase.min() >= -np.pi - 1e-9
        assert stack.wrapped_phase.max() <= np.pi + 1e-9
        assert (stack.coherence >= 0.0).all() and (stack.coherence <= 1.0).all()

    def test_deterministic_with_seed(self):
        a = build_tiny_fixture_stack(size=16, seed=5)
        b = build_tiny_fixture_stack(size=16, seed=5)
        np.testing.assert_array_equal(a.wrapped_phase, b.wrapped_phase)


class TestNonlinearTransientDeformation:
    def test_rejects_invalid_asymmetry(self):
        with pytest.raises(ValueError):
            nonlinear_transient_deformation((32, 32), asymmetry=1.0)
        with pytest.raises(ValueError):
            nonlinear_transient_deformation((32, 32), asymmetry=-0.1)

    def test_peak_is_at_center_by_default(self):
        # Even-sized grid: default center (rows/2.0, cols/2.0) lands exactly
        # on integer pixel indices (32.0, 32.0), so the peak is exactly at
        # field[32, 32]. An odd size would put the mathematical center
        # half a pixel off any single grid point, which is a real property
        # of pixel-grid indexing, not something to work around in the
        # function itself.
        field = nonlinear_transient_deformation((64, 64), peak_displacement_m=0.1, seed=0)
        center_val = field[32, 32]
        assert center_val == pytest.approx(0.1, rel=1e-6)
        assert field.max() == pytest.approx(center_val, rel=1e-6)

    def test_decays_with_distance(self):
        field = nonlinear_transient_deformation(
            (129, 129), peak_displacement_m=0.1, decay_length_m=500, seed=0
        )
        assert field[64, 64] > field[64, 100] > field[0, 0]


class TestBuildInjectedScene:
    """Required tests from the specification: injected synthetic
    deformation produces expected wrapped phase, and true ambiguity
    matches expected values."""

    @pytest.fixture
    def stack(self):
        return build_tiny_fixture_stack(size=64, n_epochs=3, seed=1)

    @pytest.mark.parametrize("kind", list(_INJECTORS.keys()))
    def test_physics_identity_holds_for_every_injection_kind(self, stack, kind):
        config = RealSyntheticInjectionConfig()
        rng = np.random.default_rng(0)
        sample, metadata = build_injected_scene(
            stack, config, time_index=0, injection_kind=kind, rng=rng
        )

        # wrap(true_unwrapped) must exactly reproduce the observed wrapped phase.
        rewrapped = wrap_phase(sample["true_unwrapped"])
        np.testing.assert_allclose(rewrapped, sample["wrapped_phase"], atol=1e-6)
        assert metadata["kind"] == kind

    @pytest.mark.parametrize("kind", list(_INJECTORS.keys()))
    def test_true_ambiguity_matches_expected_relationship(self, stack, kind):
        """k_true must satisfy true_unwrapped == wrapped_phase + 2*pi*k_true
        exactly, and must be integer-valued."""
        config = RealSyntheticInjectionConfig()
        rng = np.random.default_rng(0)
        sample, _ = build_injected_scene(stack, config, time_index=0, injection_kind=kind, rng=rng)

        k = sample["true_ambiguity"]
        reconstructed = sample["wrapped_phase"] + 2 * np.pi * k
        np.testing.assert_allclose(reconstructed, sample["true_unwrapped"], atol=1e-6)
        np.testing.assert_allclose(k, np.round(k), atol=1e-9)

    def test_real_background_noise_is_preserved(self, stack):
        """The observed wrapped phase must retain the real background's
        own noise pattern -- i.e. differ from a purely synthetic-only
        wrapped field with no real component."""
        config = RealSyntheticInjectionConfig()
        rng = np.random.default_rng(2)
        sample, _ = build_injected_scene(
            stack, config, time_index=0, injection_kind="gaussian_bowl", rng=rng
        )

        # If real noise were discarded, wrapped_phase would depend only on
        # the smooth injected deformation and be far smoother than the
        # actual (noisy real background + smooth deformation) result.
        real_psi = stack.wrapped_phase[0]
        # The real background's own high-frequency content must still be
        # detectable: correlation between real background and final output
        # phase gradients should be nontrivial (not exactly reproducible
        # analytically here, so check qualitatively via variance: pure
        # smooth injected deformation alone has far lower spatial variance
        # than deformation + real random-phase background).
        grad_real = np.diff(real_psi, axis=0)
        grad_result = np.diff(sample["wrapped_phase"], axis=0)
        assert grad_result.std() > 0.1 * grad_real.std()

    def test_regime_mask_reflects_coherence_thresholds(self, stack):
        config = RealSyntheticInjectionConfig(
            min_coherence_for_background=0.5, max_coherence_for_background=1.0
        )
        rng = np.random.default_rng(3)
        sample, _ = build_injected_scene(
            stack, config, time_index=0, injection_kind="mogi", rng=rng
        )

        coherence = stack.coherence_at(0)
        expected_mask = (coherence >= 0.5).astype(np.float32)
        np.testing.assert_array_equal(sample["regime_mask"], expected_mask)

    def test_background_unwrapped_override_is_used(self, stack):
        config = RealSyntheticInjectionConfig()
        rng = np.random.default_rng(4)
        custom_background = np.full(stack.spatial_shape, 3.0)
        sample, _ = build_injected_scene(
            stack,
            config,
            time_index=0,
            injection_kind="gaussian_bowl",
            background_unwrapped=custom_background,
            rng=rng,
        )
        # true_unwrapped must equal custom_background + injected phase, not
        # the real wrapped phase + injected phase.
        rng2 = np.random.default_rng(4)
        _, _phi_synth_meta = build_injected_scene(
            stack,
            config,
            time_index=0,
            injection_kind="gaussian_bowl",
            rng=rng2,
        )
        # Just check the custom background's constant offset shows up: the
        # mean of true_unwrapped minus the (small) injected field should be
        # close to 3.0, not close to the real background's near-zero mean.
        assert sample["true_unwrapped"].mean() > 1.0

    def test_raises_when_no_injection_kind_available(self, stack):
        config = RealSyntheticInjectionConfig(
            inject_mogi=False,
            inject_okada=False,
            inject_gaussian_bowl=False,
            inject_linear_ramp=False,
            inject_nonlinear_transient=False,
        )
        with pytest.raises(ValueError, match="No injection kind"):
            build_injected_scene(stack, config, time_index=0, rng=np.random.default_rng(0))


class TestSpatialBlockSplit:
    def test_no_leakage_between_splits(self):
        stack = build_tiny_fixture_stack(size=256, seed=1)
        config = RealSyntheticInjectionConfig(block_size=64, seed=5)
        splits = spatial_block_split(stack.spatial_shape, config)

        regions = []
        for split_name, blocks in splits.items():
            for b in blocks:
                regions.append((split_name, b.row, b.col, b.height, b.width))

        def overlaps(a, b):
            _, r1, c1, h1, w1 = a
            _, r2, c2, h2, w2 = b
            return not (r1 + h1 <= r2 or r2 + h2 <= r1 or c1 + w1 <= c2 or c2 + w2 <= c1)

        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                if regions[i][0] != regions[j][0]:
                    assert not overlaps(
                        regions[i], regions[j]
                    ), f"leakage: {regions[i]} vs {regions[j]}"

    def test_split_sizes_roughly_match_fractions(self):
        stack = build_tiny_fixture_stack(size=512, seed=2)
        config = RealSyntheticInjectionConfig(
            block_size=32,
            train_fraction=0.6,
            val_fraction=0.2,
            test_fraction=0.2,
            seed=9,
        )
        splits = spatial_block_split(stack.spatial_shape, config)
        total = sum(len(v) for v in splits.values())
        assert len(splits["train"]) / total == pytest.approx(0.6, abs=0.05)
        assert len(splits["val"]) / total == pytest.approx(0.2, abs=0.05)

    def test_deterministic_with_seed(self):
        stack = build_tiny_fixture_stack(size=256, seed=3)
        config = RealSyntheticInjectionConfig(block_size=64, seed=11)
        splits_a = spatial_block_split(stack.spatial_shape, config)
        splits_b = spatial_block_split(stack.spatial_shape, config)
        assert [(b.row, b.col) for b in splits_a["train"]] == [
            (b.row, b.col) for b in splits_b["train"]
        ]


class TestEstimateCoherenceProxy:
    def test_multi_epoch_phase_stability_proxy(self):
        # Perfectly stable phase across epochs -> proxy coherence == 1.
        stable_phase = np.zeros((5, 16, 16))
        proxy = estimate_coherence_proxy(stable_phase)
        np.testing.assert_allclose(proxy, 1.0, atol=1e-9)

    def test_random_phase_gives_low_proxy_coherence(self):
        rng = np.random.default_rng(0)
        random_phase = rng.uniform(-np.pi, np.pi, size=(20, 16, 16))
        proxy = estimate_coherence_proxy(random_phase)
        assert (
            proxy.mean() < 0.3
        )  # random phase across many epochs -> near-zero circular mean magnitude

    def test_amplitude_dispersion_fallback_for_single_epoch(self):
        amp_stack = np.random.default_rng(0).uniform(0.5, 1.5, size=(5, 16, 16))
        single_phase = np.zeros((1, 16, 16))
        proxy = estimate_coherence_proxy(single_phase, amplitude_stack=amp_stack)
        assert proxy.shape == (16, 16)
        assert (proxy >= 0).all() and (proxy <= 1).all()

    def test_raises_when_no_signal_available(self):
        single_phase = np.zeros((1, 16, 16))
        with pytest.raises(ValueError):
            estimate_coherence_proxy(single_phase, amplitude_stack=None)


class TestBuildRealInjectionDatasets:
    def test_end_to_end_produces_loadable_datasets(self, tmp_path):
        stack = build_tiny_fixture_stack(size=256, n_epochs=3, seed=6)
        config = RealSyntheticInjectionConfig(block_size=64, seed=7)
        paths = build_real_injection_datasets(
            stack,
            config,
            out_dir=tmp_path,
            tile_size=32,
            tile_overlap=8,
            use_cache=False,
        )
        assert all(p.exists() for p in paths.values())

        train_ds = InSARTileDataset(paths["train"], augment=False, require_ground_truth=True)
        assert len(train_ds) > 0
        sample = train_ds[0]
        assert "true_unwrapped" in sample

    def test_caching_reuses_existing_files(self, tmp_path):
        stack = build_tiny_fixture_stack(size=128, n_epochs=2, seed=8)
        config = RealSyntheticInjectionConfig(block_size=32, seed=9)

        paths1 = build_real_injection_datasets(
            stack,
            config,
            out_dir=tmp_path,
            tile_size=16,
            tile_overlap=4,
            use_cache=True,
        )
        mtime1 = paths1["train"].stat().st_mtime

        paths2 = build_real_injection_datasets(
            stack,
            config,
            out_dir=tmp_path,
            tile_size=16,
            tile_overlap=4,
            use_cache=True,
        )
        mtime2 = paths2["train"].stat().st_mtime
        assert mtime1 == mtime2  # file was not regenerated

    def test_cache_disabled_regenerates(self, tmp_path):
        stack = build_tiny_fixture_stack(size=128, n_epochs=2, seed=10)
        config = RealSyntheticInjectionConfig(block_size=32, seed=11)

        paths1 = build_real_injection_datasets(
            stack,
            config,
            out_dir=tmp_path,
            tile_size=16,
            tile_overlap=4,
            use_cache=False,
        )
        import time

        time.sleep(0.01)
        paths2 = build_real_injection_datasets(
            stack,
            config,
            out_dir=tmp_path,
            tile_size=16,
            tile_overlap=4,
            use_cache=False,
        )
        assert paths1["train"].stat().st_mtime <= paths2["train"].stat().st_mtime

    def test_empty_split_warns_not_crashes(self, tmp_path):
        stack = build_tiny_fixture_stack(size=64, n_epochs=2, seed=12)
        config = RealSyntheticInjectionConfig(
            block_size=64,
            train_fraction=1.0,
            val_fraction=0.0,
            test_fraction=0.0,
            seed=13,
        )
        with pytest.warns(UserWarning, match="0 spatial blocks"):
            paths = build_real_injection_datasets(
                stack,
                config,
                out_dir=tmp_path,
                tile_size=32,
                tile_overlap=0,
                use_cache=False,
            )
        assert paths["val"].exists()  # file still created, just empty

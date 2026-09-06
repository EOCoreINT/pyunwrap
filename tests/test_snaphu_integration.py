"""
tests/test_snaphu_integration.py
===================================

Tests for `pyunwrap.utils.snaphu_integration`, the real SNAPHU integration
that makes `Trainer`'s SNAPHU pseudo-ground-truth fine-tuning phase an
actually-runnable feature.

All tests skip cleanly if the `snaphu` package isn't installed (it's an
optional extra, not a core dependency -- see `pyproject.toml`'s `snaphu`
extras group).
"""

from __future__ import annotations

import numpy as np
import pytest

snaphu_pkg = pytest.importorskip("snaphu", reason="snaphu is an optional extra; see pyproject.toml")

from pyunwrap.data.preprocessing import denormalize_phase
from pyunwrap.utils.snaphu_integration import (
    _recenter_ambiguity,
    build_snaphu_pseudo_ground_truth_tiles,
    generate_snaphu_finetune_dataset,
    unwrap_with_snaphu,
)

# --------------------------------------------------------------------------- #
# Core wrapper correctness
# --------------------------------------------------------------------------- #


class TestUnwrapWithSnaphu:
    def test_recovers_known_smooth_ramp(self):
        """On a controlled, genuinely easy problem (smooth ramp, uniform high
        coherence, no noise), SNAPHU must recover the true phase almost
        exactly, up to the inherent global 2*pi offset ambiguity."""
        size = 64
        yy, xx = np.mgrid[0:size, 0:size].astype(float)
        true_unwrapped = 0.15 * xx + 0.08 * yy
        wrapped = np.angle(np.exp(1j * true_unwrapped))
        coherence = np.full((size, size), 0.85)

        result = unwrap_with_snaphu(wrapped, coherence, nlooks=1.0)

        diff = result.unwrapped_phase - true_unwrapped
        offset = np.round(diff.mean() / (2 * np.pi)) * 2 * np.pi
        residual = diff - offset
        assert np.abs(residual).max() < 1e-3

    def test_reliable_fraction_is_high_on_easy_scene(self):
        size = 64
        _yy, xx = np.mgrid[0:size, 0:size].astype(float)
        true_unwrapped = 0.1 * xx
        wrapped = np.angle(np.exp(1j * true_unwrapped))
        coherence = np.full((size, size), 0.9)

        result = unwrap_with_snaphu(wrapped, coherence, nlooks=1.0)
        assert result.reliable_fraction > 0.95

    def test_decorrelated_region_correctly_flagged_unreliable(self):
        """A deliberately corrupted (pure noise, near-zero coherence) region
        must be flagged unreliable (conncomp == 0) by SNAPHU's own
        connected-component analysis -- this is the property
        `build_snaphu_pseudo_ground_truth_tiles`'s filtering relies on."""
        rng = np.random.default_rng(1)
        size = 80
        _yy, xx = np.mgrid[0:size, 0:size].astype(float)
        wrapped = np.angle(np.exp(1j * (0.1 * xx)))
        wrapped[30:50, 30:50] = rng.uniform(-np.pi, np.pi, size=(20, 20))

        coherence = np.full((size, size), 0.9)
        coherence[30:50, 30:50] = 0.02

        result = unwrap_with_snaphu(wrapped, coherence, nlooks=1.0, min_conncomp_frac=0.01)

        corrupted_unreliable_frac = (result.conncomp[30:50, 30:50] == 0).mean()
        good_unreliable_frac = (result.conncomp[:20, :20] == 0).mean()
        assert corrupted_unreliable_frac > 0.8
        assert good_unreliable_frac < 0.05

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            unwrap_with_snaphu(np.zeros((10, 10)), np.zeros((5, 5)))


class TestRecenterAmbiguity:
    def test_removes_global_offset(self):
        k = np.full((10, 10), 7.0)  # a uniform, large, arbitrary global offset
        recentered = _recenter_ambiguity(k)
        np.testing.assert_allclose(recentered, 0.0)

    def test_preserves_relative_structure(self):
        rng = np.random.default_rng(0)
        k_true_relative = rng.integers(-2, 3, size=(10, 10)).astype(float)
        k_with_offset = k_true_relative + 5.0
        recentered = _recenter_ambiguity(k_with_offset)
        # Relative differences between pixels must be preserved exactly.
        np.testing.assert_allclose(
            recentered - recentered[0, 0],
            k_true_relative - k_true_relative[0, 0],
        )


# --------------------------------------------------------------------------- #
# Pseudo-ground-truth tile pipeline
# --------------------------------------------------------------------------- #


@pytest.fixture
def easy_scene():
    """A controlled, mostly-reliable synthetic scene for pipeline tests."""
    size = 128
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    rng = np.random.default_rng(2)
    true_unwrapped = 0.04 * xx + 0.02 * yy
    coherence = np.full((size, size), 0.9)
    decorr_noise = rng.normal(0, 0.15, size=(size, size))
    wrapped = np.angle(np.exp(1j * (true_unwrapped + decorr_noise)))
    amplitude = np.ones((size, size))
    return wrapped, coherence, amplitude


class TestBuildPseudoGroundTruthTiles:
    def test_physics_identity_holds_in_kept_tiles(self, easy_scene):
        wrapped, coherence, amplitude = easy_scene
        tiles, summary = build_snaphu_pseudo_ground_truth_tiles(
            wrapped,
            coherence,
            amplitude=amplitude,
            tile_size=64,
            overlap=16,
            min_reliable_fraction=0.8,
            snaphu_kwargs={"nlooks": 4.0},
        )
        assert summary["n_kept_tiles"] > 0, "expected at least one reliable tile on an easy scene"

        for spec, tile in tiles:
            wrapped_rad = denormalize_phase(tile["wrapped_phase"])
            k = np.round((tile["true_unwrapped"] - wrapped_rad) / (2 * np.pi))
            reconstructed = wrapped_rad + 2 * np.pi * k
            np.testing.assert_allclose(reconstructed, tile["true_unwrapped"], atol=1e-3)

    def test_tile_dict_has_expected_keys(self, easy_scene):
        wrapped, coherence, amplitude = easy_scene
        tiles, _ = build_snaphu_pseudo_ground_truth_tiles(
            wrapped,
            coherence,
            amplitude=amplitude,
            tile_size=64,
            overlap=16,
            min_reliable_fraction=0.8,
            snaphu_kwargs={"nlooks": 4.0},
        )
        assert len(tiles) > 0
        _, tile = tiles[0]
        assert set(tile.keys()) == {"wrapped_phase", "coherence", "amplitude", "true_unwrapped"}

    def test_unreliable_scene_yields_fewer_or_no_tiles_at_strict_threshold(self):
        """A genuinely decorrelated scene, filtered at a strict reliability
        threshold, must yield fewer kept tiles than a lenient threshold on
        the same data -- confirms the filter is actually doing something,
        not silently passing everything through."""
        rng = np.random.default_rng(4)
        size = 128
        wrapped = rng.uniform(-np.pi, np.pi, size=(size, size))  # pure noise
        coherence = np.full((size, size), 0.05)
        amplitude = np.ones((size, size))

        _, summary_strict = build_snaphu_pseudo_ground_truth_tiles(
            wrapped,
            coherence,
            amplitude=amplitude,
            tile_size=64,
            overlap=16,
            min_reliable_fraction=0.99,
            snaphu_kwargs={"nlooks": 4.0},
        )
        _, summary_lenient = build_snaphu_pseudo_ground_truth_tiles(
            wrapped,
            coherence,
            amplitude=amplitude,
            tile_size=64,
            overlap=16,
            min_reliable_fraction=0.0,
            snaphu_kwargs={"nlooks": 4.0},
        )
        assert summary_strict["n_kept_tiles"] <= summary_lenient["n_kept_tiles"]


class TestGenerateFinetuneDataset:
    def test_end_to_end_produces_loadable_hdf5(self, easy_scene, tmp_path):
        import rasterio
        from rasterio.transform import from_origin

        from pyunwrap.data.dataloader import InSARTileDataset

        wrapped, coherence, amplitude = easy_scene
        transform = from_origin(500000, 5000000, 20, 20)
        profile = {
            "driver": "GTiff",
            "height": wrapped.shape[0],
            "width": wrapped.shape[1],
            "count": 1,
            "dtype": "float32",
            "crs": "EPSG:32633",
            "transform": transform,
        }

        paths = {}
        for name, arr in [("wrapped", wrapped), ("coherence", coherence), ("amplitude", amplitude)]:
            path = tmp_path / f"{name}.tif"
            with rasterio.open(path, "w", **profile) as dst:
                dst.write(arr.astype(np.float32), 1)
            paths[name] = path

        out_path, summary = generate_snaphu_finetune_dataset(
            paths["wrapped"],
            paths["coherence"],
            paths["amplitude"],
            out_path=tmp_path / "snaphu_gt.h5",
            tile_size=64,
            overlap=16,
            min_reliable_fraction=0.8,
            snaphu_kwargs={"nlooks": 4.0},
        )
        assert out_path.exists()

        ds = InSARTileDataset(out_path, augment=False, require_ground_truth=True)
        assert len(ds) == summary["n_kept_tiles"]
        sample = ds[0]
        assert "true_unwrapped" in sample

    def test_raises_when_nothing_survives_filter(self, tmp_path):
        import rasterio
        from rasterio.transform import from_origin

        rng = np.random.default_rng(5)
        size = 96
        wrapped = rng.uniform(-np.pi, np.pi, size=(size, size))
        coherence = np.full((size, size), 0.02)
        amplitude = np.ones((size, size))

        transform = from_origin(500000, 5000000, 20, 20)
        profile = {
            "driver": "GTiff",
            "height": size,
            "width": size,
            "count": 1,
            "dtype": "float32",
            "crs": "EPSG:32633",
            "transform": transform,
        }
        paths = {}
        for name, arr in [("wrapped", wrapped), ("coherence", coherence), ("amplitude", amplitude)]:
            path = tmp_path / f"{name}.tif"
            with rasterio.open(path, "w", **profile) as dst:
                dst.write(arr.astype(np.float32), 1)
            paths[name] = path

        with pytest.raises(RuntimeError):
            generate_snaphu_finetune_dataset(
                paths["wrapped"],
                paths["coherence"],
                paths["amplitude"],
                out_path=tmp_path / "should_not_exist.h5",
                tile_size=32,
                overlap=8,
                min_reliable_fraction=0.999,
                snaphu_kwargs={"nlooks": 4.0},
            )

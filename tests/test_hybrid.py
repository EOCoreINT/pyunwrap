"""
tests/test_hybrid.py
=======================

Tests for `pyunwrap.inference.hybrid` (Strategy 4: hybrid, regime-aware
unwrapper). Covers the required specification tests -- high-coherence
routes to classical, low-coherence routes to learned, the final output
re-wraps consistently, and the SNAPHU-unavailable fallback still works --
plus real correctness checks (the wrap-consistency invariant is verified
numerically, not just claimed) and the merge-strategy distinction between
`"confidence_weighted"` and `"hard_switch"`.
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from pyunwrap.inference.hybrid import (
    REGIME_CLASSICAL,
    REGIME_LEARNED,
    REGIME_UNCERTAIN,
    HybridUnwrapper,
    HybridUnwrapperConfig,
    SkimageClassicalAdapter,
    build_default_classical_adapter,
)
from pyunwrap.inference.unwrapper import PhaseUnwrapper
from pyunwrap.models.ambiguity_net import AmbiguityNet
from pyunwrap.synthetic.generator import wrap_phase


def _write_geotiff(path, array):
    transform = from_origin(500_000, 5_000_000, 20, 20)
    profile = {
        "driver": "GTiff",
        "height": array.shape[0],
        "width": array.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:32633",
        "transform": transform,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(np.float32), 1)
    return path


@pytest.fixture
def split_coherence_scene(tmp_path):
    """A controlled scene: left half high-coherence + a gentle ramp (easy
    for classical), right half low-coherence + noise (hard for classical,
    the regime this project's own benchmarks show the learned model
    winning in)."""
    size = 64
    true_unwrapped = np.zeros((size, size))
    true_unwrapped[:, : size // 2] = 0.02 * np.arange(size)[None, : size // 2]
    rng = np.random.default_rng(0)
    true_unwrapped[:, size // 2 :] += rng.uniform(-8, 8, size=(size, size // 2))
    wrapped = wrap_phase(true_unwrapped)

    coherence = np.zeros((size, size))
    coherence[:, : size // 2] = 0.95
    coherence[:, size // 2 :] = 0.05
    amplitude = np.ones((size, size))

    paths = {
        "wrapped": _write_geotiff(tmp_path / "wrapped.tif", wrapped),
        "coherence": _write_geotiff(tmp_path / "coherence.tif", coherence),
        "amplitude": _write_geotiff(tmp_path / "amplitude.tif", amplitude),
    }
    return paths, size


@pytest.fixture
def learned_unwrapper():
    model = AmbiguityNet(pretrained=False, k_max=10.0)
    return PhaseUnwrapper(model=model, device="cpu")


class TestHybridUnwrapperConfig:
    def test_defaults_match_spec(self):
        cfg = HybridUnwrapperConfig()
        assert cfg.use_hybrid_mode is True
        assert cfg.high_coherence_threshold == 0.75
        assert cfg.low_coherence_threshold == 0.35
        assert cfg.high_gradient_threshold is None
        assert cfg.prefer_learned_when_uncertain is True
        assert cfg.merge_strategy == "confidence_weighted"
        assert cfg.fallback_to_snaphu is True
        assert cfg.fallback_to_pyunwrap is True

    def test_rejects_inverted_thresholds(self):
        with pytest.raises(ValueError):
            HybridUnwrapperConfig(high_coherence_threshold=0.3, low_coherence_threshold=0.7)

    def test_rejects_out_of_range_thresholds(self):
        with pytest.raises(ValueError):
            HybridUnwrapperConfig(high_coherence_threshold=1.5)

    def test_rejects_unknown_merge_strategy(self):
        with pytest.raises(ValueError):
            HybridUnwrapperConfig(merge_strategy="average")

    def test_rejects_nonpositive_gradient_threshold(self):
        with pytest.raises(ValueError):
            HybridUnwrapperConfig(high_gradient_threshold=0.0)


class TestClassifyRegimes:
    def test_high_coherence_routes_to_classical(self, learned_unwrapper):
        hybrid = HybridUnwrapper(learned_unwrapper, classical_adapter=SkimageClassicalAdapter())
        coherence = np.full((8, 8), 0.9)
        regime = hybrid.classify_regimes(coherence)
        assert (regime == REGIME_CLASSICAL).all()

    def test_low_coherence_routes_to_learned(self, learned_unwrapper):
        hybrid = HybridUnwrapper(learned_unwrapper, classical_adapter=SkimageClassicalAdapter())
        coherence = np.full((8, 8), 0.1)
        regime = hybrid.classify_regimes(coherence)
        assert (regime == REGIME_LEARNED).all()

    def test_moderate_coherence_is_uncertain(self, learned_unwrapper):
        hybrid = HybridUnwrapper(learned_unwrapper, classical_adapter=SkimageClassicalAdapter())
        coherence = np.full((8, 8), 0.5)  # between low=0.35 and high=0.75
        regime = hybrid.classify_regimes(coherence)
        assert (regime == REGIME_UNCERTAIN).all()

    def test_high_gradient_demotes_high_coherence_to_uncertain(self, learned_unwrapper):
        config = HybridUnwrapperConfig(high_gradient_threshold=1.0)
        hybrid = HybridUnwrapper(
            learned_unwrapper, classical_adapter=SkimageClassicalAdapter(), config=config
        )
        coherence = np.full((8, 8), 0.9)
        gradient = np.full((8, 8), 5.0)  # exceeds the threshold
        regime = hybrid.classify_regimes(coherence, gradient=gradient)
        assert (regime == REGIME_UNCERTAIN).all()

    def test_low_coherence_overrides_gradient_classification(self, learned_unwrapper):
        """Low coherence must always route to REGIME_LEARNED regardless of
        gradient -- gradient only refines the high-coherence branch."""
        config = HybridUnwrapperConfig(high_gradient_threshold=1.0)
        hybrid = HybridUnwrapper(
            learned_unwrapper, classical_adapter=SkimageClassicalAdapter(), config=config
        )
        coherence = np.full((8, 8), 0.1)
        gradient = np.full((8, 8), 5.0)
        regime = hybrid.classify_regimes(coherence, gradient=gradient)
        assert (regime == REGIME_LEARNED).all()


class TestBuildDefaultClassicalAdapter:
    def test_returns_snaphu_adapter_when_available(self, monkeypatch):
        import pyunwrap.inference.hybrid as hybrid_module

        monkeypatch.setattr(hybrid_module, "_HAS_SNAPHU", True)
        adapter = build_default_classical_adapter()
        assert type(adapter).__name__ == "SnaphuAdapter"

    def test_falls_back_to_skimage_when_snaphu_unavailable(self, monkeypatch):
        """Required spec test: if SNAPHU is unavailable, a real classical
        adapter (not a crash) must still be produced."""
        import pyunwrap.inference.hybrid as hybrid_module

        monkeypatch.setattr(hybrid_module, "_HAS_SNAPHU", False)
        monkeypatch.setattr(hybrid_module, "_HAS_SKIMAGE", True)
        adapter = build_default_classical_adapter()
        assert isinstance(adapter, SkimageClassicalAdapter)

    def test_raises_when_neither_available(self, monkeypatch):
        import pyunwrap.inference.hybrid as hybrid_module

        monkeypatch.setattr(hybrid_module, "_HAS_SNAPHU", False)
        monkeypatch.setattr(hybrid_module, "_HAS_SKIMAGE", False)
        with pytest.raises(ImportError):
            build_default_classical_adapter()


class TestSkimageClassicalAdapter:
    def test_produces_reasonable_unwrap_on_easy_case(self):
        size = 32
        _yy, xx = np.mgrid[0:size, 0:size].astype(float)
        true_unwrapped = 0.05 * xx
        wrapped = wrap_phase(true_unwrapped)
        coherence = np.full((size, size), 0.9)

        adapter = SkimageClassicalAdapter()
        unwrapped, reliability = adapter.unwrap(wrapped, coherence)
        assert unwrapped.shape == wrapped.shape
        assert reliability.shape == wrapped.shape
        assert np.abs(unwrapped - true_unwrapped).max() < 1e-6

    def test_reliability_reflects_coherence_threshold(self):
        wrapped = np.zeros((16, 16))
        coherence = np.zeros((16, 16))
        coherence[:8, :] = 0.9  # above default threshold (0.3)
        coherence[8:, :] = 0.1  # below

        adapter = SkimageClassicalAdapter()
        _, reliability = adapter.unwrap(wrapped, coherence)
        assert (reliability[:8, :] == 1.0).all()
        assert (reliability[8:, :] == 0.0).all()


class TestHybridUnwrapperEndToEnd:
    """Uses SkimageClassicalAdapter explicitly (not the real SNAPHU
    adapter) so these tests are deterministic and don't depend on SNAPHU's
    own (slower, external-binary) behavior -- SNAPHU-specific integration
    is covered separately where the snaphu package is actually exercised
    (test_snaphu_integration.py) and is not re-tested here."""

    def test_final_output_rewraps_consistently(self, split_coherence_scene, learned_unwrapper):
        """The specification's core correctness requirement: final output
        must re-wrap consistently with the observed wrapped phase."""
        paths, size = split_coherence_scene
        hybrid = HybridUnwrapper(learned_unwrapper, classical_adapter=SkimageClassicalAdapter())
        result = hybrid.unwrap(
            paths["wrapped"], paths["coherence"], paths["amplitude"], tile_size=size, overlap=0
        )

        diff = np.angle(np.exp(1j * (wrap_phase(result.unwrapped_phase) - result.wrapped_phase)))
        assert np.abs(diff).max() < 1e-6

    def test_high_coherence_region_routes_to_classical(
        self, split_coherence_scene, learned_unwrapper
    ):
        paths, size = split_coherence_scene
        hybrid = HybridUnwrapper(learned_unwrapper, classical_adapter=SkimageClassicalAdapter())
        result = hybrid.unwrap(
            paths["wrapped"], paths["coherence"], paths["amplitude"], tile_size=size, overlap=0
        )

        left_half = result.regime_mask[:, : size // 2]
        assert (left_half == REGIME_CLASSICAL).all()

    def test_low_coherence_region_routes_to_learned(self, split_coherence_scene, learned_unwrapper):
        paths, size = split_coherence_scene
        hybrid = HybridUnwrapper(learned_unwrapper, classical_adapter=SkimageClassicalAdapter())
        result = hybrid.unwrap(
            paths["wrapped"], paths["coherence"], paths["amplitude"], tile_size=size, overlap=0
        )

        right_half = result.regime_mask[:, size // 2 :]
        assert (right_half == REGIME_LEARNED).all()

    def test_hard_switch_gives_clean_provenance_separation(
        self, split_coherence_scene, learned_unwrapper
    ):
        """Unlike confidence_weighted (which can let a method with very
        high local confidence marginally win outside its own assigned
        regime), hard_switch must give exact regime-to-provenance equality."""
        paths, size = split_coherence_scene
        config = HybridUnwrapperConfig(merge_strategy="hard_switch")
        hybrid = HybridUnwrapper(
            learned_unwrapper, classical_adapter=SkimageClassicalAdapter(), config=config
        )
        result = hybrid.unwrap(
            paths["wrapped"], paths["coherence"], paths["amplitude"], tile_size=size, overlap=0
        )

        np.testing.assert_array_equal(result.regime_mask, result.provenance_mask)

    def test_confidence_map_in_valid_range(self, split_coherence_scene, learned_unwrapper):
        paths, size = split_coherence_scene
        hybrid = HybridUnwrapper(learned_unwrapper, classical_adapter=SkimageClassicalAdapter())
        result = hybrid.unwrap(
            paths["wrapped"], paths["coherence"], paths["amplitude"], tile_size=size, overlap=0
        )
        assert (result.confidence_map >= 0.0).all()
        assert (result.confidence_map <= 1.0 + 1e-9).all()

    def test_use_hybrid_mode_false_never_constructs_classical_adapter(self, learned_unwrapper):
        config = HybridUnwrapperConfig(use_hybrid_mode=False)
        hybrid = HybridUnwrapper(learned_unwrapper, config=config)
        assert hybrid.classical_adapter is None

    def test_use_hybrid_mode_false_result_is_learned_only(
        self, split_coherence_scene, learned_unwrapper
    ):
        paths, size = split_coherence_scene
        config = HybridUnwrapperConfig(use_hybrid_mode=False)
        hybrid = HybridUnwrapper(learned_unwrapper, config=config)
        result = hybrid.unwrap(
            paths["wrapped"], paths["coherence"], paths["amplitude"], tile_size=size, overlap=0
        )
        assert (result.regime_mask == REGIME_LEARNED).all()
        assert (result.provenance_mask == REGIME_LEARNED).all()

        diff = np.angle(np.exp(1j * (wrap_phase(result.unwrapped_phase) - result.wrapped_phase)))
        assert np.abs(diff).max() < 1e-6


class TestHybridUnwrapperFallbacks:
    def test_classical_adapter_failure_falls_back_to_learned(
        self, split_coherence_scene, learned_unwrapper
    ):
        class BrokenAdapter:
            def unwrap(self, wrapped_phase, coherence, amplitude=None):
                raise RuntimeError("simulated classical adapter failure")

        paths, size = split_coherence_scene
        hybrid = HybridUnwrapper(learned_unwrapper, classical_adapter=BrokenAdapter())
        result = hybrid.unwrap(
            paths["wrapped"], paths["coherence"], paths["amplitude"], tile_size=size, overlap=0
        )
        # Must have degraded gracefully to a learned-only result, not raised.
        assert (result.regime_mask == REGIME_LEARNED).all()

    def test_classical_adapter_failure_raises_when_fallback_disabled(
        self, split_coherence_scene, learned_unwrapper
    ):
        class BrokenAdapter:
            def unwrap(self, wrapped_phase, coherence, amplitude=None):
                raise RuntimeError("simulated classical adapter failure")

        paths, size = split_coherence_scene
        config = HybridUnwrapperConfig(fallback_to_pyunwrap=False)
        hybrid = HybridUnwrapper(
            learned_unwrapper, classical_adapter=BrokenAdapter(), config=config
        )
        with pytest.raises(RuntimeError, match="simulated classical adapter failure"):
            hybrid.unwrap(
                paths["wrapped"], paths["coherence"], paths["amplitude"], tile_size=size, overlap=0
            )

    def test_learned_model_failure_falls_back_to_classical(
        self, split_coherence_scene, learned_unwrapper, monkeypatch
    ):
        def broken_unwrap(*args, **kwargs):
            raise RuntimeError("simulated learned model failure")

        monkeypatch.setattr(learned_unwrapper, "unwrap", broken_unwrap)
        paths, size = split_coherence_scene
        hybrid = HybridUnwrapper(learned_unwrapper, classical_adapter=SkimageClassicalAdapter())
        result = hybrid.unwrap(
            paths["wrapped"], paths["coherence"], paths["amplitude"], tile_size=size, overlap=0
        )
        assert (result.regime_mask == REGIME_CLASSICAL).all()

    def test_learned_model_failure_raises_when_fallback_disabled(
        self, split_coherence_scene, learned_unwrapper, monkeypatch
    ):
        def broken_unwrap(*args, **kwargs):
            raise RuntimeError("simulated learned model failure")

        monkeypatch.setattr(learned_unwrapper, "unwrap", broken_unwrap)
        paths, size = split_coherence_scene
        config = HybridUnwrapperConfig(fallback_to_snaphu=False)
        hybrid = HybridUnwrapper(
            learned_unwrapper, classical_adapter=SkimageClassicalAdapter(), config=config
        )
        with pytest.raises(RuntimeError, match="simulated learned model failure"):
            hybrid.unwrap(
                paths["wrapped"], paths["coherence"], paths["amplitude"], tile_size=size, overlap=0
            )

    def test_no_classical_adapter_available_warns_and_degrades(
        self, learned_unwrapper, monkeypatch
    ):
        import pyunwrap.inference.hybrid as hybrid_module

        monkeypatch.setattr(hybrid_module, "_HAS_SNAPHU", False)
        monkeypatch.setattr(hybrid_module, "_HAS_SKIMAGE", False)
        with pytest.warns(UserWarning, match="No classical adapter available"):
            hybrid = HybridUnwrapper(learned_unwrapper)
        assert hybrid.classical_adapter is None

    def test_no_classical_adapter_available_raises_when_fallback_disabled(
        self, learned_unwrapper, monkeypatch
    ):
        import pyunwrap.inference.hybrid as hybrid_module

        monkeypatch.setattr(hybrid_module, "_HAS_SNAPHU", False)
        monkeypatch.setattr(hybrid_module, "_HAS_SKIMAGE", False)
        config = HybridUnwrapperConfig(fallback_to_pyunwrap=False)
        with pytest.raises(ImportError):
            HybridUnwrapper(learned_unwrapper, config=config)

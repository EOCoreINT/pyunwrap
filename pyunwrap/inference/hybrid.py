"""
pyunwrap.inference.hybrid
=============================

Strategy 4: a hybrid, regime-aware phase unwrapper.

This module reframes this project's own goal, based directly on its own
benchmark evidence (see `docs/experiments.md` and the project's release
notes): `pyunwrap`'s learned model does not beat classical SNAPHU-style
unwrapping across the board, and there is no evidence it should be
expected to. It reliably wins in specifically the regime where classical
statistical-cost unwrapping's core assumption (locally smooth, internally
consistent phase) breaks down hardest: low-coherence decorrelation. The
correct goal is therefore not "replace SNAPHU everywhere" but "route each
region of a scene to whichever method is actually good there"::

    high coherence, low gradient  -> classical unwrapping (SNAPHU or an
                                      equivalent classical adapter), which
                                      is mature, fast, and already reliable
                                      in this regime
    low coherence / high decorrelation -> the learned model, which this
                                      project's own benchmarks show
                                      winning specifically here
    high coherence + high gradient (an "uncertain boundary" regime, only
    classified when `high_gradient_threshold` is set) -> a confidence-
                                      weighted blend of both, favoring the
                                      learned model by default
                                      (`prefer_learned_when_uncertain=True`)

Correctness invariant this module is built around
-------------------------------------------------------
Merging never averages raw phase values directly -- doing so would not, in
general, satisfy `wrap(merged_phase) == observed_wrapped_phase`, which is
the one invariant every other part of this project (the model's own output
construction, tile merging in `pyunwrap.inference.unwrapper`, the SNAPHU
pseudo-label re-centering in `pyunwrap.utils.snaphu_integration`) treats as
non-negotiable. Instead, both the classical and learned results are first
converted to their own integer ambiguity maps (`k_classical`, `k_learned`),
merged in that integer space, and the final phase is reconstructed as
`wrapped_phase + 2*pi*round(k_merged)` -- which satisfies the wrap
identity exactly, by construction, regardless of how the two `k` maps were
blended.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from pyunwrap.inference.unwrapper import PhaseUnwrapper, UnwrapResult
from pyunwrap.synthetic.generator import compute_ambiguity
from pyunwrap.utils.snaphu_integration import _HAS_SNAPHU, unwrap_with_snaphu

try:
    from skimage.restoration import unwrap_phase as _skimage_unwrap_phase

    _HAS_SKIMAGE = True
except ImportError:  # pragma: no cover
    _HAS_SKIMAGE = False

try:
    import rasterio

    _HAS_RASTERIO = True
except ImportError:  # pragma: no cover
    _HAS_RASTERIO = False

#: Regime labels used throughout this module's regime_mask/provenance_mask outputs.
REGIME_CLASSICAL = 0
REGIME_LEARNED = 1
REGIME_UNCERTAIN = 2


# --------------------------------------------------------------------------- #
# Classical unwrapper adapters
# --------------------------------------------------------------------------- #


@runtime_checkable
class ClassicalUnwrapperAdapter(Protocol):
    """Interface any classical unwrapper must implement to plug into
    `HybridUnwrapper`."""

    def unwrap(
        self, wrapped_phase: np.ndarray, coherence: np.ndarray, amplitude: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Unwrap `wrapped_phase` and return `(unwrapped_phase, reliability)`,
        both the same shape as `wrapped_phase`, `reliability` in `[0, 1]`."""
        ...


@dataclass
class SnaphuAdapter:
    """Wraps `pyunwrap.utils.snaphu_integration.unwrap_with_snaphu` as a
    `ClassicalUnwrapperAdapter`.

    Attributes:
        nlooks: Forwarded to `unwrap_with_snaphu`.
        snaphu_kwargs: Extra keyword arguments forwarded to
            `unwrap_with_snaphu` (e.g. `cost`, `init`).
    """

    nlooks: float = 4.0
    snaphu_kwargs: dict = field(default_factory=dict)

    def unwrap(
        self, wrapped_phase: np.ndarray, coherence: np.ndarray, amplitude: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        result = unwrap_with_snaphu(
            wrapped_phase, coherence, amplitude=amplitude, nlooks=self.nlooks, **self.snaphu_kwargs
        )
        reliability = (result.conncomp != 0).astype(np.float64)
        return result.unwrapped_phase, reliability


@dataclass
class SkimageClassicalAdapter:
    """A real, different classical unwrapping algorithm (Herraez et al.'s
    quality-guided path-following method, via
    `skimage.restoration.unwrap_phase`), used automatically as the
    classical adapter when SNAPHU isn't installed -- so `HybridUnwrapper`
    and its tests still work without requiring the optional `snaphu`
    package, per this project's requirement that CPU-only, SNAPHU-free
    execution remain possible.

    This is explicitly **not** a substitute for SNAPHU: it is a genuinely
    different classical algorithm (not a mock/fake), but its reliability
    signal here is a simple coherence-threshold heuristic, not SNAPHU's
    own connected-component analysis -- callers relying on precise
    reliability semantics should install `snaphu` and use `SnaphuAdapter`.

    Attributes:
        coherence_reliability_threshold: Pixels with coherence above this
            value are reported as reliable (`1.0`); at or below, unreliable
            (`0.0`).
    """

    coherence_reliability_threshold: float = 0.3

    def unwrap(
        self, wrapped_phase: np.ndarray, coherence: np.ndarray, amplitude: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if not _HAS_SKIMAGE:
            raise ImportError(
                "scikit-image is required for SkimageClassicalAdapter (the automatic "
                "fallback used when snaphu is not installed)."
            )
        unwrapped = _skimage_unwrap_phase(wrapped_phase)
        reliability = (coherence > self.coherence_reliability_threshold).astype(np.float64)
        return unwrapped, reliability


def build_default_classical_adapter() -> ClassicalUnwrapperAdapter:
    """Return `SnaphuAdapter()` if the `snaphu` package is installed,
    otherwise `SkimageClassicalAdapter()`, printing which was chosen.

    Returns:
        A `ClassicalUnwrapperAdapter`.

    Raises:
        ImportError: If neither `snaphu` nor `scikit-image` is available.
    """
    if _HAS_SNAPHU:
        print("[HybridUnwrapper] Using SNAPHU as the classical unwrapper adapter.")
        return SnaphuAdapter()
    if _HAS_SKIMAGE:
        print(
            "[HybridUnwrapper] snaphu is not installed; falling back to a scikit-image "
            "quality-guided classical unwrapper (a real, different classical algorithm, "
            "not SNAPHU itself -- see SkimageClassicalAdapter's docstring for the difference)."
        )
        return SkimageClassicalAdapter()
    raise ImportError(
        "Neither the 'snaphu' package nor 'scikit-image' is installed; HybridUnwrapper "
        "requires at least one classical adapter to be available. Install one of them, "
        "or pass an explicit classical_adapter implementing ClassicalUnwrapperAdapter."
    )


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass
class HybridUnwrapperConfig:
    """Configuration for `HybridUnwrapper`.

    Attributes:
        use_hybrid_mode: Master on/off switch. `False` makes
            `HybridUnwrapper.unwrap` behave as a thin, always-learned-only
            wrapper around the underlying `PhaseUnwrapper` (no classical
            adapter is even constructed), for a clean, zero-overhead
            opt-out.
        high_coherence_threshold: Pixels with coherence at or above this
            value are classified `REGIME_CLASSICAL` (unless also flagged
            high-gradient; see `high_gradient_threshold`).
        low_coherence_threshold: Pixels with coherence at or below this
            value are classified `REGIME_LEARNED`.
        high_gradient_threshold: Optional. If set, high-coherence pixels
            whose local phase gradient (estimated from the classical
            result, the most reliable single unwrapped estimate available
            for this purpose) exceeds this value (radians/pixel) are
            classified `REGIME_UNCERTAIN` instead of `REGIME_CLASSICAL` --
            steep gradients are exactly where classical unwrapping's own
            Nyquist assumption can fail even at high coherence. `None`
            disables this refinement (high coherence alone -> classical,
            regardless of gradient).
        prefer_learned_when_uncertain: For `REGIME_UNCERTAIN` pixels,
            whether the learned model's ambiguity is favored in the merge
            (`True`, the default) or the classical result is
            (`False`).
        merge_strategy: `"confidence_weighted"` (blend `k_classical` and
            `k_learned` per-pixel, weighted by each method's own
            confidence, then round) or `"hard_switch"` (each pixel takes
            its assigned regime's `k` outright, with `REGIME_UNCERTAIN`
            resolved by `prefer_learned_when_uncertain`).
        fallback_to_snaphu: If the learned model's forward pass raises, and
            this is `True`, fall back to a classical-only result (with a
            printed warning) rather than propagating the exception.
        fallback_to_pyunwrap: If the classical adapter raises (including at
            construction time, e.g. neither `snaphu` nor `scikit-image` is
            installed and no explicit adapter was given), and this is
            `True`, fall back to a learned-only result (with a printed
            warning) rather than propagating the exception.

    Raises:
        ValueError: If thresholds are out of `[0, 1]` or inverted, if
            `merge_strategy` is not one of the two supported values, or if
            `high_gradient_threshold` is given but not positive.
    """

    use_hybrid_mode: bool = True
    high_coherence_threshold: float = 0.75
    low_coherence_threshold: float = 0.35
    high_gradient_threshold: float | None = None
    prefer_learned_when_uncertain: bool = True
    merge_strategy: str = "confidence_weighted"
    fallback_to_snaphu: bool = True
    fallback_to_pyunwrap: bool = True

    def __post_init__(self) -> None:
        if not (0.0 <= self.low_coherence_threshold <= self.high_coherence_threshold <= 1.0):
            raise ValueError(
                "Require 0 <= low_coherence_threshold <= high_coherence_threshold <= 1, got "
                f"low={self.low_coherence_threshold}, high={self.high_coherence_threshold}"
            )
        if self.merge_strategy not in ("confidence_weighted", "hard_switch"):
            raise ValueError(
                f"merge_strategy must be 'confidence_weighted' or 'hard_switch', got "
                f"{self.merge_strategy!r}"
            )
        if self.high_gradient_threshold is not None and self.high_gradient_threshold <= 0:
            raise ValueError(
                f"high_gradient_threshold must be positive if set, got {self.high_gradient_threshold}"
            )


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #


@dataclass
class HybridUnwrapResult:
    """Full-scene output of `HybridUnwrapper.unwrap`.

    Attributes:
        unwrapped_phase: Final merged unwrapped phase, radians. Guaranteed
            (by construction, not by post-hoc checking) to satisfy
            `wrap(unwrapped_phase) == wrapped_phase` exactly -- see the
            module docstring.
        ambiguity_map: Final merged integer ambiguity map.
        confidence_map: Per-pixel confidence in `[0, 1]` of the merged
            result (the confidence of whichever method's `k` ultimately
            dominated that pixel).
        regime_mask: Per-pixel regime classification (`REGIME_CLASSICAL`,
            `REGIME_LEARNED`, or `REGIME_UNCERTAIN`), i.e. what
            `classify_regimes` decided *before* merging.
        uncertainty_map: The learned model's own Monte Carlo Dropout
            ambiguity uncertainty (see `PhaseUnwrapper`), all zeros where
            the learned model didn't contribute or MC-Dropout wasn't used.
        provenance_mask: Per-pixel record of which method's `k` actually
            dominated the final merged value at that pixel
            (`REGIME_CLASSICAL` or `REGIME_LEARNED` -- never
            `REGIME_UNCERTAIN`, since every pixel's merge resolves to a
            definite dominant contributor even under
            `merge_strategy="confidence_weighted"`). This can differ from
            `regime_mask` for `REGIME_UNCERTAIN` pixels, which is exactly
            the point of tracking both.
        wrapped_phase: The original input wrapped phase, radians, cropped
            to match the other arrays' shape.
        geotiff_profile: Source GeoTIFF rasterio profile, for
            `save_geotiff`.
    """

    unwrapped_phase: np.ndarray
    ambiguity_map: np.ndarray
    confidence_map: np.ndarray
    regime_mask: np.ndarray
    uncertainty_map: np.ndarray
    provenance_mask: np.ndarray
    wrapped_phase: np.ndarray
    geotiff_profile: dict

    def save_geotiff(self, path: str | Path, array_name: str = "unwrapped_phase") -> Path:
        """Write one of this result's arrays to disk as a single-band
        GeoTIFF, reusing the source raster's georeferencing.

        Args:
            path: Output GeoTIFF path.
            array_name: Which attribute to save.

        Returns:
            The path written to.

        Raises:
            ImportError: If `rasterio` is not installed.
        """
        if not _HAS_RASTERIO:
            raise ImportError("rasterio is required to save GeoTIFF output.")
        array = getattr(self, array_name)
        profile = dict(self.geotiff_profile)
        profile.update(dtype=rasterio.float32, count=1, height=array.shape[0], width=array.shape[1])
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(array.astype(np.float32), 1)
        return path


# --------------------------------------------------------------------------- #
# HybridUnwrapper
# --------------------------------------------------------------------------- #


class HybridUnwrapper:
    """Regime-aware unwrapper routing between a classical adapter and a
    learned `pyunwrap` model.

    Example:
        >>> from pyunwrap.inference.unwrapper import PhaseUnwrapper
        >>> learned = PhaseUnwrapper(model=trained_model, device="cpu")
        >>> hybrid = HybridUnwrapper(learned)  # auto-selects SNAPHU or the scikit-image fallback
        >>> result = hybrid.unwrap("wrapped.tif", "coherence.tif", "amplitude.tif")
        >>> result.save_geotiff("unwrapped.tif")
    """

    def __init__(
        self,
        learned_unwrapper: PhaseUnwrapper,
        classical_adapter: ClassicalUnwrapperAdapter | None = None,
        config: HybridUnwrapperConfig | None = None,
    ) -> None:
        """
        Args:
            learned_unwrapper: A configured `PhaseUnwrapper` wrapping a
                (trained) `AmbiguityNet`, or an ONNX export.
            classical_adapter: A `ClassicalUnwrapperAdapter`. If `None` and
                `config.use_hybrid_mode` is `True`,
                `build_default_classical_adapter()` is used (SNAPHU if
                installed, else a scikit-image fallback).
            config: Behavior configuration. Defaults to
                `HybridUnwrapperConfig()`.
        """
        self.learned_unwrapper = learned_unwrapper
        self.config = config if config is not None else HybridUnwrapperConfig()

        self.classical_adapter: ClassicalUnwrapperAdapter | None = None
        if classical_adapter is not None:
            self.classical_adapter = classical_adapter
        elif self.config.use_hybrid_mode:
            try:
                self.classical_adapter = build_default_classical_adapter()
            except ImportError as exc:
                if self.config.fallback_to_pyunwrap:
                    warnings.warn(
                        f"No classical adapter available ({exc}); HybridUnwrapper will "
                        "always route to the learned model until one is provided. Set "
                        "fallback_to_pyunwrap=False to raise instead of degrading silently.",
                        stacklevel=2,
                    )
                else:
                    raise

    def classify_regimes(
        self, coherence: np.ndarray, gradient: np.ndarray | None = None
    ) -> np.ndarray:
        """Classify each pixel into `REGIME_CLASSICAL`, `REGIME_LEARNED`,
        or `REGIME_UNCERTAIN`.

        Args:
            coherence: Coherence map, `[0, 1]`.
            gradient: Optional local phase-gradient magnitude map, same
                shape, radians/pixel. Only used if
                `config.high_gradient_threshold` is set.

        Returns:
            An `int32` array, same shape as `coherence`, with values from
            `{REGIME_CLASSICAL, REGIME_LEARNED, REGIME_UNCERTAIN}`.
        """
        regime = np.full(coherence.shape, REGIME_UNCERTAIN, dtype=np.int32)
        high_coh = coherence >= self.config.high_coherence_threshold
        low_coh = coherence <= self.config.low_coherence_threshold

        if self.config.high_gradient_threshold is not None and gradient is not None:
            high_grad = gradient > self.config.high_gradient_threshold
            regime[high_coh & ~high_grad] = REGIME_CLASSICAL
            # high_coh & high_grad is left as REGIME_UNCERTAIN.
        else:
            regime[high_coh] = REGIME_CLASSICAL

        regime[low_coh] = REGIME_LEARNED  # applied after, so low-coherence always wins any overlap
        return regime

    def unwrap(
        self,
        wrapped_phase_path: str | Path,
        coherence_path: str | Path,
        amplitude_path: str | Path,
        tile_size: int = 512,
        overlap: int = 64,
    ) -> HybridUnwrapResult:
        """Run hybrid, regime-aware unwrapping over a full-scene interferogram.

        Args:
            wrapped_phase_path: Path to the wrapped-phase GeoTIFF, radians.
            coherence_path: Path to the coherence GeoTIFF, `[0, 1]`.
            amplitude_path: Path to the amplitude GeoTIFF.
            tile_size: Tile edge length for the learned model's tiled
                inference (see `PhaseUnwrapper.unwrap`'s docstring for the
                multiple-of-32 constraint).
            overlap: Overlap between adjacent learned-inference tiles.

        Returns:
            A `HybridUnwrapResult`.

        Raises:
            RuntimeError: If both the learned model and the classical
                adapter fail and their respective fallback flags are both
                `False` (or the failing one's fallback is `False`).
        """
        if not self.config.use_hybrid_mode or self.classical_adapter is None:
            learned_only = self.learned_unwrapper.unwrap(
                wrapped_phase_path,
                coherence_path,
                amplitude_path,
                tile_size=tile_size,
                overlap=overlap,
            )
            return self._learned_only_result(learned_only)

        learned_result: UnwrapResult | None
        try:
            learned_result = self.learned_unwrapper.unwrap(
                wrapped_phase_path,
                coherence_path,
                amplitude_path,
                tile_size=tile_size,
                overlap=overlap,
            )
        except Exception as exc:
            if not self.config.fallback_to_snaphu:
                raise
            print(
                f"[HybridUnwrapper] Learned model failed ({exc}); falling back to classical-only."
            )
            learned_result = None

        wrapped_phase, coherence, amplitude = self._load_rasters(
            wrapped_phase_path, coherence_path, amplitude_path
        )

        if learned_result is None:
            classical_unwrapped, classical_reliability = self.classical_adapter.unwrap(
                wrapped_phase, coherence, amplitude
            )
            return self._classical_only_result(
                wrapped_phase, classical_unwrapped, classical_reliability
            )

        try:
            classical_unwrapped, classical_reliability = self.classical_adapter.unwrap(
                wrapped_phase, coherence, amplitude
            )
        except Exception as exc:
            if not self.config.fallback_to_pyunwrap:
                raise
            print(
                f"[HybridUnwrapper] Classical adapter failed ({exc}); falling back to learned-only."
            )
            return self._learned_only_result(learned_result)

        return self._merge(
            wrapped_phase, coherence, learned_result, classical_unwrapped, classical_reliability
        )

    # ----------------------------------------------------------------- #
    # Internals
    # ----------------------------------------------------------------- #

    @staticmethod
    def _load_rasters(
        wrapped_phase_path: str | Path, coherence_path: str | Path, amplitude_path: str | Path
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not _HAS_RASTERIO:
            raise ImportError("rasterio is required to load input GeoTIFFs.")
        with rasterio.open(wrapped_phase_path) as src:
            wrapped_phase = src.read(1).astype(np.float64)
        with rasterio.open(coherence_path) as src:
            coherence = src.read(1).astype(np.float64)
        with rasterio.open(amplitude_path) as src:
            amplitude = src.read(1).astype(np.float64)
        return wrapped_phase, coherence, amplitude

    def _learned_only_result(self, learned_result: UnwrapResult) -> HybridUnwrapResult:
        shape = learned_result.unwrapped_phase.shape
        return HybridUnwrapResult(
            unwrapped_phase=learned_result.unwrapped_phase,
            ambiguity_map=learned_result.ambiguity_map,
            confidence_map=1.0 - learned_result.residue_prob,
            regime_mask=np.full(shape, REGIME_LEARNED, dtype=np.int32),
            uncertainty_map=learned_result.uncertainty,
            provenance_mask=np.full(shape, REGIME_LEARNED, dtype=np.int32),
            wrapped_phase=learned_result.wrapped_phase,
            geotiff_profile=learned_result.geotiff_profile,
        )

    def _classical_only_result(
        self,
        wrapped_phase: np.ndarray,
        classical_unwrapped: np.ndarray,
        classical_reliability: np.ndarray,
    ) -> HybridUnwrapResult:
        shape = wrapped_phase.shape
        k_classical = compute_ambiguity(classical_unwrapped, wrapped_phase)
        reconstructed = wrapped_phase + 2.0 * np.pi * k_classical
        return HybridUnwrapResult(
            unwrapped_phase=reconstructed,
            ambiguity_map=k_classical,
            confidence_map=classical_reliability,
            regime_mask=np.full(shape, REGIME_CLASSICAL, dtype=np.int32),
            uncertainty_map=np.zeros(shape),
            provenance_mask=np.full(shape, REGIME_CLASSICAL, dtype=np.int32),
            wrapped_phase=wrapped_phase,
            geotiff_profile={},
        )

    def _merge(
        self,
        wrapped_phase: np.ndarray,
        coherence: np.ndarray,
        learned_result: UnwrapResult,
        classical_unwrapped: np.ndarray,
        classical_reliability: np.ndarray,
    ) -> HybridUnwrapResult:
        # The learned model's tiled inference can crop to a slightly
        # different shape than the raw input rasters (see
        # PhaseUnwrapper.unwrap); crop everything else to match rather than
        # assume they're already identical.
        h, w = learned_result.unwrapped_phase.shape
        wrapped_phase_c = wrapped_phase[:h, :w]
        coherence_c = coherence[:h, :w]
        classical_unwrapped_c = classical_unwrapped[:h, :w]
        classical_reliability_c = classical_reliability[:h, :w]

        k_learned = learned_result.ambiguity_map
        k_classical = compute_ambiguity(classical_unwrapped_c, wrapped_phase_c)
        confidence_learned = 1.0 - learned_result.residue_prob
        confidence_classical = classical_reliability_c

        gradient = None
        if self.config.high_gradient_threshold is not None:
            grad_y, grad_x = np.gradient(classical_unwrapped_c)
            gradient = np.sqrt(grad_y**2 + grad_x**2)
        regime = self.classify_regimes(coherence_c, gradient=gradient)

        if self.config.merge_strategy == "hard_switch":
            k_merged = np.where(regime == REGIME_CLASSICAL, k_classical, k_learned)
            uncertain_mask = regime == REGIME_UNCERTAIN
            if self.config.prefer_learned_when_uncertain:
                k_merged = np.where(uncertain_mask, k_learned, k_merged)
                provenance = np.where(
                    regime == REGIME_CLASSICAL, REGIME_CLASSICAL, REGIME_LEARNED
                ).astype(np.int32)
            else:
                k_merged = np.where(uncertain_mask, k_classical, k_merged)
                provenance = np.where(
                    regime == REGIME_LEARNED, REGIME_LEARNED, REGIME_CLASSICAL
                ).astype(np.int32)
            confidence_map = np.where(
                provenance == REGIME_CLASSICAL, confidence_classical, confidence_learned
            )
        else:  # "confidence_weighted"
            w_learned = confidence_learned.copy()
            w_classical = confidence_classical.copy()
            # Within each method's own assigned regime, boost its weight
            # so a clear regime assignment isn't overridden by a noisy
            # confidence estimate alone.
            w_learned = np.where(regime == REGIME_LEARNED, np.maximum(w_learned, 0.9), w_learned)
            w_classical = np.where(
                regime == REGIME_CLASSICAL, np.maximum(w_classical, 0.9), w_classical
            )
            if self.config.prefer_learned_when_uncertain:
                w_learned = np.where(
                    regime == REGIME_UNCERTAIN, np.maximum(w_learned, 0.6), w_learned
                )
            else:
                w_classical = np.where(
                    regime == REGIME_UNCERTAIN, np.maximum(w_classical, 0.6), w_classical
                )

            total_weight = w_learned + w_classical + 1e-9
            w_learned_norm = w_learned / total_weight
            w_classical_norm = w_classical / total_weight

            k_merged = np.round(w_learned_norm * k_learned + w_classical_norm * k_classical)
            provenance = np.where(
                w_learned_norm >= w_classical_norm, REGIME_LEARNED, REGIME_CLASSICAL
            ).astype(np.int32)
            confidence_map = np.maximum(
                w_learned_norm * confidence_learned, w_classical_norm * confidence_classical
            ) + np.minimum(
                w_learned_norm * confidence_learned, w_classical_norm * confidence_classical
            )

        # Reconstructed directly from the (cropped) OBSERVED wrapped phase,
        # never from a blend of the two methods' unwrapped phase values --
        # this is what guarantees wrap(unwrapped_phase) == wrapped_phase
        # exactly, regardless of how k_merged was computed above.
        final_unwrapped = wrapped_phase_c + 2.0 * np.pi * k_merged

        return HybridUnwrapResult(
            unwrapped_phase=final_unwrapped,
            ambiguity_map=k_merged,
            confidence_map=confidence_map,
            regime_mask=regime,
            uncertainty_map=learned_result.uncertainty,
            provenance_mask=provenance,
            wrapped_phase=wrapped_phase_c,
            geotiff_profile=learned_result.geotiff_profile,
        )


# --------------------------------------------------------------------------- #
# CLI (pyunwrap-unwrap)
# --------------------------------------------------------------------------- #


def build_argparser():
    """Build the CLI argument parser for `pyunwrap-unwrap`."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Unwrap an InSAR interferogram, optionally with hybrid regime-aware routing."
    )
    parser.add_argument(
        "--wrapped-phase", type=str, required=True, help="Path to wrapped-phase GeoTIFF."
    )
    parser.add_argument("--coherence", type=str, required=True, help="Path to coherence GeoTIFF.")
    parser.add_argument("--amplitude", type=str, required=True, help="Path to amplitude GeoTIFF.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a trained AmbiguityNet state_dict (.pt).",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output unwrapped-phase GeoTIFF path."
    )
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--k-max", type=float, default=10.0)
    parser.add_argument("--device", type=str, default=None, choices=[None, "cuda", "cpu"])
    parser.add_argument(
        "--hybrid",
        dest="hybrid",
        action="store_true",
        default=True,
        help="Enable hybrid regime-aware routing (default).",
    )
    parser.add_argument(
        "--no-hybrid",
        dest="hybrid",
        action="store_false",
        help="Disable hybrid routing; use the learned model alone.",
    )
    parser.add_argument("--high-coherence-threshold", type=float, default=0.75)
    parser.add_argument("--low-coherence-threshold", type=float, default=0.35)
    return parser


def main() -> None:
    """CLI entry point (`pyunwrap-unwrap`): parse args, load a checkpoint, and unwrap."""
    import torch

    from pyunwrap.models.ambiguity_net import AmbiguityNet

    parser = build_argparser()
    args = parser.parse_args()

    model = AmbiguityNet(pretrained=False, k_max=args.k_max)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    learned_unwrapper = PhaseUnwrapper(model=model, device=args.device)

    config = HybridUnwrapperConfig(
        use_hybrid_mode=args.hybrid,
        high_coherence_threshold=args.high_coherence_threshold,
        low_coherence_threshold=args.low_coherence_threshold,
    )
    hybrid = HybridUnwrapper(learned_unwrapper, config=config)
    result = hybrid.unwrap(
        args.wrapped_phase,
        args.coherence,
        args.amplitude,
        tile_size=args.tile_size,
        overlap=args.overlap,
    )
    result.save_geotiff(args.output)
    print(f"Wrote unwrapped phase to {args.output}")

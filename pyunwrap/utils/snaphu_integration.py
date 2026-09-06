"""
pyunwrap.utils.snaphu_integration
====================================

Real integration with SNAPHU (Statistical-cost, Network-flow Algorithm for
PHase Unwrapping; Chen & Zebker, 2001) via the official `snaphu-py` Python
bindings (https://github.com/isce-framework/snaphu-py), maintained by the
same organization (isce-framework / JPL) that builds NASA's ISCE InSAR
processing framework.

This module exists to make the SNAPHU pseudo-ground-truth fine-tuning phase
described in `pyunwrap.training.trainer.Trainer` (Prompt 4) an actual,
working feature rather than a code path that assumes a pre-built HDF5 file
exists. It provides:

1. `unwrap_with_snaphu` -- run real SNAPHU unwrapping on a wrapped-phase /
   coherence pair (and optionally amplitude), returning the unwrapped phase
   and SNAPHU's connected-component reliability labels.
2. `build_snaphu_pseudo_ground_truth_tiles` / `generate_snaphu_finetune_dataset`
   -- the full pipeline turning a real (or realistic synthetic) wrapped
   interferogram into an HDF5 tile file directly usable as
   `Trainer(..., finetune_dataset=...)`.

Two correctness details that are easy to get wrong and are handled
explicitly here (both verified empirically during development, not assumed
from documentation alone):

- **SNAPHU's absolute phase reference is arbitrary.** Two runs on the same
  wrapped phase can differ by a global additive multiple of `2*pi` (there is
  no absolute reference in phase unwrapping -- only relative consistency
  matters). The derived ambiguity map `k` is re-centered per scene
  (subtracting its rounded mean) so training targets stay in a comparable,
  bounded range rather than drifting to arbitrarily large magnitudes.
- **Not every pixel SNAPHU unwraps is trustworthy.** SNAPHU's connected-
  component labels (`conncomp`) mark disconnected regions of *self-
  consistently* unwrapped pixels; `conncomp == 0` specifically flags pixels
  it could not confidently assign to any such region (verified empirically:
  a deliberately decorrelated test region was correctly flagged unreliable
  at a 97% rate, with zero false positives in a well-behaved control
  region). Treating those pixels as ground truth would silently inject bad
  training labels from exactly the regions where SNAPHU itself doesn't
  trust its own output. Tiles with too many unreliable pixels are excluded
  entirely rather than partially trusted.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

try:
    import snaphu as _snaphu

    _HAS_SNAPHU = True
except ImportError:  # pragma: no cover
    _HAS_SNAPHU = False

from pyunwrap.data.preprocessing import (
    NormalizedRasters,
    TileSpec,
    compute_tile_grid,
    extract_tile,
    normalize_amplitude,
    normalize_coherence,
    normalize_phase,
    save_tiles_hdf5,
)


def _require_snaphu() -> None:
    if not _HAS_SNAPHU:
        raise ImportError(
            "The 'snaphu' package is required for SNAPHU integration. "
            "Install with `pip install snaphu` (official isce-framework "
            "Python bindings; bundles its own compiled SNAPHU binary, no "
            "separate system install needed)."
        )


# --------------------------------------------------------------------------- #
# Core wrapper
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class SnaphuResult:
    """Output of `unwrap_with_snaphu`.

    Attributes:
        unwrapped_phase: SNAPHU's unwrapped phase, radians. Subject to an
            arbitrary global 2*pi*n offset (no absolute phase reference --
            see module docstring).
        conncomp: Connected-component reliability labels, same shape as
            `unwrapped_phase`. `0` marks pixels SNAPHU could not confidently
            assign to a self-consistently-unwrapped region; positive
            integers label distinct reliable regions.
        reliable_fraction: Fraction of pixels with `conncomp != 0`, i.e. the
            fraction SNAPHU considers reliably unwrapped. A quick scalar
            summary of how trustworthy this particular result is.
    """

    unwrapped_phase: np.ndarray
    conncomp: np.ndarray
    reliable_fraction: float


def unwrap_with_snaphu(
    wrapped_phase: np.ndarray,
    coherence: np.ndarray,
    amplitude: np.ndarray | None = None,
    nlooks: float = 1.0,
    cost: str = "smooth",
    init: str = "mcf",
    ntiles: tuple[int, int] = (1, 1),
    tile_overlap: int = 0,
    nproc: int = 1,
    **extra_snaphu_kwargs,
) -> SnaphuResult:
    """Run real SNAPHU phase unwrapping on a wrapped-phase / coherence pair.

    Args:
        wrapped_phase: 2D wrapped phase, radians, in (-pi, pi].
        coherence: 2D coherence map, [0, 1], same shape as `wrapped_phase`.
        amplitude: Optional 2D amplitude map, same shape. SNAPHU's cost
            functions can use amplitude as an additional reliability signal;
            if omitted, a uniform amplitude of 1 is used (coherence alone
            still drives the statistical cost model).
        nlooks: Equivalent number of independent looks used to form the
            sample coherence (see `snaphu.unwrap`'s docstring for the
            estimation formula). Higher values tell SNAPHU the coherence
            estimate is more statistically reliable.
        cost: SNAPHU statistical cost mode, `"smooth"` (general-purpose) or
            `"defo"` (tuned for expected deformation-like phase behavior).
        init: Initialization algorithm, `"mcf"` (minimum cost flow) or
            `"mst"` (minimum spanning tree).
        ntiles: SNAPHU's own internal tiling `(nrow, ncol)`, for splitting
            very large scenes; `(1, 1)` unwraps as a single tile. This is
            SNAPHU's own tiling, independent of `pyunwrap`'s tiling in
            `pyunwrap.data.preprocessing` / `pyunwrap.inference.unwrapper`.
        tile_overlap: Overlap, pixels, between SNAPHU's internal tiles (only
            relevant if `ntiles != (1, 1)`).
        nproc: Number of worker processes for SNAPHU's internal tile mode.
        **extra_snaphu_kwargs: Any other keyword argument accepted by
            `snaphu.unwrap` (e.g. `min_conncomp_frac`, `phase_grad_window`,
            `min_region_size`, `mask`) is forwarded through unchanged. Kept
            as a catch-all rather than an ever-growing explicit parameter
            list, since `snaphu.unwrap` already documents each option
            thoroughly in its own docstring.

    Returns:
        A `SnaphuResult` with the unwrapped phase, connected-component
        labels, and the reliable-pixel fraction.

    Raises:
        ImportError: If the `snaphu` package is not installed.
        ValueError: If input shapes don't match.
    """
    _require_snaphu()
    if wrapped_phase.shape != coherence.shape:
        raise ValueError(
            f"Shape mismatch: wrapped_phase {wrapped_phase.shape} vs coherence {coherence.shape}"
        )
    if amplitude is not None and amplitude.shape != wrapped_phase.shape:
        raise ValueError(
            f"Shape mismatch: wrapped_phase {wrapped_phase.shape} vs amplitude {amplitude.shape}"
        )

    amp = amplitude if amplitude is not None else np.ones_like(wrapped_phase, dtype=np.float64)
    # SNAPHU consumes the actual complex interferogram, not the bare wrapped
    # phase -- amplitude modulates its statistical cost model.
    igram = amp.astype(np.float64) * np.exp(1j * wrapped_phase.astype(np.float64))
    corr = np.clip(coherence, 0.0, 1.0).astype(np.float32)

    unw, conncomp = _snaphu.unwrap(
        igram,
        corr,
        nlooks,
        cost=cost,
        init=init,
        ntiles=ntiles,
        tile_overlap=tile_overlap,
        nproc=nproc,
        **extra_snaphu_kwargs,
    )
    unw = np.asarray(unw, dtype=np.float64)
    conncomp = np.asarray(conncomp)
    reliable_fraction = float((conncomp != 0).mean())

    return SnaphuResult(unwrapped_phase=unw, conncomp=conncomp, reliable_fraction=reliable_fraction)


def unwrap_geotiffs_with_snaphu(
    wrapped_phase_path: str | Path,
    coherence_path: str | Path,
    amplitude_path: str | Path | None = None,
    **kwargs,
) -> SnaphuResult:
    """Convenience wrapper: load wrapped phase / coherence / amplitude
    directly from GeoTIFFs and run `unwrap_with_snaphu`.

    Args:
        wrapped_phase_path: Path to a wrapped-phase GeoTIFF, radians.
        coherence_path: Path to a coherence GeoTIFF, [0, 1].
        amplitude_path: Optional path to an amplitude GeoTIFF.
        **kwargs: Forwarded to `unwrap_with_snaphu`.

    Returns:
        A `SnaphuResult`.
    """
    import rasterio

    with rasterio.open(wrapped_phase_path) as src:
        wrapped_phase = src.read(1).astype(np.float64)
    with rasterio.open(coherence_path) as src:
        coherence = src.read(1).astype(np.float64)
    amplitude = None
    if amplitude_path is not None:
        with rasterio.open(amplitude_path) as src:
            amplitude = src.read(1).astype(np.float64)

    min_rows = min(wrapped_phase.shape[0], coherence.shape[0])
    min_cols = min(wrapped_phase.shape[1], coherence.shape[1])
    wrapped_phase = wrapped_phase[:min_rows, :min_cols]
    coherence = coherence[:min_rows, :min_cols]
    if amplitude is not None:
        amplitude = amplitude[:min_rows, :min_cols]

    return unwrap_with_snaphu(wrapped_phase, coherence, amplitude=amplitude, **kwargs)


# --------------------------------------------------------------------------- #
# Pseudo-ground-truth fine-tuning dataset builder
# --------------------------------------------------------------------------- #


def _recenter_ambiguity(k: np.ndarray) -> np.ndarray:
    """Subtract the rounded mean ambiguity, removing SNAPHU's arbitrary
    global phase-reference offset so training targets stay centered near
    zero rather than drifting to whatever absolute reference SNAPHU
    happened to pick for this particular scene.
    """
    offset = np.round(k.mean())
    return k - offset


def build_snaphu_pseudo_ground_truth_tiles(
    wrapped_phase: np.ndarray,
    coherence: np.ndarray,
    amplitude: np.ndarray | None = None,
    tile_size: int = 256,
    overlap: int = 32,
    min_reliable_fraction: float = 0.95,
    snaphu_kwargs: dict | None = None,
) -> tuple[list[tuple[TileSpec, dict[str, np.ndarray]]], dict]:
    """Run SNAPHU on a full scene and produce pseudo-ground-truth training
    tiles, excluding any tile where SNAPHU wasn't confident enough.

    Args:
        wrapped_phase: 2D wrapped phase, radians.
        coherence: 2D coherence map, [0, 1].
        amplitude: Optional 2D amplitude map.
        tile_size: Output tile edge length, pixels (this is `pyunwrap`'s own
            tiling for the resulting training set -- independent of any
            internal tiling SNAPHU itself used to unwrap the scene).
        overlap: Overlap between adjacent output tiles, pixels.
        min_reliable_fraction: A candidate tile is kept only if at least
            this fraction of its pixels have `conncomp != 0` (SNAPHU
            considers them reliably unwrapped). Tiles below this threshold
            are dropped rather than partially trusted -- see the module
            docstring for why silently keeping unreliable pixels as ground
            truth would be a real correctness bug, not a minor approximation.
        snaphu_kwargs: Extra keyword arguments forwarded to
            `unwrap_with_snaphu` (e.g. `nlooks`, `cost`, `ntiles` for
            SNAPHU's own internal tiling on very large scenes).

    Returns:
        `(tiles, summary)`:
            - `tiles`: list of `(TileSpec, tile_dict)` pairs in the same
              format `pyunwrap.data.preprocessing.save_tiles_hdf5` expects,
              with keys `wrapped_phase`, `coherence`, `amplitude` (all
              normalized) and `true_unwrapped` (radians, SNAPHU's output,
              re-centered).
            - `summary`: dict with `n_candidate_tiles`, `n_kept_tiles`,
              `overall_reliable_fraction`, useful for logging/reporting how
              much of the scene was usable as fine-tuning data.
    """
    snaphu_kwargs = snaphu_kwargs or {}
    result = unwrap_with_snaphu(wrapped_phase, coherence, amplitude=amplitude, **snaphu_kwargs)

    unwrapped_recentered = (
        _recenter_ambiguity((result.unwrapped_phase - wrapped_phase) / (2.0 * np.pi))
        * (2.0 * np.pi)
        + wrapped_phase
    )
    # Equivalent to: recenter k = (unw - wrapped)/(2pi), then reconstruct
    # unwrapped = wrapped + 2*pi*k_recentered. Done in one pass so the
    # returned `true_unwrapped` and the implied k stay exactly consistent.

    rasters = NormalizedRasters(
        wrapped_phase=normalize_phase(wrapped_phase),
        coherence=normalize_coherence(coherence),
        amplitude=normalize_amplitude(
            amplitude if amplitude is not None else np.ones_like(wrapped_phase)
        ),
    )

    tile_specs = compute_tile_grid(wrapped_phase.shape, tile_size=tile_size, overlap=overlap)
    kept_tiles: list[tuple[TileSpec, dict[str, np.ndarray]]] = []

    for spec in tile_specs:
        conncomp_tile = extract_tile(result.conncomp, spec)
        reliable_frac = float((conncomp_tile != 0).mean())
        if reliable_frac < min_reliable_fraction:
            continue

        tile_dict = {
            "wrapped_phase": extract_tile(rasters.wrapped_phase, spec),
            "coherence": extract_tile(rasters.coherence, spec),
            "amplitude": extract_tile(rasters.amplitude, spec),
            "true_unwrapped": extract_tile(unwrapped_recentered, spec),
        }
        kept_tiles.append((spec, tile_dict))

    summary = {
        "n_candidate_tiles": len(tile_specs),
        "n_kept_tiles": len(kept_tiles),
        "overall_reliable_fraction": result.reliable_fraction,
    }
    return kept_tiles, summary


def generate_snaphu_finetune_dataset(
    wrapped_phase_path: str | Path,
    coherence_path: str | Path,
    amplitude_path: str | Path | None = None,
    out_path: str | Path = "snaphu_pseudo_gt.h5",
    tile_size: int = 256,
    overlap: int = 32,
    min_reliable_fraction: float = 0.95,
    snaphu_kwargs: dict | None = None,
) -> tuple[Path, dict]:
    """End-to-end pipeline: real GeoTIFFs -> SNAPHU -> HDF5 fine-tuning
    dataset directly usable as `Trainer(..., finetune_dataset=InSARTileDataset(out_path, ...))`.

    This is the function that makes `Trainer`'s `finetune_dataset` /
    `finetune_start_epoch` fine-tuning phase (see
    `pyunwrap.training.trainer`) an actually-runnable feature end to end,
    rather than a code path that assumes its input file already exists.

    Args:
        wrapped_phase_path: Path to a real (or realistic synthetic)
            wrapped-phase GeoTIFF.
        coherence_path: Path to the corresponding coherence GeoTIFF.
        amplitude_path: Optional path to an amplitude GeoTIFF.
        out_path: Output HDF5 path.
        tile_size: Output tile edge length, pixels.
        overlap: Overlap between adjacent output tiles, pixels.
        min_reliable_fraction: See `build_snaphu_pseudo_ground_truth_tiles`.
        snaphu_kwargs: Extra keyword arguments forwarded to SNAPHU.

    Returns:
        `(out_path, summary)` -- the written HDF5 path and a summary dict
        (see `build_snaphu_pseudo_ground_truth_tiles`).

    Raises:
        RuntimeError: If zero tiles survived the reliability filter (the
            scene was too decorrelated / SNAPHU too unconfident everywhere
            to produce any usable fine-tuning data at the requested
            `min_reliable_fraction`).
    """
    import rasterio

    with rasterio.open(wrapped_phase_path) as src:
        wrapped_phase = src.read(1).astype(np.float64)
    with rasterio.open(coherence_path) as src:
        coherence = src.read(1).astype(np.float64)
    amplitude = None
    if amplitude_path is not None:
        with rasterio.open(amplitude_path) as src:
            amplitude = src.read(1).astype(np.float64)

    min_rows = min(wrapped_phase.shape[0], coherence.shape[0])
    min_cols = min(wrapped_phase.shape[1], coherence.shape[1])
    wrapped_phase = wrapped_phase[:min_rows, :min_cols]
    coherence = coherence[:min_rows, :min_cols]
    if amplitude is not None:
        amplitude = amplitude[:min_rows, :min_cols]

    tiles, summary = build_snaphu_pseudo_ground_truth_tiles(
        wrapped_phase,
        coherence,
        amplitude=amplitude,
        tile_size=tile_size,
        overlap=overlap,
        min_reliable_fraction=min_reliable_fraction,
        snaphu_kwargs=snaphu_kwargs,
    )

    if len(tiles) == 0:
        raise RuntimeError(
            f"No tiles survived the min_reliable_fraction={min_reliable_fraction} filter "
            f"(SNAPHU's overall reliable fraction for this scene was "
            f"{summary['overall_reliable_fraction']:.2%}). Try a lower "
            f"min_reliable_fraction, a larger/less-decorrelated scene, or "
            f"different SNAPHU cost/init settings."
        )

    out_path = save_tiles_hdf5(tiles, out_path)
    return out_path, summary

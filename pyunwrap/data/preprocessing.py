"""
pyunwrap.data.preprocessing
=============================

Preprocessing and tiling pipeline for large InSAR rasters.

Real interferograms are frequently far too large (e.g. 20,000 x 20,000 pixels)
to fit in GPU memory or be processed by a U-Net in a single forward pass. This
module provides:

1. Normalization of the three input rasters (wrapped phase, coherence,
   amplitude) into ranges suitable for neural network training.
2. A sliding-window tiling function that cuts massive rasters into overlapping
   patches, so a model trained on small tiles (e.g. 256x256) can later be
   applied to arbitrarily large scenes (see `pyunwrap.inference.unwrapper`,
   which is responsible for stitching tiles back together).
3. Fast on-disk storage of tiles as either raw `.npy`/`.npz` stacks or a
   single chunked HDF5 file (preferred for large tile counts, since it avoids
   the filesystem overhead of millions of tiny files).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np

try:
    import rasterio

    _HAS_RASTERIO = True
except ImportError:  # pragma: no cover
    _HAS_RASTERIO = False

try:
    import h5py

    _HAS_H5PY = True
except ImportError:  # pragma: no cover
    _HAS_H5PY = False


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #


def normalize_phase(phase_rad: np.ndarray) -> np.ndarray:
    """Scale wrapped phase from (-pi, pi] radians to [-1, 1].

    Args:
        phase_rad: Wrapped phase, radians, expected in (-pi, pi].

    Returns:
        Phase scaled to [-1, 1], same shape/dtype family (float32).
    """
    return (phase_rad / np.pi).astype(np.float32)


def denormalize_phase(phase_norm: np.ndarray) -> np.ndarray:
    """Inverse of `normalize_phase`: [-1, 1] -> radians in (-pi, pi]."""
    return (phase_norm * np.pi).astype(np.float64)


def normalize_coherence(coherence: np.ndarray) -> np.ndarray:
    """Clip coherence into [0, 1] and cast to float32.

    Coherence is already naturally bounded in [0, 1]; this mainly guards
    against small numerical overshoot from upstream processing (e.g.
    multilooking) and ensures a consistent dtype for the network input.

    Args:
        coherence: Interferometric coherence map.

    Returns:
        Coherence clipped to [0, 1], float32.
    """
    return np.clip(coherence, 0.0, 1.0).astype(np.float32)


def normalize_amplitude(
    amplitude: np.ndarray,
    percentile_clip: tuple[float, float] = (1.0, 99.0),
) -> np.ndarray:
    """Log-scale SAR amplitude and rescale to roughly [0, 1] via robust percentile clipping.

    Raw SAR amplitude has an extremely heavy-tailed distribution (a handful of
    strong corner reflectors / urban scatterers can be orders of magnitude
    brighter than the median). Log-scaling followed by percentile clipping
    (rather than min/max) prevents those outliers from crushing the dynamic
    range of the rest of the scene.

    Args:
        amplitude: Raw (linear-scale) SAR amplitude, non-negative.
        percentile_clip: (low, high) percentiles used to set the normalization
            range after log-scaling.

    Returns:
        Normalized amplitude, approximately in [0, 1], float32.
    """
    amp = np.clip(amplitude, a_min=1e-6, a_max=None)
    log_amp = np.log1p(amp)
    lo, hi = np.percentile(log_amp, percentile_clip)
    if hi <= lo:
        hi = lo + 1e-6
    norm = np.clip((log_amp - lo) / (hi - lo), 0.0, 1.0)
    return norm.astype(np.float32)


@dataclasses.dataclass
class NormalizedRasters:
    """Container for a set of normalized input rasters ready for tiling."""

    wrapped_phase: np.ndarray  # [-1, 1]
    coherence: np.ndarray  # [0, 1]
    amplitude: np.ndarray  # ~[0, 1]


def load_and_normalize_rasters(
    wrapped_phase_path: str | Path,
    coherence_path: str | Path,
    amplitude_path: str | Path,
) -> NormalizedRasters:
    """Load wrapped phase, coherence, and amplitude GeoTIFFs and normalize them.

    Args:
        wrapped_phase_path: Path to a single-band wrapped-phase GeoTIFF, radians.
        coherence_path: Path to a single-band coherence GeoTIFF, [0, 1].
        amplitude_path: Path to a single-band amplitude GeoTIFF, linear scale.

    Returns:
        `NormalizedRasters` with all three bands normalized and (if shapes
        differ slightly due to processing artifacts) cropped to a common
        minimum shape.

    Raises:
        ImportError: If rasterio is not installed.
    """
    if not _HAS_RASTERIO:
        raise ImportError("rasterio is required to load GeoTIFF rasters.")

    with rasterio.open(wrapped_phase_path) as src:
        phase = src.read(1).astype(np.float64)
    with rasterio.open(coherence_path) as src:
        coherence = src.read(1).astype(np.float64)
    with rasterio.open(amplitude_path) as src:
        amplitude = src.read(1).astype(np.float64)

    # Defensive crop: some pipelines produce rasters that differ by a pixel or
    # two due to independent resampling; align to the common minimum extent
    # rather than failing outright.
    min_rows = min(phase.shape[0], coherence.shape[0], amplitude.shape[0])
    min_cols = min(phase.shape[1], coherence.shape[1], amplitude.shape[1])
    phase = phase[:min_rows, :min_cols]
    coherence = coherence[:min_rows, :min_cols]
    amplitude = amplitude[:min_rows, :min_cols]

    return NormalizedRasters(
        wrapped_phase=normalize_phase(phase),
        coherence=normalize_coherence(coherence),
        amplitude=normalize_amplitude(amplitude),
    )


# --------------------------------------------------------------------------- #
# Sliding-window tiling
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class TileSpec:
    """Location and extent of one tile within a larger raster.

    Attributes:
        row: Top-row pixel offset of the tile within the source raster.
        col: Left-column pixel offset of the tile within the source raster.
        height: Tile height in pixels (may be smaller than `tile_size` at
            bottom/right borders if `pad_borders=False`).
        width: Tile width in pixels.
    """

    row: int
    col: int
    height: int
    width: int


def compute_tile_grid(
    raster_shape: tuple[int, int],
    tile_size: int = 256,
    overlap: int = 32,
) -> list[TileSpec]:
    """Compute the (row, col) grid of overlapping tile locations covering a raster.

    Tiles are stepped by `tile_size - overlap` pixels in each dimension so
    that neighboring tiles overlap by exactly `overlap` pixels; this overlap
    is what lets `pyunwrap.inference.unwrapper` later Hanning-blend the
    integer ambiguity maps across tile boundaries without discontinuities.

    The final row/column of tiles is shifted inward (rather than padded) so
    every tile stays fully inside the raster and has the full `tile_size`
    extent, at the cost of extra overlap along the bottom/right edges.

    Args:
        raster_shape: (rows, cols) of the full raster.
        tile_size: Tile edge length in pixels (square tiles).
        overlap: Overlap between adjacent tiles, in pixels. Must be smaller
            than `tile_size`.

    Returns:
        List of `TileSpec`, in row-major order, fully covering the raster.

    Raises:
        ValueError: If `overlap >= tile_size`, or the raster is smaller than
            one tile in either dimension.
    """
    if overlap >= tile_size:
        raise ValueError(f"overlap ({overlap}) must be smaller than tile_size ({tile_size}).")
    rows, cols = raster_shape
    if rows < tile_size or cols < tile_size:
        raise ValueError(
            f"Raster shape {raster_shape} is smaller than tile_size {tile_size} "
            "in at least one dimension."
        )

    stride = tile_size - overlap

    def _starts(extent: int) -> list[int]:
        starts = list(range(0, extent - tile_size + 1, stride))
        last_possible = extent - tile_size
        if starts[-1] != last_possible:
            starts.append(last_possible)  # ensure full coverage to the far edge
        return starts

    row_starts = _starts(rows)
    col_starts = _starts(cols)

    tiles = [
        TileSpec(row=r, col=c, height=tile_size, width=tile_size)
        for r in row_starts
        for c in col_starts
    ]
    return tiles


def extract_tile(array: np.ndarray, spec: TileSpec) -> np.ndarray:
    """Extract a single tile from a 2D array given a `TileSpec`.

    Args:
        array: Source 2D array.
        spec: Tile location/extent.

    Returns:
        The extracted tile, shape (spec.height, spec.width).
    """
    return array[spec.row : spec.row + spec.height, spec.col : spec.col + spec.width]


def iter_tiles(
    rasters: NormalizedRasters,
    true_unwrapped: np.ndarray | None = None,
    tile_size: int = 256,
    overlap: int = 32,
) -> Iterator[tuple[TileSpec, dict[str, np.ndarray]]]:
    """Iterate over all tiles of a (normalized) raster set, yielding tile data.

    Args:
        rasters: Normalized wrapped phase / coherence / amplitude rasters.
        true_unwrapped: Optional ground-truth unwrapped phase (radians), same
            shape as `rasters.wrapped_phase` — only available for
            synthetic/training data, not real inference inputs.
        tile_size: Tile edge length, pixels.
        overlap: Overlap between adjacent tiles, pixels.

    Yields:
        (TileSpec, dict) pairs, where the dict has keys `"wrapped_phase"`,
        `"coherence"`, `"amplitude"`, and (if provided) `"true_unwrapped"`.
    """
    shape = rasters.wrapped_phase.shape
    for spec in compute_tile_grid(shape, tile_size=tile_size, overlap=overlap):
        tile = {
            "wrapped_phase": extract_tile(rasters.wrapped_phase, spec),
            "coherence": extract_tile(rasters.coherence, spec),
            "amplitude": extract_tile(rasters.amplitude, spec),
        }
        if true_unwrapped is not None:
            tile["true_unwrapped"] = extract_tile(true_unwrapped, spec)
        yield spec, tile


# --------------------------------------------------------------------------- #
# Tile persistence
# --------------------------------------------------------------------------- #


def save_tiles_npz(
    tiles: Sequence[tuple[TileSpec, dict[str, np.ndarray]]],
    out_dir: str | Path,
    prefix: str = "tile",
) -> list[Path]:
    """Save each tile as an individual compressed `.npz` file.

    Simple and portable, but inefficient for very large tile counts (millions
    of small files stress most filesystems); prefer `save_tiles_hdf5` for
    large-scale training sets.

    Args:
        tiles: Sequence of (TileSpec, tile_dict) pairs, e.g. from `iter_tiles`.
        out_dir: Output directory (created if it doesn't exist).
        prefix: Filename prefix; files are named `{prefix}_{row}_{col}.npz`.

    Returns:
        List of paths written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for spec, tile in tiles:
        path = out_dir / f"{prefix}_{spec.row:06d}_{spec.col:06d}.npz"
        np.savez_compressed(path, row=spec.row, col=spec.col, **tile)
        written.append(path)
    return written


def save_tiles_hdf5(
    tiles: Sequence[tuple[TileSpec, dict[str, np.ndarray]]],
    out_path: str | Path,
    compression: str = "gzip",
    compression_opts: int = 4,
) -> Path:
    """Save all tiles into a single chunked HDF5 file for fast, scalable I/O.

    Layout: one top-level group per tile, named `tile_{index:08d}`, each
    containing datasets for every key in the tile dict plus `row`/`col`
    attributes recording the tile's position in the source raster (needed to
    reassemble spatial context, e.g. for the DataLoader's positional metadata
    or for debugging).

    Args:
        tiles: Sequence of (TileSpec, tile_dict) pairs.
        out_path: Output `.h5` file path.
        compression: h5py compression filter, e.g. "gzip" or "lzf".
        compression_opts: Compression level (for "gzip": 0-9).

    Returns:
        The path written to.

    Raises:
        ImportError: If h5py is not installed.
    """
    if not _HAS_H5PY:
        raise ImportError("h5py is required for HDF5 tile storage.")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_path, "w") as f:
        f.attrs["n_tiles"] = len(tiles)
        for idx, (spec, tile) in enumerate(tiles):
            grp = f.create_group(f"tile_{idx:08d}")
            grp.attrs["row"] = spec.row
            grp.attrs["col"] = spec.col
            grp.attrs["height"] = spec.height
            grp.attrs["width"] = spec.width
            for key, value in tile.items():
                grp.create_dataset(
                    key,
                    data=value.astype(np.float32),
                    compression=compression,
                    compression_opts=compression_opts if compression == "gzip" else None,
                )
    return out_path


def tile_and_save(
    wrapped_phase_path: str | Path,
    coherence_path: str | Path,
    amplitude_path: str | Path,
    out_path: str | Path,
    true_unwrapped: np.ndarray | None = None,
    tile_size: int = 256,
    overlap: int = 32,
    storage: str = "hdf5",
) -> Path:
    """End-to-end convenience function: load, normalize, tile, and persist.

    Args:
        wrapped_phase_path: Path to wrapped-phase GeoTIFF.
        coherence_path: Path to coherence GeoTIFF.
        amplitude_path: Path to amplitude GeoTIFF.
        out_path: Output path — a directory (for `storage="npz"`) or a single
            `.h5` file (for `storage="hdf5"`).
        true_unwrapped: Optional ground-truth unwrapped phase array, radians,
            matching the raw (unnormalized) raster shape.
        tile_size: Tile edge length, pixels.
        overlap: Overlap between adjacent tiles, pixels.
        storage: `"hdf5"` or `"npz"`.

    Returns:
        The path (file or directory) that was written.

    Raises:
        ValueError: If `storage` is not one of `"hdf5"` or `"npz"`.
    """
    rasters = load_and_normalize_rasters(wrapped_phase_path, coherence_path, amplitude_path)
    tiles = list(
        iter_tiles(rasters, true_unwrapped=true_unwrapped, tile_size=tile_size, overlap=overlap)
    )

    if storage == "hdf5":
        return save_tiles_hdf5(tiles, out_path)
    elif storage == "npz":
        save_tiles_npz(tiles, out_path)
        return Path(out_path)
    else:
        raise ValueError(f"Unknown storage backend: {storage!r}. Use 'hdf5' or 'npz'.")

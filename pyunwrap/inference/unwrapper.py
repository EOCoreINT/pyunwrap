"""
pyunwrap.inference.unwrapper
===============================

`PhaseUnwrapper`: production inference pipeline for applying a trained
`AmbiguityNet` to arbitrarily large interferograms.

Because the network was trained on small tiles (e.g. 256x256), a large
interferogram (e.g. 20,000x20,000) must be processed tile-by-tile. Naively
averaging the *unwrapped phase* in overlapping regions is unsafe: two
overlapping tiles can correctly agree on the wrapped phase while disagreeing
on the integer ambiguity by exactly the amount needed to make an *average of
phases* fall between two valid cycles, creating a physically meaningless
result. This module instead:

1. Averages the **integer ambiguity maps** `k` in overlapping regions
   (never the phase directly).
2. Uses a **Hanning window** per tile for smooth spatial blending, so no
   tile boundary is ever a hard cut.
3. **Weights the blend by each tile's residue probability** (the model's own
   uncertainty output), so confident tiles dominate uncertain ones in
   overlap regions.
4. Reconstructs the global unwrapped phase as `phi = psi + 2*pi*k_merged`
   from the *original, un-tiled* wrapped phase raster, so the reconstruction
   is always exactly consistent with the true input data everywhere (not
   just within each tile).
"""

from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path

import numpy as np
import torch

try:
    import rasterio

    _HAS_RASTERIO = True
except ImportError:  # pragma: no cover
    _HAS_RASTERIO = False

from pyunwrap.data.preprocessing import (
    NormalizedRasters,
    TileSpec,
    compute_tile_grid,
    extract_tile,
    load_and_normalize_rasters,
)
from pyunwrap.models.ambiguity_net import AmbiguityNet
from pyunwrap.utils.deployment import InferenceEngine, ModelCache, zenodo_download_url

#: Total downsampling stride of `AmbiguityNet`'s ResNet-34 encoder (2 stem +
#: maxpool stages, then 4 residual stages: 2*2*2*2*2 = 32). Tile sizes must
#: be a multiple of this -- see `PhaseUnwrapper.unwrap`'s docstring.
_ENCODER_STRIDE = 32


def _round_to_multiple(value: int, multiple: int) -> int:
    """Round `value` to the nearest positive multiple of `multiple` (used to
    suggest a valid `tile_size` in the `ValueError` message below)."""
    rounded = round(value / multiple) * multiple
    return max(rounded, multiple)


# --------------------------------------------------------------------------- #
# Merge weighting
# --------------------------------------------------------------------------- #


def hanning_window_2d(size: int) -> np.ndarray:
    """2D separable Hanning window, useful as a *reference* taper shape.

    Not used directly for tile blending -- see `build_tile_window`, which
    only tapers edges that actually border a neighboring tile. A plain
    Hanning window goes to exactly zero at *every* edge of the tile,
    including edges that sit on the outer boundary of the whole scene (where
    no neighboring tile exists to pick up the lost weight), which would
    collapse border pixels to ~0/~0. This function is kept as a simple,
    standalone utility (e.g. for visualization or testing the taper shape in
    isolation).

    Args:
        size: Tile edge length, pixels.

    Returns:
        2D array, shape (size, size), values in [0, 1], peaking at the
        tile's center.
    """
    win_1d = np.hanning(size)
    if size <= 1 or np.allclose(win_1d, 0.0):
        win_1d = np.ones(size)
    return np.outer(win_1d, win_1d)


def _edge_aware_ramp_1d(
    length: int, overlap: int, taper_start: bool, taper_end: bool
) -> np.ndarray:
    """1D taper: a raised-cosine (Hanning-style) ramp of width `overlap` at
    each end that actually needs to blend with a neighbor, and flat 1.0
    everywhere else (including ends that sit on the scene's outer boundary
    and therefore must NOT be tapered to zero).

    Args:
        length: Length of this axis (tile_size).
        overlap: Ramp width, pixels (should not exceed length // 2).
        taper_start: Whether the start of this axis borders another tile
            (and therefore needs a rising ramp from 0 -> 1).
        taper_end: Whether the end of this axis borders another tile
            (needs a falling ramp from 1 -> 0).

    Returns:
        1D array, shape (length,), values in [0, 1].
    """
    win = np.ones(length, dtype=np.float64)
    ov = max(int(overlap), 1)
    ov = min(ov, length // 2) if length >= 2 else 0
    if ov == 0:
        return win

    # Raised-cosine ramp from 0 -> 1 over `ov` samples (the rising half of a
    # Hanning window), used at both ends (mirrored for the falling ramp).
    ramp_up = 0.5 - 0.5 * np.cos(np.pi * (np.arange(ov) + 1) / (ov + 1))

    if taper_start:
        win[:ov] = ramp_up
    if taper_end:
        win[-ov:] = ramp_up[::-1]
    return win


def build_tile_window(
    spec: TileSpec,
    tile_size: int,
    overlap: int,
    raster_shape: tuple[int, int],
) -> np.ndarray:
    """Build the 2D blend weight window for one tile, tapering only the edges
    that actually border a neighboring tile.

    This is the key fix that makes Hanning-style blending safe to use across
    an entire scene: edges on the true outer boundary of the raster (where
    `spec.row == 0`, `spec.col == 0`, or the tile reaches the raster's far
    edge) are left at full weight (1.0) rather than tapered to zero, since no
    neighboring tile exists there to pick up the lost weight.

    Args:
        spec: This tile's location/extent (a `pyunwrap.data.preprocessing.TileSpec`).
        tile_size: Tile edge length, pixels.
        overlap: Nominal overlap width used to size the taper ramp, pixels.
        raster_shape: (rows, cols) of the full scene, used to detect which
            edges of `spec` sit on the scene's outer boundary.

    Returns:
        2D array, shape (spec.height, spec.width), values in [0, 1].
    """
    rows, cols = raster_shape
    top_interior = spec.row > 0
    bottom_interior = (spec.row + spec.height) < rows
    left_interior = spec.col > 0
    right_interior = (spec.col + spec.width) < cols

    row_win = _edge_aware_ramp_1d(
        spec.height, overlap, taper_start=top_interior, taper_end=bottom_interior
    )
    col_win = _edge_aware_ramp_1d(
        spec.width, overlap, taper_start=left_interior, taper_end=right_interior
    )
    return np.outer(row_win, col_win)


@dataclasses.dataclass
class TileInferenceResult:
    """Raw per-tile model output, before merging into the global raster.

    Attributes:
        k_hat: Integer ambiguity map for this tile, shape (H, W).
        residue_prob: Residue/uncertainty probability map, shape (H, W).
        k_std: Per-pixel standard deviation of `k_hat` across Monte Carlo
            Dropout passes (uncertainty estimate), shape (H, W).
    """

    k_hat: np.ndarray
    residue_prob: np.ndarray
    k_std: np.ndarray


# --------------------------------------------------------------------------- #
# PhaseUnwrapper
# --------------------------------------------------------------------------- #


class PhaseUnwrapper:
    """Production tiled-inference pipeline for InSAR phase unwrapping.

    Supports two backends:
        - `"torch"`: a loaded `AmbiguityNet` PyTorch module (supports Monte
          Carlo Dropout uncertainty via repeated stochastic forward passes).
        - `"onnx"`: an exported ONNX model run through
          `pyunwrap.utils.deployment.InferenceEngine` (fast, portable, no
          PyTorch dependency at inference time; MC Dropout uncertainty is
          unavailable since exported graphs run in eval mode).

    Example:
        >>> unwrapper = PhaseUnwrapper(model=trained_model, device="cuda")
        >>> result = unwrapper.unwrap(
        ...     wrapped_phase_path="wrapped.tif",
        ...     coherence_path="coherence.tif",
        ...     amplitude_path="amplitude.tif",
        ...     tile_size=512, overlap=64,
        ... )
        >>> result.unwrapped_phase.shape
        (H, W)
    """

    def __init__(
        self,
        model: AmbiguityNet | None = None,
        onnx_path: str | Path | None = None,
        device: str | None = None,
        mc_dropout_passes: int = 5,
    ) -> None:
        """
        Args:
            model: A loaded `AmbiguityNet` PyTorch module. Mutually exclusive
                with `onnx_path`.
            onnx_path: Path to an exported ONNX model. Mutually exclusive
                with `model`.
            device: `"cuda"` or `"cpu"` for the torch backend; ignored for
                the ONNX backend (handled by `InferenceEngine`'s own
                GPU-then-CPU-then-OpenVINO fallback).
            mc_dropout_passes: Number of stochastic forward passes used to
                estimate per-pixel ambiguity uncertainty via Monte Carlo
                Dropout. Only applies to the torch backend.

        Raises:
            ValueError: If both or neither of `model`/`onnx_path` are given.
        """
        if (model is None) == (onnx_path is None):
            raise ValueError("Provide exactly one of `model` or `onnx_path`.")

        self.mc_dropout_passes = mc_dropout_passes

        if model is not None:
            self.backend = "torch"
            self.device = (
                torch.device(device)
                if device
                else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
            self.model = model.to(self.device)
            self.engine = None
        else:
            self.backend = "onnx"
            self.model = None
            self.engine = InferenceEngine(onnx_path, prefer_gpu=True)
            print(f"[PhaseUnwrapper] ONNX backend initialized: {self.engine.backend}")

    @classmethod
    def from_pretrained(
        cls,
        name: str,
        zenodo_record_id: str | None = None,
        filename: str | None = None,
        cache_dir: str | Path | None = None,
        device: str | None = None,
    ) -> PhaseUnwrapper:
        """Load a `PhaseUnwrapper` from a named pretrained ONNX model, downloading
        and caching it from Zenodo on first use.

        Args:
            name: Local cache filename, e.g. `"pyunwrap-v1.onnx"`. Also used
                to build the default remote filename if `filename` is omitted.
            zenodo_record_id: Numeric Zenodo record id to download from, if
                the model is not already cached. Required unless the model
                is already present in the cache.
            filename: Remote filename within the Zenodo record, if different
                from `name`.
            cache_dir: Override the default `~/.pyunwrap/models/` cache dir.
            device: Passed through to `InferenceEngine` selection logic.

        Returns:
            A `PhaseUnwrapper` backed by the cached/downloaded ONNX model.

        Raises:
            RuntimeError: If the model is not cached and either
                `zenodo_record_id` is not provided, or the download fails.
        """
        cache = ModelCache(cache_dir=cache_dir)
        if cache.is_cached(name):
            path = cache.local_path(name)
        else:
            if zenodo_record_id is None:
                raise RuntimeError(
                    f"Model '{name}' is not cached at {cache.local_path(name)} and no "
                    "`zenodo_record_id` was provided to download it from."
                )
            url = zenodo_download_url(zenodo_record_id, filename or name)
            path = cache.get_or_download(name, url)
        return cls(onnx_path=path, device=device)

    # ----------------------------------------------------------------- #
    # Per-tile forward pass
    # ----------------------------------------------------------------- #

    def _run_tile_torch(self, x: np.ndarray) -> TileInferenceResult:
        """Run one tile through the PyTorch backend, with MC Dropout uncertainty.

        Args:
            x: Tile input, shape [3, H, W] (wrapped phase norm, coherence,
                amplitude).

        Returns:
            `TileInferenceResult` for this tile.
        """
        x_t = torch.from_numpy(x).float().unsqueeze(0).to(self.device)

        if self.mc_dropout_passes > 1:
            self._enable_mc_dropout(self.model)
            k_samples = []
            residue_samples = []
            with torch.no_grad():
                for _ in range(self.mc_dropout_passes):
                    out = self.model(x_t)
                    k_samples.append(out.k_hat.squeeze().cpu().numpy())
                    residue_samples.append(out.residue_prob.squeeze().cpu().numpy())
            self.model.eval()  # restore standard eval mode after MC Dropout sampling

            k_stack = np.stack(k_samples, axis=0)
            k_hat = np.median(k_stack, axis=0)  # median is robust to occasional off-by-one flips
            k_std = k_stack.std(axis=0)
            residue_prob = np.mean(residue_samples, axis=0)
        else:
            self.model.eval()
            with torch.no_grad():
                out = self.model(x_t)
            k_hat = out.k_hat.squeeze().cpu().numpy()
            residue_prob = out.residue_prob.squeeze().cpu().numpy()
            k_std = np.zeros_like(k_hat)

        return TileInferenceResult(k_hat=k_hat, residue_prob=residue_prob, k_std=k_std)

    @staticmethod
    def _enable_mc_dropout(model: torch.nn.Module) -> None:
        """Put the model in eval mode except for any `Dropout*` layers, which
        are switched to train mode so they remain stochastic across the
        repeated forward passes used for Monte Carlo Dropout uncertainty
        estimation.

        Note: `AmbiguityNet` as defined in `ambiguity_net.py` does not
        currently include explicit Dropout layers (BatchNorm provides some
        stochasticity in train mode, but is not a substitute for Dropout).
        This method is written to correctly enable MC Dropout uncertainty
        estimation for any future revision of the architecture that adds
        `nn.Dropout`/`nn.Dropout2d` layers, without requiring any change to
        `PhaseUnwrapper`'s calling code.
        """
        model.eval()
        for module in model.modules():
            if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
                module.train()

    def _run_tile_onnx(self, x: np.ndarray) -> TileInferenceResult:
        """Run one tile through the ONNX backend (no MC Dropout support).

        Args:
            x: Tile input, shape [3, H, W].

        Returns:
            `TileInferenceResult` for this tile (`k_std` is all zeros, since
            exported ONNX graphs run deterministically in eval mode).
        """
        x_batch = x[np.newaxis, ...].astype(np.float32)
        outputs = self.engine.run(x_batch)
        k_hat, _k_cont, residue_prob, _phi_hat = outputs
        return TileInferenceResult(
            k_hat=k_hat[0, 0],
            residue_prob=residue_prob[0, 0],
            k_std=np.zeros_like(k_hat[0, 0]),
        )

    def _run_tile(self, x: np.ndarray) -> TileInferenceResult:
        if self.backend == "torch":
            return self._run_tile_torch(x)
        return self._run_tile_onnx(x)

    # ----------------------------------------------------------------- #
    # Smart tile merging
    # ----------------------------------------------------------------- #

    def _merge_tiles(
        self,
        raster_shape: tuple[int, int],
        tile_size: int,
        overlap: int,
        rasters: NormalizedRasters,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run inference on every tile and Hanning/residue-weighted-merge the
        integer ambiguity maps into a single global raster.

        Args:
            raster_shape: (rows, cols) of the full scene.
            tile_size: Tile edge length, pixels.
            overlap: Overlap between adjacent tiles, pixels.
            rasters: Normalized wrapped phase / coherence / amplitude for the
                full scene.

        Returns:
            (k_merged, residue_prob_merged, k_uncertainty_merged): three
            arrays of shape `raster_shape`, being the globally-merged integer
            ambiguity map, residue probability map, and MC-Dropout ambiguity
            standard deviation map, respectively.
        """
        tile_specs = compute_tile_grid(raster_shape, tile_size=tile_size, overlap=overlap)

        k_accum = np.zeros(raster_shape, dtype=np.float64)
        residue_accum = np.zeros(raster_shape, dtype=np.float64)
        uncertainty_accum = np.zeros(raster_shape, dtype=np.float64)
        weight_accum = np.zeros(raster_shape, dtype=np.float64)

        for spec in tile_specs:
            wrapped_tile = extract_tile(rasters.wrapped_phase, spec)
            coherence_tile = extract_tile(rasters.coherence, spec)
            amplitude_tile = extract_tile(rasters.amplitude, spec)
            x = np.stack([wrapped_tile, coherence_tile, amplitude_tile], axis=0)

            result = self._run_tile(x)

            # Per-pixel blend weight: an edge-aware raised-cosine taper
            # (spatial smoothness across genuine tile overlaps, full weight
            # at the scene's outer boundary -- see `build_tile_window`)
            # times (1 - residue_prob) (down-weight pixels the model itself
            # flags as uncertain), with a small floor so a tile is never
            # given exactly zero weight everywhere.
            window = build_tile_window(spec, tile_size, overlap, raster_shape)
            confidence = np.clip(1.0 - result.residue_prob, 0.05, 1.0)
            tile_weight = window * confidence

            row_sl = slice(spec.row, spec.row + spec.height)
            col_sl = slice(spec.col, spec.col + spec.width)

            k_accum[row_sl, col_sl] += result.k_hat * tile_weight
            residue_accum[row_sl, col_sl] += result.residue_prob * tile_weight
            uncertainty_accum[row_sl, col_sl] += result.k_std * tile_weight
            weight_accum[row_sl, col_sl] += tile_weight

        # Every pixel is covered by at least one tile by construction of
        # `compute_tile_grid`, so weight_accum should never be exactly zero;
        # guard anyway against a degenerate all-uncertain-everywhere corner case.
        weight_accum = np.clip(weight_accum, 1e-8, None)

        k_merged = k_accum / weight_accum
        residue_merged = residue_accum / weight_accum
        uncertainty_merged = uncertainty_accum / weight_accum

        # The merged ambiguity map must be integer-valued for the final
        # `phi = psi + 2*pi*k` reconstruction to remain physically valid;
        # round only at the very end, after all continuous blending is done.
        k_merged = np.round(k_merged)

        return k_merged, residue_merged, uncertainty_merged

    # ----------------------------------------------------------------- #
    # Top-level API
    # ----------------------------------------------------------------- #

    def unwrap(
        self,
        wrapped_phase_path: str | Path,
        coherence_path: str | Path,
        amplitude_path: str | Path,
        tile_size: int = 512,
        overlap: int = 64,
        generate_report: bool = False,
        report_out_dir: str | Path | None = None,
    ) -> UnwrapResult:
        """Run tiled inference over a full-scene interferogram and reconstruct
        the global unwrapped phase.

        Args:
            wrapped_phase_path: Path to the wrapped-phase GeoTIFF, radians.
            coherence_path: Path to the coherence GeoTIFF, [0, 1].
            amplitude_path: Path to the amplitude GeoTIFF.
            tile_size: Tile edge length used for inference, pixels. Must be a
                multiple of the encoder's total downsampling stride (32, for
                the ResNet-34 encoder) -- see the "Tile size constraint" note
                below.
            overlap: Overlap between adjacent tiles, pixels.
            generate_report: If True, trigger the analytics/report pipeline
                (`pyunwrap.analytics.report_generator`) after inference.
                Wrapped in try/except so a reporting failure never prevents
                the core unwrapped-phase result from being returned.
            report_out_dir: Output directory for the generated report, if
                `generate_report=True`. Defaults to a `report/` subdirectory
                next to `wrapped_phase_path`.

        Returns:
            An `UnwrapResult` with the unwrapped phase, merged ambiguity map,
            residue probability map, and uncertainty map, all at full scene
            resolution.

        Raises:
            ValueError: If `tile_size` is not a multiple of the encoder
                stride (32).

        Tile size constraint
        ----------------------
        `AmbiguityNet`'s ResNet-34 encoder downsamples by a total factor of
        32 across its 5 stages. The PyTorch model handles arbitrary input
        sizes at *runtime* via an explicit `F.interpolate` safety net when a
        decoder skip connection doesn't line up exactly (see
        `ambiguity_net.py`). However, an ONNX graph traced from a specific
        input size only captures the control-flow branch actually taken
        during tracing: if traced at a size divisible by 32 (where every
        skip connection lines up exactly), the safety-net branch is never
        recorded in the graph, and the exported model then fails (with an
        opaque shape-mismatch error deep in ONNX Runtime, not a helpful one)
        on any tile size that *isn't* a multiple of 32. Rather than let that
        surface as a cryptic backend error, `tile_size` is validated
        up front for both backends, so the constraint is enforced
        consistently regardless of which one is in use.
        """
        if tile_size % _ENCODER_STRIDE != 0:
            raise ValueError(
                f"tile_size={tile_size} must be a multiple of {_ENCODER_STRIDE} "
                f"(the ResNet-34 encoder's total downsampling stride). "
                f"Try tile_size={_round_to_multiple(tile_size, _ENCODER_STRIDE)} instead. "
                "See the `unwrap()` docstring's 'Tile size constraint' note for why."
            )

        rasters = load_and_normalize_rasters(wrapped_phase_path, coherence_path, amplitude_path)
        raster_shape = rasters.wrapped_phase.shape

        k_merged, residue_merged, uncertainty_merged = self._merge_tiles(
            raster_shape,
            tile_size,
            overlap,
            rasters,
        )

        # Reconstruct from the ORIGINAL (un-tiled, un-normalized) wrapped
        # phase, so the final result is exactly consistent with the true
        # input data everywhere -- not merely within each tile's own
        # (possibly slightly re-normalized) view of it.
        with rasterio.open(wrapped_phase_path) as src:
            wrapped_phase_rad = src.read(1).astype(np.float64)
            profile = src.profile

        min_rows = min(wrapped_phase_rad.shape[0], k_merged.shape[0])
        min_cols = min(wrapped_phase_rad.shape[1], k_merged.shape[1])
        wrapped_phase_rad = wrapped_phase_rad[:min_rows, :min_cols]
        k_merged = k_merged[:min_rows, :min_cols]
        residue_merged = residue_merged[:min_rows, :min_cols]
        uncertainty_merged = uncertainty_merged[:min_rows, :min_cols]

        unwrapped_phase = wrapped_phase_rad + 2.0 * np.pi * k_merged

        result = UnwrapResult(
            unwrapped_phase=unwrapped_phase,
            ambiguity_map=k_merged,
            residue_prob=residue_merged,
            uncertainty=uncertainty_merged,
            wrapped_phase=wrapped_phase_rad,
            geotiff_profile=profile,
        )

        if generate_report:
            self._try_generate_report(result, wrapped_phase_path, report_out_dir)

        return result

    def _try_generate_report(
        self,
        result: UnwrapResult,
        wrapped_phase_path: str | Path,
        report_out_dir: str | Path | None,
    ) -> None:
        """Best-effort report generation; failures are logged, never raised,
        so a broken plotting/reporting dependency can never take down a
        production inference call. See `pyunwrap.analytics.report_generator`
        (Prompt 7) for the actual report pipeline.
        """
        try:
            from pyunwrap.analytics import (
                phase_stats,  # noqa: F401  (Prompt 6)
                report_generator,
            )

            out_dir = (
                Path(report_out_dir)
                if report_out_dir
                else Path(wrapped_phase_path).parent / "report"
            )
            report_generator.generate_report(result, out_dir)  # type: ignore[attr-defined]
        except Exception as exc:
            warnings.warn(
                f"Report generation failed and was skipped (core inference result is "
                f"still valid and returned): {exc}"
            )


@dataclasses.dataclass
class UnwrapResult:
    """Full-scene output of `PhaseUnwrapper.unwrap`.

    Attributes:
        unwrapped_phase: Reconstructed unwrapped phase, radians, full scene
            resolution.
        ambiguity_map: Merged integer ambiguity map k, same shape.
        residue_prob: Merged residue/uncertainty probability map, [0, 1].
        uncertainty: Merged Monte Carlo Dropout ambiguity standard deviation
            map (all zeros for the ONNX backend, which is deterministic).
        wrapped_phase: The original input wrapped phase, radians, cropped to
            match the other arrays' shape.
        geotiff_profile: The source wrapped-phase GeoTIFF's rasterio profile
            (CRS, transform, etc.), for writing results back out with the
            same georeferencing.
    """

    unwrapped_phase: np.ndarray
    ambiguity_map: np.ndarray
    residue_prob: np.ndarray
    uncertainty: np.ndarray
    wrapped_phase: np.ndarray
    geotiff_profile: dict

    def save_geotiff(self, path: str | Path, array_name: str = "unwrapped_phase") -> Path:
        """Write one of this result's arrays to disk as a single-band GeoTIFF,
        reusing the source raster's georeferencing.

        Args:
            path: Output GeoTIFF path.
            array_name: Which attribute to save (`"unwrapped_phase"`,
                `"ambiguity_map"`, `"residue_prob"`, or `"uncertainty"`).

        Returns:
            The path written to.
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

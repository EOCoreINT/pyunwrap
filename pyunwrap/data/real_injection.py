"""
pyunwrap.data.real_injection
================================

Strategy 1: a real-data + synthetic-injection training pipeline.

Training exclusively on `pyunwrap.synthetic.generator`'s fully synthetic
scenes is a real, documented limitation of this project (see
`docs/experiments.md` and the README's "Project status" section): real SAR
imagery has sensor-specific speckle statistics, real atmospheric artifacts,
and real decorrelation patterns that a synthetic generator only
approximates. This module builds training samples a different way: start
from a REAL (or, for testing, real-*like*) wrapped-phase/amplitude/
coherence stack, inject synthetic deformation with an *exactly known*
ground truth on top of it, and derive the integer ambiguity label from
that -- so the network trains against real noise texture with a real,
verifiable label, rather than an entirely synthetic scene.

Honest scope note: this module has been developed and tested against a
small, clearly-labeled synthetic *fixture* standing in for real data (see
`build_tiny_fixture_stack`), not against genuine satellite imagery -- this
sandbox's network access does not reach any real SAR data provider
(Copernicus Data Space, ASF Vertex, etc.), a limitation already documented
elsewhere in this project. The pipeline is written to accept genuinely
real `.npy`/GeoTIFF stacks with the documented shapes and has no
dependency on synthetic data internally once a stack is loaded; validating
it against real Sentinel-1 data is real future work, not something this
module can claim to have already done.

How ground truth is constructed
-----------------------------------
Given a real observed wrapped phase `psi_real` (one epoch of the loaded
stack) and a synthetic deformation field `phi_synth` (unwrapped, radians)::

    phi_base  = background_unwrapped if given, else psi_real itself
                (i.e. by default, the real background is treated as having
                zero ambiguity of its own -- this is a documented modeling
                choice, not a hidden default: if you have a better estimate
                of the real background's true unwrapped phase, e.g. from a
                classical unwrapper's output on the real data, pass it as
                `background_unwrapped` for a more accurate label)
    phi_true  = phi_base + phi_synth
    psi_observed = wrap(phi_true)
    k_true    = round((phi_true - psi_observed) / (2*pi))

Because `phi_base` is built from the *real* observed phase, `psi_observed`
still contains real speckle, decorrelation, and atmospheric texture --
`preserve_real_noise=True` (the default) is enforced by construction here,
not as a post-hoc filter.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np

try:
    import rasterio

    _HAS_RASTERIO = True
except ImportError:  # pragma: no cover
    _HAS_RASTERIO = False

from pyunwrap.data.preprocessing import (
    TileSpec,
    compute_tile_grid,
    extract_tile,
    normalize_amplitude,
    normalize_coherence,
    normalize_phase,
    save_tiles_hdf5,
)
from pyunwrap.synthetic.generator import (
    WAVELENGTH_M,
    compute_ambiguity,
    displacement_to_phase,
    gaussian_bowl_deformation,
    mogi_point_source,
    okada_dislocation,
    orbital_ramp,
    wrap_phase,
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

InjectionKind = Literal["gaussian_bowl", "mogi", "okada", "linear_ramp", "nonlinear_transient"]


@dataclass
class RealSyntheticInjectionConfig:
    """Configuration for the real-data + synthetic-injection pipeline.

    Attributes:
        inject_mogi: Whether Mogi point-source injection is enabled.
        inject_okada: Whether Okada fault-dislocation injection is enabled.
        inject_gaussian_bowl: Whether Gaussian subsidence-bowl injection is
            enabled.
        inject_linear_ramp: Whether residual orbital/atmospheric linear
            ramp injection is enabled.
        inject_nonlinear_transient: Whether the nonlinear transient
            deformation model is enabled (see
            `nonlinear_transient_deformation`'s docstring for exactly what
            this simplified model represents).
        min_coherence_for_background: Real background pixels with
            coherence below this value are excluded from injected training
            tiles (see `build_real_injection_tiles`'s `min_reliable_fraction`-
            style filtering).
        max_coherence_for_background: Real background pixels with
            coherence above this value are excluded (rarely needed; present
            for symmetry and for deliberately targeting a specific
            coherence band, e.g. to build a low-coherence-only dataset).
        preserve_real_noise: Enforced by construction in this module (see
            module docstring) rather than actually toggleable; kept as an
            explicit field so it is visible in serialized configs and cache
            hashes, and so a future contributor who wires in an alternative
            (noise-replacing) injection path has an obvious flag to gate it
            on. Currently, setting this to `False` raises `NotImplementedError`
            rather than silently doing nothing.
        wrap_after_injection: Whether to wrap `phi_true` after injection to
            produce the observed phase. `True` is the physically correct
            setting for essentially all real use (an interferogram sensor
            only ever measures wrapped phase); `False` is provided for
            debugging/inspection of the pre-wrap injected field only, and
            such samples are not valid training tiles.
        spatial_split_strategy: Only `"blocks"` is currently implemented
            (non-overlapping spatial blocks assigned wholesale to a single
            split, preventing pixel-level train/val/test leakage between
            spatially adjacent pixels). Reserved as a string (not a bool)
            for future strategies.
        train_fraction: Fraction of spatial blocks assigned to training.
        val_fraction: Fraction assigned to validation.
        test_fraction: Fraction assigned to test. The three fractions must
            sum to `1.0`.
        seed: Seed for injection randomization and the train/val/test block
            shuffle, for full reproducibility.
        block_size: Edge length, pixels, of each spatial block used by the
            `"blocks"` split strategy.
        wavelength_m: Radar wavelength used to convert injected
            displacement (meters) to phase (radians); defaults to Sentinel-1
            C-band.

    Raises:
        ValueError: If the three split fractions don't sum to `1.0`, if any
            is negative, if `min_coherence_for_background >
            max_coherence_for_background`, or if `block_size < 1`.
    """

    inject_mogi: bool = True
    inject_okada: bool = True
    inject_gaussian_bowl: bool = True
    inject_linear_ramp: bool = True
    inject_nonlinear_transient: bool = True
    min_coherence_for_background: float = 0.0
    max_coherence_for_background: float = 1.0
    preserve_real_noise: bool = True
    wrap_after_injection: bool = True
    spatial_split_strategy: str = "blocks"
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 42
    block_size: int = 64
    wavelength_m: float = WAVELENGTH_M["C-band"]

    def __post_init__(self) -> None:
        total = self.train_fraction + self.val_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"train_fraction + val_fraction + test_fraction must sum to 1.0, got {total}"
            )
        for name, value in [
            ("train_fraction", self.train_fraction),
            ("val_fraction", self.val_fraction),
            ("test_fraction", self.test_fraction),
        ]:
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        if self.min_coherence_for_background > self.max_coherence_for_background:
            raise ValueError(
                "min_coherence_for_background must be <= max_coherence_for_background, got "
                f"{self.min_coherence_for_background} > {self.max_coherence_for_background}"
            )
        if self.block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {self.block_size}")
        if self.spatial_split_strategy != "blocks":
            raise ValueError(
                f"spatial_split_strategy must be 'blocks' (the only strategy currently "
                f"implemented), got {self.spatial_split_strategy!r}"
            )
        if not self.preserve_real_noise:
            raise NotImplementedError(
                "preserve_real_noise=False is not implemented -- this pipeline's entire "
                "purpose is training against real noise texture; a noise-replacing path "
                "would need separate, explicit implementation, not a silent no-op."
            )
        if not self.wrap_after_injection:
            warnings.warn(
                "wrap_after_injection=False produces samples with an unwrapped 'observed' "
                "phase, which is not physically valid as a training tile (real sensors only "
                "measure wrapped phase). Use this only for debugging/inspecting the injected "
                "field directly.",
                stacklevel=2,
            )

    def enabled_injection_kinds(self) -> list[InjectionKind]:
        """Return the list of injection kinds enabled by this config, in a
        fixed, deterministic order."""
        kinds: list[InjectionKind] = []
        if self.inject_gaussian_bowl:
            kinds.append("gaussian_bowl")
        if self.inject_mogi:
            kinds.append("mogi")
        if self.inject_okada:
            kinds.append("okada")
        if self.inject_linear_ramp:
            kinds.append("linear_ramp")
        if self.inject_nonlinear_transient:
            kinds.append("nonlinear_transient")
        return kinds

    def config_hash(self) -> str:
        """A short, stable hash of this config's field values, for cache
        filenames -- two configs with identical field values produce the
        same hash regardless of construction order or process.

        Returns:
            A 16-character hex digest.
        """
        payload = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Real data loading
# --------------------------------------------------------------------------- #


@dataclass
class RealDataStack:
    """A loaded real (or real-like) SAR data stack, ready for injection.

    Attributes:
        amplitude: Amplitude stack, shape `(T, H, W)`.
        wrapped_phase: Wrapped phase stack, radians, shape `(T, H, W)`,
            values expected in `(-pi, pi]` (not enforced at load time, since
            some real products store phase in `[0, 2*pi)` or degrees --
            callers should wrap/convert before constructing this object;
            `build_real_injection_tiles` re-wraps its final output
            regardless, but the pre-injection background should already be
            in the expected convention for `wavelength_m`/displacement
            conversions to be meaningful).
        coherence: Coherence map, shape `(H, W)` (one map for the whole
            stack) or `(T, H, W)` (per-epoch).
        dem: Optional DEM, shape `(H, W)`, meters.
        incidence_angle: Optional incidence angle, either a scalar
            (radians) or a per-pixel map, shape `(H, W)`.
        temporal_baselines: Optional temporal baselines, shape `(T,)`
            (relative to a reference epoch) or `(T, T)` (pairwise).

    Raises:
        ValueError: If `amplitude` and `wrapped_phase` shapes don't match,
            or if `coherence`'s shape is neither `(H, W)` nor `(T, H, W)`
            matching the stack.
    """

    amplitude: np.ndarray
    wrapped_phase: np.ndarray
    coherence: np.ndarray
    dem: np.ndarray | None = None
    incidence_angle: np.ndarray | float | None = None
    temporal_baselines: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.amplitude.shape != self.wrapped_phase.shape:
            raise ValueError(
                f"amplitude shape {self.amplitude.shape} must match wrapped_phase shape "
                f"{self.wrapped_phase.shape}"
            )
        if self.amplitude.ndim != 3:
            raise ValueError(
                f"amplitude/wrapped_phase must be 3D (T, H, W), got shape {self.amplitude.shape}"
            )
        t, h, w = self.amplitude.shape
        if self.coherence.shape not in {(h, w), (t, h, w)}:
            raise ValueError(
                f"coherence shape must be (H, W)={(h, w)} or (T, H, W)={(t, h, w)}, "
                f"got {self.coherence.shape}"
            )
        if self.dem is not None and self.dem.shape != (h, w):
            raise ValueError(f"dem shape must be (H, W)={(h, w)}, got {self.dem.shape}")

    @property
    def n_epochs(self) -> int:
        return self.amplitude.shape[0]

    @property
    def spatial_shape(self) -> tuple[int, int]:
        return self.amplitude.shape[1], self.amplitude.shape[2]

    def coherence_at(self, time_index: int) -> np.ndarray:
        """Return the 2D coherence map applicable to `time_index` (handles
        both the shared-`(H,W)` and per-epoch-`(T,H,W)` storage forms)."""
        if self.coherence.ndim == 2:
            return self.coherence
        return self.coherence[time_index]


def load_real_stack(
    amplitude_path: str | Path,
    wrapped_phase_path: str | Path,
    coherence_path: str | Path,
    dem_path: str | Path | None = None,
    incidence_angle: str | Path | float | None = None,
    temporal_baselines_path: str | Path | None = None,
) -> RealDataStack:
    """Load a real (or real-like) data stack from disk.

    Supports `.npy` (loaded directly via `numpy.load`) and
    `.tif`/`.tiff`/GeoTIFF (loaded via `rasterio`, each band treated as one
    stack epoch for the 3D arrays). Zarr/HDF5 are not supported here since
    this repository has no existing zarr dependency and no established
    HDF5 *raw-stack* convention (`pyunwrap.data.preprocessing`'s HDF5
    format is a *tiled* format for already-injected/normalized training
    tiles, a different thing from a raw multi-epoch stack) -- adding either
    is real, scoped future work, not a silent gap: attempting to load an
    unsupported extension raises `ValueError` immediately rather than
    guessing.

    Args:
        amplitude_path: Path to the amplitude stack.
        wrapped_phase_path: Path to the wrapped-phase stack, radians.
        coherence_path: Path to the coherence map/stack.
        dem_path: Optional path to a DEM.
        incidence_angle: Optional path to an incidence-angle map, or a bare
            float (a single scalar incidence angle for the whole scene).
        temporal_baselines_path: Optional path to a `.npy` array of
            temporal baselines.

    Returns:
        A validated `RealDataStack`.

    Raises:
        ValueError: If any path has an unsupported extension.
        ImportError: If a `.tif`/`.tiff` path is given but `rasterio` is
            not installed.
    """
    amplitude = _load_array(amplitude_path)
    wrapped_phase = _load_array(wrapped_phase_path)
    coherence = _load_array(coherence_path)
    dem = _load_array(dem_path) if dem_path is not None else None

    resolved_incidence: np.ndarray | float | None
    if isinstance(incidence_angle, (int, float)):
        resolved_incidence = float(incidence_angle)
    elif incidence_angle is not None:
        resolved_incidence = _load_array(incidence_angle)
    else:
        resolved_incidence = None

    temporal_baselines = None
    if temporal_baselines_path is not None:
        temporal_baselines = np.load(temporal_baselines_path)

    return RealDataStack(
        amplitude=amplitude,
        wrapped_phase=wrapped_phase,
        coherence=coherence,
        dem=dem,
        incidence_angle=resolved_incidence,
        temporal_baselines=temporal_baselines,
    )


def _load_array(path: str | Path) -> np.ndarray:
    """Load a single array from `.npy` or `.tif`/`.tiff`, based on extension.

    Args:
        path: File path.

    Returns:
        Loaded array. GeoTIFFs with a single band are squeezed to 2D;
        multi-band GeoTIFFs are returned as `(bands, H, W)`.

    Raises:
        ValueError: If the extension is not `.npy`, `.tif`, or `.tiff`.
        ImportError: If a GeoTIFF is requested but `rasterio` is unavailable.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    if suffix in (".tif", ".tiff"):
        if not _HAS_RASTERIO:
            raise ImportError(
                f"rasterio is required to load GeoTIFF files (got {path}); install it via "
                "the base pyunwrap dependencies."
            )
        with rasterio.open(path) as src:
            data = src.read()  # (bands, H, W)
        return data[0] if data.shape[0] == 1 else data
    raise ValueError(
        f"Unsupported file extension {suffix!r} for {path}. Supported: .npy, .tif, .tiff."
    )


# --------------------------------------------------------------------------- #
# Deformation injection
# --------------------------------------------------------------------------- #


def nonlinear_transient_deformation(
    shape: tuple[int, int],
    pixel_spacing_m: float = 20.0,
    center: tuple[float, float] | None = None,
    peak_displacement_m: float = 0.05,
    decay_length_m: float = 1500.0,
    asymmetry: float = 0.3,
    seed: int | None = None,
) -> np.ndarray:
    """A simplified, physically-motivated nonlinear transient deformation
    model: an exponential (rather than Gaussian) radial decay with a random
    directional asymmetry, representing signals such as afterslip or a
    slow-slip transient whose spatial falloff is measurably heavier-tailed
    than a smooth Gaussian bowl's.

    This is a deliberately simple model, in the same spirit as this
    project's existing "simplified physically plausible fault model"
    framing for Okada dislocation -- it is not a rigorous viscoelastic or
    rate-and-state afterslip simulation, and should not be presented as one.
    It exists to give the injection pipeline spatial variety distinct from
    the smooth Gaussian bowl and axisymmetric Mogi source already available.

    Args:
        shape: `(rows, cols)` of the output grid.
        pixel_spacing_m: Ground pixel spacing, meters.
        center: `(row, col)` center of the transient; defaults to the grid
            center.
        peak_displacement_m: Peak LOS displacement at the center, meters.
        decay_length_m: Characteristic exponential decay length, meters.
        asymmetry: Directional asymmetry strength, `[0, 1)`; `0` is
            perfectly axisymmetric, larger values stretch the pattern more
            strongly along a (seeded-)random direction.
        seed: Seed for the random asymmetry direction.

    Returns:
        2D array of LOS displacement, meters, shape `shape`.

    Raises:
        ValueError: If `asymmetry` is outside `[0, 1)`.
    """
    if not (0.0 <= asymmetry < 1.0):
        raise ValueError(f"asymmetry must be in [0, 1), got {asymmetry}")

    rows, cols = shape
    if center is None:
        center = (rows / 2.0, cols / 2.0)
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi)

    yy, xx = np.mgrid[0:rows, 0:cols].astype(np.float64)
    dy_m = (yy - center[0]) * pixel_spacing_m
    dx_m = (xx - center[1]) * pixel_spacing_m

    # Rotate into the asymmetry axis, then stretch one axis to break
    # axisymmetry (a purely radial exponential would look too similar to a
    # rescaled Gaussian bowl to add real variety to the injection set).
    rot_x = dx_m * np.cos(theta) + dy_m * np.sin(theta)
    rot_y = -dx_m * np.sin(theta) + dy_m * np.cos(theta)
    stretched_r = np.sqrt(rot_x**2 + ((1.0 - asymmetry) * rot_y) ** 2)

    displacement = peak_displacement_m * np.exp(-stretched_r / decay_length_m)
    return displacement


def _inject_gaussian_bowl(shape, config, rng) -> tuple[np.ndarray, dict]:
    amplitude_m = float(rng.uniform(0.02, 0.15)) * (1 if rng.random() < 0.5 else -1)
    sigma_m = float(rng.uniform(400.0, 1200.0))
    displacement = gaussian_bowl_deformation(shape, amplitude_m=amplitude_m, sigma_m=sigma_m)
    phase = displacement_to_phase(displacement, wavelength_m=config.wavelength_m)
    return phase, {"kind": "gaussian_bowl", "amplitude_m": amplitude_m, "sigma_m": sigma_m}


def _inject_mogi(shape, config, rng) -> tuple[np.ndarray, dict]:
    volume_change_m3 = float(rng.uniform(-1.0e6, 1.0e6))
    source_depth_m = float(rng.uniform(1500.0, 4000.0))
    center = (float(rng.uniform(0.3, 0.7)) * shape[0], float(rng.uniform(0.3, 0.7)) * shape[1])
    displacement = mogi_point_source(
        shape,
        source_depth_m=source_depth_m,
        volume_change_m3=volume_change_m3,
        center=center,
    )
    phase = displacement_to_phase(displacement, wavelength_m=config.wavelength_m)
    return phase, {
        "kind": "mogi",
        "volume_change_m3": volume_change_m3,
        "source_depth_m": source_depth_m,
        "center": center,
    }


def _inject_okada(shape, config, rng) -> tuple[np.ndarray, dict]:
    strike_deg = float(rng.uniform(0, 360))
    dip_deg = float(rng.uniform(30, 89))
    rake_deg = float(rng.uniform(-180, 180))
    slip_m = float(rng.uniform(0.1, 0.8))
    length_m = float(rng.uniform(2000, 5000))
    width_m = float(rng.uniform(1000, 3000))
    depth_m = float(rng.uniform(3000, 7000))
    # Explicit keyword arguments (not **a homogeneous dict[str, float]) so
    # mypy can verify each against okada_dislocation's actual (mixed
    # float/tuple) parameter types.
    displacement = okada_dislocation(
        shape,
        strike_deg=strike_deg,
        dip_deg=dip_deg,
        rake_deg=rake_deg,
        slip_m=slip_m,
        length_m=length_m,
        width_m=width_m,
        depth_m=depth_m,
    )
    phase = displacement_to_phase(displacement, wavelength_m=config.wavelength_m)
    return phase, {
        "kind": "okada",
        "strike_deg": strike_deg,
        "dip_deg": dip_deg,
        "rake_deg": rake_deg,
        "slip_m": slip_m,
        "length_m": length_m,
        "width_m": width_m,
        "depth_m": depth_m,
    }


def _inject_linear_ramp(shape, config, rng) -> tuple[np.ndarray, dict]:
    seed = int(rng.integers(0, 2**31 - 1))
    amplitude_rad = float(rng.uniform(0.5, 2.5))
    phase = orbital_ramp(shape, amplitude_rad=amplitude_rad, seed=seed)
    return phase, {"kind": "linear_ramp", "amplitude_rad": amplitude_rad, "seed": seed}


def _inject_nonlinear_transient(shape, config, rng) -> tuple[np.ndarray, dict]:
    seed = int(rng.integers(0, 2**31 - 1))
    peak_displacement_m = float(rng.uniform(0.01, 0.08))
    decay_length_m = float(rng.uniform(800.0, 2500.0))
    asymmetry = float(rng.uniform(0.0, 0.6))
    displacement = nonlinear_transient_deformation(
        shape,
        peak_displacement_m=peak_displacement_m,
        decay_length_m=decay_length_m,
        asymmetry=asymmetry,
        seed=seed,
    )
    phase = displacement_to_phase(displacement, wavelength_m=config.wavelength_m)
    return phase, {
        "kind": "nonlinear_transient",
        "peak_displacement_m": peak_displacement_m,
        "decay_length_m": decay_length_m,
        "asymmetry": asymmetry,
        "seed": seed,
    }


_INJECTORS = {
    "gaussian_bowl": _inject_gaussian_bowl,
    "mogi": _inject_mogi,
    "okada": _inject_okada,
    "linear_ramp": _inject_linear_ramp,
    "nonlinear_transient": _inject_nonlinear_transient,
}


def estimate_coherence_proxy(
    phase_stack: np.ndarray, amplitude_stack: np.ndarray | None = None
) -> np.ndarray:
    """Estimate a coherence proxy when real coherence is unavailable.

    Uses temporal phase stability (the magnitude of the circular mean of
    `exp(1j * phase)` across the time axis) when the stack has more than
    one epoch, since true coherence is fundamentally a measure of
    phase-stability between acquisitions and this is the closest
    single-stack proxy for it. Falls back to a normalized inverse
    amplitude-dispersion proxy (lower dispersion -> higher proxy
    coherence) when only a single epoch is available and amplitude is
    given; raises if neither signal is available, rather than fabricating
    a uniform/meaningless map silently.

    Args:
        phase_stack: Wrapped phase stack, shape `(T, H, W)`.
        amplitude_stack: Optional amplitude stack, shape `(T, H, W)`, used
            as the fallback proxy when `T == 1`.

    Returns:
        A `(H, W)` proxy coherence map, values in `[0, 1]`.

    Raises:
        ValueError: If `phase_stack` has only one epoch and
            `amplitude_stack` is not provided (or also has only one
            epoch), since neither proxy is computable from a single static
            image.
    """
    t = phase_stack.shape[0]
    if t > 1:
        mean_vector = np.mean(np.exp(1j * phase_stack), axis=0)
        return np.clip(np.abs(mean_vector), 0.0, 1.0)

    if amplitude_stack is not None and amplitude_stack.shape[0] > 1:
        mean_amp = amplitude_stack.mean(axis=0)
        std_amp = amplitude_stack.std(axis=0)
        dispersion = std_amp / (mean_amp + 1e-9)
        # Normalize dispersion to a [0, 1] proxy coherence via a smooth
        # monotonic map; the exact scale is a heuristic (documented as
        # such), not a calibrated physical relationship.
        return np.clip(1.0 - dispersion / (dispersion.max() + 1e-9), 0.0, 1.0)

    raise ValueError(
        "Cannot estimate a coherence proxy: phase_stack has only 1 epoch and no "
        "multi-epoch amplitude_stack was provided. Supply real coherence, a multi-epoch "
        "phase stack, or a multi-epoch amplitude stack."
    )


# --------------------------------------------------------------------------- #
# Core injection + spatial splitting
# --------------------------------------------------------------------------- #


def build_injected_scene(
    real_stack: RealDataStack,
    config: RealSyntheticInjectionConfig,
    time_index: int = -1,
    injection_kind: InjectionKind | None = None,
    background_unwrapped: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    """Build one full-scene injected sample from a real data stack.

    See the module docstring for exactly how `true_unwrapped`/`true_ambiguity`
    are derived from the real background phase and the injected field.

    Args:
        real_stack: The loaded real (or real-like) data stack.
        config: Injection configuration.
        time_index: Which epoch of `real_stack` to use as the real
            amplitude/phase/coherence background.
        injection_kind: Which deformation model to inject. If `None`, one
            is chosen uniformly at random from `config.enabled_injection_kinds()`.
        background_unwrapped: Optional best-estimate real unwrapped
            background phase (see module docstring); defaults to the real
            wrapped phase taken as-is (implicit zero ambiguity for the
            real background).
        rng: Random generator for injection parameter randomization.
            Defaults to `np.random.default_rng(config.seed)`.

    Returns:
        `(sample, metadata)`:
            - `sample`: dict with keys `"wrapped_phase"`, `"coherence"`,
              `"amplitude"`, `"true_unwrapped"`, `"true_ambiguity"`,
              `"regime_mask"` -- all raw (un-normalized, un-tiled) full-scene
              arrays, shape matching `real_stack.spatial_shape`.
            - `metadata`: injection parameters (deformation kind, its
              randomized physical parameters, and the RNG seed used),
              suitable for logging/reproducibility.

    Raises:
        ValueError: If `config.enabled_injection_kinds()` is empty and
            `injection_kind` was not given explicitly.
    """
    rng = rng if rng is not None else np.random.default_rng(config.seed)

    enabled = config.enabled_injection_kinds()
    if injection_kind is None:
        if len(enabled) == 0:
            raise ValueError(
                "No injection kind was given and config has every inject_* flag disabled; "
                "nothing to inject."
            )
        injection_kind = enabled[int(rng.integers(0, len(enabled)))]

    shape = real_stack.spatial_shape
    psi_real = real_stack.wrapped_phase[time_index]
    amplitude = real_stack.amplitude[time_index]
    coherence = real_stack.coherence_at(time_index)

    phi_base = background_unwrapped if background_unwrapped is not None else psi_real

    phi_synth, injection_metadata = _INJECTORS[injection_kind](shape, config, rng)
    phi_true = phi_base + phi_synth

    if config.wrap_after_injection:
        psi_observed = wrap_phase(phi_true)
    else:
        psi_observed = phi_true  # invalid as a training tile; see config docstring

    k_true = compute_ambiguity(phi_true, psi_observed)

    regime_mask = (
        (coherence >= config.min_coherence_for_background)
        & (coherence <= config.max_coherence_for_background)
    ).astype(np.float32)

    sample = {
        "wrapped_phase": psi_observed.astype(np.float64),
        "coherence": coherence.astype(np.float64),
        "amplitude": amplitude.astype(np.float64),
        "true_unwrapped": phi_true.astype(np.float64),
        "true_ambiguity": k_true.astype(np.float64),
        "regime_mask": regime_mask,
    }
    metadata = {"time_index": time_index, **injection_metadata}
    return sample, metadata


def spatial_block_split(
    shape: tuple[int, int], config: RealSyntheticInjectionConfig
) -> dict[str, list[TileSpec]]:
    """Partition a scene into non-overlapping spatial blocks and assign
    whole blocks to train/val/test, preventing pixel-level leakage between
    spatially adjacent pixels in different splits.

    Args:
        shape: `(H, W)` of the scene to split.
        config: Provides `block_size`, the three split fractions, and
            `seed`.

    Returns:
        `{"train": [...], "val": [...], "test": [...]}`, each a list of
        `TileSpec` block coordinates (non-overlapping, covering the full
        scene up to edge remainder handled by `compute_tile_grid`).
    """
    blocks = compute_tile_grid(shape, tile_size=config.block_size, overlap=0)
    rng = np.random.default_rng(config.seed)
    order = rng.permutation(len(blocks))

    n_train = round(len(blocks) * config.train_fraction)
    n_val = round(len(blocks) * config.val_fraction)

    train_idx = order[:n_train]
    val_idx = order[n_train : n_train + n_val]
    test_idx = order[n_train + n_val :]

    return {
        "train": [blocks[i] for i in train_idx],
        "val": [blocks[i] for i in val_idx],
        "test": [blocks[i] for i in test_idx],
    }


# --------------------------------------------------------------------------- #
# End-to-end pipeline: real stack -> cached HDF5 tile files
# --------------------------------------------------------------------------- #


def build_real_injection_datasets(
    real_stack: RealDataStack,
    config: RealSyntheticInjectionConfig,
    out_dir: str | Path,
    n_scenes_per_split_multiplier: int = 1,
    tile_size: int = 64,
    tile_overlap: int = 16,
    use_cache: bool = True,
) -> dict[str, Path]:
    """End-to-end pipeline: a real data stack -> spatially-split, tiled,
    cached HDF5 files directly usable as `Trainer`'s `train_dataset`/
    `val_dataset` (via `InSARTileDataset`).

    Args:
        real_stack: The loaded real (or real-like) data stack.
        config: Injection configuration.
        out_dir: Directory for cached output files.
        n_scenes_per_split_multiplier: How many independent injected
            scenes (each a fresh random deformation draw, injected into
            the same real background at a randomly chosen time index) to
            generate per split before tiling. `1` uses each split's own
            spatial region exactly once; higher values inject multiple
            different synthetic deformations into the same real spatial
            region for that split, for more training diversity from a
            single real stack.
        tile_size: Output tile edge length, pixels, for the final
            `InSARTileDataset`-compatible HDF5 files.
        tile_overlap: Overlap between adjacent output tiles, pixels.
        use_cache: If `True` and a cache file matching
            `config.config_hash()` already exists in `out_dir`, it is
            reused rather than regenerated.

    Returns:
        `{"train": Path, "val": Path, "test": Path}`, each an HDF5 file
        directly loadable via `InSARTileDataset`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config_hash = config.config_hash()

    paths = {
        split: out_dir / f"real_injection_{split}_{config_hash}.h5"
        for split in ("train", "val", "test")
    }
    if use_cache and all(p.exists() for p in paths.values()):
        print(f"[real_injection] Using cached datasets (config hash {config_hash}).")
        return paths

    splits = spatial_block_split(real_stack.spatial_shape, config)
    rng = np.random.default_rng(config.seed)

    for split_name, blocks in splits.items():
        if len(blocks) == 0:
            warnings.warn(
                f"Split '{split_name}' received 0 spatial blocks (fraction too small "
                f"relative to block_size/scene size); its output file will be empty.",
                stacklevel=2,
            )
        all_tiles: list[tuple[TileSpec, dict[str, np.ndarray]]] = []
        for _ in range(n_scenes_per_split_multiplier):
            time_index = int(rng.integers(0, real_stack.n_epochs))
            full_sample, _metadata = build_injected_scene(
                real_stack,
                config,
                time_index=time_index,
                rng=rng,
            )
            norm = {
                "wrapped_phase": normalize_phase(full_sample["wrapped_phase"]),
                "coherence": normalize_coherence(full_sample["coherence"]),
                "amplitude": normalize_amplitude(full_sample["amplitude"]),
            }
            for block in blocks:
                tile_dict = {
                    "wrapped_phase": extract_tile(norm["wrapped_phase"], block),
                    "coherence": extract_tile(norm["coherence"], block),
                    "amplitude": extract_tile(norm["amplitude"], block),
                    "true_unwrapped": extract_tile(full_sample["true_unwrapped"], block),
                }
                all_tiles.append((block, tile_dict))

        save_tiles_hdf5(all_tiles, paths[split_name])
        print(f"[real_injection] {split_name}: {len(all_tiles)} tiles -> {paths[split_name]}")

    return paths


# --------------------------------------------------------------------------- #
# Test fixture
# --------------------------------------------------------------------------- #


def build_tiny_fixture_stack(size: int = 64, n_epochs: int = 3, seed: int = 0) -> RealDataStack:
    """Build a small, clearly-labeled **real-like** (not literally real)
    fixture `RealDataStack`, for fast, dependency-free tests.

    This is explicitly NOT real satellite data -- see the module docstring
    for why no real SAR data is reachable from this project's development
    environment. It exists purely so `load_real_stack`, injection, and
    spatial splitting can be exercised by tests without requiring a real
    data file on disk. It deliberately does *not* reuse
    `pyunwrap.synthetic.generator.InSARSyntheticGenerator` (which already
    has its own, separate, extensive test coverage) -- this fixture is
    intentionally simpler and noisier, standing in for "some real stack
    with unknown, unmodeled statistics" rather than for
    `pyunwrap`'s own synthetic model.

    Args:
        size: Spatial edge length, pixels.
        n_epochs: Number of stack epochs.
        seed: RNG seed.

    Returns:
        A `RealDataStack` with random (not physically simulated) phase,
        amplitude, and coherence.
    """
    rng = np.random.default_rng(seed)
    amplitude = rng.uniform(0.3, 1.0, size=(n_epochs, size, size))
    wrapped_phase = wrap_phase(rng.uniform(-np.pi, np.pi, size=(n_epochs, size, size)))
    coherence = np.clip(rng.normal(0.6, 0.2, size=(size, size)), 0.0, 1.0)
    dem = rng.uniform(0, 500, size=(size, size))
    return RealDataStack(
        amplitude=amplitude,
        wrapped_phase=wrapped_phase,
        coherence=coherence,
        dem=dem,
    )

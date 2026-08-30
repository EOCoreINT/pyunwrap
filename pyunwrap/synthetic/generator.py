"""
pyunwrap.synthetic.generator
=============================

Realistic synthetic InSAR (Interferometric Synthetic Aperture Radar) data generator.

This module produces physically-motivated triplets of:

    (wrapped_phase, true_unwrapped_phase, coherence_map)

that are used to train `AmbiguityNet` to predict the discrete integer ambiguity
map ``k`` such that ``true_unwrapped_phase = wrapped_phase + 2*pi*k``.

The simulator composes several independent physical phase contributions:

1. Deformation phase (Gaussian bowl / Okada fault dislocation / Mogi point source)
2. Topographic phase (from a real DEM + a randomized perpendicular baseline)
3. Atmospheric phase (Kolmogorov power-law spectral noise -> tropospheric delay)
4. Orbital phase ramps (residual baseline errors, linear + quadratic)
5. Decorrelation noise (phase noise whose variance is coherence-dependent)

All contributions are summed in radians, then wrapped into (-pi, pi] to form the
observed wrapped phase. The unwrapped sum (before wrapping) is retained as ground
truth, from which the integer ambiguity map ``k`` is derived exactly.

A "pseudo-real" data strategy is also provided: real L-band (ALOS-2) unwrapped
phase can be rewrapped to a simulated C-band (Sentinel-1-like) wavelength to
produce ground-truth-backed *real* interferometric structure, helping bridge the
sim-to-real domain gap.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Literal

import numpy as np

try:
    import rasterio
    from rasterio.enums import Resampling

    _HAS_RASTERIO = True
except ImportError:  # pragma: no cover - rasterio is a hard dependency in prod,
    _HAS_RASTERIO = False  # but we degrade gracefully for lightweight unit tests.


# --------------------------------------------------------------------------- #
# Physical constants
# --------------------------------------------------------------------------- #

# Common SAR carrier wavelengths, in meters.
WAVELENGTH_M = {
    "C-band": 0.0555,  # Sentinel-1
    "L-band": 0.2360,  # ALOS-2
    "X-band": 0.0310,  # TerraSAR-X
}


@dataclasses.dataclass
class SyntheticSample:
    """Container for one generated synthetic InSAR sample.

    Attributes:
        wrapped_phase: Wrapped interferometric phase, radians, in (-pi, pi].
        unwrapped_phase: True unwrapped phase, radians (ground truth). This is
            the *actually observed* phase (physical signal + decorrelation
            noise) before wrapping -- i.e. exactly what `wrapped_phase + 2*pi
            * ambiguity` reconstructs. It intentionally includes noise: phase
            unwrapping resolves the 2*pi ambiguity of the observed signal, it
            does not denoise it. The noise-free signal is available separately
            in `components["clean_phase"]`.
        ambiguity: Integer ambiguity map k such that
            unwrapped_phase = wrapped_phase + 2*pi*k. Ground truth for training.
        coherence: Simulated interferometric coherence, in [0, 1].
        amplitude: Simulated (log-scaled) SAR backscatter amplitude.
        components: Dict of the individual phase contributions (radians), kept
            for diagnostics/visualization (e.g. isolating the atmosphere term).
    """

    wrapped_phase: np.ndarray
    unwrapped_phase: np.ndarray
    ambiguity: np.ndarray
    coherence: np.ndarray
    amplitude: np.ndarray
    components: dict


def wrap_phase(phase: np.ndarray) -> np.ndarray:
    """Wrap a real-valued phase array into (-pi, pi].

    Uses the numerically robust ``angle(exp(1j * phase))`` identity rather than a
    manual modulo, which avoids edge-case sign errors at exact multiples of pi.

    Args:
        phase: Unwrapped phase in radians, any shape.

    Returns:
        Wrapped phase, same shape, values in (-pi, pi].
    """
    return np.angle(np.exp(1j * phase))


def compute_ambiguity(unwrapped_phase: np.ndarray, wrapped_phase: np.ndarray) -> np.ndarray:
    """Derive the exact integer ambiguity map k from unwrapped and wrapped phase.

    k = (unwrapped_phase - wrapped_phase) / (2*pi), rounded to the nearest integer
    to remove floating point residue. This is the ground-truth label AmbiguityNet
    is trained to regress.

    Args:
        unwrapped_phase: True unwrapped phase, radians.
        wrapped_phase: Corresponding wrapped phase, radians, in (-pi, pi].

    Returns:
        Integer-valued (but float-dtype) ambiguity map k, same shape as input.
    """
    k = (unwrapped_phase - wrapped_phase) / (2.0 * np.pi)
    return np.round(k)


# --------------------------------------------------------------------------- #
# 1. Deformation models
# --------------------------------------------------------------------------- #


def gaussian_bowl_deformation(
    shape: tuple[int, int],
    pixel_spacing_m: float = 20.0,
    center: tuple[float, float] | None = None,
    amplitude_m: float = 0.10,
    sigma_m: float = 800.0,
    los_incidence_deg: float = 34.0,
) -> np.ndarray:
    """Generate a Gaussian subsidence/uplift bowl (e.g. mining, groundwater extraction).

    The vertical surface displacement follows a 2D Gaussian; the result is
    projected into radar line-of-sight (LOS) and converted to phase for a
    single-band wavelength (wavelength is applied by the caller via
    `displacement_to_phase`, so this function returns *meters* of LOS
    displacement, not phase).

    Args:
        shape: (rows, cols) of the output grid.
        pixel_spacing_m: Ground pixel spacing in meters.
        center: (row, col) center of the bowl in pixels; defaults to grid center.
        amplitude_m: Peak vertical displacement in meters (negative = subsidence).
        sigma_m: Gaussian standard deviation in meters (controls bowl width).
        los_incidence_deg: Radar incidence angle used for the vertical -> LOS
            projection (a simple cosine projection; ignores azimuth look vector
            for simplicity, which is adequate for synthetic training data).

    Returns:
        2D array of LOS displacement in meters, shape `shape`.
    """
    rows, cols = shape
    if center is None:
        center = (rows / 2.0, cols / 2.0)
    yy, xx = np.mgrid[0:rows, 0:cols].astype(np.float64)
    dy_m = (yy - center[0]) * pixel_spacing_m
    dx_m = (xx - center[1]) * pixel_spacing_m
    r2 = dx_m**2 + dy_m**2
    vertical_disp = amplitude_m * np.exp(-r2 / (2.0 * sigma_m**2))
    los_disp = vertical_disp * math.cos(math.radians(los_incidence_deg))
    return los_disp


def okada_dislocation(
    shape: tuple[int, int],
    pixel_spacing_m: float = 20.0,
    strike_deg: float = 0.0,
    dip_deg: float = 90.0,
    rake_deg: float = 0.0,
    slip_m: float = 1.0,
    length_m: float = 10_000.0,
    width_m: float = 5_000.0,
    depth_m: float = 5_000.0,
    fault_center: tuple[float, float] | None = None,
    poisson_ratio: float = 0.25,
    los_vector: tuple[float, float, float] = (0.38, -0.08, 0.92),
) -> np.ndarray:
    """Okada (1985) rectangular dislocation surface displacement, projected to LOS.

    Implements the analytic elastic half-space solution for a finite rectangular
    fault plane, used to simulate coseismic earthquake deformation. This is a
    from-scratch implementation of the classic Okada formulation (not a wrapper
    around a third-party geodesy package), so `pyunwrap` has no hard runtime
    dependency beyond NumPy for this component.

    Args:
        shape: (rows, cols) of the output grid.
        pixel_spacing_m: Ground pixel spacing in meters.
        strike_deg: Fault strike angle, degrees clockwise from north.
        dip_deg: Fault dip angle, degrees from horizontal (90 = vertical fault).
        rake_deg: Slip rake angle, degrees (0 = pure left-lateral strike-slip,
            90 = pure thrust).
        slip_m: Total slip on the fault plane, meters.
        length_m: Fault length along strike, meters.
        width_m: Fault width along dip, meters.
        depth_m: Depth to the fault's top edge, meters (positive down).
        fault_center: (row, col) surface projection of the fault center in
            pixels; defaults to grid center.
        poisson_ratio: Poisson's ratio of the elastic half-space (~0.25 typical).
        los_vector: Unit vector (east, north, up) of the radar line of sight,
            used to project the 3D (east, north, up) displacement field into a
            scalar LOS displacement.

    Returns:
        2D array of LOS displacement in meters, shape `shape`.
    """
    rows, cols = shape
    if fault_center is None:
        fault_center = (rows / 2.0, cols / 2.0)

    yy, xx = np.mgrid[0:rows, 0:cols].astype(np.float64)
    # Local east/north coordinates (meters) relative to the fault's surface center.
    north_m = -(yy - fault_center[0]) * pixel_spacing_m  # image row increases southward
    east_m = (xx - fault_center[1]) * pixel_spacing_m

    strike = math.radians(strike_deg)
    dip = math.radians(dip_deg)
    rake = math.radians(rake_deg)

    # Rotate observation points into the fault-strike-aligned coordinate frame.
    cos_s, sin_s = math.cos(strike), math.sin(strike)
    x_strike = north_m * cos_s + east_m * sin_s
    y_strike = -north_m * sin_s + east_m * cos_s

    # Okada's convention: origin at the midpoint of the fault's bottom edge
    # surface projection. Shift so length/width are centered.
    x = x_strike + length_m / 2.0
    p = y_strike * math.cos(dip) + depth_m * math.sin(dip)

    U1 = slip_m * math.cos(rake)  # strike-slip component
    U2 = slip_m * math.sin(rake)  # dip-slip component
    L = length_m
    W = width_m
    nu = poisson_ratio
    alpha = (
        1.0 - 2.0 * nu
    )  # standard Okada "alpha" = (lambda + mu) / (lambda + 2*mu) with nu=0.25 -> alpha=0.5... see note below.
    # NOTE: For an isotropic Poisson solid, Okada's alpha = (lambda+mu)/(lambda+2mu).
    # For nu = lambda / (2*(lambda+mu)), algebra gives alpha = 1 - 2*nu is NOT exact;
    # the correct closed form is alpha = 1/(2*(1-nu)). Using the correct expression:
    alpha = 1.0 / (2.0 * (1.0 - nu))

    def _ux_strike_slip(xi: np.ndarray, eta: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Chinnery-style strike-slip displacement kernel (simplified far-field form)."""
        R = np.sqrt(xi**2 + eta**2 + q**2) + 1e-9
        return (
            -U1
            / (2.0 * math.pi)
            * (xi * q / (R * (R + eta) + 1e-9) + np.arctan2(xi * eta, q * R + 1e-9))
        )

    def _uy_strike_slip(xi: np.ndarray, eta: np.ndarray, q: np.ndarray) -> np.ndarray:
        R = np.sqrt(xi**2 + eta**2 + q**2) + 1e-9
        return (
            -U1
            / (2.0 * math.pi)
            * (
                (eta * math.cos(dip) + q * math.sin(dip)) * q / (R * (R + eta) + 1e-9)
                + q * math.cos(dip) / (R + eta + 1e-9)
                - alpha / 2.0 * np.log(R + eta + 1e-9)
            )
        )

    def _uz_strike_slip(xi: np.ndarray, eta: np.ndarray, q: np.ndarray) -> np.ndarray:
        R = np.sqrt(xi**2 + eta**2 + q**2) + 1e-9
        return (
            -U1
            / (2.0 * math.pi)
            * (
                (eta * math.sin(dip) - q * math.cos(dip)) * q / (R * (R + eta) + 1e-9)
                + q * math.sin(dip) / (R + eta + 1e-9)
                - alpha / 2.0 * np.log(R + eta + 1e-9)
            )
        )

    def _dipslip_terms(xi: np.ndarray, eta: np.ndarray, q: np.ndarray):
        """Chinnery-style dip-slip kernels."""
        R = np.sqrt(xi**2 + eta**2 + q**2) + 1e-9
        ux = (
            -U2
            / (2.0 * math.pi)
            * (
                q / R
                - alpha
                / 2.0
                * np.arcsin(np.clip(q * (2 * R + xi) / ((R + xi) * (R + q) + 1e-9), -1, 1))
            )
        )
        uy = (
            -U2
            / (2.0 * math.pi)
            * (
                (eta * math.cos(dip) + q * math.sin(dip)) * q / (R * (R + eta) + 1e-9) * 0.0
                + math.cos(dip) * np.arctan2(xi * eta, q * R + 1e-9)
                - alpha / 2.0 * xi * q / (R * (R + eta) + 1e-9)
            )
        )
        uz = (
            -U2
            / (2.0 * math.pi)
            * (
                math.sin(dip) * np.arctan2(xi * eta, q * R + 1e-9)
                - alpha / 2.0 * xi * q / (R * (R + eta) + 1e-9) * math.tan(dip)
            )
        )
        return ux, uy, uz

    # Rectangular integration bounds (Chinnery's method): sum the point-source
    # kernel over the 4 corners of the fault plane with alternating sign.
    xi1, xi2 = x - L, x
    eta1, eta2 = p - W, p
    q = y_strike * math.sin(dip) - depth_m * math.cos(dip)

    def eval_corner(xi, eta, sign):
        ux = sign * (_ux_strike_slip(xi, eta, q) + _dipslip_terms(xi, eta, q)[0])
        uy = sign * (_uy_strike_slip(xi, eta, q) + _dipslip_terms(xi, eta, q)[1])
        uz = sign * (_uz_strike_slip(xi, eta, q) + _dipslip_terms(xi, eta, q)[2])
        return ux, uy, uz

    ux_total = uy_total = uz_total = np.zeros_like(q)
    for xi, sxi in ((xi1, -1.0), (xi2, 1.0)):
        for eta, seta in ((eta1, -1.0), (eta2, 1.0)):
            sign = sxi * seta
            ux, uy, uz = eval_corner(xi, eta, sign)
            ux_total += ux
            uy_total += uy
            uz_total += uz

    # Rotate the strike-aligned (ux, uy) horizontal displacement back to (east, north).
    east_disp = ux_total * sin_s + uy_total * cos_s
    north_disp = ux_total * cos_s - uy_total * sin_s
    up_disp = uz_total

    los_e, los_n, los_u = los_vector
    los_disp = east_disp * los_e + north_disp * los_n + up_disp * los_u
    return los_disp


def mogi_point_source(
    shape: tuple[int, int],
    pixel_spacing_m: float = 20.0,
    source_depth_m: float = 3_000.0,
    volume_change_m3: float = 1.0e6,
    center: tuple[float, float] | None = None,
    poisson_ratio: float = 0.25,
    los_vector: tuple[float, float, float] = (0.38, -0.08, 0.92),
) -> np.ndarray:
    """Mogi (1958) point-source deformation model for volcanic inflation/deflation.

    Models a spherical pressure source embedded in an elastic half-space,
    producing the classic axisymmetric radial "bullseye" deformation pattern
    seen in volcanic InSAR interferograms.

    Args:
        shape: (rows, cols) of the output grid.
        pixel_spacing_m: Ground pixel spacing in meters.
        source_depth_m: Depth of the point source below the surface, meters.
        volume_change_m3: Volume change of the source; positive = inflation.
        center: (row, col) surface projection of the source; default grid center.
        poisson_ratio: Poisson's ratio of the half-space.
        los_vector: Unit vector (east, north, up) of the radar line of sight.

    Returns:
        2D array of LOS displacement in meters, shape `shape`.
    """
    rows, cols = shape
    if center is None:
        center = (rows / 2.0, cols / 2.0)
    yy, xx = np.mgrid[0:rows, 0:cols].astype(np.float64)
    dy_m = (yy - center[0]) * pixel_spacing_m
    dx_m = (xx - center[1]) * pixel_spacing_m
    r = np.sqrt(dx_m**2 + dy_m**2)
    d = source_depth_m

    # Mogi (1958): displacement field of a point pressure source.
    # Standard form: u_r = (3/4) * (dV/pi) * r / (r^2 + d^2)^(3/2)
    #                u_z = (3/4) * (dV/pi) * d / (r^2 + d^2)^(3/2)
    # (the (1-nu) elastic prefactor is commonly folded into dV; kept explicit here)
    denom = (r**2 + d**2) ** 1.5 + 1e-9
    coeff = (3.0 / 4.0) * volume_change_m3 / math.pi * (1.0 - poisson_ratio) / (1.0 - poisson_ratio)
    u_r = coeff * r / denom
    u_z = coeff * d / denom

    with np.errstate(invalid="ignore", divide="ignore"):
        cos_theta = np.divide(dx_m, r, out=np.zeros_like(r), where=r > 1e-6)
        sin_theta = np.divide(dy_m, r, out=np.zeros_like(r), where=r > 1e-6)
    u_east = u_r * cos_theta
    u_north = u_r * sin_theta

    los_e, los_n, los_u = los_vector
    los_disp = u_east * los_e + u_north * los_n + u_z * los_u
    return los_disp


def displacement_to_phase(
    los_displacement_m: np.ndarray, wavelength_m: float = WAVELENGTH_M["C-band"]
) -> np.ndarray:
    """Convert line-of-sight displacement (meters) to interferometric phase (radians).

    phase = -4*pi/lambda * los_displacement  (the factor of 2 for two-way travel
    combined with the standard InSAR sign convention where LOS motion toward the
    sensor is positive phase).

    Args:
        los_displacement_m: LOS displacement, meters.
        wavelength_m: Radar carrier wavelength, meters.

    Returns:
        Phase contribution, radians (unwrapped, i.e. not yet limited to [-pi, pi]).
    """
    return -4.0 * math.pi / wavelength_m * los_displacement_m


# --------------------------------------------------------------------------- #
# 2. Topographic phase from a DEM
# --------------------------------------------------------------------------- #


def load_dem(dem_path: str | Path, out_shape: tuple[int, int] | None = None) -> np.ndarray:
    """Load a real DEM from a GeoTIFF, optionally resampled to a target shape.

    Args:
        dem_path: Path to a GeoTIFF DEM.
        out_shape: Optional (rows, cols) to resample to via bilinear resampling.

    Returns:
        2D elevation array, meters.

    Raises:
        ImportError: If rasterio is not installed.
    """
    if not _HAS_RASTERIO:
        raise ImportError("rasterio is required to load DEMs. Install with `pip install rasterio`.")
    with rasterio.open(dem_path) as src:
        if out_shape is not None:
            dem = src.read(
                1,
                out_shape=out_shape,
                resampling=Resampling.bilinear,
            ).astype(np.float64)
        else:
            dem = src.read(1).astype(np.float64)
    return dem


def topographic_phase(
    dem_m: np.ndarray,
    perpendicular_baseline_m: float,
    range_distance_m: float = 800_000.0,
    incidence_deg: float = 34.0,
    wavelength_m: float = WAVELENGTH_M["C-band"],
) -> np.ndarray:
    """Simulate the topographic phase component from a DEM and a perpendicular baseline.

    Uses the standard "DEM error" style formula relating unmodeled topography to
    interferometric phase via the perpendicular baseline:

        phase_topo = -4*pi / (lambda * range * sin(incidence)) * B_perp * elevation

    Args:
        dem_m: DEM elevation, meters (can be relative "unmodeled" elevation).
        perpendicular_baseline_m: Perpendicular baseline of the interferometric
            pair, meters. Randomized per-sample to diversify training data.
        range_distance_m: Sensor-to-target slant range, meters.
        incidence_deg: Radar incidence angle, degrees.
        wavelength_m: Radar carrier wavelength, meters.

    Returns:
        Topographic phase contribution, radians (unwrapped).
    """
    sin_inc = math.sin(math.radians(incidence_deg))
    factor = -4.0 * math.pi * perpendicular_baseline_m / (wavelength_m * range_distance_m * sin_inc)
    return factor * dem_m


# --------------------------------------------------------------------------- #
# 3. Atmospheric phase (Kolmogorov fractal noise)
# --------------------------------------------------------------------------- #


def kolmogorov_atmospheric_noise(
    shape: tuple[int, int],
    pixel_spacing_m: float = 20.0,
    beta: float = 8.0 / 3.0,
    amplitude_rad: float = 1.5,
    seed: int | None = None,
) -> np.ndarray:
    """Generate spatially correlated tropospheric delay noise via a Kolmogorov
    power-law spectrum, i.e. fractal (1/f^beta) noise.

    The turbulent troposphere is well described by a power spectral density
    P(f) ~ f^(-beta), with beta ~ 8/3 (Kolmogorov turbulence) to 11/3 depending
    on the vertical structure assumed. This is generated by filtering white
    noise in the Fourier domain.

    Args:
        shape: (rows, cols) of the output grid.
        pixel_spacing_m: Ground pixel spacing, meters (affects spatial frequency scaling).
        beta: Power-law exponent of the 2D radial power spectrum.
        amplitude_rad: Approximate standard deviation of the resulting phase
            screen, radians, after normalization.
        seed: Optional RNG seed for reproducibility.

    Returns:
        2D array of atmospheric phase noise, radians.
    """
    rng = np.random.default_rng(seed)
    rows, cols = shape

    # White noise seed field.
    white = rng.standard_normal((rows, cols))
    white_fft = np.fft.fft2(white)

    # Radial spatial-frequency grid (cycles / meter).
    fy = np.fft.fftfreq(rows, d=pixel_spacing_m)
    fx = np.fft.fftfreq(cols, d=pixel_spacing_m)
    fyy, fxx = np.meshgrid(fy, fx, indexing="ij")
    f_radial = np.sqrt(fyy**2 + fxx**2)
    f_radial[0, 0] = (
        f_radial[f_radial > 0].min() if np.any(f_radial > 0) else 1.0
    )  # avoid div by 0 at DC

    # 2D power spectrum P(f) ~ f^-beta  ->  amplitude filter ~ f^(-beta/2).
    amplitude_filter = f_radial ** (-beta / 2.0)
    amplitude_filter[0, 0] = 0.0  # remove DC offset (mean-zero screen)

    filtered_fft = white_fft * amplitude_filter
    screen = np.real(np.fft.ifft2(filtered_fft))

    # Normalize to the requested amplitude (standard deviation).
    screen -= screen.mean()
    current_std = screen.std()
    if current_std > 1e-12:
        screen *= amplitude_rad / current_std
    return screen


# --------------------------------------------------------------------------- #
# 4. Orbital ramps
# --------------------------------------------------------------------------- #


def orbital_ramp(
    shape: tuple[int, int],
    linear_coeffs: tuple[float, float] | None = None,
    quadratic_coeffs: tuple[float, float, float] | None = None,
    amplitude_rad: float = 2.0,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a residual orbital phase ramp (linear + quadratic across the scene).

    Residual baseline/orbit errors typically manifest as a smooth low-order
    polynomial trend across an interferogram. This synthesizes that artifact so
    the network learns to fold it into the ambiguity map like any other smooth
    phase contribution, rather than mistaking it for genuine deformation.

    Args:
        shape: (rows, cols) of the output grid.
        linear_coeffs: (a_row, a_col) linear ramp coefficients (rad per pixel).
            Randomized if None.
        quadratic_coeffs: (b_row2, b_col2, b_rowcol) quadratic coefficients
            (rad per pixel^2). Randomized if None.
        amplitude_rad: Overall scale used when randomizing coefficients.
        seed: Optional RNG seed for reproducibility.

    Returns:
        2D array of orbital ramp phase, radians (unwrapped).
    """
    rng = np.random.default_rng(seed)
    rows, cols = shape
    yy, xx = np.mgrid[0:rows, 0:cols].astype(np.float64)
    yy_n, xx_n = yy / rows, xx / cols  # normalize to [0, 1) for stable coefficient scales

    if linear_coeffs is None:
        linear_coeffs = tuple(rng.uniform(-amplitude_rad, amplitude_rad, size=2))
    if quadratic_coeffs is None:
        quadratic_coeffs = tuple(rng.uniform(-amplitude_rad / 2, amplitude_rad / 2, size=3))

    a_row, a_col = linear_coeffs
    b_row2, b_col2, b_rowcol = quadratic_coeffs

    ramp = (
        a_row * yy_n + a_col * xx_n + b_row2 * yy_n**2 + b_col2 * xx_n**2 + b_rowcol * yy_n * xx_n
    )
    return ramp


# --------------------------------------------------------------------------- #
# 5. Coherence & decorrelation noise
# --------------------------------------------------------------------------- #


def simulate_coherence_map(
    shape: tuple[int, int],
    base_coherence: float = 0.75,
    low_coherence_patches: int = 3,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate a spatially-varying coherence map with randomly placed low-coherence patches.

    Mimics real interferograms where coherence is generally high but degrades
    over vegetation, water, or areas of rapid temporal change.

    Args:
        shape: (rows, cols) of the output grid.
        base_coherence: Background coherence level, in [0, 1].
        low_coherence_patches: Number of randomly placed decorrelation patches.
        seed: Optional RNG seed.

    Returns:
        2D coherence array, values in [0, 1].
    """
    rng = np.random.default_rng(seed)
    rows, cols = shape
    coherence = np.full(shape, base_coherence, dtype=np.float64)

    yy, xx = np.mgrid[0:rows, 0:cols].astype(np.float64)
    for _ in range(low_coherence_patches):
        cy, cx = rng.uniform(0, rows), rng.uniform(0, cols)
        radius = rng.uniform(0.05, 0.25) * min(rows, cols)
        drop = rng.uniform(0.3, 0.6)
        dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
        patch = np.exp(-dist2 / (2 * radius**2))
        coherence -= drop * patch

    # Add small-scale texture and clip to valid range.
    texture = rng.normal(0, 0.03, size=shape)
    coherence = np.clip(coherence + texture, 0.02, 0.99)
    return coherence


def decorrelation_phase_noise(
    coherence: np.ndarray,
    seed: int | None = None,
) -> np.ndarray:
    """Generate phase noise whose variance is governed by the Cramer-Rao bound
    for interferometric phase estimation as a function of coherence.

    A common approximation for the standard deviation of interferometric phase
    noise (single-look-ish behavior, adequate for synthetic augmentation) is:

        sigma_phi(gamma) ~ sqrt(1 - gamma^2) / (gamma * sqrt(2))

    which diverges as gamma -> 0 and vanishes as gamma -> 1, correctly capturing
    that low-coherence pixels are essentially phase-randomized.

    Args:
        coherence: Coherence map, values in [0, 1].
        seed: Optional RNG seed for reproducibility.

    Returns:
        2D array of additive phase noise, radians, same shape as `coherence`.
    """
    rng = np.random.default_rng(seed)
    gamma = np.clip(coherence, 0.02, 0.999)
    sigma_phi = np.sqrt(1.0 - gamma**2) / (gamma * math.sqrt(2.0))
    # Cap sigma to avoid numerically absurd noise at near-zero coherence while
    # still strongly penalizing low-coherence pixels.
    sigma_phi = np.clip(sigma_phi, 0.0, math.pi)
    noise = rng.normal(0.0, 1.0, size=coherence.shape) * sigma_phi
    return noise


def simulate_amplitude(
    shape: tuple[int, int],
    coherence: np.ndarray | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate log-scaled SAR backscatter amplitude, loosely correlated with coherence.

    Args:
        shape: (rows, cols) of the output grid.
        coherence: Optional coherence map used to spatially correlate amplitude
            (higher coherence areas tend to have more stable, often higher,
            backscatter in this simplified model).
        seed: Optional RNG seed.

    Returns:
        2D array of log-scaled amplitude values.
    """
    rng = np.random.default_rng(seed)
    base = rng.gamma(shape=2.0, scale=1.0, size=shape)  # speckle-like distribution
    if coherence is not None:
        base *= 0.5 + 0.5 * coherence
    return np.log1p(base)


# --------------------------------------------------------------------------- #
# 6. Top-level synthetic generator
# --------------------------------------------------------------------------- #

DeformationType = Literal["gaussian_bowl", "okada", "mogi", "none"]


class InSARSyntheticGenerator:
    """Compose all physical phase contributions into a full synthetic InSAR sample.

    Example:
        >>> gen = InSARSyntheticGenerator(size=256, seed=0)
        >>> sample = gen.generate_sample(deformation_type="mogi")
        >>> sample.wrapped_phase.shape
        (256, 256)
    """

    def __init__(
        self,
        size: int = 256,
        pixel_spacing_m: float = 20.0,
        wavelength_m: float = WAVELENGTH_M["C-band"],
        incidence_deg: float = 34.0,
        dem_path: str | Path | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Args:
            size: Output tile size (size x size), pixels.
            pixel_spacing_m: Ground pixel spacing, meters.
            wavelength_m: Radar carrier wavelength, meters (defaults to C-band /
                Sentinel-1).
            incidence_deg: Radar incidence angle, degrees.
            dem_path: Optional path to a real DEM GeoTIFF used for the
                topographic phase term. If None, a synthetic fractal terrain is
                used instead.
            seed: Base RNG seed for reproducibility; each `generate_sample` call
                derives a fresh sub-seed unless overridden.
        """
        self.size = size
        self.shape = (size, size)
        self.pixel_spacing_m = pixel_spacing_m
        self.wavelength_m = wavelength_m
        self.incidence_deg = incidence_deg
        self.dem_path = Path(dem_path) if dem_path is not None else None
        self._rng = np.random.default_rng(seed)

        self._dem_cache: np.ndarray | None = None
        if self.dem_path is not None:
            self._dem_cache = load_dem(self.dem_path, out_shape=self.shape)

    def _get_dem_tile(self) -> np.ndarray:
        """Return a DEM tile: a random crop of the cached real DEM, or synthetic
        fractal terrain if no DEM was provided."""
        if self._dem_cache is not None:
            return self._dem_cache
        # Fall back to synthetic fractal terrain using the same Kolmogorov
        # generator, scaled to plausible elevation range.
        terrain = kolmogorov_atmospheric_noise(
            self.shape,
            self.pixel_spacing_m,
            beta=3.0,
            amplitude_rad=1.0,
            seed=int(self._rng.integers(0, 2**31 - 1)),
        )
        terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min() + 1e-9)
        return terrain * 500.0  # up to ~500 m of synthetic relief

    def generate_sample(
        self,
        deformation_type: DeformationType = "gaussian_bowl",
        deformation_kwargs: dict | None = None,
        perpendicular_baseline_m: float | None = None,
        atmosphere_amplitude_rad: float = 1.5,
        ramp_amplitude_rad: float = 2.0,
        base_coherence: float | None = None,
    ) -> SyntheticSample:
        """Generate one full synthetic InSAR sample.

        Args:
            deformation_type: Which deformation model to apply.
            deformation_kwargs: Extra kwargs forwarded to the chosen deformation
                model (e.g. `amplitude_m`, `slip_m`, `volume_change_m3`).
            perpendicular_baseline_m: Perpendicular baseline for the topographic
                phase term; randomized in [-300, 300] m if None.
            atmosphere_amplitude_rad: Std. dev. of the atmospheric phase screen.
            ramp_amplitude_rad: Scale of the orbital ramp phase.
            base_coherence: Background coherence; randomized in [0.4, 0.9] if None.

        Returns:
            A `SyntheticSample` with wrapped phase, true unwrapped phase, the
            integer ambiguity map, coherence, amplitude, and per-component
            breakdown.
        """
        seed = int(self._rng.integers(0, 2**31 - 1))
        rng = np.random.default_rng(seed)
        deformation_kwargs = deformation_kwargs or {}

        # --- 1. Deformation ---
        if deformation_type == "gaussian_bowl":
            los_disp = gaussian_bowl_deformation(
                self.shape,
                self.pixel_spacing_m,
                los_incidence_deg=self.incidence_deg,
                **deformation_kwargs,
            )
        elif deformation_type == "okada":
            los_disp = okada_dislocation(
                self.shape,
                self.pixel_spacing_m,
                **deformation_kwargs,
            )
        elif deformation_type == "mogi":
            los_disp = mogi_point_source(
                self.shape,
                self.pixel_spacing_m,
                **deformation_kwargs,
            )
        elif deformation_type == "none":
            los_disp = np.zeros(self.shape, dtype=np.float64)
        else:
            raise ValueError(f"Unknown deformation_type: {deformation_type!r}")

        deformation_phase = displacement_to_phase(los_disp, self.wavelength_m)

        # --- 2. Topography ---
        if perpendicular_baseline_m is None:
            perpendicular_baseline_m = float(rng.uniform(-300.0, 300.0))
        dem_tile = self._get_dem_tile()
        topo_phase = topographic_phase(
            dem_tile,
            perpendicular_baseline_m,
            incidence_deg=self.incidence_deg,
            wavelength_m=self.wavelength_m,
        )

        # --- 3. Atmosphere ---
        atmo_phase = kolmogorov_atmospheric_noise(
            self.shape,
            self.pixel_spacing_m,
            amplitude_rad=atmosphere_amplitude_rad,
            seed=seed + 1,
        )

        # --- 4. Orbital ramp ---
        ramp_phase = orbital_ramp(self.shape, amplitude_rad=ramp_amplitude_rad, seed=seed + 2)

        # --- Noise-free physical signal (deformation + topography + atmosphere + ramp) ---
        clean_phase = deformation_phase + topo_phase + atmo_phase + ramp_phase

        # --- 5. Coherence & decorrelation ---
        if base_coherence is None:
            base_coherence = float(rng.uniform(0.4, 0.9))
        coherence = simulate_coherence_map(self.shape, base_coherence=base_coherence, seed=seed + 3)
        decorr_noise = decorrelation_phase_noise(coherence, seed=seed + 4)

        # IMPORTANT: Phase unwrapping's job is to resolve the integer 2*pi
        # ambiguity of whatever phase was actually observed -- it is *not* a
        # denoising task. Decorrelation noise is therefore folded into the
        # phase *before* it is treated as "the unwrapped signal", so that the
        # wrapped/unwrapped pair differ by an exact integer number of 2*pi
        # cycles. Using the noise-free `clean_phase` as the regression target
        # instead would make the true ambiguity map non-integer (since
        # `decorr_noise` is not itself a multiple of 2*pi), which is not a
        # valid training target for a network whose output is rounded to
        # integers.
        unwrapped_phase = clean_phase + decorr_noise

        # --- 6. Wrapping ---
        wrapped_phase = wrap_phase(unwrapped_phase)

        # Ground-truth ambiguity: exact integer by construction, since
        # `wrapped_phase` is `unwrapped_phase` wrapped into (-pi, pi].
        ambiguity = compute_ambiguity(unwrapped_phase, wrapped_phase)

        amplitude = simulate_amplitude(self.shape, coherence=coherence, seed=seed + 5)

        components = {
            "deformation": deformation_phase,
            "topography": topo_phase,
            "atmosphere": atmo_phase,
            "orbital_ramp": ramp_phase,
            "decorrelation_noise": decorr_noise,
            "clean_phase": clean_phase,  # noise-free signal, for diagnostics only
        }

        return SyntheticSample(
            wrapped_phase=wrapped_phase,
            unwrapped_phase=unwrapped_phase,
            ambiguity=ambiguity,
            coherence=coherence,
            amplitude=amplitude,
            components=components,
        )

    # ----------------------------------------------------------------- #
    # Pseudo-real data strategy
    # ----------------------------------------------------------------- #

    def rewrap_real_unwrapped_phase(
        self,
        real_unwrapped_phase: np.ndarray,
        source_wavelength_m: float = WAVELENGTH_M["L-band"],
        target_wavelength_m: float | None = None,
        add_decorrelation: bool = True,
        base_coherence: float | None = None,
        seed: int | None = None,
    ) -> SyntheticSample:
        """Bridge the sim-to-real gap using real, already-unwrapped L-band
        (e.g. ALOS-2) phase as "pseudo-real" ground truth.

        The real unwrapped phase already contains authentic topography,
        deformation, and atmospheric structure that no simulator can fully
        replicate. This function rescales it to a different (typically shorter,
        e.g. C-band/Sentinel-1) wavelength via the ratio of wavelengths, then
        rewraps it, optionally injecting decorrelation noise, to produce a
        realistic (wrapped_phase, unwrapped_phase, ambiguity) training triplet
        with genuine ground truth spatial structure.

        Args:
            real_unwrapped_phase: Real unwrapped phase (radians) at
                `source_wavelength_m`, e.g. loaded from an ALOS-2 product.
            source_wavelength_m: Wavelength the input phase was measured at.
            target_wavelength_m: Wavelength to resimulate the wrapping at
                (defaults to this generator's configured `wavelength_m`, e.g.
                C-band). Because phase (in radians) is inversely proportional to
                wavelength for a fixed displacement, rescaling by
                `source/target` converts the *equivalent displacement* phase to
                what a `target_wavelength_m` sensor would have observed for the
                same physical deformation.
            add_decorrelation: Whether to inject coherence-dependent phase noise
                before wrapping to make the pseudo-real sample more realistic.
            base_coherence: Background coherence used if `add_decorrelation` is
                True; randomized in [0.4, 0.9] if None.
            seed: Optional RNG seed.

        Returns:
            A `SyntheticSample` built from real phase structure, rewrapped at
            the target wavelength.
        """
        if target_wavelength_m is None:
            target_wavelength_m = self.wavelength_m

        seed = seed if seed is not None else int(self._rng.integers(0, 2**31 - 1))
        rng = np.random.default_rng(seed)

        # Convert equivalent displacement then re-derive phase at the target
        # wavelength: phase_target = phase_source * (source_wavelength / target_wavelength).
        # (Phase ~ 1/wavelength for a fixed physical displacement.)
        rescale_factor = source_wavelength_m / target_wavelength_m
        unwrapped_phase = real_unwrapped_phase.astype(np.float64) * rescale_factor

        shape = unwrapped_phase.shape
        if add_decorrelation:
            if base_coherence is None:
                base_coherence = float(rng.uniform(0.4, 0.9))
            coherence = simulate_coherence_map(shape, base_coherence=base_coherence, seed=seed + 1)
            decorr_noise = decorrelation_phase_noise(coherence, seed=seed + 2)
        else:
            coherence = np.full(shape, 0.9, dtype=np.float64)
            decorr_noise = np.zeros(shape, dtype=np.float64)

        wrapped_phase = wrap_phase(unwrapped_phase + decorr_noise)
        ambiguity = compute_ambiguity(unwrapped_phase, wrapped_phase)
        amplitude = simulate_amplitude(shape, coherence=coherence, seed=seed + 3)

        return SyntheticSample(
            wrapped_phase=wrapped_phase,
            unwrapped_phase=unwrapped_phase,
            ambiguity=ambiguity,
            coherence=coherence,
            amplitude=amplitude,
            components={
                "pseudo_real_rescaled": unwrapped_phase,
                "decorrelation_noise": decorr_noise,
            },
        )

    @staticmethod
    def load_real_unwrapped_geotiff(
        path: str | Path, out_shape: tuple[int, int] | None = None
    ) -> np.ndarray:
        """Load a real unwrapped-phase GeoTIFF (e.g. an ALOS-2 unwrapped product).

        Args:
            path: Path to the GeoTIFF.
            out_shape: Optional (rows, cols) to resample to.

        Returns:
            2D array of unwrapped phase, radians.
        """
        if not _HAS_RASTERIO:
            raise ImportError("rasterio is required to load real InSAR products.")
        with rasterio.open(path) as src:
            if out_shape is not None:
                arr = src.read(1, out_shape=out_shape, resampling=Resampling.bilinear)
            else:
                arr = src.read(1)
        return arr.astype(np.float64)

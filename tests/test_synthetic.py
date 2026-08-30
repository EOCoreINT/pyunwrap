"""
tests/test_synthetic.py
==========================

Unit tests for `pyunwrap.synthetic.generator`: verifies the core physics
identity (`wrap(unwrapped) == wrapped`), the Kolmogorov atmospheric noise
power spectrum, and the pseudo-real (L-band -> C-band) rewrapping strategy.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyunwrap.synthetic.generator import (
    WAVELENGTH_M,
    InSARSyntheticGenerator,
    compute_ambiguity,
    decorrelation_phase_noise,
    displacement_to_phase,
    gaussian_bowl_deformation,
    kolmogorov_atmospheric_noise,
    mogi_point_source,
    okada_dislocation,
    orbital_ramp,
    simulate_coherence_map,
    wrap_phase,
)

# --------------------------------------------------------------------------- #
# Core wrap/ambiguity identity
# --------------------------------------------------------------------------- #


class TestWrapPhaseIdentity:
    """`wrap(unwrapped_phase) == wrapped_phase` must hold exactly for every
    sample the generator produces -- this is the hard physics constraint the
    whole package is built around, so it gets the most thorough coverage.
    """

    @pytest.mark.parametrize("deformation_type", ["gaussian_bowl", "okada", "mogi", "none"])
    def test_wrap_matches_generated_wrapped_phase(self, deformation_type):
        gen = InSARSyntheticGenerator(size=64, seed=42)
        sample = gen.generate_sample(deformation_type=deformation_type)

        rewrapped = wrap_phase(sample.unwrapped_phase)
        np.testing.assert_allclose(rewrapped, sample.wrapped_phase, atol=1e-9)

    @pytest.mark.parametrize("deformation_type", ["gaussian_bowl", "okada", "mogi", "none"])
    def test_ambiguity_reconstruction_identity(self, deformation_type):
        """wrapped + 2*pi*k must reconstruct the true unwrapped phase exactly."""
        gen = InSARSyntheticGenerator(size=64, seed=7)
        sample = gen.generate_sample(deformation_type=deformation_type)

        reconstructed = sample.wrapped_phase + 2.0 * np.pi * sample.ambiguity
        np.testing.assert_allclose(reconstructed, sample.unwrapped_phase, atol=1e-9)

    def test_ambiguity_is_integer_valued(self, synthetic_sample):
        k = synthetic_sample.ambiguity
        np.testing.assert_allclose(k, np.round(k), atol=1e-9)

    def test_wrapped_phase_is_bounded(self, synthetic_sample):
        wrapped = synthetic_sample.wrapped_phase
        assert wrapped.min() >= -math.pi - 1e-9
        assert wrapped.max() <= math.pi + 1e-9

    def test_wrap_phase_is_idempotent(self):
        """wrap(wrap(x)) == wrap(x): wrapping an already-wrapped value must be a no-op."""
        x = np.linspace(-10 * math.pi, 10 * math.pi, 1000)
        once = wrap_phase(x)
        twice = wrap_phase(once)
        np.testing.assert_allclose(once, twice, atol=1e-12)

    def test_compute_ambiguity_matches_manual_construction(self):
        """A controlled, noise-free construction: build phase = wrapped + 2*pi*k_true
        directly, and verify compute_ambiguity recovers exactly k_true."""
        rng = np.random.default_rng(0)
        wrapped = rng.uniform(-math.pi, math.pi, size=(32, 32))
        k_true = rng.integers(-8, 9, size=(32, 32)).astype(float)
        unwrapped = wrapped + 2 * math.pi * k_true

        k_recovered = compute_ambiguity(unwrapped, wrapped)
        np.testing.assert_allclose(k_recovered, k_true, atol=1e-9)


# --------------------------------------------------------------------------- #
# Deformation models
# --------------------------------------------------------------------------- #


class TestDeformationModels:
    def test_gaussian_bowl_peaks_at_center(self):
        shape = (65, 65)  # odd size so there's an exact center pixel
        disp = gaussian_bowl_deformation(shape, amplitude_m=0.1, sigma_m=500.0)
        center_val = disp[32, 32]
        corner_val = disp[0, 0]
        assert abs(center_val) > abs(corner_val)

    def test_gaussian_bowl_zero_amplitude_gives_zero_displacement(self):
        disp = gaussian_bowl_deformation((32, 32), amplitude_m=0.0)
        np.testing.assert_allclose(disp, 0.0)

    def test_mogi_produces_radially_symmetric_pattern(self):
        """A Mogi source's underlying 3D displacement field is axisymmetric.
        To test that cleanly, use a purely-vertical LOS vector (0, 0, 1): with
        no horizontal LOS component, the projected displacement reduces to
        the vertical component alone, which depends only on radius and must
        therefore be identical at any two points equidistant from the source.
        (With the default, more realistic slanted LOS vector, north/south and
        east/west values are *expected* to differ -- the horizontal LOS
        components pick up the sign-antisymmetric radial horizontal
        displacement -- so that vector is deliberately not used here.)"""
        shape = (101, 101)
        disp = mogi_point_source(
            shape,
            center=(50, 50),
            volume_change_m3=1e6,
            source_depth_m=2000,
            los_vector=(0.0, 0.0, 1.0),
        )
        north = disp[40, 50]
        south = disp[60, 50]
        east = disp[50, 60]
        west = disp[50, 40]
        assert north == pytest.approx(south, rel=1e-9)
        assert east == pytest.approx(west, rel=1e-9)
        assert north == pytest.approx(east, rel=1e-9)  # also radially symmetric across axes

    def test_okada_dislocation_produces_finite_output(self):
        """Regression-style smoke test for the from-scratch Okada implementation:
        must produce a finite, non-degenerate displacement field for standard
        earthquake-scale fault parameters."""
        disp = okada_dislocation(
            (64, 64),
            strike_deg=30.0,
            dip_deg=60.0,
            rake_deg=90.0,
            slip_m=2.0,
            length_m=8000.0,
            width_m=4000.0,
            depth_m=3000.0,
        )
        assert np.all(np.isfinite(disp))
        assert disp.std() > 0.0  # not a degenerate all-constant field

    def test_displacement_to_phase_sign_convention(self):
        """Positive LOS displacement (toward sensor) should map to a
        consistent, nonzero phase sign per the -4*pi/lambda convention."""
        phase_pos = displacement_to_phase(np.array([0.01]), wavelength_m=WAVELENGTH_M["C-band"])
        phase_neg = displacement_to_phase(np.array([-0.01]), wavelength_m=WAVELENGTH_M["C-band"])
        assert phase_pos[0] == -phase_neg[0]
        assert phase_pos[0] != 0.0


# --------------------------------------------------------------------------- #
# Atmospheric (Kolmogorov) noise
# --------------------------------------------------------------------------- #


class TestKolmogorovNoise:
    def test_power_spectrum_slope_matches_beta(self):
        """The 2D radial power spectral density of the generated screen should
        follow P(f) ~ f^(-beta); fit the slope in log-log space and check it's
        close to the requested beta."""
        beta = 8.0 / 3.0
        screen = kolmogorov_atmospheric_noise(
            (256, 256),
            pixel_spacing_m=20.0,
            beta=beta,
            amplitude_rad=2.0,
            seed=1,
        )
        fft = np.fft.fft2(screen)
        psd = np.abs(fft) ** 2

        fy = np.fft.fftfreq(256, d=20.0)
        fx = np.fft.fftfreq(256, d=20.0)
        fyy, fxx = np.meshgrid(fy, fx, indexing="ij")
        f_radial = np.sqrt(fyy**2 + fxx**2)

        # Restrict the fit to a mid-frequency band, avoiding the DC bin and
        # the highest frequencies (aliasing/discretization noise dominates there).
        mask = (f_radial > 1e-5) & (f_radial < 0.01)
        log_f = np.log(f_radial[mask])
        log_p = np.log(psd[mask] + 1e-30)
        measured_slope, _intercept = np.polyfit(log_f, log_p, 1)

        assert measured_slope == pytest.approx(-beta, abs=0.5)

    def test_output_shape_and_zero_mean(self):
        screen = kolmogorov_atmospheric_noise((64, 64), amplitude_rad=1.0, seed=0)
        assert screen.shape == (64, 64)
        assert abs(screen.mean()) < 1e-9

    def test_amplitude_controls_std(self):
        screen_small = kolmogorov_atmospheric_noise((128, 128), amplitude_rad=0.5, seed=2)
        screen_large = kolmogorov_atmospheric_noise((128, 128), amplitude_rad=2.0, seed=2)
        assert screen_small.std() == pytest.approx(0.5, rel=0.05)
        assert screen_large.std() == pytest.approx(2.0, rel=0.05)


# --------------------------------------------------------------------------- #
# Coherence / decorrelation
# --------------------------------------------------------------------------- #


class TestCoherenceAndDecorrelation:
    def test_coherence_map_bounded(self):
        coh = simulate_coherence_map((64, 64), base_coherence=0.7, seed=0)
        assert coh.min() >= 0.0
        assert coh.max() <= 1.0

    def test_decorrelation_noise_variance_increases_as_coherence_drops(self):
        """Lower coherence should produce higher-variance phase noise, per the
        sigma_phi(gamma) formula (Cramer-Rao-style approximation)."""
        high_coh = np.full((64, 64), 0.95)
        low_coh = np.full((64, 64), 0.15)
        noise_high = decorrelation_phase_noise(high_coh, seed=0)
        noise_low = decorrelation_phase_noise(low_coh, seed=0)
        assert noise_low.std() > noise_high.std()


# --------------------------------------------------------------------------- #
# Orbital ramps
# --------------------------------------------------------------------------- #


class TestOrbitalRamp:
    def test_zero_coefficients_give_zero_ramp(self):
        ramp = orbital_ramp((32, 32), linear_coeffs=(0.0, 0.0), quadratic_coeffs=(0.0, 0.0, 0.0))
        np.testing.assert_allclose(ramp, 0.0)

    def test_linear_ramp_is_monotonic_along_its_axis(self):
        ramp = orbital_ramp((32, 32), linear_coeffs=(0.0, 1.0), quadratic_coeffs=(0.0, 0.0, 0.0))
        row = ramp[0, :]
        assert np.all(np.diff(row) >= -1e-12)  # monotonically non-decreasing along columns


# --------------------------------------------------------------------------- #
# Pseudo-real (L-band -> C-band) rewrapping strategy
# --------------------------------------------------------------------------- #


class TestPseudoRealStrategy:
    def test_rewrap_preserves_identity(self, small_generator):
        rng = np.random.default_rng(3)
        fake_real_phase = rng.normal(0, 15, size=(64, 64)).cumsum(axis=0).cumsum(axis=1) * 0.01

        sample = small_generator.rewrap_real_unwrapped_phase(
            fake_real_phase,
            source_wavelength_m=WAVELENGTH_M["L-band"],
            target_wavelength_m=WAVELENGTH_M["C-band"],
            add_decorrelation=False,
        )
        reconstructed = sample.wrapped_phase + 2 * np.pi * sample.ambiguity
        np.testing.assert_allclose(reconstructed, sample.unwrapped_phase, atol=1e-9)

    def test_rescale_factor_matches_wavelength_ratio(self, small_generator):
        rng = np.random.default_rng(4)
        fake_real_phase = rng.normal(0, 1, size=(64, 64))
        sample = small_generator.rewrap_real_unwrapped_phase(
            fake_real_phase,
            source_wavelength_m=WAVELENGTH_M["L-band"],
            target_wavelength_m=WAVELENGTH_M["C-band"],
            add_decorrelation=False,
        )
        expected_factor = WAVELENGTH_M["L-band"] / WAVELENGTH_M["C-band"]
        actual_factor = sample.components["pseudo_real_rescaled"][0, 0] / fake_real_phase[0, 0]
        assert actual_factor == pytest.approx(expected_factor, rel=1e-6)

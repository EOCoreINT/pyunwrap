"""
tests/test_physics.py
========================

Physics-correctness tests for `pyunwrap.analytics.phase_stats` and
`pyunwrap.analytics.explainability`: residue detection against a
hand-constructed phase field with a known, exact residue; Nyquist gradient
flagging against a hand-constructed steep-gradient field; and the
Integrated-Gradients completeness axiom.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyunwrap.analytics.phase_stats import (
    compare_residues,
    compute_error_distribution,
    detect_residues,
    gradient_analysis_summary,
    nyquist_violation_map,
    phase_gradient_magnitude,
    residue_summary,
)

# --------------------------------------------------------------------------- #
# Residue detection: known-answer tests
# --------------------------------------------------------------------------- #


def _make_phase_vortex(size: int = 20, center: tuple[float, float] = (10.3, 10.3)) -> np.ndarray:
    """Construct a phase field with exactly one +1 topological residue.

    `phase = atan2(y - y0, x - x0)` is the canonical branch-point/vortex
    field: winding once around `(y0, x0)` sweeps the phase through exactly
    2*pi, which is precisely the condition a residue detector must flag. The
    center is placed off-grid (non-integer pixel coordinates) so the
    winding point doesn't coincide exactly with a sample location.
    """
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    y0, x0 = center
    return np.arctan2(yy - y0, xx - x0)


class TestResidueDetection:
    def test_known_vortex_produces_exactly_one_residue(self):
        vortex = _make_phase_vortex(size=20, center=(10.3, 10.3))
        residue_map, clusters_df = detect_residues(vortex)

        assert (residue_map != 0).sum() == 1
        assert len(clusters_df) == 1
        assert clusters_df.iloc[0]["charge_sign"] == 1
        # The detected plaquette should sit at the vortex's integer-floor location.
        assert clusters_df.iloc[0]["centroid_row"] == pytest.approx(10.0, abs=1.0)
        assert clusters_df.iloc[0]["centroid_col"] == pytest.approx(10.0, abs=1.0)

    def test_negative_vortex_produces_negative_charge(self):
        """Negating a known +1 vortex field must flip the detected residue's sign."""
        vortex = _make_phase_vortex(size=20, center=(10.3, 10.3))
        negated = -vortex
        residue_map, clusters_df = detect_residues(negated)
        assert (residue_map != 0).sum() == 1
        assert clusters_df.iloc[0]["charge_sign"] == -1

    def test_smooth_field_has_zero_residues(self):
        yy, xx = np.mgrid[0:32, 0:32].astype(float)
        smooth_phase = 0.01 * (xx**2 + yy**2)
        smooth_wrapped = np.angle(np.exp(1j * smooth_phase))
        residue_map, clusters_df = detect_residues(smooth_wrapped)
        assert (residue_map != 0).sum() == 0
        assert len(clusters_df) == 0

    def test_residue_detection_is_invariant_to_wrapping(self):
        """Per the topological-invariance argument (see `losses.py`), the
        residue count of a phase field must be identical whether computed on
        the wrapped or unwrapped version -- adding 2*pi*k anywhere cannot
        change the wrapped-gradient loop sums."""
        vortex = _make_phase_vortex(size=20)
        wrapped_vortex = np.angle(np.exp(1j * vortex))
        map_unwrapped, _ = detect_residues(vortex)
        map_wrapped, _ = detect_residues(wrapped_vortex)
        np.testing.assert_array_equal(map_unwrapped, map_wrapped)

    def test_residue_summary_matches_detect_residues(self):
        vortex = _make_phase_vortex(size=20)
        summary = residue_summary(vortex)
        assert summary["n_positive_clusters"] == 1
        assert summary["n_negative_clusters"] == 0
        assert summary["net_charge"] == 1

    def test_compare_residues_flags_spurious_residues(self):
        """A predicted field with an INSERTED extra vortex (relative to the
        input's residue content) must be flagged as containing spurious
        residues."""
        size = 40
        # "Input": smooth, no residues.
        yy, xx = np.mgrid[0:size, 0:size].astype(float)
        clean = 0.005 * (xx**2 + yy**2)
        clean_wrapped = np.angle(np.exp(1j * clean))

        # "Predicted": same smooth field PLUS a spurious vortex the input never had.
        vortex = np.arctan2(yy - 20.3, xx - 20.3)
        predicted = clean + vortex

        result = compare_residues(clean_wrapped, predicted)
        assert result["spurious_residue_pixels"] > 0
        assert result["predicted"]["n_positive_pixels"] > result["input"]["n_positive_pixels"]


# --------------------------------------------------------------------------- #
# Nyquist gradient analysis: known-answer tests
# --------------------------------------------------------------------------- #


class TestNyquistGradientAnalysis:
    def test_known_steep_gradient_is_measured_correctly(self):
        _yy, xx = np.mgrid[0:50, 0:50].astype(float)
        steep = 5.0 * xx  # exact gradient magnitude = 5.0 rad/pixel everywhere (interior)
        grad_mag = phase_gradient_magnitude(steep)
        assert grad_mag[25, 25] == pytest.approx(5.0, abs=0.05)

    def test_gradient_exceeding_pi_is_flagged(self):
        _yy, xx = np.mgrid[0:50, 0:50].astype(float)
        steep = 5.0 * xx  # 5 > pi everywhere
        violations = nyquist_violation_map(steep)
        assert violations.mean() > 0.9  # nearly all interior pixels flagged

    def test_flat_field_has_no_violations(self):
        flat = np.zeros((50, 50))
        violations = nyquist_violation_map(flat)
        assert violations.sum() == 0

    def test_gradient_summary_reports_consistent_stats(self):
        _yy, xx = np.mgrid[0:50, 0:50].astype(float)
        steep = 5.0 * xx
        summary = gradient_analysis_summary(steep)
        assert summary["max_gradient"] == pytest.approx(5.0, abs=0.1)
        assert summary["n_nyquist_violations"] > 0
        assert 0.0 <= summary["pct_nyquist_violations"] <= 100.0


# --------------------------------------------------------------------------- #
# Error distribution: known-answer tests
# --------------------------------------------------------------------------- #


class TestErrorDistribution:
    def test_recovers_known_gaussian_noise_std(self):
        rng = np.random.default_rng(0)
        true_phase = np.zeros((100, 100))
        known_std = 0.2
        predicted = true_phase + rng.normal(0, known_std, size=(100, 100))

        _hist_df, summary = compute_error_distribution(predicted, true_phase)
        assert summary["std_error"] == pytest.approx(known_std, rel=0.1)
        assert summary["mean_error"] == pytest.approx(0.0, abs=0.05)

    def test_zero_error_for_identical_arrays(self):
        phase = np.random.default_rng(1).normal(size=(20, 20))
        _hist_df, summary = compute_error_distribution(phase, phase)
        assert summary["rmse"] == pytest.approx(0.0, abs=1e-12)
        assert summary["pct_under_0p1_rad"] == pytest.approx(100.0)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            compute_error_distribution(np.zeros((10, 10)), np.zeros((5, 5)))


# --------------------------------------------------------------------------- #
# Explainability: completeness axiom
# --------------------------------------------------------------------------- #


class TestIntegratedGradientsCompleteness:
    def test_attribution_sums_to_output_difference(self, tiny_model):
        import torch

        from pyunwrap.analytics.explainability import integrated_gradients

        x = torch.rand(1, 3, 64, 64)
        baseline = torch.zeros_like(x)
        target_fn = lambda out: out.k_continuous.sum()

        result = integrated_gradients(
            tiny_model, x, target_fn=target_fn, baseline=baseline, steps=100
        )

        with torch.no_grad():
            target_x = target_fn(tiny_model(x)).item()
            target_baseline = target_fn(tiny_model(baseline)).item()
        expected_diff = target_x - target_baseline

        attribution_sum = float(result.attribution_map.sum())
        rel_error = abs(attribution_sum - expected_diff) / (abs(expected_diff) + 1e-8)
        assert rel_error < 0.05  # Riemann-sum approximation tolerance at 100 steps

    def test_channel_importance_shares_sum_to_one(self, tiny_model, dummy_input):
        from pyunwrap.analytics.explainability import integrated_gradients

        result = integrated_gradients(tiny_model, dummy_input[:1], steps=20)
        assert sum(result.channel_importance.values()) == pytest.approx(1.0, abs=1e-6)
        assert set(result.channel_importance.keys()) == {"wrapped_phase", "coherence", "amplitude"}


class TestGradCAM:
    def test_saliency_map_shape_and_range(self, tiny_model, dummy_input):
        from pyunwrap.analytics.explainability import grad_cam

        cam_result = grad_cam(tiny_model, dummy_input[:1])
        assert cam_result.saliency_map.shape == dummy_input.shape[-2:]
        assert cam_result.saliency_map.min() >= 0.0
        assert cam_result.saliency_map.max() <= 1.0 + 1e-6


class TestReliabilityDiagram:
    def test_well_calibrated_uncertainty_gives_high_correlation(self):
        from pyunwrap.analytics.explainability import reliability_diagram

        rng = np.random.default_rng(0)
        uncertainty = rng.uniform(0, 1, size=5000)
        error = uncertainty * 2 + rng.normal(
            0, 0.1, size=5000
        )  # strongly correlated by construction

        _bins_df, summary = reliability_diagram(uncertainty, error, n_bins=10)
        assert summary["spearman_correlation"] > 0.9

    def test_uncorrelated_uncertainty_gives_near_zero_correlation(self):
        from pyunwrap.analytics.explainability import reliability_diagram

        rng = np.random.default_rng(1)
        uncertainty = rng.uniform(0, 1, size=5000)
        error = rng.normal(0, 1, size=5000)  # independent of uncertainty by construction

        _bins_df, summary = reliability_diagram(uncertainty, error, n_bins=10)
        assert abs(summary["spearman_correlation"]) < 0.15

    def test_shape_mismatch_raises(self):
        from pyunwrap.analytics.explainability import reliability_diagram

        with pytest.raises(ValueError):
            reliability_diagram(np.zeros(10), np.zeros(5))

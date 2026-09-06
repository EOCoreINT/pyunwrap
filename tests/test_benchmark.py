"""
tests/test_benchmark.py
==========================

Tests for `pyunwrap.analytics.benchmark`, the pyunwrap-vs-SNAPHU comparison
harness. Skips cleanly if `snaphu` isn't installed (optional extra).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("snaphu", reason="snaphu is an optional extra; see pyproject.toml")

from pyunwrap.analytics.benchmark import (
    _best_fit_offset_rmse,
    compare_against_snaphu,
    run_benchmark_suite,
)
from pyunwrap.synthetic.generator import InSARSyntheticGenerator


class TestBestFitOffsetRmse:
    def test_zero_for_identical_arrays(self):
        arr = np.random.default_rng(0).normal(size=(20, 20))
        _residual, rmse = _best_fit_offset_rmse(arr, arr)
        assert rmse == pytest.approx(0.0, abs=1e-10)

    def test_removes_known_global_offset(self):
        """A predicted field that's the true field plus an arbitrary global
        2*pi*n offset must score as if it were a perfect prediction."""
        true = np.random.default_rng(1).normal(size=(20, 20))
        predicted = true + 6 * 2 * np.pi  # arbitrary integer multiple of 2*pi
        _residual, rmse = _best_fit_offset_rmse(predicted, true)
        assert rmse == pytest.approx(0.0, abs=1e-10)

    def test_does_not_hide_genuine_error(self):
        """A predicted field with genuine (non-offset) error must NOT be
        scored as zero -- confirms the offset removal isn't accidentally
        absorbing real error too."""
        rng = np.random.default_rng(2)
        true = np.zeros((30, 30))
        predicted = true + rng.normal(0, 0.5, size=(30, 30))  # genuine per-pixel noise
        _residual, rmse = _best_fit_offset_rmse(predicted, true)
        assert rmse == pytest.approx(0.5, rel=0.15)


class TestCompareAgainstSnaphu:
    def test_runs_end_to_end_on_easy_scene(self, tiny_model):
        """Smoke test: both methods run, both return sensible (finite,
        non-negative) metrics, on a genuinely easy controlled scene."""
        gen = InSARSyntheticGenerator(size=64, seed=10)
        sample = gen.generate_sample(
            deformation_type="none",
            base_coherence=0.95,
            atmosphere_amplitude_rad=0.1,
            ramp_amplitude_rad=0.1,
        )
        result = compare_against_snaphu(
            tiny_model,
            sample,
            scene_id="test_scene",
            deformation_type="none",
            device="cpu",
            snaphu_nlooks=4.0,
        )
        assert result.snaphu is not None
        assert np.isfinite(result.snaphu.rmse_rad) and result.snaphu.rmse_rad >= 0
        assert np.isfinite(result.pyunwrap.rmse_rad) and result.pyunwrap.rmse_rad >= 0
        assert 0.0 <= result.snaphu.reliable_fraction <= 1.0
        assert result.pyunwrap.reliable_fraction == 1.0

    def test_rejects_non_multiple_of_32_scene_size(self, tiny_model):
        gen = InSARSyntheticGenerator(size=50, seed=11)  # not a multiple of 32
        sample = gen.generate_sample(deformation_type="none")
        with pytest.raises(ValueError):
            compare_against_snaphu(tiny_model, sample, "x", "none")


class TestRunBenchmarkSuite:
    def test_returns_dataframe_with_expected_columns(self, tiny_model):
        # Use a small size (multiple of 32) to keep this fast.
        df = run_benchmark_suite(tiny_model, size=64, device="cpu", seed=20, snaphu_nlooks=4.0)
        assert len(df) == 7  # 7 scenarios defined in run_benchmark_suite
        expected_cols = {
            "scene_id",
            "deformation_type",
            "mean_coherence",
            "pct_nyquist_violations",
            "max_abs_ambiguity",
            "pyunwrap_rmse_rad",
            "pyunwrap_pct_under_0p1_rad",
            "pyunwrap_runtime_s",
            "snaphu_rmse_rad",
            "snaphu_pct_under_0p1_rad",
            "snaphu_reliable_fraction",
            "snaphu_runtime_s",
        }
        assert expected_cols.issubset(set(df.columns))
        assert (df["pyunwrap_rmse_rad"] >= 0).all()
        assert (df["snaphu_rmse_rad"] >= 0).all()


class TestDecideWinner:
    def test_clear_winner_declared(self):
        from pyunwrap.analytics.benchmark import _decide_winner

        assert _decide_winner(classical_rmse=10.0, learned_rmse=1.0, hybrid_rmse=5.0) == "learned"

    def test_near_tie_is_inconclusive(self):
        from pyunwrap.analytics.benchmark import _decide_winner

        assert (
            _decide_winner(classical_rmse=1.0, learned_rmse=1.02, hybrid_rmse=1.01)
            == "inconclusive"
        )

    def test_handles_missing_classical(self):
        from pyunwrap.analytics.benchmark import _decide_winner

        assert _decide_winner(classical_rmse=None, learned_rmse=5.0, hybrid_rmse=1.0) == "hybrid"


class TestHybridBenchmarkSuite:
    def test_runs_end_to_end_with_expected_columns(self, tiny_model):
        from pyunwrap.analytics.benchmark import run_hybrid_benchmark_suite

        df = run_hybrid_benchmark_suite(
            tiny_model, size=64, device="cpu", seed=30, snaphu_nlooks=4.0
        )
        assert len(df) == 7
        expected_cols = {
            "scene_id",
            "deformation_type",
            "mean_coherence",
            "pct_nyquist_violations",
            "classical_rmse_rad",
            "learned_rmse_rad",
            "hybrid_rmse_rad",
            "winner",
        }
        assert expected_cols.issubset(set(df.columns))
        assert df["winner"].isin(["classical", "learned", "hybrid", "inconclusive"]).all()

    def test_regime_report_never_claims_universal_winner(self, tiny_model):
        from pyunwrap.analytics.benchmark import generate_regime_report, run_hybrid_benchmark_suite

        df = run_hybrid_benchmark_suite(
            tiny_model, size=64, device="cpu", seed=31, snaphu_nlooks=4.0
        )
        report = generate_regime_report(df)
        assert "does not claim any single method wins across the board" in report
        # Every outcome category present in the data must be named explicitly.
        for outcome in df["winner"].unique():
            assert outcome in report.lower() or outcome.capitalize() in report

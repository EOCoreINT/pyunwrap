"""
tests/test_losses.py
=======================

Tests for `pyunwrap.models.losses`, focused on the new
`CoherenceWeightedPhaseSmoothnessLoss` / `SmoothnessConfig` (Strategy 3:
coherence-weighted, edge-preserving phase-smoothness regularizer) and its
opt-in integration into `PhysicsInformedUnwrapLoss` as Component 5.

`PhysicsInformedUnwrapLoss`'s original four components already have
indirect coverage via `tests/test_model.py`; this file is additive, not a
replacement.
"""

from __future__ import annotations

import pytest
import torch

from pyunwrap.models.ambiguity_net import AmbiguityNet
from pyunwrap.models.losses import (
    CoherenceWeightedPhaseSmoothnessLoss,
    PhysicsInformedUnwrapLoss,
    SmoothnessConfig,
)


def _make_field(batch=1, h=32, w=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, 1, h, w, generator=g)


class TestSmoothnessConfig:
    def test_defaults_match_spec(self):
        cfg = SmoothnessConfig()
        assert cfg.use_smoothness_loss is True
        assert cfg.smoothness_weight == 0.1
        assert cfg.smoothness_norm == "huber"
        assert cfg.edge_tau == 1.0
        assert cfg.apply_in_high_coherence_only is True
        assert cfg.coherence_threshold == 0.5
        assert cfg.penalize_ambiguity_gradients is True

    def test_rejects_invalid_norm(self):
        with pytest.raises(ValueError):
            SmoothnessConfig(smoothness_norm="l2")  # not a supported option

    def test_rejects_invalid_coherence_threshold(self):
        with pytest.raises(ValueError):
            SmoothnessConfig(coherence_threshold=1.5)
        with pytest.raises(ValueError):
            SmoothnessConfig(coherence_threshold=-0.1)

    def test_rejects_nonpositive_edge_tau(self):
        with pytest.raises(ValueError):
            SmoothnessConfig(edge_tau=0.0)
        with pytest.raises(ValueError):
            SmoothnessConfig(edge_tau=-1.0)


class TestCoherenceWeightedPhaseSmoothnessLoss:
    def test_disabled_returns_exact_zero(self):
        loss_fn = CoherenceWeightedPhaseSmoothnessLoss(SmoothnessConfig(use_smoothness_loss=False))
        phi_hat = _make_field()
        psi = _make_field(seed=1)
        k_hat = torch.zeros_like(phi_hat)
        gamma = torch.ones_like(phi_hat)
        result = loss_fn(phi_hat, psi, k_hat, gamma)
        assert result.item() == 0.0

    def test_degenerate_single_pixel_input_returns_zero_not_nan(self):
        """A 1x1 (or otherwise gradient-less) tile has no interior gradient
        pixels; the loss must return a clean zero, not NaN from an empty
        tensor's .mean()."""
        loss_fn = CoherenceWeightedPhaseSmoothnessLoss()
        phi_hat = torch.randn(1, 1, 1, 1)
        psi = torch.randn(1, 1, 1, 1)
        k_hat = torch.zeros_like(phi_hat)
        gamma = torch.ones_like(phi_hat)
        result = loss_fn(phi_hat, psi, k_hat, gamma)
        assert torch.isfinite(result)
        assert result.item() == 0.0

    def test_is_differentiable(self):
        loss_fn = CoherenceWeightedPhaseSmoothnessLoss()
        phi_hat = _make_field()
        phi_hat.requires_grad_(True)
        psi = _make_field(seed=1)
        k_hat = _make_field(seed=2).requires_grad_(True)
        gamma = torch.rand(1, 1, 32, 32)

        result = loss_fn(phi_hat, psi, k_hat, gamma)
        result.backward()

        assert phi_hat.grad is not None
        assert torch.isfinite(phi_hat.grad).all()
        assert k_hat.grad is not None
        assert torch.isfinite(k_hat.grad).all()

    def test_high_coherence_noisy_region_is_penalized_more_than_smooth(self):
        """A noisy phi_hat in a high-coherence region must incur a larger
        penalty than a smooth (constant) one under identical coherence."""
        loss_fn = CoherenceWeightedPhaseSmoothnessLoss(
            SmoothnessConfig(apply_in_high_coherence_only=False, penalize_ambiguity_gradients=False)
        )
        gamma = torch.full((1, 1, 32, 32), 0.95)
        k_hat = torch.zeros(1, 1, 32, 32)
        psi = torch.zeros(1, 1, 32, 32)

        smooth = torch.zeros(1, 1, 32, 32)
        noisy = torch.randn(1, 1, 32, 32) * 2.0

        penalty_smooth = loss_fn(smooth, psi, k_hat, gamma)
        penalty_noisy = loss_fn(noisy, psi, k_hat, gamma)
        assert penalty_noisy.item() > penalty_smooth.item()

    def test_low_coherence_receives_weaker_penalty_than_high_coherence(self):
        """The identical noisy phi_hat pattern must be penalized less when
        coherence is uniformly low than when it's uniformly high."""
        loss_fn = CoherenceWeightedPhaseSmoothnessLoss(
            SmoothnessConfig(apply_in_high_coherence_only=False, penalize_ambiguity_gradients=False)
        )
        torch.manual_seed(3)
        noisy = torch.randn(1, 1, 32, 32) * 2.0
        k_hat = torch.zeros_like(noisy)
        psi = torch.zeros_like(noisy)

        gamma_high = torch.full((1, 1, 32, 32), 0.95)
        gamma_low = torch.full((1, 1, 32, 32), 0.05)

        penalty_high = loss_fn(noisy, psi, k_hat, gamma_high)
        penalty_low = loss_fn(noisy, psi, k_hat, gamma_low)
        assert penalty_low.item() < penalty_high.item()

    def test_high_coherence_only_mask_zeroes_low_coherence_pixels(self):
        """With apply_in_high_coherence_only=True and coherence entirely
        below threshold, the penalty must be exactly zero -- not just
        smaller."""
        loss_fn = CoherenceWeightedPhaseSmoothnessLoss(
            SmoothnessConfig(
                apply_in_high_coherence_only=True,
                coherence_threshold=0.5,
                penalize_ambiguity_gradients=False,
            )
        )
        torch.manual_seed(4)
        noisy = torch.randn(1, 1, 32, 32) * 2.0
        k_hat = torch.zeros_like(noisy)
        psi = torch.zeros_like(noisy)
        gamma_all_low = torch.full((1, 1, 32, 32), 0.1)

        penalty = loss_fn(noisy, psi, k_hat, gamma_all_low)
        assert penalty.item() == 0.0

    def test_strong_discontinuity_is_not_destroyed_by_smoothing(self):
        """A genuine large, spatially-consistent step edge (e.g. a
        deformation boundary) must receive a much smaller *per-unit-gradient*
        penalty than equivalent-magnitude but spatially-incoherent noise, due
        to the edge-preservation factor -- this is the entire point of the
        edge-aware term over a plain (existing) L2 smoothness penalty."""
        edge_tau = 0.5
        loss_fn = CoherenceWeightedPhaseSmoothnessLoss(
            SmoothnessConfig(
                apply_in_high_coherence_only=False,
                penalize_ambiguity_gradients=False,
                edge_tau=edge_tau,
                smoothness_norm="huber",
            )
        )
        gamma = torch.full((1, 1, 32, 32), 0.95)
        k_hat = torch.zeros(1, 1, 32, 32)
        psi = torch.zeros(1, 1, 32, 32)

        # A clean step edge: left half low, right half high, large jump exactly at the boundary.
        step = torch.zeros(1, 1, 32, 32)
        step[:, :, :, 16:] = 6.0  # a single sharp, large, spatially-coherent jump

        # Same total gradient "energy" but spread across many small, incoherent jumps instead.
        torch.manual_seed(5)
        noisy_same_scale = torch.randn(1, 1, 32, 32) * 6.0

        penalty_step = loss_fn(step, psi, k_hat, gamma)
        penalty_noisy = loss_fn(noisy_same_scale, psi, k_hat, gamma)

        # The edge-preserving penalty on the single coherent large jump should
        # be substantially smaller than on widespread large incoherent jumps,
        # since exp(-|grad|/tau) suppresses the (rare) very large gradient at
        # the step much more selectively than it suppresses the (common)
        # large gradients everywhere in the noisy field's penalty mean.
        assert penalty_step.item() < penalty_noisy.item()

    def test_huber_vs_l1_differ(self):
        torch.manual_seed(6)
        phi_hat = torch.randn(1, 1, 32, 32) * 3.0
        psi = torch.zeros_like(phi_hat)
        k_hat = torch.zeros_like(phi_hat)
        gamma = torch.full_like(phi_hat, 0.9)

        loss_huber = CoherenceWeightedPhaseSmoothnessLoss(
            SmoothnessConfig(
                smoothness_norm="huber",
                apply_in_high_coherence_only=False,
                penalize_ambiguity_gradients=False,
            )
        )
        loss_l1 = CoherenceWeightedPhaseSmoothnessLoss(
            SmoothnessConfig(
                smoothness_norm="l1",
                apply_in_high_coherence_only=False,
                penalize_ambiguity_gradients=False,
            )
        )
        assert (
            loss_huber(phi_hat, psi, k_hat, gamma).item()
            != loss_l1(phi_hat, psi, k_hat, gamma).item()
        )

    def test_penalize_ambiguity_gradients_toggle_changes_result(self):
        torch.manual_seed(7)
        phi_hat = torch.randn(1, 1, 32, 32)
        psi = torch.zeros_like(phi_hat)
        k_hat = torch.randn(1, 1, 32, 32) * 2.0  # nonzero k gradient
        gamma = torch.full_like(phi_hat, 0.9)

        loss_with = CoherenceWeightedPhaseSmoothnessLoss(
            SmoothnessConfig(penalize_ambiguity_gradients=True, apply_in_high_coherence_only=False)
        )
        loss_without = CoherenceWeightedPhaseSmoothnessLoss(
            SmoothnessConfig(penalize_ambiguity_gradients=False, apply_in_high_coherence_only=False)
        )
        assert (
            loss_with(phi_hat, psi, k_hat, gamma).item()
            > loss_without(phi_hat, psi, k_hat, gamma).item()
        )

    def test_edge_mask_and_regime_mask_zero_out_masked_regions(self):
        torch.manual_seed(8)
        phi_hat = torch.randn(1, 1, 32, 32) * 3.0
        psi = torch.zeros_like(phi_hat)
        k_hat = torch.zeros_like(phi_hat)
        gamma = torch.full_like(phi_hat, 0.9)

        loss_fn = CoherenceWeightedPhaseSmoothnessLoss(
            SmoothnessConfig(apply_in_high_coherence_only=False, penalize_ambiguity_gradients=False)
        )
        zero_mask = torch.zeros_like(phi_hat)
        result = loss_fn(phi_hat, psi, k_hat, gamma, edge_mask=zero_mask)
        assert result.item() == 0.0

        result2 = loss_fn(phi_hat, psi, k_hat, gamma, regime_mask=zero_mask)
        assert result2.item() == 0.0


class TestPhysicsInformedUnwrapLossIntegration:
    def test_default_construction_disables_component_5(self):
        """PhysicsInformedUnwrapLoss() with no arguments must be exactly
        equivalent to the pre-Strategy-3 4-component loss."""
        torch.manual_seed(9)
        model = AmbiguityNet(pretrained=False, k_max=10.0)
        x = torch.rand(2, 3, 32, 32) * 2 - 1  # batch>1: BatchNorm requires it in train mode
        out = model(x)
        k_true = torch.zeros(2, 1, 32, 32)
        coherence = torch.rand(2, 1, 32, 32)

        criterion = PhysicsInformedUnwrapLoss()
        result = criterion(out, k_true=k_true, wrapped_phase_norm=x[:, 0:1], coherence=coherence)
        assert result.edge_smoothness.item() == 0.0
        assert criterion.edge_smoothness_loss is None

    def test_smoothness_config_enables_and_contributes_to_total(self):
        torch.manual_seed(10)
        model = AmbiguityNet(pretrained=False, k_max=10.0)
        x = torch.rand(2, 3, 32, 32) * 2 - 1
        out = model(x)
        k_true = torch.zeros(2, 1, 32, 32)
        coherence = torch.rand(2, 1, 32, 32)

        criterion_off = PhysicsInformedUnwrapLoss()
        criterion_on = PhysicsInformedUnwrapLoss(
            smoothness_config=SmoothnessConfig(smoothness_weight=1.0)
        )

        result_off = criterion_off(
            out, k_true=k_true, wrapped_phase_norm=x[:, 0:1], coherence=coherence
        )
        result_on = criterion_on(
            out, k_true=k_true, wrapped_phase_norm=x[:, 0:1], coherence=coherence
        )

        assert result_on.edge_smoothness.item() >= 0.0
        # Enabling a nonzero-weighted extra component must change the total
        # unless the component itself happens to be exactly zero (extremely
        # unlikely with random weights/coherence, but guard explicitly).
        if result_on.edge_smoothness.item() > 0:
            assert result_on.total.item() != result_off.total.item()

    def test_as_dict_includes_edge_smoothness_key(self):
        torch.manual_seed(11)
        model = AmbiguityNet(pretrained=False, k_max=10.0)
        x = torch.rand(2, 3, 32, 32) * 2 - 1
        out = model(x)
        k_true = torch.zeros(2, 1, 32, 32)
        coherence = torch.rand(2, 1, 32, 32)
        criterion = PhysicsInformedUnwrapLoss()
        result = criterion(out, k_true=k_true, wrapped_phase_norm=x[:, 0:1], coherence=coherence)
        d = result.as_dict()
        assert "loss/edge_smoothness" in d
        assert d["loss/edge_smoothness"] == 0.0

    def test_backward_through_full_composite_loss_with_component_5(self):
        torch.manual_seed(12)
        model = AmbiguityNet(pretrained=False, k_max=10.0)
        x = (torch.rand(2, 3, 32, 32) * 2 - 1).requires_grad_(False)
        out = model(x)
        k_true = torch.randint(-2, 3, (2, 1, 32, 32)).float()
        coherence = torch.rand(2, 1, 32, 32)

        criterion = PhysicsInformedUnwrapLoss(smoothness_config=SmoothnessConfig())
        result = criterion(out, k_true=k_true, wrapped_phase_norm=x[:, 0:1], coherence=coherence)
        result.total.backward()

        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0
        assert all(torch.isfinite(g).all() for g in grads)

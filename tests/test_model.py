"""
tests/test_model.py
======================

Unit tests for `pyunwrap.models.ambiguity_net` (architecture) and
`pyunwrap.models.losses` (physics-informed loss). Uses `pretrained=False`
throughout, since network access to download ImageNet weights is not
guaranteed in CI.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from pyunwrap.models.ambiguity_net import AmbiguityNet, round_ste
from pyunwrap.models.losses import (
    PhysicsInformedUnwrapLoss,
    discrete_laplacian,
    wrap_phase_torch,
)

# --------------------------------------------------------------------------- #
# Architecture
# --------------------------------------------------------------------------- #


class TestAmbiguityNetShapes:
    def test_output_shapes(self, tiny_model, dummy_input):
        out = tiny_model(dummy_input)
        batch, _, h, w = dummy_input.shape
        assert out.k_hat.shape == (batch, 1, h, w)
        assert out.k_continuous.shape == (batch, 1, h, w)
        assert out.residue_prob.shape == (batch, 1, h, w)
        assert out.phi_hat.shape == (batch, 1, h, w)

    @pytest.mark.parametrize("size", [64, 96, 200])
    def test_handles_various_input_sizes(self, tiny_model, size):
        """Sizes not divisible by the encoder's stride-32 must still work at
        the PyTorch level (the ONNX-export constraint from Prompt 5 is a
        separate, backend-specific limitation, tested in test_pipeline.py)."""
        x = torch.rand(1, 3, size, size)
        out = tiny_model(x)
        assert out.k_hat.shape[-2:] == (size, size)

    def test_k_hat_is_integer_valued(self, tiny_model, dummy_input):
        out = tiny_model(dummy_input)
        torch.testing.assert_close(out.k_hat, torch.round(out.k_hat))

    def test_k_continuous_bounded_by_k_max(self, dummy_input):
        model = AmbiguityNet(pretrained=False, k_max=5.0)
        model.eval()
        out = model(dummy_input)
        assert out.k_continuous.abs().max().item() <= 5.0 + 1e-4

    def test_residue_prob_in_unit_interval(self, tiny_model, dummy_input):
        out = tiny_model(dummy_input)
        assert out.residue_prob.min().item() >= 0.0
        assert out.residue_prob.max().item() <= 1.0

    def test_phi_hat_reconstruction_formula(self, tiny_model, dummy_input):
        """phi_hat must equal wrapped_phase_rad + 2*pi*k_hat exactly."""
        out = tiny_model(dummy_input)
        wrapped_rad = dummy_input[:, 0:1] * math.pi
        expected = wrapped_rad + 2 * math.pi * out.k_hat
        torch.testing.assert_close(out.phi_hat, expected, atol=1e-4, rtol=1e-4)


class TestRoundSTE:
    def test_forward_rounds(self):
        x = torch.tensor([1.2, 1.5, -1.5, -1.9, 0.49])
        rounded = round_ste(x)
        torch.testing.assert_close(rounded, torch.round(x))

    def test_backward_is_identity(self):
        """The straight-through estimator's gradient must pass through
        unchanged, as if round() were the identity function."""
        x = torch.tensor([2.3], requires_grad=True)
        y = round_ste(x) * 3.0
        y.backward()
        assert x.grad.item() == pytest.approx(3.0)


# --------------------------------------------------------------------------- #
# Loss function
# --------------------------------------------------------------------------- #


@pytest.fixture
def loss_inputs(dummy_input):
    """A (k_true, wrapped_phase_norm, coherence) tuple matching `dummy_input`'s
    batch/spatial shape, for loss-function tests."""
    batch, _, h, w = dummy_input.shape
    k_true = torch.randint(-5, 6, (batch, 1, h, w)).float()
    wrapped_phase_norm = dummy_input[:, 0:1]
    coherence = dummy_input[:, 1:2]
    return k_true, wrapped_phase_norm, coherence


class TestPhysicsInformedUnwrapLoss:
    def test_ambiguity_loss_zero_for_perfect_prediction(self, loss_inputs):
        """Component 1 (ambiguity MSE) must be exactly 0 when k_continuous == k_true."""
        k_true, wrapped_phase_norm, coherence = loss_inputs

        class FakeOutput:
            pass

        fake = FakeOutput()
        fake.k_hat = k_true.clone()
        fake.k_continuous = k_true.clone()
        fake.residue_prob = torch.full_like(k_true, 0.5)
        fake.phi_hat = wrapped_phase_norm * math.pi + 2 * math.pi * k_true

        criterion = PhysicsInformedUnwrapLoss()
        result = criterion(
            fake, k_true=k_true, wrapped_phase_norm=wrapped_phase_norm, coherence=coherence
        )
        assert result.ambiguity.item() == pytest.approx(0.0, abs=1e-8)

    def test_rewrap_consistency_near_zero_by_construction(
        self, tiny_model, dummy_input, loss_inputs
    ):
        """Component 2 must be ~0 regardless of k correctness, since phi_hat is
        built by literally adding an integer multiple of 2*pi (see the
        `losses.py` module docstring's architectural-invariance argument).
        Uses realistic (non-clamped-random) generator data, since heavily
        clamped uniform random data can pin many pixels to exactly +-pi and
        hit the (harmless) floating-point wraparound edge case."""
        from pyunwrap.data.preprocessing import (
            normalize_amplitude,
            normalize_coherence,
            normalize_phase,
        )
        from pyunwrap.synthetic.generator import InSARSyntheticGenerator

        gen = InSARSyntheticGenerator(size=64, seed=9)
        sample = gen.generate_sample(deformation_type="mogi")
        x = (
            torch.stack(
                [
                    torch.from_numpy(normalize_phase(sample.wrapped_phase)),
                    torch.from_numpy(normalize_coherence(sample.coherence)),
                    torch.from_numpy(normalize_amplitude(sample.amplitude)),
                ]
            )
            .unsqueeze(0)
            .float()
        )
        k_true = torch.from_numpy(sample.ambiguity).float().unsqueeze(0).unsqueeze(0)

        out = tiny_model(x)
        criterion = PhysicsInformedUnwrapLoss()
        result = criterion(out, k_true=k_true, wrapped_phase_norm=x[:, 0:1], coherence=x[:, 1:2])
        assert result.rewrap_consistency.item() < 1e-6

    def test_gradients_flow_through_ste(self, tiny_model, dummy_input, loss_inputs):
        k_true, wrapped_phase_norm, coherence = loss_inputs
        out = tiny_model(dummy_input)
        criterion = PhysicsInformedUnwrapLoss()
        result = criterion(
            out, k_true=k_true, wrapped_phase_norm=wrapped_phase_norm, coherence=coherence
        )
        result.total.backward()
        grad_norm = sum(p.grad.norm().item() for p in tiny_model.parameters() if p.grad is not None)
        assert grad_norm > 0.0

    def test_residue_penalty_flags_isolated_spike(self):
        """Component 4's Laplacian penalty must give a higher value for a
        field with one isolated pixel spike than for a smooth field."""
        smooth = torch.zeros(1, 1, 10, 10)
        spiked = smooth.clone()
        spiked[0, 0, 5, 5] = 3.0
        assert discrete_laplacian(spiked).pow(2).mean() > discrete_laplacian(smooth).pow(2).mean()

    def test_total_loss_is_weighted_sum(self, tiny_model, dummy_input, loss_inputs):
        k_true, wrapped_phase_norm, coherence = loss_inputs
        out = tiny_model(dummy_input)
        criterion = PhysicsInformedUnwrapLoss(
            weight_ambiguity=0.5,
            weight_rewrap=0.3,
            weight_smoothness=0.1,
            weight_residue=0.1,
        )
        result = criterion(
            out, k_true=k_true, wrapped_phase_norm=wrapped_phase_norm, coherence=coherence
        )
        expected_total = (
            0.5 * result.ambiguity
            + 0.3 * result.rewrap_consistency
            + 0.1 * result.smoothness
            + 0.1 * result.residue
        )
        torch.testing.assert_close(result.total, expected_total, atol=1e-6, rtol=1e-6)


class TestWrapPhaseTorch:
    def test_matches_numpy_wrap_phase(self):
        """The differentiable torch wrap must agree with the numpy reference
        implementation used in the synthetic generator."""
        from pyunwrap.synthetic.generator import wrap_phase as wrap_phase_numpy

        x_np = np.linspace(-10, 10, 500)
        x_torch = torch.from_numpy(x_np)
        np.testing.assert_allclose(
            wrap_phase_torch(x_torch).numpy(),
            wrap_phase_numpy(x_np),
            atol=1e-6,
        )

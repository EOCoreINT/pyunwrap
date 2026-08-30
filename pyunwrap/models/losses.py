"""
pyunwrap.models.losses
========================

`PhysicsInformedUnwrapLoss`: the composite loss function used to train
`AmbiguityNet`.

The loss combines four components:

1. **Ambiguity loss** -- supervised MSE between the continuous ambiguity
   prediction and the ground-truth integer ambiguity map. This is the main
   corrective training signal.
2. **Re-wrapping consistency** -- checks that `wrap(wrapped_phase +
   2*pi*round(k))` reproduces the observed wrapped phase.
3. **Coherence-weighted smoothness** -- penalizes the spatial gradient of the
   reconstructed unwrapped phase, weighted per-pixel by coherence (strict in
   high-coherence areas, relaxed in low-coherence areas where genuine sharp
   gradients / noise are expected).
4. **Residue penalty** -- discourages isolated, unsupported jumps in the
   predicted ambiguity map (the pattern that produces spurious "new"
   singularities not present in the input data).

Important note on Components 2 and 4 (read before tuning weights)
-------------------------------------------------------------------
Because `AmbiguityNet.forward` builds `phi_hat = wrapped_phase + 2*pi *
round_ste(k)` directly (see `ambiguity_net.py`), two things are true by
construction, independent of whether `k` is *correct*:

- `wrap(phi_hat)` is **exactly** `wrapped_phase` to floating-point precision,
  for *any* integer-valued `k_hat` -- adding an integer multiple of `2*pi`
  before wrapping can never change the wrapped result. So a *literal* re-wrap
  consistency loss (Component 2) is architecturally guaranteed to be ~0
  regardless of prediction quality, and mainly serves as (a) a numerical
  sanity check / regression test, and (b) a channel for the straight-through
  estimator's gradient to reach `k_continuous` during backprop. It is *not* a
  substitute for the supervised ambiguity loss (Component 1).
- Likewise, because `phi_hat` is built by literally *adding* an integer
  field to a real-valued phase (not by any wrap-then-patch operation), it is
  a genuine single-valued function on the pixel grid: the four edge
  differences around any 2x2 pixel loop telescope to exactly zero by basic
  algebra. In other words, `phi_hat` can *never* contain a classical
  (Goldstein-sense) topological residue that isn't already present in the
  input wrapped phase -- that invariance is a property of any unwrapping
  scheme that only adds integer cycles, not a training outcome.

  What *can* go wrong -- and what actually causes the boundary/checkerburard
  artifacts this loss is meant to prevent -- is an **isolated, spatially
  unsupported flip** in the predicted ambiguity map `k_hat` (e.g. one pixel
  jumps by +-1 relative to every neighbor with no coherence/gradient
  evidence for it). This produces a large, physically implausible jump in
  `phi_hat` at that pixel even though no formal topological residue exists.
  Component 4 is therefore implemented as a penalty on the discrete
  Laplacian of `k_hat` (its local second-order variation), which directly
  targets exactly this failure mode while still permitting the smooth,
  multi-pixel `k` transitions that genuine large deformations require.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from pyunwrap.models.ambiguity_net import AmbiguityNetOutput


def wrap_phase_torch(phase: torch.Tensor) -> torch.Tensor:
    """Differentiable phase wrapping into (-pi, pi], mirroring
    `pyunwrap.synthetic.generator.wrap_phase` for use inside the training
    graph.

    Uses `atan2(sin(x), cos(x))`, which is differentiable everywhere except
    at the exact +-pi discontinuity (a measure-zero set that PyTorch's
    autograd handles gracefully, same as `torch.round`'s flat-zero-gradient
    plateaus elsewhere in this module).

    Args:
        phase: Real-valued phase tensor, radians, any shape.

    Returns:
        Wrapped phase, same shape, values in (-pi, pi].
    """
    return torch.atan2(torch.sin(phase), torch.cos(phase))


def spatial_gradients(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute forward-difference spatial gradients of a [B, 1, H, W] tensor.

    Args:
        x: Input tensor, shape [B, 1, H, W].

    Returns:
        (grad_x, grad_y): gradients along width and height, shapes
        [B, 1, H, W-1] and [B, 1, H-1, W] respectively.
    """
    grad_x = x[:, :, :, 1:] - x[:, :, :, :-1]
    grad_y = x[:, :, 1:, :] - x[:, :, :-1, :]
    return grad_x, grad_y


def discrete_laplacian(x: torch.Tensor) -> torch.Tensor:
    """4-neighbor discrete Laplacian of a [B, 1, H, W] tensor (interior pixels only).

    `laplacian[i, j] = x[i+1,j] + x[i-1,j] + x[i,j+1] + x[i,j-1] - 4*x[i,j]`.
    Large magnitudes flag isolated pixels that disagree sharply with all
    four of their neighbors -- exactly the "single stuck pixel" pattern that
    creates spurious ambiguity-map residues.

    Args:
        x: Input tensor, shape [B, 1, H, W].

    Returns:
        Laplacian, shape [B, 1, H-2, W-2] (interior pixels only, via
        `conv2d` with a fixed 3x3 kernel and no padding).
    """
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=x.dtype,
        device=x.device,
    ).view(1, 1, 3, 3)
    return F.conv2d(x, kernel, padding=0)


@dataclass
class PhysicsLossOutput:
    """Structured breakdown of the composite physics-informed loss.

    Attributes:
        total: Weighted sum of all four components (the value to call
            `.backward()` on).
        ambiguity: Component 1 -- supervised ambiguity MSE.
        rewrap_consistency: Component 2 -- re-wrapping consistency MSE.
        smoothness: Component 3 -- coherence-weighted smoothness penalty.
        residue: Component 4 -- ambiguity-map Laplacian residue penalty.
    """

    total: torch.Tensor
    ambiguity: torch.Tensor
    rewrap_consistency: torch.Tensor
    smoothness: torch.Tensor
    residue: torch.Tensor

    def as_dict(self) -> dict[str, float]:
        """Detach and convert every component to a plain Python float, for logging."""
        return {
            "loss/total": float(self.total.detach().cpu()),
            "loss/ambiguity": float(self.ambiguity.detach().cpu()),
            "loss/rewrap_consistency": float(self.rewrap_consistency.detach().cpu()),
            "loss/smoothness": float(self.smoothness.detach().cpu()),
            "loss/residue": float(self.residue.detach().cpu()),
        }


class PhysicsInformedUnwrapLoss(nn.Module):
    """Composite physics-informed loss for training `AmbiguityNet`.

    Example:
        >>> criterion = PhysicsInformedUnwrapLoss()
        >>> out = model(x)  # AmbiguityNetOutput
        >>> loss = criterion(out, k_true=batch["true_ambiguity"],
        ...                   wrapped_phase_norm=batch["wrapped_phase"],
        ...                   coherence=batch["coherence"])
        >>> loss.total.backward()
    """

    def __init__(
        self,
        weight_ambiguity: float = 0.5,
        weight_rewrap: float = 0.3,
        weight_smoothness: float = 0.1,
        weight_residue: float = 0.1,
    ) -> None:
        """
        Args:
            weight_ambiguity: Weight for Component 1 (ambiguity MSE).
            weight_rewrap: Weight for Component 2 (re-wrap consistency).
            weight_smoothness: Weight for Component 3 (coherence-weighted
                smoothness).
            weight_residue: Weight for Component 4 (ambiguity-map residue /
                Laplacian penalty).
        """
        super().__init__()
        self.weight_ambiguity = weight_ambiguity
        self.weight_rewrap = weight_rewrap
        self.weight_smoothness = weight_smoothness
        self.weight_residue = weight_residue

    def forward(
        self,
        pred: AmbiguityNetOutput,
        k_true: torch.Tensor,
        wrapped_phase_norm: torch.Tensor,
        coherence: torch.Tensor,
    ) -> PhysicsLossOutput:
        """Compute all four loss components and their weighted total.

        Args:
            pred: Output of `AmbiguityNet.forward` (device matches inputs).
            k_true: Ground-truth integer ambiguity map, [B, 1, H, W]
                (e.g. `batch["true_ambiguity"]` from `InSARTileDataset`).
            wrapped_phase_norm: The *normalized* ([-1, 1]) wrapped phase
                input channel, [B, 1, H, W] (e.g. `batch["wrapped_phase"]`).
            coherence: Coherence map, [B, 1, H, W], values in [0, 1]
                (e.g. `batch["coherence"]`).

        Returns:
            `PhysicsLossOutput` with the total (weighted) loss and each
            individual component, all differentiable tensors except where
            noted.
        """
        device = pred.k_hat.device
        k_true = k_true.to(device)
        wrapped_phase_norm = wrapped_phase_norm.to(device)
        coherence = coherence.to(device)

        # --- Component 1: Ambiguity loss ---
        # Supervised MSE against the continuous (pre-rounding) prediction:
        # this is the primary training signal and is fully differentiable
        # without needing the straight-through estimator.
        loss_ambiguity = F.mse_loss(pred.k_continuous, k_true)

        # --- Component 2: Re-wrapping consistency ---
        wrapped_phase_rad = wrapped_phase_norm * math.pi
        psi_pred = wrap_phase_torch(pred.phi_hat)
        loss_rewrap = F.mse_loss(psi_pred, wrapped_phase_rad)

        # --- Component 3: Coherence-weighted smoothness ---
        grad_x, grad_y = spatial_gradients(pred.phi_hat)
        gamma_x = coherence[:, :, :, 1:]  # align coherence weight to grad_x's shifted grid
        gamma_y = coherence[:, :, 1:, :]
        loss_smoothness = (
            (gamma_x * grad_x.pow(2)).mean() + (gamma_y * grad_y.pow(2)).mean()
        ) / 2.0

        # --- Component 4: Residue penalty (ambiguity-map Laplacian) ---
        # See module docstring for why this operates on k_hat's Laplacian
        # rather than a literal topological residue count on phi_hat.
        k_laplacian = discrete_laplacian(pred.k_hat)
        loss_residue = k_laplacian.pow(2).mean()

        total = (
            self.weight_ambiguity * loss_ambiguity
            + self.weight_rewrap * loss_rewrap
            + self.weight_smoothness * loss_smoothness
            + self.weight_residue * loss_residue
        )

        return PhysicsLossOutput(
            total=total,
            ambiguity=loss_ambiguity,
            rewrap_consistency=loss_rewrap,
            smoothness=loss_smoothness,
            residue=loss_residue,
        )

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
from typing import Literal

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
class SmoothnessConfig:
    """Configuration for `CoherenceWeightedPhaseSmoothnessLoss`.

    Attributes:
        use_smoothness_loss: Master on/off switch. When `False`,
            `CoherenceWeightedPhaseSmoothnessLoss.forward` returns an exact
            `0.0` tensor (still differentiable-shaped, so it composes safely
            into a larger weighted sum without special-casing by the caller).
        smoothness_weight: Weight this component contributes when composed
            into a larger loss (e.g. `PhysicsInformedUnwrapLoss`'s
            `weight_edge_smoothness`). Stored here too so the config is a
            complete, self-contained record of "how this term behaves."
        smoothness_norm: `"huber"` (robust to genuine large jumps at real
            discontinuities) or `"l1"`. Deliberately does *not* offer plain
            L2/MSE here -- squared error on spatial gradients punishes a
            single genuine large deformation edge as heavily as many small
            noisy ones, which is exactly the over-smoothing failure mode
            this loss exists to avoid (the *existing*, simpler smoothness
            term in `PhysicsInformedUnwrapLoss` already covers the L2 case).
        edge_tau: Temperature for the edge-preservation factor
            `exp(-|grad| / edge_tau)`. Smaller values suppress the
            smoothness penalty more aggressively around large gradients
            (preserving sharper edges); larger values smooth more broadly.
        apply_in_high_coherence_only: If `True`, zero out the penalty
            wherever `coherence <= coherence_threshold` (low-coherence
            regions are exactly where genuine unmodeled sharp
            structure/noise is expected -- see this project's own
            `docs/experiments.md` and README for why "low coherence" is a
            named target failure mode, not a region to smooth away).
        coherence_threshold: Threshold used by
            `apply_in_high_coherence_only`.
        penalize_ambiguity_gradients: If `True`, additionally penalize
            spatial gradients of the integer ambiguity map `k_hat` itself
            (not just the reconstructed phase `phi_hat`), using the same
            robust norm and edge weighting. This is a complementary,
            gradient-based view of the same "don't let ambiguity flip
            without local support" goal the existing residue/Laplacian
            term (Component 4) already targets via curvature rather than
            gradient -- the two are not redundant (a smooth *ramp* in k has
            zero Laplacian but nonzero gradient) and are independently
            toggleable.
        huber_delta: Transition point between the quadratic and linear
            regimes of the Huber norm, only used when `smoothness_norm ==
            "huber"`.
    """

    use_smoothness_loss: bool = True
    smoothness_weight: float = 0.1
    smoothness_norm: Literal["huber", "l1"] = "huber"
    edge_tau: float = 1.0
    apply_in_high_coherence_only: bool = True
    coherence_threshold: float = 0.5
    penalize_ambiguity_gradients: bool = True
    huber_delta: float = 1.0

    def __post_init__(self) -> None:
        if self.smoothness_norm not in ("huber", "l1"):
            raise ValueError(
                f"smoothness_norm must be 'huber' or 'l1', got {self.smoothness_norm!r}"
            )
        if not (0.0 <= self.coherence_threshold <= 1.0):
            raise ValueError(
                f"coherence_threshold must be in [0, 1], got {self.coherence_threshold}"
            )
        if self.edge_tau <= 0.0:
            raise ValueError(f"edge_tau must be > 0, got {self.edge_tau}")


def _robust_norm(x: torch.Tensor, kind: Literal["huber", "l1"], huber_delta: float) -> torch.Tensor:
    """Apply an elementwise robust penalty to `x` (typically a spatial
    gradient), without reducing over any dimension.

    Args:
        x: Input tensor, any shape.
        kind: `"huber"` or `"l1"`.
        huber_delta: Huber transition point (ignored for `"l1"`).

    Returns:
        Elementwise-penalized tensor, same shape as `x`.
    """
    if kind == "l1":
        return x.abs()
    return F.huber_loss(x, torch.zeros_like(x), delta=huber_delta, reduction="none")


class CoherenceWeightedPhaseSmoothnessLoss(nn.Module):
    """Edge-preserving, coherence-weighted smoothness penalty on the
    reconstructed unwrapped phase (and optionally the ambiguity map).

    This is a separate, more sophisticated sibling to
    `PhysicsInformedUnwrapLoss`'s existing (simpler, unconditional L2)
    smoothness component, not a replacement for it -- see this module's
    top docstring and `SmoothnessConfig` for how the two differ and why
    both exist. Composed into `PhysicsInformedUnwrapLoss` as an optional,
    separately-weighted term when a `SmoothnessConfig` is supplied;
    unaffected (and not computed) when it isn't, so existing training runs
    that never pass one see byte-identical loss values to before this class
    was added.

    The penalty for a pixel is, schematically::

        weight = coherence * exp(-|grad(phi_hat)| / edge_tau)
        penalty = weight * robust_norm(grad(phi_hat))

    High coherence and a small local gradient both increase the penalty
    (genuinely flat, reliable regions should be smooth); a large local
    gradient suppresses it via the exponential edge term (don't erase real
    discontinuities); low coherence suppresses it via the coherence factor
    and, optionally, a hard threshold mask.

    Example:
        >>> config = SmoothnessConfig(smoothness_norm="huber", edge_tau=0.5)
        >>> smoothness_loss = CoherenceWeightedPhaseSmoothnessLoss(config)
        >>> penalty = smoothness_loss(phi_hat, psi, k_hat, gamma)
        >>> penalty.backward()
    """

    def __init__(self, config: SmoothnessConfig | None = None) -> None:
        """
        Args:
            config: Behavior configuration. Defaults to `SmoothnessConfig()`
                (enabled, Huber norm, high-coherence-only) if not given.
        """
        super().__init__()
        self.config = config if config is not None else SmoothnessConfig()

    def forward(
        self,
        phi_hat: torch.Tensor,
        psi: torch.Tensor,
        k_hat: torch.Tensor,
        gamma: torch.Tensor,
        edge_mask: torch.Tensor | None = None,
        regime_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the edge-preserving, coherence-weighted smoothness penalty.

        Args:
            phi_hat: Reconstructed unwrapped phase, [B, 1, H, W], radians.
            psi: Observed wrapped phase, [B, 1, H, W], radians. Accepted
                for interface symmetry with the rest of the physics loss
                (and potential future use, e.g. penalizing disagreement
                with wrapped-phase-derived edges directly) but the edge
                indicator here deliberately comes from `phi_hat`'s own
                gradient rather than `psi`'s: `psi` is wrapped, so its raw
                spatial gradient has artificial +-2*pi jumps at every wrap
                boundary that are not real edges and would incorrectly
                suppress smoothing there.
            k_hat: Predicted integer ambiguity map, [B, 1, H, W]. Only used
                if `config.penalize_ambiguity_gradients` is `True`.
            gamma: Coherence map, [B, 1, H, W], values in [0, 1].
            edge_mask: Optional externally-supplied edge/confidence mask,
                [B, 1, H, W] or broadcastable, multiplied into the final
                per-pixel weight (e.g. a mask from an edge detector or a
                known-deformation-boundary prior). `1.0` where the input is
                silent (no effect).
            regime_mask: Optional externally-supplied regime mask (e.g.
                from `pyunwrap.data.real_injection`, marking
                injected-deformation regions), same broadcasting rules as
                `edge_mask`. `1.0` where silent.

        Returns:
            A scalar tensor: the mean penalty across all valid gradient
            pixels, or an exact `0.0` (still a tensor, still safe to add
            into a larger weighted sum) if the loss is disabled, or if the
            input is too small to have any interior gradient pixels.
        """
        if not self.config.use_smoothness_loss:
            return torch.zeros((), device=phi_hat.device, dtype=phi_hat.dtype)

        total = torch.zeros((), device=phi_hat.device, dtype=phi_hat.dtype)
        total = total + self._directional_penalty(phi_hat, gamma, edge_mask, regime_mask)

        if self.config.penalize_ambiguity_gradients:
            # k_hat is produced via a straight-through estimator (see
            # ambiguity_net.round_ste), so it remains differentiable here
            # exactly as it does for the existing residue/Laplacian term.
            total = total + self._directional_penalty(k_hat, gamma, edge_mask, regime_mask)

        return total

    def _directional_penalty(
        self,
        field: torch.Tensor,
        gamma: torch.Tensor,
        edge_mask: torch.Tensor | None,
        regime_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Shared x/y gradient-penalty computation, used for both `phi_hat`
        and (optionally) `k_hat`."""
        grad_x, grad_y = spatial_gradients(field)
        if grad_x.numel() == 0 or grad_y.numel() == 0:
            # Degenerate input (e.g. a 1-pixel-wide tile): no interior
            # gradient pixels exist. Return exact zero rather than NaN from
            # an empty-tensor .mean().
            return torch.zeros((), device=field.device, dtype=field.dtype)

        gamma_x = gamma[:, :, :, 1:]
        gamma_y = gamma[:, :, 1:, :]

        edge_factor_x = torch.exp(-grad_x.abs() / self.config.edge_tau)
        edge_factor_y = torch.exp(-grad_y.abs() / self.config.edge_tau)

        weight_x = gamma_x * edge_factor_x
        weight_y = gamma_y * edge_factor_y

        if self.config.apply_in_high_coherence_only:
            high_coh_x = (gamma_x > self.config.coherence_threshold).to(field.dtype)
            high_coh_y = (gamma_y > self.config.coherence_threshold).to(field.dtype)
            weight_x = weight_x * high_coh_x
            weight_y = weight_y * high_coh_y

        if edge_mask is not None:
            weight_x = weight_x * edge_mask[:, :, :, 1:]
            weight_y = weight_y * edge_mask[:, :, 1:, :]
        if regime_mask is not None:
            weight_x = weight_x * regime_mask[:, :, :, 1:]
            weight_y = weight_y * regime_mask[:, :, 1:, :]

        penalty_x = weight_x * _robust_norm(
            grad_x, self.config.smoothness_norm, self.config.huber_delta
        )
        penalty_y = weight_y * _robust_norm(
            grad_y, self.config.smoothness_norm, self.config.huber_delta
        )

        return (penalty_x.mean() + penalty_y.mean()) / 2.0


@dataclass
class PhysicsLossOutput:
    """Structured breakdown of the composite physics-informed loss.

    Attributes:
        total: Weighted sum of all active components (the value to call
            `.backward()` on).
        ambiguity: Component 1 -- supervised ambiguity MSE.
        rewrap_consistency: Component 2 -- re-wrapping consistency MSE.
        smoothness: Component 3 -- coherence-weighted smoothness penalty
            (simple, unconditional L2 gradient penalty).
        residue: Component 4 -- ambiguity-map Laplacian residue penalty.
        edge_smoothness: Component 5 -- edge-preserving, coherence-weighted
            smoothness via `CoherenceWeightedPhaseSmoothnessLoss`. Exact
            `0.0` whenever `PhysicsInformedUnwrapLoss` was constructed
            without a `smoothness_config` (the default), so existing
            training runs see byte-identical `total` values to before this
            component existed.
    """

    total: torch.Tensor
    ambiguity: torch.Tensor
    rewrap_consistency: torch.Tensor
    smoothness: torch.Tensor
    residue: torch.Tensor
    edge_smoothness: torch.Tensor

    def as_dict(self) -> dict[str, float]:
        """Detach and convert every component to a plain Python float, for logging."""
        return {
            "loss/total": float(self.total.detach().cpu()),
            "loss/ambiguity": float(self.ambiguity.detach().cpu()),
            "loss/rewrap_consistency": float(self.rewrap_consistency.detach().cpu()),
            "loss/smoothness": float(self.smoothness.detach().cpu()),
            "loss/residue": float(self.residue.detach().cpu()),
            "loss/edge_smoothness": float(self.edge_smoothness.detach().cpu()),
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
        smoothness_config: SmoothnessConfig | None = None,
        weight_edge_smoothness: float | None = None,
    ) -> None:
        """
        Args:
            weight_ambiguity: Weight for Component 1 (ambiguity MSE).
            weight_rewrap: Weight for Component 2 (re-wrap consistency).
            weight_smoothness: Weight for Component 3 (coherence-weighted
                smoothness, simple/unconditional L2 gradient penalty).
            weight_residue: Weight for Component 4 (ambiguity-map residue /
                Laplacian penalty).
            smoothness_config: Optional `SmoothnessConfig` enabling
                Component 5, the edge-preserving
                `CoherenceWeightedPhaseSmoothnessLoss`. `None` (the
                default) disables Component 5 entirely -- it is not
                constructed, not evaluated, and contributes exactly `0.0`
                to `total`, so existing code calling
                `PhysicsInformedUnwrapLoss()` with no arguments gets
                byte-identical behavior to before this parameter existed.
                Pass `SmoothnessConfig()` (or a customized one) to opt in.
            weight_edge_smoothness: Weight for Component 5. Defaults to
                `smoothness_config.smoothness_weight` when
                `smoothness_config` is given and this is left as `None`;
                has no effect when `smoothness_config` is `None`.
        """
        super().__init__()
        self.weight_ambiguity = weight_ambiguity
        self.weight_rewrap = weight_rewrap
        self.weight_smoothness = weight_smoothness
        self.weight_residue = weight_residue

        self.smoothness_config = smoothness_config
        self.edge_smoothness_loss: CoherenceWeightedPhaseSmoothnessLoss | None
        if smoothness_config is not None:
            self.edge_smoothness_loss = CoherenceWeightedPhaseSmoothnessLoss(smoothness_config)
            self.weight_edge_smoothness = (
                weight_edge_smoothness
                if weight_edge_smoothness is not None
                else smoothness_config.smoothness_weight
            )
        else:
            self.edge_smoothness_loss = None
            self.weight_edge_smoothness = 0.0

    def forward(
        self,
        pred: AmbiguityNetOutput,
        k_true: torch.Tensor,
        wrapped_phase_norm: torch.Tensor,
        coherence: torch.Tensor,
        edge_mask: torch.Tensor | None = None,
        regime_mask: torch.Tensor | None = None,
    ) -> PhysicsLossOutput:
        """Compute all loss components and their weighted total.

        Args:
            pred: Output of `AmbiguityNet.forward` (device matches inputs).
            k_true: Ground-truth integer ambiguity map, [B, 1, H, W]
                (e.g. `batch["true_ambiguity"]` from `InSARTileDataset`).
            wrapped_phase_norm: The *normalized* ([-1, 1]) wrapped phase
                input channel, [B, 1, H, W] (e.g. `batch["wrapped_phase"]`).
            coherence: Coherence map, [B, 1, H, W], values in [0, 1]
                (e.g. `batch["coherence"]`).
            edge_mask: Optional edge/confidence mask forwarded to Component
                5 (`CoherenceWeightedPhaseSmoothnessLoss`), if enabled.
                Ignored when `smoothness_config` was not supplied.
            regime_mask: Optional regime mask (e.g. from
                `pyunwrap.data.real_injection`) forwarded to Component 5.
                Ignored when `smoothness_config` was not supplied.

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

        # --- Component 5: Edge-preserving coherence-weighted smoothness (optional) ---
        if self.edge_smoothness_loss is not None:
            loss_edge_smoothness = self.edge_smoothness_loss(
                pred.phi_hat,
                wrapped_phase_rad,
                pred.k_hat,
                coherence,
                edge_mask=edge_mask,
                regime_mask=regime_mask,
            )
        else:
            loss_edge_smoothness = torch.zeros((), device=device, dtype=pred.phi_hat.dtype)

        total = (
            self.weight_ambiguity * loss_ambiguity
            + self.weight_rewrap * loss_rewrap
            + self.weight_smoothness * loss_smoothness
            + self.weight_residue * loss_residue
            + self.weight_edge_smoothness * loss_edge_smoothness
        )

        return PhysicsLossOutput(
            total=total,
            ambiguity=loss_ambiguity,
            rewrap_consistency=loss_rewrap,
            smoothness=loss_smoothness,
            residue=loss_residue,
            edge_smoothness=loss_edge_smoothness,
        )

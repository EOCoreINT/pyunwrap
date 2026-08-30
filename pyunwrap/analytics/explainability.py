"""
pyunwrap.analytics.explainability
====================================

Explainability tools for `AmbiguityNet`:

1. **Grad-CAM**: spatial saliency over the decoder's final feature map,
   showing *where* in the tile the network focused when producing its
   ambiguity prediction.
2. **Integrated Gradients (per-channel)**: attributes the prediction back to
   each of the 3 input channels (wrapped phase, coherence, amplitude),
   answering *which physical quantity* the network relied on most --
   directly useful for sanity-checking that the model is using coherence to
   modulate its confidence near decorrelated areas, rather than ignoring it.
3. **Uncertainty calibration**: a reliability diagram binning Monte Carlo
   Dropout ambiguity variance (from `pyunwrap.inference.unwrapper`) against
   the actual observed phase error, plus a scalar calibration metric.

No third-party explainability library (e.g. Captum) is used -- both Grad-CAM
and Integrated Gradients are implemented directly against `AmbiguityNet`'s
forward pass with plain PyTorch autograd, keeping `pyunwrap`'s dependency
footprint unchanged from `pyproject.toml`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from pyunwrap.models.ambiguity_net import AmbiguityNet, AmbiguityNetOutput

#: Human-readable names for the 3 stacked input channels, in the order
#: `AmbiguityNet`/`InSARTileDataset` use throughout `pyunwrap`.
INPUT_CHANNEL_NAMES = ("wrapped_phase", "coherence", "amplitude")


# --------------------------------------------------------------------------- #
# Grad-CAM
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class GradCAMResult:
    """Output of `grad_cam`.

    Attributes:
        saliency_map: 2D array, same H/W as the input tile, values in
            [0, 1] (min-max normalized), where higher values indicate
            greater influence on the target scalar.
        raw_cam: The un-normalized, un-upsampled class activation map at the
            target layer's native (downsampled) resolution, for diagnostics.
    """

    saliency_map: np.ndarray
    raw_cam: np.ndarray


def grad_cam(
    model: AmbiguityNet,
    x: torch.Tensor,
    target_layer: torch.nn.Module | None = None,
    target_fn: Callable[[AmbiguityNetOutput], torch.Tensor] | None = None,
) -> GradCAMResult:
    """Compute a Grad-CAM spatial saliency map for one input tile.

    Args:
        model: A trained `AmbiguityNet` (will be switched to eval mode
            internally, gradients enabled for this call regardless of the
            caller's `torch.no_grad()` context).
        x: Input tensor, shape [1, 3, H, W] (single tile, already on the
            model's device).
        target_layer: The layer whose output feature map Grad-CAM is
            computed over. Defaults to `model.dec0` (the final decoder
            block, immediately before the two 1x1 output heads), which is
            the natural choice for explaining *where* the ambiguity
            prediction came from at full spatial resolution.
        target_fn: Function mapping the model's `AmbiguityNetOutput` to a
            scalar tensor to explain. Defaults to the sum of
            `k_continuous.abs()` over the whole tile (i.e. "what drove the
            magnitude of the predicted ambiguity everywhere"). Pass a
            function that indexes a specific pixel/region to explain a
            localized prediction instead (e.g. a difficult 2*pi jump).

    Returns:
        A `GradCAMResult` with the upsampled, normalized saliency map.
    """
    model.eval()
    if target_layer is None:
        target_layer = model.dec0
    if target_fn is None:
        target_fn = lambda out: out.k_continuous.abs().sum()

    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}

    def _forward_hook(_module, _input, output):
        activations["value"] = output

    def _backward_hook(_module, _grad_input, grad_output):
        gradients["value"] = grad_output[0]

    fwd_handle = target_layer.register_forward_hook(_forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(_backward_hook)

    try:
        x = x.clone().requires_grad_(True)
        out = model(x)
        target = target_fn(out)
        model.zero_grad(set_to_none=True)
        target.backward()

        act = activations["value"]  # [1, C, h, w]
        grad = gradients["value"]  # [1, C, h, w]

        # Standard Grad-CAM weighting: global-average-pool the gradients over
        # space to get one importance weight per channel, then take a
        # ReLU'd weighted sum of the activation maps.
        weights = grad.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        raw_cam = F.relu((weights * act).sum(dim=1, keepdim=True))  # [1, 1, h, w]
        raw_cam_np = raw_cam.detach().squeeze().cpu().numpy()

        upsampled = F.interpolate(raw_cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        upsampled_np = upsampled.detach().squeeze().cpu().numpy()
        cam_min, cam_max = upsampled_np.min(), upsampled_np.max()
        if cam_max - cam_min > 1e-12:
            saliency = (upsampled_np - cam_min) / (cam_max - cam_min)
        else:
            saliency = np.zeros_like(upsampled_np)

        return GradCAMResult(saliency_map=saliency, raw_cam=raw_cam_np)
    finally:
        fwd_handle.remove()
        bwd_handle.remove()


# --------------------------------------------------------------------------- #
# Integrated Gradients (per-channel attribution)
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class IntegratedGradientsResult:
    """Output of `integrated_gradients`.

    Attributes:
        attribution_map: [3, H, W] array of signed per-pixel, per-channel
            attributions (same units/sign convention as the standard
            Integrated Gradients formulation: sums approximately to
            `target(x) - target(baseline)` over all pixels and channels).
        channel_importance: dict mapping each of `INPUT_CHANNEL_NAMES` to its
            total absolute attribution share, normalized to sum to 1.0 --
            the direct answer to "which input channel did the model rely on".
    """

    attribution_map: np.ndarray
    channel_importance: dict[str, float]


def integrated_gradients(
    model: AmbiguityNet,
    x: torch.Tensor,
    target_fn: Callable[[AmbiguityNetOutput], torch.Tensor] | None = None,
    baseline: torch.Tensor | None = None,
    steps: int = 50,
) -> IntegratedGradientsResult:
    """Compute per-channel Integrated Gradients attribution for one input tile.

    Integrated Gradients (Sundararajan et al., 2017) attributes a model's
    output to its inputs by integrating the gradient of the output along a
    straight-line path from a `baseline` input to the actual input `x`. It
    satisfies a completeness axiom (attributions sum to the output
    difference), which is what makes the resulting per-channel shares
    meaningfully comparable to each other.

    Args:
        model: A trained `AmbiguityNet`.
        x: Input tensor, shape [1, 3, H, W].
        target_fn: Function mapping `AmbiguityNetOutput` to a scalar tensor
            to explain. Defaults to the sum of `k_continuous` over the tile.
        baseline: Reference input to integrate from, same shape as `x`.
            Defaults to an all-zeros tensor (phase=0, coherence=0,
            amplitude=0), a physically neutral "no information" reference.
        steps: Number of linear interpolation steps used to approximate the
            path integral (Riemann sum); higher is more accurate but
            proportionally more forward/backward passes.

    Returns:
        An `IntegratedGradientsResult` with the full attribution map and the
        summarized per-channel importance shares.
    """
    model.eval()
    if target_fn is None:
        target_fn = lambda out: out.k_continuous.sum()
    if baseline is None:
        baseline = torch.zeros_like(x)

    alphas = torch.linspace(0.0, 1.0, steps, device=x.device)
    grads_accum = torch.zeros_like(x)

    for alpha in alphas:
        interpolated = baseline + alpha * (x - baseline)
        interpolated = interpolated.clone().requires_grad_(True)
        out = model(interpolated)
        target = target_fn(out)
        model.zero_grad(set_to_none=True)
        grad = torch.autograd.grad(target, interpolated)[0]
        grads_accum += grad

    avg_grads = grads_accum / steps
    attribution = ((x - baseline) * avg_grads).detach()  # [1, 3, H, W]
    attribution_np = attribution.squeeze(0).cpu().numpy()  # [3, H, W]

    abs_totals = np.abs(attribution_np).sum(axis=(1, 2))
    total = abs_totals.sum()
    if total > 1e-12:
        shares = abs_totals / total
    else:
        shares = np.full(3, 1.0 / 3.0)

    channel_importance = {name: float(share) for name, share in zip(INPUT_CHANNEL_NAMES, shares)}

    return IntegratedGradientsResult(
        attribution_map=attribution_np, channel_importance=channel_importance
    )


# --------------------------------------------------------------------------- #
# Uncertainty calibration
# --------------------------------------------------------------------------- #


def reliability_diagram(
    predicted_uncertainty: np.ndarray,
    actual_error: np.ndarray,
    n_bins: int = 10,
) -> tuple[pd.DataFrame, dict]:
    """Bin Monte Carlo Dropout uncertainty against actual phase error to
    assess calibration.

    A well-calibrated uncertainty estimate should be *monotonically*
    associated with actual error: tiles/pixels the model reports higher
    uncertainty for should, on average, have higher actual error. This
    function bins pixels by predicted uncertainty into `n_bins` equal-width
    bins and reports the mean actual error within each bin.

    Args:
        predicted_uncertainty: Per-pixel uncertainty estimate (e.g.
            `UnwrapResult.uncertainty`, the MC Dropout ambiguity std, or
            `AmbiguityNetOutput.residue_prob`), any shape.
        actual_error: Per-pixel actual absolute phase error, same shape as
            `predicted_uncertainty`.
        n_bins: Number of equal-width uncertainty bins.

    Returns:
        `(bins_df, summary)`:
            - `bins_df`: `pandas.DataFrame` with columns `["bin_center",
              "mean_predicted_uncertainty", "mean_actual_error", "n_pixels"]`,
              one row per non-empty bin.
            - `summary`: dict with `"spearman_correlation"` (rank
              correlation between predicted uncertainty and actual error --
              the key calibration diagnostic, robust to any monotonic
              miscalibration of scale) and `"n_bins_used"`.
    """
    pred = predicted_uncertainty.ravel()
    err = actual_error.ravel()
    if pred.shape != err.shape:
        raise ValueError(
            f"Shape mismatch: uncertainty {predicted_uncertainty.shape} vs error {actual_error.shape}"
        )

    lo, hi = float(pred.min()), float(pred.max())
    if hi - lo < 1e-12:
        # Degenerate case: uncertainty is constant everywhere (e.g. an
        # untrained model, or a backend without MC Dropout support) -- no
        # meaningful binning is possible.
        return (
            pd.DataFrame(
                columns=[
                    "bin_center",
                    "mean_predicted_uncertainty",
                    "mean_actual_error",
                    "n_pixels",
                ]
            ),
            {"spearman_correlation": float("nan"), "n_bins_used": 0},
        )

    edges = np.linspace(lo, hi, n_bins + 1)
    bin_idx = np.clip(np.digitize(pred, edges) - 1, 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        rows.append(
            {
                "bin_center": float(0.5 * (edges[b] + edges[b + 1])),
                "mean_predicted_uncertainty": float(pred[mask].mean()),
                "mean_actual_error": float(err[mask].mean()),
                "n_pixels": int(mask.sum()),
            }
        )
    bins_df = pd.DataFrame(rows)

    spearman = _spearman_correlation(pred, err)

    return bins_df, {"spearman_correlation": spearman, "n_bins_used": len(bins_df)}


def _spearman_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Spearman rank correlation between two 1D arrays without a
    SciPy dependency (rank both arrays, then Pearson-correlate the ranks).

    Args:
        a: First array.
        b: Second array, same length.

    Returns:
        Spearman's rho, in [-1, 1], or `nan` if either array is constant.
    """
    rank_a = pd.Series(a).rank().to_numpy()
    rank_b = pd.Series(b).rank().to_numpy()
    if rank_a.std() < 1e-12 or rank_b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(rank_a, rank_b)[0, 1])

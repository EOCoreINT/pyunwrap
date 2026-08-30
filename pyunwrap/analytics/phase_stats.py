"""
pyunwrap.analytics.phase_stats
=================================

Scientific analytics for phase-unwrapping quality assessment:

1. **Residue detection**: locates classical Goldstein-sense phase
   singularities (residues) in a wrapped phase field, via the standard
   closed-loop wrapped-gradient-sum test, with `scikit-image` used to
   cluster and label connected residue regions (isolated residues rarely
   occur perfectly alone -- decorrelation tends to produce small clusters).
2. **Phase gradient / Nyquist analysis**: flags pixels where the spatial
   gradient of the unwrapped phase exceeds the Nyquist sampling limit (pi
   radians/pixel), the classical condition under which naive phase
   unwrapping becomes ill-posed.
3. **Error distribution**: compares a predicted unwrapped phase against
   ground truth and summarizes the error distribution.

All public functions return results as `pandas.DataFrame`/plain dicts (never
raw numpy scalars in the returned records), so results are directly usable
in `pyunwrap.analytics.report_generator` (Prompt 7) and JSON-serializable
for logging/API responses.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pandas as pd
from skimage.measure import label, regionprops

# --------------------------------------------------------------------------- #
# Residue detection
# --------------------------------------------------------------------------- #


def wrap_phase(phase: np.ndarray) -> np.ndarray:
    """Wrap a real-valued phase array into (-pi, pi] (local copy of the
    identity used throughout `pyunwrap`, kept dependency-free here so
    `phase_stats` doesn't need to import the synthetic generator)."""
    return np.angle(np.exp(1j * phase))


def compute_residue_map(phase: np.ndarray) -> np.ndarray:
    """Compute the classical Goldstein-sense residue charge at every interior
    2x2 pixel plaquette of a phase field.

    For each 2x2 block of pixels, sums the wrapped phase differences
    traversed around the closed loop (top->right->bottom->left->top). For a
    consistent, singularity-free field this sum is exactly 0; a genuine
    phase residue produces a sum of exactly +-2*pi (charge +-1). This test is
    invariant to whether `phase` is wrapped or unwrapped (see
    `pyunwrap.models.losses` module docstring for the telescoping-sum
    argument), so it may be applied directly to either.

    Args:
        phase: 2D phase array, radians (wrapped or unwrapped).

    Returns:
        2D integer-valued charge array, shape (H-1, W-1): `residue[i, j]` is
        the charge of the loop with top-left corner at `(i, j)`. Values are
        (up to floating point rounding) in `{-1, 0, 1}`.
    """
    top = phase[:-1, :-1]
    right = phase[:-1, 1:]
    bottom = phase[1:, 1:]
    left = phase[1:, :-1]

    loop_sum = (
        wrap_phase(right - top)
        + wrap_phase(bottom - right)
        + wrap_phase(left - bottom)
        + wrap_phase(top - left)
    )
    return np.round(loop_sum / (2.0 * np.pi)).astype(np.int32)


@dataclasses.dataclass
class ResidueCluster:
    """One connected cluster of same-sign residues, as identified by
    `skimage.measure.label`.

    Attributes:
        charge_sign: `+1` or `-1`.
        n_pixels: Number of residue-charge pixels in this cluster.
        centroid_row: Cluster centroid row (in the residue-map's coordinate
            frame, i.e. offset by 0.5 pixel from the original phase array).
        centroid_col: Cluster centroid column.
    """

    charge_sign: int
    n_pixels: int
    centroid_row: float
    centroid_col: float


def detect_residues(phase: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    """Detect and cluster phase residues in a phase field.

    Args:
        phase: 2D phase array, radians (wrapped or unwrapped -- see
            `compute_residue_map`).

    Returns:
        `(residue_map, clusters_df)`:
            - `residue_map`: the raw per-plaquette charge array from
              `compute_residue_map`, shape (H-1, W-1).
            - `clusters_df`: a `pandas.DataFrame` with one row per connected
              same-sign residue cluster, columns
              `["charge_sign", "n_pixels", "centroid_row", "centroid_col"]`.
              Empty (zero rows, correct columns) if no residues are present.
    """
    residue_map = compute_residue_map(phase)

    clusters: list[ResidueCluster] = []
    for sign in (1, -1):
        mask = residue_map == sign
        if not mask.any():
            continue
        labeled = label(
            mask, connectivity=2
        )  # 8-connectivity: diagonal residues often form dipole pairs
        for region in regionprops(labeled):
            centroid_row, centroid_col = region.centroid
            clusters.append(
                ResidueCluster(
                    charge_sign=sign,
                    n_pixels=int(region.area),
                    centroid_row=float(centroid_row),
                    centroid_col=float(centroid_col),
                )
            )

    if clusters:
        df = pd.DataFrame([dataclasses.asdict(c) for c in clusters])
    else:
        df = pd.DataFrame(columns=["charge_sign", "n_pixels", "centroid_row", "centroid_col"])

    return residue_map, df


def residue_summary(phase: np.ndarray) -> dict:
    """Compute a JSON-serializable summary of residue statistics for a phase field.

    Args:
        phase: 2D phase array, radians.

    Returns:
        Dict with keys `n_positive_clusters`, `n_negative_clusters`,
        `n_positive_pixels`, `n_negative_pixels`, `net_charge`,
        `residue_density_per_1000px` (residue *pixel* count per 1000 image
        pixels, a scale-independent density metric).
    """
    _residue_map, clusters_df = detect_residues(phase)
    n_pos_clusters = int((clusters_df["charge_sign"] == 1).sum()) if len(clusters_df) else 0
    n_neg_clusters = int((clusters_df["charge_sign"] == -1).sum()) if len(clusters_df) else 0
    n_pos_pixels = (
        int(clusters_df.loc[clusters_df["charge_sign"] == 1, "n_pixels"].sum())
        if len(clusters_df)
        else 0
    )
    n_neg_pixels = (
        int(clusters_df.loc[clusters_df["charge_sign"] == -1, "n_pixels"].sum())
        if len(clusters_df)
        else 0
    )

    total_px = phase.size
    return {
        "n_positive_clusters": n_pos_clusters,
        "n_negative_clusters": n_neg_clusters,
        "n_positive_pixels": n_pos_pixels,
        "n_negative_pixels": n_neg_pixels,
        "net_charge": n_pos_pixels - n_neg_pixels,
        "residue_density_per_1000px": 1000.0 * (n_pos_pixels + n_neg_pixels) / max(total_px, 1),
    }


def compare_residues(
    wrapped_phase: np.ndarray,
    predicted_unwrapped: np.ndarray,
) -> dict:
    """Compare residue content between the input wrapped phase and a predicted
    unwrapped phase.

    Per the topological invariance argument in
    `pyunwrap.models.losses` (adding any integer field to a phase cannot
    change its wrapped-gradient-loop residues), a *correct* unwrapping should
    reproduce exactly the same residue locations/count as the input wrapped
    phase. A materially higher residue count in the prediction indicates the
    network introduced spurious, spatially unsupported ambiguity-map jumps
    (see `pyunwrap.models.losses` Component 4's discussion) rather than
    genuine data singularities.

    Args:
        wrapped_phase: Observed wrapped phase, radians, in (-pi, pi].
        predicted_unwrapped: Model's predicted unwrapped phase, radians.

    Returns:
        Dict with `"input"` and `"predicted"` sub-dicts (each a
        `residue_summary`), plus `"spurious_residue_pixels"`: the excess
        residue-pixel count in the prediction beyond what the input data
        supports (0 if the prediction has fewer or equal residues).
    """
    input_summary = residue_summary(wrapped_phase)
    pred_summary = residue_summary(predicted_unwrapped)
    input_total = input_summary["n_positive_pixels"] + input_summary["n_negative_pixels"]
    pred_total = pred_summary["n_positive_pixels"] + pred_summary["n_negative_pixels"]
    return {
        "input": input_summary,
        "predicted": pred_summary,
        "spurious_residue_pixels": max(pred_total - input_total, 0),
    }


# --------------------------------------------------------------------------- #
# Phase gradient / Nyquist analysis
# --------------------------------------------------------------------------- #


def phase_gradient_magnitude(phase: np.ndarray) -> np.ndarray:
    """Compute the per-pixel spatial gradient magnitude of a phase field.

    Uses central differences via `np.gradient` (interior pixels are averaged
    forward/backward differences; edge pixels use one-sided differences),
    giving a full-resolution gradient-magnitude map matching `phase`'s shape.

    Args:
        phase: 2D phase array, radians.

    Returns:
        2D gradient magnitude array, radians/pixel, same shape as `phase`.
    """
    grad_y, grad_x = np.gradient(phase)
    return np.sqrt(grad_x**2 + grad_y**2)


def nyquist_violation_map(phase: np.ndarray, threshold: float = math.pi) -> np.ndarray:
    """Flag pixels where the local phase gradient exceeds the Nyquist limit.

    Classical (non-AI) unwrapping algorithms implicitly assume the true
    phase gradient never exceeds pi radians/pixel (the Nyquist/Itoh
    condition); violations indicate deformation steep enough that even a
    theoretically perfect classical unwrapper cannot resolve it from the
    wrapped phase alone without additional prior information (which is
    exactly the regime `AmbiguityNet` is designed to help with).

    Args:
        phase: 2D phase array, radians.
        threshold: Gradient magnitude threshold, radians/pixel (defaults to
            pi, the theoretical Nyquist limit).

    Returns:
        Boolean array, same shape as `phase`, True where the gradient
        magnitude exceeds `threshold`.
    """
    return phase_gradient_magnitude(phase) > threshold


def gradient_analysis_summary(phase: np.ndarray, threshold: float = math.pi) -> dict:
    """Summarize gradient statistics and Nyquist violations for a phase field.

    Args:
        phase: 2D phase array, radians.
        threshold: Nyquist violation threshold, radians/pixel.

    Returns:
        Dict with `mean_gradient`, `p95_gradient`, `max_gradient` (all
        radians/pixel), `n_nyquist_violations`, and
        `pct_nyquist_violations`.
    """
    grad_mag = phase_gradient_magnitude(phase)
    violations = grad_mag > threshold
    return {
        "mean_gradient": float(grad_mag.mean()),
        "p95_gradient": float(np.percentile(grad_mag, 95)),
        "max_gradient": float(grad_mag.max()),
        "n_nyquist_violations": int(violations.sum()),
        "pct_nyquist_violations": float(100.0 * violations.mean()),
    }


# --------------------------------------------------------------------------- #
# Error distribution
# --------------------------------------------------------------------------- #


def compute_error_distribution(
    predicted_unwrapped: np.ndarray,
    true_unwrapped: np.ndarray,
    n_bins: int = 50,
) -> tuple[pd.DataFrame, dict]:
    """Compute the error distribution between a predicted and true unwrapped phase.

    Args:
        predicted_unwrapped: Predicted unwrapped phase, radians.
        true_unwrapped: Ground-truth unwrapped phase, radians, same shape.
        n_bins: Number of histogram bins.

    Returns:
        `(histogram_df, summary)`:
            - `histogram_df`: `pandas.DataFrame` with columns
              `["bin_center", "count", "density"]`, one row per histogram bin.
            - `summary`: dict with `mean_error`, `std_error`, `rmse`,
              `mae`, `p50_abs_error`, `p95_abs_error`, `max_abs_error`,
              `pct_under_0p1_rad`, `pct_under_0p5_rad` (all radians except
              the percentages).

    Raises:
        ValueError: If the two arrays' shapes don't match.
    """
    if predicted_unwrapped.shape != true_unwrapped.shape:
        raise ValueError(
            f"Shape mismatch: predicted {predicted_unwrapped.shape} vs "
            f"true {true_unwrapped.shape}"
        )
    error = (predicted_unwrapped - true_unwrapped).ravel()
    abs_error = np.abs(error)

    counts, edges = np.histogram(error, bins=n_bins)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    density = counts / max(counts.sum(), 1)
    histogram_df = pd.DataFrame({"bin_center": bin_centers, "count": counts, "density": density})

    summary = {
        "mean_error": float(error.mean()),
        "std_error": float(error.std()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(abs_error.mean()),
        "p50_abs_error": float(np.percentile(abs_error, 50)),
        "p95_abs_error": float(np.percentile(abs_error, 95)),
        "max_abs_error": float(abs_error.max()),
        "pct_under_0p1_rad": float(100.0 * np.mean(abs_error < 0.1)),
        "pct_under_0p5_rad": float(100.0 * np.mean(abs_error < 0.5)),
    }
    return histogram_df, summary


# --------------------------------------------------------------------------- #
# Top-level convenience: full scene report
# --------------------------------------------------------------------------- #


def analyze_scene(
    wrapped_phase: np.ndarray,
    predicted_unwrapped: np.ndarray,
    true_unwrapped: np.ndarray | None = None,
) -> dict:
    """Run the full phase-statistics suite on one unwrapped scene.

    Args:
        wrapped_phase: Observed wrapped phase, radians.
        predicted_unwrapped: Model's predicted unwrapped phase, radians.
        true_unwrapped: Optional ground-truth unwrapped phase (only
            available for synthetic/validation data); if provided, the
            error-distribution analysis is included.

    Returns:
        A single JSON-serializable dict with keys `"residues"`,
        `"gradient_analysis"`, and (if `true_unwrapped` is given)
        `"error_distribution"`.
    """
    result = {
        "residues": compare_residues(wrapped_phase, predicted_unwrapped),
        "gradient_analysis": gradient_analysis_summary(predicted_unwrapped),
    }
    if true_unwrapped is not None:
        _histogram_df, error_summary = compute_error_distribution(
            predicted_unwrapped, true_unwrapped
        )
        result["error_distribution"] = error_summary
    return result

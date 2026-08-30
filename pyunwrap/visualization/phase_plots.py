"""
pyunwrap.visualization.phase_plots
======================================

Publication-quality, interactive `plotly` visualizations of phase data:

1. **3D phase surfaces** (X, Y, phase as Z) -- the standard way to visually
   inspect 2*pi jumps, atmospheric ramps, and deformation bowls in an
   interferogram, since discontinuities and curvature are far easier to spot
   in 3D than in a flat heatmap.
2. **2D heatmaps** for the residue-probability (uncertainty) map and other
   scalar rasters.

All functions return `plotly.graph_objects.Figure` objects (not files); use
`.write_html(path)` or `pyunwrap.analytics.report_generator`'s helpers to
persist them.
"""

from __future__ import annotations

import numpy as np

try:
    import plotly.graph_objects as go

    _HAS_PLOTLY = True
except ImportError:  # pragma: no cover
    _HAS_PLOTLY = False


def _require_plotly() -> None:
    if not _HAS_PLOTLY:
        raise ImportError("plotly is required for phase plots (`pip install plotly`).")


# --------------------------------------------------------------------------- #
# 3D surfaces
# --------------------------------------------------------------------------- #


def plot_phase_surface_3d(
    phase: np.ndarray,
    title: str = "Unwrapped phase surface",
    colorscale: str = "Jet",
    max_resolution: int = 300,
) -> go.Figure:
    """Render a phase field as an interactive 3D surface (X, Y, phase as Z).

    Args:
        phase: 2D phase array, radians.
        title: Plot title.
        colorscale: Plotly colorscale name.
        max_resolution: If either dimension of `phase` exceeds this, the
            array is downsampled (via striding) before plotting -- Plotly's
            WebGL surface renderer becomes sluggish well before typical
            interferogram resolutions, and a downsampled preview is far more
            useful in a report than a browser tab that hangs.

    Returns:
        A `plotly.graph_objects.Figure` with the 3D surface.
    """
    _require_plotly()
    phase = _downsample_if_needed(phase, max_resolution)

    fig = go.Figure(data=[go.Surface(z=phase, colorscale=colorscale, colorbar={"title": "rad"})])
    fig.update_layout(
        title=title,
        scene={
            "xaxis_title": "Range (px)",
            "yaxis_title": "Azimuth (px)",
            "zaxis_title": "Phase (rad)",
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
    )
    return fig


def plot_phase_surface_comparison_3d(
    predicted: np.ndarray,
    true_phase: np.ndarray,
    max_resolution: int = 200,
) -> go.Figure:
    """Render predicted and ground-truth unwrapped phase as two side-by-side
    3D subplots sharing a Z-axis range, for direct visual comparison.

    Args:
        predicted: 2D predicted unwrapped phase, radians.
        true_phase: 2D ground-truth unwrapped phase, radians, same shape.
        max_resolution: See `plot_phase_surface_3d`.

    Returns:
        A `plotly.graph_objects.Figure` with two 3D surface subplots.
    """
    _require_plotly()
    from plotly.subplots import make_subplots

    predicted = _downsample_if_needed(predicted, max_resolution)
    true_phase = _downsample_if_needed(true_phase, max_resolution)

    zmin = float(min(predicted.min(), true_phase.min()))
    zmax = float(max(predicted.max(), true_phase.max()))

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "surface"}, {"type": "surface"}]],
        subplot_titles=("Predicted unwrapped", "Ground truth unwrapped"),
    )
    fig.add_trace(
        go.Surface(z=predicted, colorscale="Jet", cmin=zmin, cmax=zmax, showscale=False),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Surface(z=true_phase, colorscale="Jet", cmin=zmin, cmax=zmax, colorbar={"title": "rad"}),
        row=1,
        col=2,
    )
    fig.update_layout(
        title="Predicted vs. ground truth unwrapped phase", margin={"l": 0, "r": 0, "b": 0, "t": 60}
    )
    return fig


def _downsample_if_needed(array: np.ndarray, max_resolution: int) -> np.ndarray:
    """Stride-downsample a 2D array so neither dimension exceeds `max_resolution`."""
    rows, cols = array.shape
    stride = max(1, max(rows, cols) // max_resolution)
    return array[::stride, ::stride] if stride > 1 else array


# --------------------------------------------------------------------------- #
# 2D heatmaps
# --------------------------------------------------------------------------- #


def plot_heatmap(
    array: np.ndarray,
    title: str = "Heatmap",
    colorscale: str = "Inferno",
    zmin: float | None = None,
    zmax: float | None = None,
    colorbar_title: str = "",
) -> go.Figure:
    """Render a 2D scalar array as an interactive Plotly heatmap.

    Args:
        array: 2D data array.
        title: Plot title.
        colorscale: Plotly colorscale name.
        zmin: Color scale minimum; defaults to `array.min()`.
        zmax: Color scale maximum; defaults to `array.max()`.
        colorbar_title: Label for the colorbar.

    Returns:
        A `plotly.graph_objects.Figure` with the heatmap.
    """
    _require_plotly()
    fig = go.Figure(
        data=go.Heatmap(
            z=array,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            colorbar={"title": colorbar_title},
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Range (px)",
        yaxis_title="Azimuth (px)",
        yaxis={"autorange": "reversed"},  # image-style origin (row 0 at top)
        margin={"l": 40, "r": 40, "b": 40, "t": 60},
    )
    return fig


def plot_residue_probability_heatmap(residue_prob: np.ndarray) -> go.Figure:
    """Convenience wrapper: heatmap of the model's residue-probability
    (uncertainty) output, with a fixed [0, 1] color scale.

    Args:
        residue_prob: 2D residue probability array, [0, 1].

    Returns:
        A `plotly.graph_objects.Figure`.
    """
    return plot_heatmap(
        residue_prob,
        title="Residue / uncertainty probability",
        colorscale="Inferno",
        zmin=0.0,
        zmax=1.0,
        colorbar_title="P(uncertain)",
    )

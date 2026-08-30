"""
pyunwrap.visualization.charts
================================

Interactive `plotly` line/bar charts for training diagnostics and error
analysis, complementing `pyunwrap.training.trainer.Visualizer`'s static
matplotlib PNGs with browser-interactive versions suitable for embedding in
the HTML report (`pyunwrap.analytics.report_generator`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go

    _HAS_PLOTLY = True
except ImportError:  # pragma: no cover
    _HAS_PLOTLY = False


def _require_plotly() -> None:
    if not _HAS_PLOTLY:
        raise ImportError("plotly is required for charts (`pip install plotly`).")


def plot_training_curve(
    epochs: list[int],
    values: list[float],
    title: str = "Training RMSE",
    y_label: str = "RMSE (rad)",
) -> go.Figure:
    """Plot a single scalar metric against epoch as an interactive line chart.

    Args:
        epochs: Epoch numbers (x-axis).
        values: Metric values (y-axis), same length as `epochs`.
        title: Plot title.
        y_label: Y-axis label.

    Returns:
        A `plotly.graph_objects.Figure`.
    """
    _require_plotly()
    fig = go.Figure(data=go.Scatter(x=epochs, y=values, mode="lines+markers"))
    fig.update_layout(
        title=title,
        xaxis_title="Epoch",
        yaxis_title=y_label,
        margin={"l": 40, "r": 20, "b": 40, "t": 60},
    )
    return fig


def plot_multi_metric_curves(
    metrics: dict[str, tuple[list[int], list[float]]],
    title: str = "Training metrics",
) -> go.Figure:
    """Plot several named metrics (e.g. from `Trainer.Visualizer.history`) on
    one shared-x chart, each metric its own (epoch, value) series (since
    validation metrics may be logged less frequently than training metrics).

    Args:
        metrics: Mapping of metric name -> `(epochs, values)`.
        title: Plot title.

    Returns:
        A `plotly.graph_objects.Figure` with one trace per metric.
    """
    _require_plotly()
    fig = go.Figure()
    for name, (epochs, values) in metrics.items():
        fig.add_trace(go.Scatter(x=epochs, y=values, mode="lines+markers", name=name))
    fig.update_layout(title=title, xaxis_title="Epoch", margin={"l": 40, "r": 20, "b": 40, "t": 60})
    return fig


def plot_residue_count_curve(epochs: list[int], residue_counts: list[int]) -> go.Figure:
    """Convenience wrapper: bar chart of validation residue count over epochs.

    Args:
        epochs: Epoch numbers at which validation ran.
        residue_counts: Residue (or residue-proxy) counts at each of `epochs`.

    Returns:
        A `plotly.graph_objects.Figure`.
    """
    _require_plotly()
    fig = go.Figure(data=go.Bar(x=epochs, y=residue_counts))
    fig.update_layout(
        title="Validation residue count over training",
        xaxis_title="Epoch",
        yaxis_title="Residue count",
        margin={"l": 40, "r": 20, "b": 40, "t": 60},
    )
    return fig


def plot_error_histogram(
    histogram_df: pd.DataFrame,
    title: str = "Phase error distribution",
) -> go.Figure:
    """Plot a phase-error histogram from
    `pyunwrap.analytics.phase_stats.compute_error_distribution`'s
    `histogram_df` output.

    Args:
        histogram_df: DataFrame with columns `["bin_center", "count", "density"]`.
        title: Plot title.

    Returns:
        A `plotly.graph_objects.Figure` bar chart.
    """
    _require_plotly()
    fig = go.Figure(
        data=go.Bar(x=histogram_df["bin_center"], y=histogram_df["density"], width=None)
    )
    fig.update_layout(
        title=title,
        xaxis_title="Error (rad)",
        yaxis_title="Density",
        margin={"l": 40, "r": 20, "b": 40, "t": 60},
    )
    return fig


def plot_reliability_diagram(
    bins_df: pd.DataFrame,
    title: str = "Uncertainty calibration",
) -> go.Figure:
    """Plot a reliability diagram from
    `pyunwrap.analytics.explainability.reliability_diagram`'s `bins_df`
    output: predicted uncertainty vs. mean actual error per bin, with a
    diagonal reference line for perfect (1:1 scale) calibration.

    Args:
        bins_df: DataFrame with columns `["bin_center",
            "mean_predicted_uncertainty", "mean_actual_error", "n_pixels"]`.
        title: Plot title.

    Returns:
        A `plotly.graph_objects.Figure`.
    """
    _require_plotly()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=bins_df["mean_predicted_uncertainty"],
            y=bins_df["mean_actual_error"],
            mode="markers+lines",
            name="Observed",
            marker={"size": np.clip(np.sqrt(bins_df["n_pixels"]), 4, 20)},
        )
    )
    if len(bins_df):
        lo = float(
            min(bins_df["mean_predicted_uncertainty"].min(), bins_df["mean_actual_error"].min())
        )
        hi = float(
            max(bins_df["mean_predicted_uncertainty"].max(), bins_df["mean_actual_error"].max())
        )
        fig.add_trace(
            go.Scatter(
                x=[lo, hi],
                y=[lo, hi],
                mode="lines",
                name="Perfect calibration",
                line={"dash": "dash"},
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Mean predicted uncertainty",
        yaxis_title="Mean actual |error| (rad)",
        margin={"l": 40, "r": 20, "b": 40, "t": 60},
    )
    return fig

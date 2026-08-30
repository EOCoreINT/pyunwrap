"""
pyunwrap.analytics.report_generator
======================================

Automated HTML reporting: assembles the phase statistics
(`pyunwrap.analytics.phase_stats`), explainability
(`pyunwrap.analytics.explainability`), and visualization
(`pyunwrap.visualization.*`) modules into a single, responsive, self-
contained HTML report via Jinja2.

Layout: Executive Summary -> Model Performance -> Phase Analysis (embedded
3D Plotly surfaces + residue/uncertainty heatmaps) -> Explainability
(embedded Grad-CAM / channel-attribution) -> Provenance footer. Each
interactive figure (Plotly/Folium) is saved as its own standalone HTML
snippet under `assets/` and embedded into the main report via an `<iframe>`,
which keeps the main report's HTML small and lets each figure be opened
independently if needed.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from jinja2 import BaseLoader, Environment

from pyunwrap.analytics import phase_stats
from pyunwrap.visualization import charts, phase_plots

if TYPE_CHECKING:  # pragma: no cover
    import torch

    from pyunwrap.inference.unwrapper import UnwrapResult
    from pyunwrap.models.ambiguity_net import AmbiguityNet


REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>pyunwrap unwrapping report</title>
<style>
  :root { --accent: #2b6cb0; --bg: #f7f8fa; --card-bg: #ffffff; --border: #e2e8f0; }
  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
         margin: 0; background: var(--bg); color: #1a202c; line-height: 1.5; }
  header { background: var(--accent); color: white; padding: 2rem; }
  header h1 { margin: 0 0 0.25rem 0; font-size: 1.6rem; }
  header p { margin: 0; opacity: 0.9; font-size: 0.9rem; }
  main { max-width: 1100px; margin: 0 auto; padding: 1.5rem; }
  section { background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px;
            padding: 1.5rem; margin-bottom: 1.5rem; }
  section h2 { margin-top: 0; font-size: 1.25rem; border-bottom: 2px solid var(--border); padding-bottom: 0.5rem; }
  .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin: 1rem 0; }
  .metric-card { background: var(--bg); border-radius: 6px; padding: 1rem; text-align: center; }
  .metric-value { font-size: 1.5rem; font-weight: 600; color: var(--accent); }
  .metric-label { font-size: 0.8rem; color: #4a5568; margin-top: 0.25rem; }
  iframe { width: 100%; height: 520px; border: 1px solid var(--border); border-radius: 6px; }
  table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.9rem; }
  th, td { text-align: left; padding: 0.5rem 0.75rem; border-bottom: 1px solid var(--border); }
  th { background: var(--bg); }
  footer { text-align: center; color: #718096; font-size: 0.8rem; padding: 2rem; }
  .figure-row { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1rem 0; }
  @media (max-width: 800px) { .figure-row { grid-template-columns: 1fr; } }
  .warning-banner { background: #fffaf0; border: 1px solid #f6ad55; color: #7b341e;
                     border-radius: 6px; padding: 0.75rem 1rem; margin-bottom: 1rem; font-size: 0.9rem; }
</style>
</head>
<body>
<header>
  <h1>pyunwrap unwrapping report</h1>
  <p>Generated {{ generated_at }}</p>
</header>
<main>

<section>
  <h2>Executive summary</h2>
  <div class="metrics-grid">
    {% for label, value in summary_metrics %}
    <div class="metric-card">
      <div class="metric-value">{{ value }}</div>
      <div class="metric-label">{{ label }}</div>
    </div>
    {% endfor %}
  </div>
  {% if warnings %}
  {% for w in warnings %}
  <div class="warning-banner">{{ w }}</div>
  {% endfor %}
  {% endif %}
</section>

{% if has_ground_truth %}
<section>
  <h2>Model performance</h2>
  <div class="metrics-grid">
    {% for label, value in performance_metrics %}
    <div class="metric-card">
      <div class="metric-value">{{ value }}</div>
      <div class="metric-label">{{ label }}</div>
    </div>
    {% endfor %}
  </div>
  <iframe src="assets/error_histogram.html"></iframe>
</section>
{% endif %}

{% if has_training_history %}
<section>
  <h2>Training history</h2>
  <iframe src="assets/training_curves.html"></iframe>
</section>
{% endif %}

<section>
  <h2>Phase analysis</h2>
  <div class="figure-row">
    <iframe src="assets/unwrapped_surface_3d.html"></iframe>
    <iframe src="assets/residue_probability.html"></iframe>
  </div>
  {% if has_ground_truth %}
  <iframe src="assets/phase_comparison_3d.html"></iframe>
  {% endif %}
  <h3>Residue summary</h3>
  <table>
    <tr><th>Metric</th><th>Input (observed)</th><th>Predicted</th></tr>
    {% for row in residue_table %}
    <tr><td>{{ row.metric }}</td><td>{{ row.input }}</td><td>{{ row.predicted }}</td></tr>
    {% endfor %}
  </table>
  <h3>Gradient / Nyquist analysis</h3>
  <table>
    {% for label, value in gradient_table %}
    <tr><td>{{ label }}</td><td>{{ value }}</td></tr>
    {% endfor %}
  </table>
  {% if map_available %}
  <h3>Interactive map</h3>
  <iframe src="assets/phase_map.html" style="height: 600px;"></iframe>
  {% endif %}
</section>

{% if has_explainability %}
<section>
  <h2>Explainability</h2>
  <h3>Input channel attribution (Integrated Gradients)</h3>
  <table>
    <tr><th>Channel</th><th>Attribution share</th></tr>
    {% for name, share in channel_importance %}
    <tr><td>{{ name }}</td><td>{{ share }}</td></tr>
    {% endfor %}
  </table>
  <div class="figure-row">
    <iframe src="assets/grad_cam.html"></iframe>
    {% if has_calibration %}
    <iframe src="assets/reliability_diagram.html"></iframe>
    {% endif %}
  </div>
</section>
{% endif %}

</main>
<footer>
  <p>pyunwrap v{{ version }} &middot; generated {{ generated_at }} &middot;
     {{ raster_shape }} scene &middot; report id {{ report_id }}</p>
</footer>
</body>
</html>
"""


def _write_html_asset(fig, assets_dir: Path, filename: str) -> None:
    """Write a Plotly figure or folium map to `assets_dir/filename` as a
    standalone HTML file. Accepts either object type transparently (both
    expose a compatible `.write_html`/`.save` surface, dispatched on
    whichever attribute is present).
    """
    path = assets_dir / filename
    if hasattr(fig, "write_html"):
        fig.write_html(str(path), include_plotlyjs="cdn")
    elif hasattr(fig, "save"):
        fig.save(str(path))
    else:  # pragma: no cover
        raise TypeError(f"Don't know how to save asset of type {type(fig)}")


def generate_report(
    result: UnwrapResult,
    out_dir: str | Path,
    true_unwrapped: np.ndarray | None = None,
    model: AmbiguityNet | None = None,
    sample_tile_input: torch.Tensor | None = None,
    training_history: dict | None = None,
    include_map: bool = True,
) -> Path:
    """Generate the full automated HTML report for one `UnwrapResult`.

    Args:
        result: The `PhaseUnwrapper.unwrap()` output to report on.
        out_dir: Output directory; the report is written to
            `out_dir/report.html`, with supporting figures under
            `out_dir/assets/`.
        true_unwrapped: Optional ground-truth unwrapped phase (only
            available for synthetic/validation scenes). If provided, the
            "Model performance" section and the predicted-vs-truth 3D
            comparison are included.
        model: Optional trained `AmbiguityNet`, used together with
            `sample_tile_input` to generate the Explainability section
            (Grad-CAM + channel attribution). Both must be provided together.
        sample_tile_input: Optional [1, 3, H, W] input tensor for a
            representative tile, used for explainability. See `model`.
        training_history: Optional dict of `{metric_name: (epochs, values)}`
            (e.g. from `pyunwrap.training.trainer.Visualizer.history`), used
            for an additional training-curves figure if provided.
        include_map: Whether to attempt building the interactive folium map
            (requires `result.geotiff_profile` to carry a valid CRS+transform;
            silently skipped with a report banner if that fails, since a
            missing/invalid CRS is common for ad hoc numpy-only workflows and
            should not break the rest of the report).

    Returns:
        Path to the generated `report.html`.

    Note on failure handling: this function does the actual report assembly
    work; `pyunwrap.inference.unwrapper.PhaseUnwrapper._try_generate_report`
    is the layer responsible for catching exceptions from this function so a
    reporting failure never takes down a production inference call. This
    function itself is written to degrade individual *sections* gracefully
    (e.g. skipping the map on a bad CRS) rather than the whole report.
    """
    out_dir = Path(out_dir)
    assets_dir = out_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    warnings_list: list[str] = []

    # --- Phase statistics ---
    analysis = phase_stats.analyze_scene(
        result.wrapped_phase,
        result.unwrapped_phase,
        true_unwrapped=true_unwrapped if true_unwrapped is not None else None,
    )
    residue_info = analysis["residues"]
    gradient_info = analysis["gradient_analysis"]

    # --- Core phase figures ---
    _write_html_asset(
        phase_plots.plot_phase_surface_3d(
            result.unwrapped_phase, title="Predicted unwrapped phase"
        ),
        assets_dir,
        "unwrapped_surface_3d.html",
    )
    _write_html_asset(
        phase_plots.plot_residue_probability_heatmap(result.residue_prob),
        assets_dir,
        "residue_probability.html",
    )

    has_ground_truth = true_unwrapped is not None
    performance_metrics: list[tuple[str, str]] = []
    if has_ground_truth:
        _write_html_asset(
            phase_plots.plot_phase_surface_comparison_3d(result.unwrapped_phase, true_unwrapped),
            assets_dir,
            "phase_comparison_3d.html",
        )
        hist_df, error_summary = phase_stats.compute_error_distribution(
            result.unwrapped_phase, true_unwrapped
        )
        _write_html_asset(charts.plot_error_histogram(hist_df), assets_dir, "error_histogram.html")
        performance_metrics = [
            ("RMSE (rad)", f"{error_summary['rmse']:.4f}"),
            ("MAE (rad)", f"{error_summary['mae']:.4f}"),
            ("% pixels < 0.1 rad", f"{error_summary['pct_under_0p1_rad']:.1f}%"),
            ("Max abs error (rad)", f"{error_summary['max_abs_error']:.4f}"),
        ]

    # --- Interactive map (best-effort; a bad/missing CRS shouldn't kill the report) ---
    map_available = False
    if include_map:
        try:
            from pyunwrap.visualization import maps as maps_module

            fmap = maps_module.build_phase_comparison_map(
                wrapped_phase=result.wrapped_phase,
                unwrapped_phase=result.unwrapped_phase,
                geotiff_profile=result.geotiff_profile,
                coherence=None,
                residue_prob=result.residue_prob,
            )
            maps_module.save_map(fmap, assets_dir / "phase_map.html")
            map_available = True
        except Exception as exc:
            warnings_list.append(f"Interactive map could not be generated and was skipped: {exc}")

    # --- Explainability (optional; needs a live model + sample tile) ---
    has_explainability = model is not None and sample_tile_input is not None
    channel_importance: list[tuple[str, str]] = []
    has_calibration = False
    if has_explainability:
        from pyunwrap.analytics import explainability

        try:
            cam_result = explainability.grad_cam(model, sample_tile_input)
            _write_html_asset(
                phase_plots.plot_heatmap(
                    cam_result.saliency_map, title="Grad-CAM saliency", colorscale="Viridis"
                ),
                assets_dir,
                "grad_cam.html",
            )
            ig_result = explainability.integrated_gradients(model, sample_tile_input, steps=30)
            channel_importance = [
                (name, f"{share * 100:.1f}%")
                for name, share in ig_result.channel_importance.items()
            ]
        except Exception as exc:
            has_explainability = False
            warnings_list.append(
                f"Explainability section could not be generated and was skipped: {exc}"
            )

    has_training_history = False
    if training_history:
        try:
            _write_html_asset(
                charts.plot_multi_metric_curves(
                    {
                        k: ([e for e, _ in v], [val for _, val in v])
                        for k, v in training_history.items()
                    }
                ),
                assets_dir,
                "training_curves.html",
            )
            has_training_history = True
        except Exception as exc:
            warnings_list.append(f"Training curves could not be embedded and were skipped: {exc}")

    # --- Assemble summary tables ---
    summary_metrics = [
        ("Scene shape", f"{result.unwrapped_phase.shape[0]} x {result.unwrapped_phase.shape[1]}"),
        ("Mean residue prob.", f"{float(result.residue_prob.mean()):.3f}"),
        ("Spurious residue px", str(residue_info["spurious_residue_pixels"])),
        ("Nyquist violations", f"{gradient_info['pct_nyquist_violations']:.2f}%"),
    ]

    residue_table = [
        {
            "metric": "Positive clusters",
            "input": residue_info["input"]["n_positive_clusters"],
            "predicted": residue_info["predicted"]["n_positive_clusters"],
        },
        {
            "metric": "Negative clusters",
            "input": residue_info["input"]["n_negative_clusters"],
            "predicted": residue_info["predicted"]["n_negative_clusters"],
        },
        {
            "metric": "Residue density (per 1000 px)",
            "input": f"{residue_info['input']['residue_density_per_1000px']:.2f}",
            "predicted": f"{residue_info['predicted']['residue_density_per_1000px']:.2f}",
        },
        {
            "metric": "Net topological charge",
            "input": residue_info["input"]["net_charge"],
            "predicted": residue_info["predicted"]["net_charge"],
        },
    ]

    gradient_table = [
        ("Mean gradient (rad/px)", f"{gradient_info['mean_gradient']:.4f}"),
        ("95th pct. gradient (rad/px)", f"{gradient_info['p95_gradient']:.4f}"),
        ("Max gradient (rad/px)", f"{gradient_info['max_gradient']:.4f}"),
        ("Nyquist violations (px)", str(gradient_info["n_nyquist_violations"])),
        ("Nyquist violations (%)", f"{gradient_info['pct_nyquist_violations']:.2f}%"),
    ]

    env = Environment(loader=BaseLoader(), autoescape=True)
    template = env.from_string(REPORT_TEMPLATE)
    html = template.render(
        generated_at=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        summary_metrics=summary_metrics,
        performance_metrics=performance_metrics,
        has_ground_truth=has_ground_truth,
        has_training_history=has_training_history,
        residue_table=residue_table,
        gradient_table=gradient_table,
        map_available=map_available,
        has_explainability=has_explainability,
        channel_importance=channel_importance,
        has_calibration=has_calibration,
        warnings=warnings_list,
        version="0.1.0",
        raster_shape=f"{result.unwrapped_phase.shape[0]}x{result.unwrapped_phase.shape[1]}",
        report_id=datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d%H%M%S"),
    )

    report_path = out_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")
    return report_path

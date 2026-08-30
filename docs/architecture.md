# pyunwrap architecture

This document gives a module-by-module overview of the package and how data
flows through it. For installation and a quick example, see the top-level
[README](../README.md).

## Data flow

```
InSARSyntheticGenerator                    real GeoTIFFs
(pyunwrap.synthetic.generator)             (wrapped phase, coherence, amplitude)
        |                                           |
        v                                           v
NormalizedRasters + tiling                 load_and_normalize_rasters
(pyunwrap.data.preprocessing)                        |
        |                                           |
        v                                           |
   HDF5 tile file                                    |
        |                                           |
        v                                           |
InSARTileDataset  --------> AmbiguityNet  <----------+  (inference: tiled,
(pyunwrap.data.dataloader)  (pyunwrap.models          via PhaseUnwrapper,
        |                    .ambiguity_net)          pyunwrap.inference
        v                          |                  .unwrapper)
Trainer.fit()                      v
(pyunwrap.training           PhysicsInformedUnwrapLoss
 .trainer)                   (pyunwrap.models.losses)
        |
        v
  checkpoints, TensorBoard logs, Visualizer plots
```

At inference time, `PhaseUnwrapper` tiles a full-scene GeoTIFF, runs
`AmbiguityNet` (or an exported ONNX model) on each tile, and merges the
predicted **integer ambiguity maps** (never the phase itself) back into a
single global raster via edge-aware, residue-probability-weighted blending.
The final unwrapped phase is always reconstructed as
`wrapped_phase + 2*pi*k_merged` directly from the original input raster, so
it is exactly consistent with the true observed data everywhere.

## The core physics constraint

Every stage of the pipeline is built around one invariant:

```
wrapped_phase + 2*pi*k == unwrapped_phase       (k integer-valued)
```

- The synthetic generator constructs samples so this holds exactly (see
  `pyunwrap.synthetic.generator`'s module docstring and `SyntheticSample`).
- `AmbiguityNet.forward` builds `phi_hat` by literally adding
  `2*pi*round_ste(k_continuous)` to the wrapped phase, so the constraint
  holds **by construction**, not as a learned behavior.
- `PhaseUnwrapper` reconstructs the final result the same way, from the
  original (un-tiled) wrapped phase raster.

Two consequences of this that are easy to misread as bugs (documented in
detail in `pyunwrap.models.losses`'s module docstring):

1. `PhysicsInformedUnwrapLoss`'s "re-wrap consistency" component is
   near-zero **regardless of whether the prediction is correct**, since
   adding any integer multiple of `2*pi` before wrapping can never change
   the wrapped result.
2. The reconstructed phase can never contain a classical topological residue
   that wasn't already present in the input wrapped phase -- adding an
   integer field is algebraically guaranteed not to introduce one. What
   *can* go wrong is an isolated, spatially-unsupported jump in the
   predicted `k` map, which is what the loss's residue component (a
   Laplacian penalty on `k`, not a literal residue count) and the
   inference-time residue-probability-weighted merging are both designed to
   suppress.

## Module reference

| Module | Responsibility |
|---|---|
| `pyunwrap.synthetic.generator` | Physically-motivated synthetic InSAR data: deformation models (Gaussian bowl, Okada, Mogi), topography, atmosphere, orbital ramps, decorrelation noise, and the pseudo-real ALOS-2 rewrapping strategy. |
| `pyunwrap.data.preprocessing` | Input normalization and full-coverage sliding-window tiling of large rasters. |
| `pyunwrap.data.dataloader` | `InSARTileDataset`: a PyTorch `Dataset` over tiled HDF5 data with physics-consistent augmentation. |
| `pyunwrap.models.ambiguity_net` | `AmbiguityNet`: the ResNet-34-encoder U-Net that predicts the integer ambiguity map. |
| `pyunwrap.models.losses` | `PhysicsInformedUnwrapLoss`: the 4-component training loss. |
| `pyunwrap.training.trainer` | `Trainer`: curriculum learning, SNAPHU fine-tuning, the training loop, and the `Visualizer` diagnostic-plotting helper. |
| `pyunwrap.inference.unwrapper` | `PhaseUnwrapper`: tiled inference, smart ambiguity-map merging, Monte Carlo Dropout uncertainty. |
| `pyunwrap.utils.deployment` | ONNX export, weight caching, and the ONNX Runtime / OpenVINO backend fallback chain. |
| `pyunwrap.analytics.phase_stats` | Residue detection, Nyquist gradient analysis, error-distribution statistics. |
| `pyunwrap.analytics.explainability` | Grad-CAM, Integrated Gradients (per-input-channel attribution), uncertainty calibration. |
| `pyunwrap.analytics.report_generator` | Assembles the above (plus `pyunwrap.visualization`) into a single HTML report. |
| `pyunwrap.visualization.*` | Interactive Plotly 3D surfaces/heatmaps, Plotly training charts, and a folium swipe-comparison map. |

## Where to look for more detail

Every public function and class has a full docstring, including the
reasoning behind non-obvious design choices (e.g. why the curriculum's
gradient filter uses a 99th percentile rather than a hard max, or why tile
sizes must be multiples of 32 for the ONNX backend). Start with
`pyunwrap/synthetic/generator.py` and `pyunwrap/models/ambiguity_net.py` if
you're new to the codebase -- everything downstream builds on the
conventions established there.

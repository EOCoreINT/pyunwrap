# API reference

Generated directly from the source docstrings. Every function/class below
includes its full `Args`/`Returns`/`Raises` documentation, and a "[source]"
link to the exact implementation on GitHub.

```{toctree}
:maxdepth: 1

synthetic
data
models
training
inference
analytics
visualization
utils
```

## Package map

| Subpackage | Responsibility |
|---|---|
| {doc}`synthetic` | Physically-motivated synthetic InSAR data: deformation models, topography, atmosphere, decorrelation. |
| {doc}`data` | Preprocessing, tiling, and the PyTorch `Dataset`. |
| {doc}`models` | `AmbiguityNet` and `PhysicsInformedUnwrapLoss`. |
| {doc}`training` | `Trainer`: curriculum learning, SNAPHU fine-tuning, the training loop. |
| {doc}`inference` | `PhaseUnwrapper`: tiled inference, smart ambiguity-map merging. |
| {doc}`analytics` | Residue stats, explainability, the SNAPHU benchmark harness, HTML reports. |
| {doc}`visualization` | Interactive maps and Plotly 3D phase surfaces. |
| {doc}`utils` | ONNX/OpenVINO deployment and the real SNAPHU integration. |

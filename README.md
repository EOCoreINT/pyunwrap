<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/logo-horizontal-dark.svg">
  <img src="assets/logo-horizontal.svg" alt="pyunwrap" width="420">
</picture>

**Physics-informed deep learning for InSAR phase unwrapping.**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![Tests: 65 passing](https://img.shields.io/badge/tests-65%20passing-brightgreen)](tests/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Status: early-stage](https://img.shields.io/badge/status-early--stage-orange)](#project-status)

[Why this exists](#why-this-exists) ·
[How it works](#how-it-works) ·
[Install](#installation) ·
[Quickstart](#quickstart) ·
[Notebooks](#notebooks) ·
[Architecture](#architecture) ·
[Citation](#citation)

</div>

---

`pyunwrap` unwraps Interferometric Synthetic Aperture Radar (InSAR) phase — the
core measurement behind satellite-based ground-deformation monitoring — using a
**physics-informed U-Net that predicts an integer ambiguity map**, not the
unwrapped phase itself. The rewrapping identity is enforced by construction,
so the model is structurally incapable of producing an output that
contradicts the observed wrapped phase; it can only be wrong about *how many*
`2π` cycles were missed, never about the physics of wrapping.

## Why this exists

Every classical phase-unwrapping algorithm — branch-cut methods, minimum-cost
flow (the approach behind SNAPHU, the field's long-standing reference tool),
Goldstein's algorithm — rests on one assumption: that the true phase gradient
between adjacent pixels never exceeds `π` radians (the Nyquist/Itoh
condition). Two situations break that assumption in practice:

- **Low coherence.** Vegetation, water, and temporal decorrelation add
  near-random phase noise. Once the local gradient becomes unreliable,
  branch-cut and MCF methods don't just get that one pixel wrong — they
  propagate the error across the connected region downstream of it.
- **Steep deformation gradients.** Earthquakes, volcanic inflation, and
  mining subsidence can produce genuine phase gradients that exceed `π`
  radians per pixel near the source. No amount of algorithmic cleverness
  recovers this from the wrapped phase alone without an external prior —
  which is exactly the gap a learned model can fill.

`pyunwrap` targets both regimes directly, and does so without inheriting the
most common failure mode of naive deep-learning approaches to this problem:
regressing the unwrapped phase directly gives a network no reason to respect
`wrap(prediction) == observed_phase`. Predicting the integer ambiguity
instead makes that identity a mathematical guarantee, not a hope.

## How it works

`AmbiguityNet` outputs a continuous ambiguity prediction, rounds it via a
straight-through estimator, and reconstructs phase directly:

```
φ̂ = ψ + 2π · round(k̂),   k̂ ∈ ℤ
```

| Approach | Failure mode |
|---|---|
| Regress `φ` directly | Nothing constrains `wrap(prediction) == ψ`; the network can output any real value |
| **Predict `k` (this package)** | `wrap(ψ + 2π·round(k̂)) == ψ` holds by construction — the network only has to get the *integer cycle count* right |

That single design decision shapes everything downstream: training uses a
physics-informed loss with a component that's provably near-zero regardless
of prediction quality (documented explicitly, not hidden — see
[`pyunwrap/models/losses.py`](pyunwrap/models/losses.py)); tiled inference
merges the **integer ambiguity map** across overlapping patches, never the
phase itself, because averaging phase directly across a tile boundary can
silently produce a value that satisfies no physical interferogram.

## Key features

**Synthetic data engine** ([`pyunwrap.synthetic`](pyunwrap/synthetic/generator.py))
— Gaussian subsidence bowls, a from-scratch Okada (1985) rectangular fault
dislocation model, a Mogi (1958) volcanic point source, DEM-driven
topographic phase, Kolmogorov-spectrum atmospheric turbulence, orbital
ramps, coherence-dependent decorrelation noise, and a pseudo-real strategy
that rewraps real L-band (ALOS-2) unwrapped phase into simulated C-band data
to help bridge the sim-to-real gap.

**`AmbiguityNet`** ([`pyunwrap.models`](pyunwrap/models/ambiguity_net.py))
— a ResNet-34-encoder U-Net with a dual head: the integer ambiguity map via
a straight-through-estimator rounding layer, and an auxiliary residue-
probability map for uncertainty. Trained with a four-component
physics-informed loss (ambiguity regression, re-wrap consistency,
coherence-weighted smoothness, ambiguity-map residue penalty).

**Curriculum training** ([`pyunwrap.training`](pyunwrap/training/trainer.py))
— a three-stage curriculum (high-coherence/low-gradient → moderate →
full difficulty), optional SNAPHU pseudo-ground-truth fine-tuning on real
data, AdamW with warmup/cosine annealing, and TensorBoard logging.

**Production inference** ([`pyunwrap.inference`](pyunwrap/inference/unwrapper.py))
— tiled processing of arbitrarily large interferograms with edge-aware,
residue-probability-weighted smart merging of the ambiguity map, Monte
Carlo Dropout uncertainty, and an ONNX Runtime → OpenVINO backend fallback
chain for deployment without a PyTorch dependency.

**Scientific analytics** ([`pyunwrap.analytics`](pyunwrap/analytics/))
— Goldstein-style residue detection and clustering, Nyquist
gradient-violation mapping, error-distribution statistics, Grad-CAM and
Integrated-Gradients explainability, and uncertainty-calibration
reliability diagrams.

**Visualization & reporting** ([`pyunwrap.visualization`](pyunwrap/visualization/))
— interactive Plotly 3D phase surfaces and heatmaps, a folium
swipe-comparison map, and a self-contained, Jinja2-templated HTML report
tying every stage together.

## Architecture

```
InSARSyntheticGenerator ──┐                     real GeoTIFFs
                           │                     (wrapped, coherence, amplitude)
                           ▼                              │
                  tiling + normalization                  ▼
                           │                     tiled inference
                           ▼                     via PhaseUnwrapper
                  InSARTileDataset                         │
                           │                                │
                           ▼                                ▼
                   Trainer.fit()  ──── AmbiguityNet ──── smart ambiguity-map
                  (curriculum,          (this is the        merging (never
                   physics loss)        same model)          the phase)
                                                              │
                                                              ▼
                                                    unwrapped phase +
                                                    analytics + report
```

See [`docs/architecture.md`](docs/architecture.md) for the full
module-by-module reference and a longer explanation of the physics
invariant every stage is built around.

## Installation

```bash
git clone https://github.com/yourusername/pyunwrap.git
cd pyunwrap
pip install -e .
```

Optional extras, installed as needed:

| Extra | Adds | Use case |
|---|---|---|
| `dev` | `pytest`, `pytest-cov`, `black`, `ruff`, `mypy` | Development, testing, linting |
| `maps` | `folium`, `leafmap` | Interactive map visualization |
| `deploy` | `openvino`, `onnx`, `onnxconverter-common`, `onnxscript` | ONNX export, OpenVINO inference |
| `notebooks` | `jupyter`, `ipykernel` | Running the example notebooks |

```bash
pip install -e ".[dev,maps,deploy,notebooks]"   # everything
```

## Quickstart

```python
from pyunwrap.synthetic.generator import InSARSyntheticGenerator
from pyunwrap.models.ambiguity_net import AmbiguityNet
from pyunwrap.inference.unwrapper import PhaseUnwrapper

# Generate a synthetic training sample (Mogi volcanic source deformation).
gen = InSARSyntheticGenerator(size=256, seed=42)
sample = gen.generate_sample(deformation_type="mogi")

# Run tiled inference on a real interferogram with a trained model.
model = AmbiguityNet(pretrained=False, k_max=10.0)  # or load your own checkpoint
unwrapper = PhaseUnwrapper(model=model, device="cuda")
result = unwrapper.unwrap(
    wrapped_phase_path="data/wrapped_phase.tif",
    coherence_path="data/coherence.tif",
    amplitude_path="data/amplitude.tif",
    tile_size=512, overlap=64,
    generate_report=True,
)
result.save_geotiff("unwrapped_output.tif")
```

Training a model end to end:

```bash
pyunwrap-train \
  --train-hdf5 train_tiles.h5 --val-hdf5 val_tiles.h5 \
  --epochs 60 --warmup-epochs 5 \
  --finetune-hdf5 snaphu_pseudo_gt.h5 --finetune-start-epoch 55 \
  --out-dir runs/pyunwrap_v1
```

## Notebooks

Two notebooks in [`notebooks/`](notebooks/) walk through the package
hands-on, checked in **pre-executed with real outputs** so they're readable
without running anything:

- [`01_training_pipeline.ipynb`](notebooks/01_training_pipeline.ipynb) —
  the complete training chain: synthetic data, tiling, `AmbiguityNet` +
  `Trainer`, training curves, and predicted-vs-ground-truth comparison.
- [`02_full_pipeline.ipynb`](notebooks/02_full_pipeline.ipynb) — the
  complete end-to-end chain: a deformation-model gallery, tiling and
  augmentation visualized, a full curriculum training run with every loss
  component plotted, tiled inference on real GeoTIFFs, residue/Nyquist/error
  analytics, Grad-CAM and Integrated-Gradients explainability, 3D and
  interactive-map visualization, ONNX deployment with a numerical
  PyTorch-vs-ONNX agreement check, and the final HTML report.

```bash
pip install -e ".[dev,maps,notebooks]"
jupyter notebook notebooks/
```

## Testing

```bash
pytest -m "not slow"              # fast unit + integration tests
pytest                             # full suite, including the end-to-end
                                    # synthetic → train → infer → report test
pytest --cov=pyunwrap --cov-report=term-missing
```

The suite includes known-answer physics tests (e.g. residue detection is
checked against a hand-constructed phase vortex with an exact, known
topological charge — not just "runs without crashing") and a tile-merging
regression test that asserts tiled-and-merged inference is *numerically
identical* to a whole-image pass, directly targeting the class of bug where
tile boundaries silently corrupt the output. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) for the full breakdown and development
setup.


## Project status

Early-stage and actively developed. The pipeline — synthetic data
generation, training, tiled inference, analytics, and reporting — is
implemented and tested end to end, but **no pretrained weights ship yet**;
`from_pretrained()` downloads from a Zenodo record you supply, and the
example notebooks train small demo models from scratch rather than loading
a benchmarked checkpoint. Monte Carlo Dropout uncertainty is currently a
no-op (`AmbiguityNet` has no `nn.Dropout` layers yet, only BatchNorm) —
tracked as a known gap, not hidden. APIs may change between minor versions
until `1.0`. See [`CHANGELOG.md`](CHANGELOG.md) for what's landed so far.

## Relationship to the wider EO stack

`pyunwrap` works standalone, but is designed to eventually sit downstream of
[`pygeofetch`](#) (interferogram acquisition/formation) and alongside
[`ps-gnn`](#) (persistent scatterer identification) in a broader open-source
InSAR processing stack.

## Citation

If `pyunwrap` is useful in your research, please cite it:

```bibtex
@software{pyunwrap2026,
  title  = {pyunwrap: Physics-Informed Deep Learning for InSAR Phase Unwrapping},
  author = {{pyunwrap contributors}},
  year   = {2026},
  url    = {https://github.com/yourusername/pyunwrap},
  note   = {Version 0.1.0}
}
```

## References

- Goldstein, R. M., Zebker, H. A., & Werner, C. L. (1988). Satellite radar
  interferometry: Two-dimensional phase unwrapping. *Radio Science*, 23(4).
- Itoh, K. (1982). Analysis of the phase unwrapping algorithm. *Applied
  Optics*, 21(14).
- Chen, C. W., & Zebker, H. A. (2001). Two-dimensional phase unwrapping with
  use of statistical models for cost functions in nonlinear optimization
  (SNAPHU). *JOSA A*, 18(2).
- Okada, Y. (1985). Surface deformation due to shear and tensile faults in a
  half-space. *Bulletin of the Seismological Society of America*, 75(4).
- Mogi, K. (1958). Relations between the eruptions of various volcanoes and
  the deformations of the ground surfaces around them. *Bulletin of the
  Earthquake Research Institute*, 36.

## Contributing

Contributions are welcome — bug reports, documentation, new deformation
models, or core improvements. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for
development setup, test/lint conventions, and the design principles worth
knowing before touching the physics-critical modules.

## License

[MIT](LICENSE)

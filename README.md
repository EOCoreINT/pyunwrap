<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/logo-horizontal-dark.svg">
  <img src="assets/logo-horizontal.svg" alt="pyunwrap" width="420">
</picture>

**A hybrid, regime-aware InSAR phase unwrapper — classical where classical is strong, learned where it isn't.**

[![PyPI](https://img.shields.io/pypi/v/pyunwrap-insar.svg)](https://pypi.org/project/pyunwrap-insar/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![Tests: 235 passing](https://img.shields.io/badge/tests-235%20passing-brightgreen)](tests/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Docs](https://readthedocs.org/projects/pyunwrap/badge/?version=latest)](https://pyunwrap.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22209219.svg)](https://doi.org/10.5281/zenodo.22209219)
[![Status: early-stage](https://img.shields.io/badge/status-early--stage-orange)](#project-status)

[Why this exists](#why-this-exists) ·
[How it works](#how-it-works) ·
[Hybrid unwrapping](#hybrid-unwrapping) ·
[Install](#installation) ·
[Quickstart](#quickstart) ·
[Notebooks](#notebooks) ·
[Architecture](#architecture) ·
[Docs](https://pyunwrap.readthedocs.io) ·
[Citation](#citation)

</div>

---

# ⚠️ Work In Progress (Pre-Alpha)
This repository is currently an active R&D playground as part of my pre-Master's roadmap for the Copernicus Master in Digital Earth. Architecture is shifting rapidly. 


`pyunwrap` unwraps Interferometric Synthetic Aperture Radar (InSAR) phase —
the core measurement behind satellite-based ground-deformation monitoring.
It is **not** a claim to replace classical unwrapping (SNAPHU) everywhere —
this project's own benchmarks (see [`docs/experiments.md`](docs/experiments.md))
show classical unwrapping winning on most scenes tested, and a learned
model winning reliably in exactly one regime: low-coherence decorrelation,
where classical statistical-cost unwrapping's core assumption breaks down
hardest. `pyunwrap`'s actual goal, reframed directly from that evidence, is
a **hybrid, regime-aware unwrapper**: route each region of a scene to
whichever method is actually good there, rather than asking one method to
be universally best.

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

## Hybrid unwrapping

[`pyunwrap.inference.hybrid.HybridUnwrapper`](pyunwrap/inference/hybrid.py)
routes each pixel to whichever method this project's own benchmarks show
winning there, rather than asking one method to win everywhere:

| Regime | Routed to | Why |
|---|---|---|
| High coherence, low gradient | Classical (SNAPHU, or a scikit-image fallback if SNAPHU isn't installed) | Mature, fast, and already reliable here |
| Low coherence / high decorrelation | Learned model | This project's own benchmarks show it winning specifically here |
| High coherence **and** high gradient (optional, via `high_gradient_threshold`) | Confidence-weighted blend | Steep gradients can violate the Nyquist assumption even at high coherence |

The merge never averages raw phase values directly — that would not, in
general, satisfy `wrap(merged_phase) == observed_phase`. Both methods'
results are first converted to their own integer ambiguity maps, blended
*there*, and the final phase is reconstructed as `wrapped_phase +
2π·round(k_merged)` — satisfying the wrap identity exactly, by
construction, regardless of how the two `k` maps were merged.

```python
from pyunwrap.inference.hybrid import HybridUnwrapper

hybrid = HybridUnwrapper(learned_unwrapper)  # auto-selects SNAPHU or a scikit-image fallback
result = hybrid.unwrap("wrapped.tif", "coherence.tif", "amplitude.tif")
result.save_geotiff("unwrapped.tif")
# result.regime_mask / .provenance_mask / .confidence_map / .uncertainty_map
# are also available, so you can see exactly which method produced which region.
```

Benchmarking follows the same principle:
[`pyunwrap.analytics.benchmark.generate_regime_report`](pyunwrap/analytics/benchmark.py)
explicitly reports which scenes classical wins, which the learned model
wins, which the hybrid router wins, and which are inconclusive — never a
single aggregate score that would hide which regime is actually driving
any claimed advantage.

## Key features

**Synthetic data engine** ([`pyunwrap.synthetic`](pyunwrap/synthetic/generator.py))
— Gaussian subsidence bowls, a from-scratch Okada (1985) rectangular fault
dislocation model, a Mogi (1958) volcanic point source, DEM-driven
topographic phase, Kolmogorov-spectrum atmospheric turbulence, orbital
ramps, coherence-dependent decorrelation noise, and a pseudo-real strategy
that rewraps real L-band (ALOS-2) unwrapped phase into simulated C-band data
to help bridge the sim-to-real gap.

**Real-data + synthetic-injection pipeline** ([`pyunwrap.data.real_injection`](pyunwrap/data/real_injection.py))
— rather than training exclusively on fully synthetic scenes, inject
synthetic deformation (Gaussian bowl, Mogi, Okada, orbital ramp, or a
nonlinear transient model) with an exactly-known ground truth into a real
(or real-like) wrapped-phase/amplitude/coherence stack, preserving real
speckle and decorrelation texture that a synthetic generator only
approximates. Spatially-blocked train/val/test splitting prevents pixel-
level leakage between splits, and outputs are cached by a hash of the
injection config.

**`AmbiguityNet`** ([`pyunwrap.models`](pyunwrap/models/ambiguity_net.py))
— a ResNet-34-encoder U-Net with a dual head: the integer ambiguity map via
a straight-through-estimator rounding layer, and an auxiliary residue-
probability map for uncertainty. Trained with a physics-informed loss
(ambiguity regression, re-wrap consistency, coherence-weighted smoothness,
ambiguity-map residue penalty, and an optional edge-preserving
`CoherenceWeightedPhaseSmoothnessLoss` that penalizes incoherent phase in
high-coherence regions without over-smoothing genuine deformation edges).

**Curriculum training with replay** ([`pyunwrap.training`](pyunwrap/training/trainer.py),
[`pyunwrap.training.curriculum`](pyunwrap/training/curriculum.py))
— the original 3-stage sequential curriculum (high-coherence/low-gradient
→ moderate → full difficulty) is joined by an optional **curriculum
replay** mode: every epoch draws a fixed-proportion mixture of
easy/moderate/hard tiles instead of a strict stage handoff, with automatic
forgetting detection that increases the easy-tile replay fraction if
easy-tier validation performance degrades. This exists because the
sequential curriculum was found, on a real training run, to cause
catastrophic forgetting of easy-regime performance once the hardest stage
took over every epoch — see [`docs/experiments.md`](docs/experiments.md).
Optional SNAPHU pseudo-ground-truth fine-tuning on real data is a
genuinely working feature, not just a code path: see
[`pyunwrap.utils.snaphu_integration`](pyunwrap/utils/snaphu_integration.py),
which runs real SNAPHU unwrapping and filters out any region SNAPHU itself
wasn't confident enough to trust as ground truth.

**Production inference** ([`pyunwrap.inference`](pyunwrap/inference/unwrapper.py))
— tiled processing of arbitrarily large interferograms with edge-aware,
residue-probability-weighted smart merging of the ambiguity map, Monte
Carlo Dropout uncertainty, and an ONNX Runtime → OpenVINO backend fallback
chain for deployment without a PyTorch dependency. Sits underneath the
[hybrid unwrapper](#hybrid-unwrapping) as the "learned" side of the route.

**Scientific analytics** ([`pyunwrap.analytics`](pyunwrap/analytics/))
— Goldstein-style residue detection and clustering, Nyquist
gradient-violation mapping, error-distribution statistics, Grad-CAM and
Integrated-Gradients explainability, uncertainty-calibration reliability
diagrams, and a fair, offset-corrected, **regime-separated**
[benchmark harness](pyunwrap/analytics/benchmark.py) against classical
SNAPHU unwrapping and the hybrid router, on data with exactly known
ground truth.

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

From PyPI (the package is distributed as `pyunwrap-insar`; the importable
module is still `pyunwrap`):

```bash
pip install pyunwrap-insar
```

```python
from pyunwrap.synthetic.generator import InSARSyntheticGenerator  # same import either way
```

From source, for development:

```bash
git clone https://github.com/EOCoreINT/pyunwrap.git
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
| `snaphu` | `snaphu` (official isce-framework bindings) | SNAPHU pseudo-ground-truth fine-tuning, benchmarking |
| `docs` | `sphinx`, `furo`, `myst-parser`, `nbsphinx`, ... | Building the [documentation site](https://pyunwrap.readthedocs.io) locally |

```bash
pip install -e ".[dev,maps,deploy,notebooks,snaphu]"   # everything, from source
pip install "pyunwrap-insar[maps,snaphu]"               # extras also work from PyPI, using the distribution name
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
  --use-curriculum-replay --smoothness-weight 0.1 \
  --out-dir runs/pyunwrap_v1
```

Or from a real (or real-like) data stack instead of pre-built HDF5 tiles,
via the [real-data injection pipeline](#hybrid-unwrapping):

```bash
pyunwrap-train --real-data-path ./my_real_stack/ --epochs 60 --out-dir runs/pyunwrap_v1
```

Hybrid inference from the command line:

```bash
pyunwrap-unwrap \
  --wrapped-phase wrapped.tif --coherence coherence.tif --amplitude amplitude.tif \
  --checkpoint model.pt --output unwrapped.tif
  # add --no-hybrid to use the learned model alone
```

## Notebooks

Three notebooks in [`notebooks/`](notebooks/) walk through the package
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
- [`03_training_bug_and_fix.ipynb`](notebooks/03_training_bug_and_fix.ipynb) —
  a genuine debugging record: a real LR-schedule bug found on a completed
  60-epoch curriculum run, the fix, a controlled before/after benchmark
  against SNAPHU, and an honest attempt to reproduce a second finding (the
  fix's catastrophic-forgetting side effect) that **does not cleanly
  replicate** at a different scale — reported as such, not smoothed over.
  See [`docs/experiments.md`](docs/experiments.md) for the original,
  larger-scale results this notebook partially reproduces, and the
  companion `resumable_train_v2.py` script it uses for chunked, resumable
  training on a single CPU core.

```bash
pip install -e ".[dev,maps,notebooks]"
jupyter notebook notebooks/
```

## Documentation

The full documentation site — installation, quickstart, the architecture
guide, both tutorial notebooks rendered with their real outputs, and the
complete API reference generated from source — is built with Sphinx and
hosted at **[pyunwrap.readthedocs.io](https://pyunwrap.readthedocs.io)**.

> [!NOTE]
> The Read the Docs badge/link above will show as unbuilt until the project
> is activated at readthedocs.org under whichever account hosts this repo —
> the `.readthedocs.yaml` config and full `docs/` source are ready to go as
> soon as that's done.

Build it locally:

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
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

## Project layout

```
pyunwrap/
├── synthetic/       # Physically-realistic InSAR data simulation
├── data/             # Preprocessing, tiling, PyTorch DataLoaders
├── models/           # Ambiguity-Net architecture + physics-informed losses
├── training/         # Curriculum-learning trainer, SNAPHU fine-tuning
├── inference/         # Tiled inference, smart merging, deployment
├── analytics/         # Residue stats, explainability, HTML reports
├── visualization/     # Interactive maps and 3D phase plots
└── utils/             # Shared helpers (deployment, caching, I/O)
notebooks/             # Pre-executed example notebooks
tests/                 # pytest suite (unit, model, integration, physics)
docs/                  # Architecture reference
assets/                # Logo and brand assets
```

## Project status

Early-stage and actively developed. **The project's goal is deliberately
not "beat SNAPHU everywhere"** — this project's own benchmarks don't
support that claim, and `pyunwrap.analytics.benchmark.generate_regime_report`
exists specifically to keep reporting honest about *where* each method
wins rather than collapsing results into one aggregate number. The actual
goal is the hybrid, regime-aware unwrapper described above: classical
where classical is strong, learned where this project's own evidence
shows it winning (low-coherence decorrelation).

The pipeline — synthetic data generation, real-data injection, curriculum
training (sequential or replay-based), tiled inference, the hybrid router,
analytics, and reporting — is implemented and tested end to end, and
includes a real SNAPHU integration (`pyunwrap.utils.snaphu_integration`)
for pseudo-ground-truth fine-tuning and a real Monte Carlo Dropout
uncertainty signal (`AmbiguityNet`'s `nn.Dropout2d` layers). A few things
worth knowing before you rely on this in production:

- **No pretrained weights ship yet.** `from_pretrained()` downloads from a
  Zenodo record you supply, and the example notebooks train small demo
  models from scratch rather than loading a benchmarked checkpoint. A real
  (if compute-modest) training run and a fair, offset-corrected benchmark
  against SNAPHU are documented in [`docs/experiments.md`](docs/experiments.md)
  — the honest result is that a lightly-trained model beats SNAPHU in
  exactly the low-coherence regime the hybrid framing targets, and loses
  on most other scenes, which is the evidence the hybrid design is based
  on rather than a shortcoming to hide.
- **No validation against real satellite (e.g. Sentinel-1) data yet.**
  Every result to date is against `pyunwrap`'s own synthetic generator (or,
  for `pyunwrap.data.real_injection`, a clearly-labeled real-*like* test
  fixture — see that module's docstring), not genuine SAR imagery.
- **The real-data injection pipeline has not been run against genuinely
  real data.** It is built and tested against a synthetic stand-in
  fixture; validating it against real Sentinel-1 stacks is real,
  unstarted future work, not something already done.

APIs may change between minor versions until `1.0`. See
[`CHANGELOG.md`](CHANGELOG.md) for what's landed so far.


## Relationship to the wider EO stack

`pyunwrap` works standalone, but is designed to eventually sit downstream of
[`pygeofetch`](#) (interferogram acquisition/formation) and alongside
[`ps-gnn`](#) (persistent scatterer identification) in a broader open-source
InSAR processing stack.

## Citation

If `pyunwrap` is useful in your research, please cite the archived release
via its Zenodo DOI:

```bibtex
@software{pyunwrap2026,
  title  = {pyunwrap: Physics-Informed Deep Learning for InSAR Phase Unwrapping},
  author = {Appiah Kubi, Samuel},
  year   = {2026},
  url    = {https://github.com/EOCoreINT/pyunwrap},
  doi    = {10.5281/zenodo.22209219},
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

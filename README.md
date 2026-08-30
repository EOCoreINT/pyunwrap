# pyunwrap

**Physics-informed deep learning for InSAR phase unwrapping.**

`pyunwrap` is a standalone, open-source Python package that unwraps Interferometric
Synthetic Aperture Radar (InSAR) phase using a **Physics-Informed U-Net ("Ambiguity-Net")**.
Unlike conventional deep-learning approaches that regress the continuous unwrapped phase
directly (and can therefore hallucinate physically inconsistent results), `pyunwrap` predicts
the discrete **integer ambiguity map** $k$ such that:

$$
\phi_{\text{unwrapped}} = \psi_{\text{wrapped}} + 2\pi k, \qquad k \in \mathbb{Z}
$$

Because the re-wrapping identity $\psi = \phi \mod 2\pi$ is enforced *by construction*, the
network cannot produce an output that disagrees with the observed wrapped phase — it can only
be wrong about *how many* $2\pi$ cycles were lost, not about the physics of wrapping itself.

This design specifically targets the two regimes where classical algorithms such as SNAPHU
struggle most:

- **Low-coherence areas** (vegetation, water, temporal decorrelation), where the phase
  gradient is noisy and standard branch-cut / minimum cost flow (MCF) methods propagate
  errors across large regions.
- **High deformation gradients** (earthquakes, volcanic inflation, mining subsidence),
  where the true phase gradient can locally exceed the Nyquist ($\pi$) sampling limit,
  violating the core assumption behind most unwrapping algorithms.

## Why predict $k$ instead of $\phi$?

| Approach | Failure mode |
|---|---|
| Regress $\phi$ directly | Network can output any real value; nothing prevents `wrap(pred) != observed_wrapped_phase` |
| Regress $k$ (this package) | `wrap(psi + 2*pi*round(k)) == psi` is guaranteed by definition; the network only has to get the *integer cycle count* right |

## Key features

- **Ambiguity-Net**: ResNet-34-encoder U-Net with a dual head — integer ambiguity map $k$
  and an auxiliary residue-probability (uncertainty) map.
- **Physics-informed loss**: combines ambiguity regression, hard re-wrapping consistency,
  coherence-weighted smoothness, and residue-count penalties.
- **Realistic synthetic data engine**: Gaussian subsidence bowls, Okada fault dislocation,
  Mogi volcanic point sources, DEM-driven topographic phase, Kolmogorov-spectrum atmospheric
  noise, orbital ramps, and coherence-dependent decorrelation noise — plus a pseudo-real data
  strategy that rewraps real L-band ALOS-2 unwrapped phase into simulated C-band data.
- **Production inference**: tiled processing of arbitrarily large interferograms with
  Hanning-windowed, uncertainty-weighted smart merging of the ambiguity map (never the phase
  directly), ONNX Runtime + OpenVINO fallback, and Monte Carlo Dropout uncertainty.
- **Scientific analytics**: residue detection, Nyquist gradient-violation flagging, error
  histograms, Grad-CAM / Integrated Gradients explainability, and uncertainty calibration.
- **Automated HTML reporting**: Jinja2-templated reports with embedded interactive Plotly 3D
  phase surfaces and Folium/Leafmap raster overlays.

## Installation

```bash
pip install -e ".[dev,maps]"
```

## Quickstart

```python
from pyunwrap.synthetic.generator import InSARSyntheticGenerator
from pyunwrap.models.ambiguity_net import AmbiguityNet
from pyunwrap.inference.unwrapper import PhaseUnwrapper

# 1. Generate a synthetic training pair
gen = InSARSyntheticGenerator(size=256, seed=42)
sample = gen.generate_sample(deformation_type="mogi")

# 2. Load a trained model and run inference on a real interferogram
unwrapper = PhaseUnwrapper.from_pretrained("pyunwrap-v1")
result = unwrapper.unwrap(
    wrapped_phase_path="data/wrapped_phase.tif",
    coherence_path="data/coherence.tif",
    generate_report=True,
)
```

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
tests/                 # pytest suite (unit, model, integration, physics)
docs/                  # Documentation (see docs/architecture.md for a full module-by-module overview)
```

See [`docs/architecture.md`](docs/architecture.md) for a diagram of how data
flows through the pipeline and a table of what each module is responsible
for. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for development setup and
[`CHANGELOG.md`](CHANGELOG.md) for release history.

## Relationship to the wider EO stack

`pyunwrap` is designed to be used standalone, but is intended to eventually sit downstream
of [`pygeofetch`](#) (interferogram acquisition/formation) and alongside
[`ps-gnn`](#) (persistent scatterer identification) in a broader open-source InSAR
processing stack.

## Status

Early-stage / actively developed. APIs may change between minor versions until `1.0`.

## License

MIT

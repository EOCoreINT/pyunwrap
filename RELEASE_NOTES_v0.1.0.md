# pyunwrap v0.1.0 — Initial Release

`pyunwrap` unwraps InSAR phase using a physics-informed U-Net that predicts
the **integer ambiguity map** rather than the unwrapped phase itself. The
core identity — `unwrapped = wrapped + 2π·k` — is enforced by construction,
so the model is structurally incapable of contradicting the observed
wrapped phase. This is the first tagged release: the full pipeline from
synthetic data through training, tiled inference, analytics, and reporting
is implemented, tested, and documented end to end.

## Highlights

- 🌀 **`AmbiguityNet`** — a ResNet-34 U-Net predicting integer phase
  ambiguity via a straight-through-estimator rounding head, with an
  auxiliary residue-probability head for uncertainty.
- 🧪 **Realistic synthetic data** — Gaussian bowls, a from-scratch Okada
  (1985) fault dislocation model, a Mogi (1958) volcanic source, DEM
  topography, Kolmogorov atmospheric noise, orbital ramps, decorrelation
  noise, and pseudo-real ALOS-2 rewrapping.
- 🎓 **Curriculum training** with optional SNAPHU pseudo-ground-truth
  fine-tuning on real data.
- 🧩 **Production tiled inference** with edge-aware, residue-weighted smart
  merging of the ambiguity map — never the phase directly — plus Monte
  Carlo Dropout uncertainty and an ONNX Runtime → OpenVINO fallback chain.
- 📊 **Full analytics suite** — residue detection, Nyquist gradient
  analysis, Grad-CAM, Integrated Gradients, uncertainty calibration, and an
  automated self-contained HTML report.
- ✅ **65 tests**, including known-answer physics tests and a tile-merging
  regression test that catches boundary-artifact bugs numerically, not
  visually.
- 📓 Two **pre-executed** example notebooks covering the training chain and
  the full end-to-end pipeline against real synthetic data.

## What's included

### Synthetic data engine
Four deformation models (Gaussian bowl, Okada, Mogi, deformation-free
control), DEM-driven topographic phase, Kolmogorov-spectrum atmospheric
turbulence, orbital ramps, and coherence-dependent decorrelation noise.
Ground-truth ambiguity is computed against the *actual observed* (noisy)
phase — unwrapping resolves `2π` ambiguity, it does not denoise — which
keeps every generated sample exactly consistent with
`wrapped + 2π·k == unwrapped` to floating-point precision.

### Data pipeline
Full-coverage sliding-window tiling, percentile-robust amplitude
normalization, and an `InSARTileDataset` with 8-fold dihedral augmentation
that recomputes the integer ambiguity map after every transform rather than
reusing a cached value, so it stays exact even at the `±π` wrap boundary.

### Model & training
`AmbiguityNet` (24M params) plus `PhysicsInformedUnwrapLoss`, a
four-component loss combining supervised ambiguity regression, re-wrap
consistency, coherence-weighted smoothness, and an ambiguity-map residue
penalty. `Trainer` implements three-stage curriculum learning, AdamW with
warmup/cosine annealing, gradient clipping, TensorBoard logging, and
optional SNAPHU-based fine-tuning.

### Inference & deployment
`PhaseUnwrapper` tiles arbitrarily large interferograms and merges results
with an edge-aware taper (only feathering edges that actually border a
neighboring tile — a plain Hanning window incorrectly zeroes scene
boundaries too) weighted by the model's own residue-probability output.
ONNX export is single-file by design, with GPU → CPU → OpenVINO backend
fallback.

### Analytics, visualization & reporting
Goldstein-style residue detection and clustering, Nyquist gradient-
violation mapping, error-distribution statistics, Grad-CAM and Integrated
Gradients explainability, uncertainty-calibration reliability diagrams,
interactive Plotly 3D surfaces, a folium swipe-comparison map, and a
Jinja2-templated HTML report — all wrapped in best-effort error handling so
a reporting failure never breaks a production inference call.

### Testing & CI
A 65-test suite (`pytest`) covering synthetic generation, model/loss
correctness, physics/analytics correctness against hand-constructed
known-answer fields (e.g. an exact phase vortex with a known topological
charge), and full-pipeline integration. GitHub Actions runs a fast-test
matrix across Python 3.10–3.12, a separate slow/integration job, and a
`ruff`/`black` lint job.

## Installation

```bash
git clone https://github.com/EOCoreINT/pyunwrap.git
cd pyunwrap
pip install -e ".[dev,maps,deploy,notebooks]"
```

See the [README](README.md) for the quickstart and the
[architecture reference](docs/architecture.md) for how data flows through
the pipeline.

## Known limitations

This is an early-stage release and is honest about where it currently
falls short:

- **No pretrained weights ship with this release.** `PhaseUnwrapper.from_pretrained()`
  downloads from a Zenodo record you supply; there is no benchmarked
  checkpoint yet.
- **Monte Carlo Dropout uncertainty is currently a no-op.** `AmbiguityNet`
  has no `nn.Dropout` layers yet (only BatchNorm), so `uncertainty` in
  `UnwrapResult` is identically zero. The residue-probability head is the
  meaningful uncertainty signal available today.
- **No published accuracy benchmark against SNAPHU or other classical
  unwrappers yet.** The physics-consistency guarantees are proven
  (exhaustively, in tests); real-world unwrapping *accuracy* on held-out
  Sentinel-1 data has not yet been formally evaluated in this repo.
- APIs may change between minor versions until `1.0`.

## Acknowledgments

Built on the shoulders of the classical InSAR literature this package's
synthetic models and design decisions are grounded in: Goldstein, Zebker &
Werner (1988); Itoh (1982); Chen & Zebker (2001, SNAPHU); Okada (1985);
Mogi (1958). Full citations in the [README](README.md#references).

## Full Changelog

See [`CHANGELOG.md`](CHANGELOG.md) for the complete, itemized history,
including every bug found and fixed during development.

**Full diff**: https://github.com/EOCoreINT/pyunwrap/commits/v0.1.0

# Changelog

All notable changes to `pyunwrap` are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project intends to adhere to [Semantic Versioning](https://semver.org/)
once it reaches `1.0`.

## [Unreleased]

## [0.1.0] - Initial release

### Added
- **Package foundation**: standard `pyunwrap/` layout (`synthetic`, `data`,
  `models`, `training`, `inference`, `analytics`, `visualization`, `utils`),
  `pyproject.toml`, `README.md`.
- **Synthetic data generator** (`pyunwrap.synthetic.generator`): Gaussian
  subsidence bowls, a from-scratch Okada (1985) rectangular dislocation
  model, a Mogi (1958) point-source model, DEM-driven topographic phase,
  Kolmogorov-spectrum atmospheric noise, orbital ramps, coherence-dependent
  decorrelation noise, and a pseudo-real L-band (ALOS-2) -> C-band rewrapping
  strategy for bridging the sim-to-real domain gap.
- **Preprocessing & tiling** (`pyunwrap.data`): input normalization, a
  full-coverage sliding-window tiler, and HDF5/`.npz` tile persistence.
- **`InSARTileDataset`** (`pyunwrap.data.dataloader`): a PyTorch `Dataset`
  with 8-fold dihedral augmentation that recomputes the integer ambiguity map
  after each transform, keeping it exactly consistent with the augmented
  wrapped/unwrapped pair.
- **`AmbiguityNet`** (`pyunwrap.models.ambiguity_net`): a ResNet-34-encoder
  U-Net predicting the discrete integer phase-ambiguity map via a
  straight-through-estimator rounding head, plus an auxiliary residue-
  probability (uncertainty) head.
- **`PhysicsInformedUnwrapLoss`** (`pyunwrap.models.losses`): a 4-component
  loss (ambiguity MSE, re-wrap consistency, coherence-weighted smoothness,
  ambiguity-map residue/Laplacian penalty), with an explicit module-level
  explanation of which components are architecturally near-trivial by
  construction and why.
- **`Trainer`** (`pyunwrap.training.trainer`): 3-stage curriculum learning,
  SNAPHU pseudo-ground-truth fine-tuning, AdamW + warmup/cosine LR
  scheduling, gradient clipping, TensorBoard logging, and a `Visualizer` for
  training curves and phase comparison plots.
- **`PhaseUnwrapper`** (`pyunwrap.inference.unwrapper`): tiled inference over
  arbitrarily large interferograms with Hanning/residue-probability-weighted
  smart merging of the integer ambiguity map (never the phase directly),
  Monte Carlo Dropout uncertainty, and ONNX/OpenVINO backend support.
- **Deployment utilities** (`pyunwrap.utils.deployment`): ONNX export with
  optional FP16 quantization, a Zenodo-backed local weight cache, and a
  GPU -> CPU -> OpenVINO inference-backend fallback chain.
- **Analytics** (`pyunwrap.analytics`): Goldstein-style residue detection and
  clustering, Nyquist gradient-violation flagging, error-distribution
  analysis, Grad-CAM and Integrated-Gradients explainability, and
  uncertainty-calibration reliability diagrams.
- **Visualization & reporting** (`pyunwrap.visualization`,
  `pyunwrap.analytics.report_generator`): interactive Plotly 3D phase
  surfaces and heatmaps, a folium swipe-comparison map, and a Jinja2-
  templated, self-contained HTML report tying it all together.
- **Test suite** (`tests/`): 65 tests across synthetic generation, model/loss
  correctness, physics/analytics correctness (validated against
  hand-constructed known-answer fields, e.g. an exact phase vortex), and
  full-pipeline integration -- including a regression test that
  tiled-and-merged inference is numerically identical to a whole-image pass,
  directly targeting the tile-boundary artifact class of bug.
- **CI** (`.github/workflows/tests.yml`): fast-test matrix across Python
  3.10-3.12, a separate slow/integration job, and a `ruff`/`black` lint job.

### Fixed during development (see commit history / build log for details)
- Synthetic generator: ground-truth ambiguity is now computed against the
  actual *noisy* observed phase (what gets wrapped), not the noise-free
  signal -- the previous definition made the "ground truth" integer
  ambiguity map non-integer whenever decorrelation noise was present.
- `Trainer`'s `Visualizer`: fixed a length-mismatch crash caused by
  assuming every metric is logged every epoch (validation metrics are
  logged less frequently than training metrics).
- `Trainer`'s curriculum "easy" tile filter: switched from a hard max
  gradient threshold to a 99th-percentile threshold, since the ground-truth
  unwrapped phase legitimately contains spatially-uncorrelated decorrelation
  noise that otherwise misclassified nearly every tile as "hard."
- ONNX export: forced single-file output (`external_data=False`); the
  default multi-file (`.onnx` + `.onnx.data`) output was incompatible with
  the single-artifact-per-model assumption in `ModelCache`.
- Tile merging: replaced a plain Hanning window (which tapers to exactly
  zero at *every* tile edge, including the outer boundary of the whole
  scene) with an edge-aware taper that only feathers edges bordering an
  actual neighboring tile, fixing a bug that collapsed scene-border pixels
  toward zero.
- ONNX export: added an explicit `tile_size % 32 == 0` validation in
  `PhaseUnwrapper.unwrap`, since the exported graph's shape-mismatch safety
  net is a data-dependent branch that isn't captured for tile sizes not
  divisible by the encoder's stride.
- Report generator: the  "Training history" section's chart asset was being
  generated but never actually embedded in the report template.

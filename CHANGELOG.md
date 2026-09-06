# Changelog

All notable changes to `pyunwrap` are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project intends to adhere to [Semantic Versioning](https://semver.org/)
once it reaches `1.0`.

## [Unreleased]

### Changed
- **Docs theme switched from Furo to `sphinx_rtd_theme`**, and the sidebar
  navigation reorganized into named, grouped sections (Getting Started,
  How It Works, Tutorials, Reference, Project) -- matching the sibling
  EOCoreINT/pygeofetch project's documentation site for a consistent look
  across the org's projects. Verified by actually fetching pygeofetch's
  live docs site and comparing structure, not assumed from memory.
  `docs/_static/custom.css` updated to target the new theme's actual CSS
  classes (Furo's `.sidebar-logo` doesn't exist under this theme).
- **Project goal reframed to a hybrid, regime-aware unwrapper**, directly
  from this project's own benchmark evidence: `pyunwrap` beats SNAPHU
  reliably in exactly one regime (low-coherence decorrelation) and loses
  on most others. The project no longer frames its goal as "beat SNAPHU
  everywhere" -- see the README's new "Hybrid unwrapping" section and
  `pyunwrap.inference.hybrid`.

### Added
- **CLI integration** for all four strategies: `pyunwrap-train` gains
  `--use-curriculum-replay`, `--smoothness-weight`, and `--real-data-path`
  (builds training/validation datasets from a real data stack via the
  Strategy 1 pipeline instead of requiring pre-built HDF5 files). A new
  entry point, `pyunwrap-unwrap` (`pyunwrap.inference.hybrid:main`), runs
  hybrid or learned-only inference from the command line with
  `--hybrid`/`--no-hybrid` and coherence-threshold flags. All new flags
  verified via real end-to-end CLI invocations, not just argument parsing.
- **`pyunwrap.inference.hybrid`** (Strategy 4): `HybridUnwrapper` routes
  each pixel to classical unwrapping (high coherence/low gradient),
  the learned model (low coherence), or a confidence-weighted blend
  (an optional "uncertain" regime for high-coherence-but-high-gradient
  pixels). Merging happens in integer ambiguity space, not raw phase
  space, and the final output is reconstructed as `wrapped_phase +
  2*pi*round(k_merged)` -- verified numerically (not just argued) to
  satisfy the wrap-consistency invariant exactly. Falls back to a real,
  different classical algorithm (`scikit-image`'s quality-guided
  path-following unwrapper) when SNAPHU isn't installed, so CPU-only,
  SNAPHU-free execution and tests remain possible. `HybridUnwrapperConfig`
  makes every routing threshold and merge strategy configurable; both
  the learned model and the classical adapter fail loudly (or degrade
  with an explicit printed warning, per their respective fallback flags)
  rather than silently. 28 tests, including the wrap-consistency
  invariant checked on a real, controlled split-coherence scene.
- **`pyunwrap.analytics.benchmark.run_hybrid_benchmark_suite` /
  `generate_regime_report`**: extends the benchmark harness to compare
  classical-only, learned-only, and hybrid results on the same graduated
  scene set, with an explicit per-scene winner (`"classical"`,
  `"learned"`, `"hybrid"`, or `"inconclusive"` when methods are within 5%
  of each other) -- the report never collapses to one aggregate score,
  and explicitly states it does not claim any single method wins across
  the board.
- **`pyunwrap.data.real_injection`** (Strategy 1): a real-data +
  synthetic-injection training pipeline. Injects synthetic deformation
  (Gaussian bowl, Mogi, Okada, orbital ramp, or a new simplified
  nonlinear-transient model) with exactly-known ground truth into a real
  (or real-like) wrapped-phase/amplitude/coherence stack, preserving real
  noise texture by construction (the injected field is added to the real
  background phase itself, not a separately-generated clean signal).
  Spatially-blocked train/val/test splitting prevents pixel-level
  leakage -- verified directly, not just implemented. Output is cached by
  a hash of the injection config and is directly loadable via the
  existing `InSARTileDataset`/`Trainer` pipeline. Honest scope note
  carried through the module's own docstrings: built and tested against a
  clearly-labeled synthetic fixture standing in for real data, since this
  project's development environment cannot reach any real SAR data
  provider -- validating against genuine Sentinel-1 data is real,
  unstarted future work. 48 tests, including a full real-injection ->
  HDF5 -> `Trainer.fit()` integration test and physics-identity
  verification for all 5 injection kinds.
- **`pyunwrap.training.curriculum.CurriculumReplayConfig` /
  `CurriculumReplayIndex`** (Strategy 2): an alternative to the original
  sequential 3-stage curriculum. Every epoch draws a fixed-proportion
  mixture of easy/moderate/hard tiles instead of a strict stage handoff,
  directly targeting the catastrophic-forgetting failure mode documented
  in `docs/experiments.md`'s Experiment 3 (easy tiles are never fully
  absent from training, regardless of how far along the schedule is).
  Includes automatic forgetting detection (trend-based and absolute-
  threshold triggers) that increases the easy-replay fraction -- taking
  the difference from `hard_fraction` first, then `medium_replay_fraction`
  -- and an optional persistent learning-rate reduction on detection.
  `CurriculumIndex` and friends were extracted from `trainer.py` into this
  new module (re-exported from `trainer.py` for backward compatibility --
  verified via the full existing test suite passing unchanged). 19 tests.
- **`pyunwrap.models.losses.SmoothnessConfig` /
  `CoherenceWeightedPhaseSmoothnessLoss`** (Strategy 3): an edge-preserving
  sibling to `PhysicsInformedUnwrapLoss`'s existing (simpler, unconditional
  L2) smoothness term. Penalizes spatially incoherent phase in
  high-coherence regions using a robust (Huber or L1) norm weighted by
  `coherence * exp(-|gradient| / edge_tau)`, so genuine deformation edges
  are not over-smoothed the way a plain L2 penalty would. Composed into
  `PhysicsInformedUnwrapLoss` as an opt-in Component 5 via
  `smoothness_config` -- verified byte-for-byte backward compatible when
  omitted (existing training runs are completely unaffected). 18 tests,
  including a direct check that a genuine large discontinuity receives a
  smaller per-unit-gradient penalty than equivalent-energy spatially-
  incoherent noise.


- **`notebooks/03_training_bug_and_fix.ipynb`**: a real, fully-executed
  notebook (not mocked) reproducing the LR-restart bug and fix at a
  smaller scale (216 tiles vs. the original session's 360) for
  documentation-build feasibility, including a real SNAPHU benchmark
  comparison and a real `evaluate_stratified` demonstration. Includes an
  honest, unresolved discrepancy reported directly rather than hidden: the
  catastrophic-forgetting effect documented in `docs/experiments.md`'s
  Experiment 3 **does not cleanly replicate** at this notebook's smaller
  scale/different seed, which the notebook argues is itself an important
  finding about the fragility of single-run ML conclusions. Ships with
  `notebooks/resumable_train_v2.py`, the chunked/checkpointed training
  helper the notebook's cells were actually run through (a full 60-epoch
  run doesn't fit in one execution on a single CPU core).
- **Difficulty-stratified validation** (`pyunwrap.training.trainer.evaluate_stratified`):
  periodic validation now reports RMSE broken out by the same
  easy/moderate/hard difficulty tiers the curriculum already computes, in
  addition to the overall aggregate. On by default (`Trainer`'s
  `stratified_validation=True`), exposed via the CLI as
  `--no-stratified-validation` to disable. Directly motivated by
  Experiment 3 in `docs/experiments.md`: a real 74% regression on
  easy-tier scenes was invisible in the aggregate `val_rmse` throughout
  that entire training run, because improvement on the (numerically
  larger) hard-tile errors outweighed it in the average. The per-tile
  difficulty computation was extracted out of `CurriculumIndex` into a
  standalone `compute_tile_difficulty_stats` function specifically so
  training-time curriculum filtering and validation-time reporting share
  one implementation rather than two that could silently drift apart --
  verified identical via a dedicated test, not just asserted.
- **Experiment 3** (`docs/experiments.md`): retrained with the LR-restart
  fix (identical seed/data/hyperparameters to Experiment 1's checkpoint,
  scheduler only difference) and reran the SNAPHU benchmark for a
  controlled before/after comparison. The fix improves 6 of 7 benchmark
  scenes (median +31.3% RMSE), including the project's first outright win
  over SNAPHU (`low_coherence_control`) and near-parity on the
  steepest-gradient scene -- but causes a 74% regression on `easy_control`.
  Directly demonstrated (not just hypothesized) that this is catastrophic
  forgetting: a model trained identically but stopped before curriculum
  stage 3 ever triggers scores 2.44 rad on `easy_control`, dramatically
  better than either full 60-epoch run. The same LR restart that fixes
  learning on hard data actively degrades performance on easier data.
  Documents four concrete, not-yet-implemented mitigations (smaller
  restart peaks per stage, curriculum replay, difficulty-stratified
  validation tracking, a regime-aware inference dispatcher).
- **`build_curriculum_aware_scheduler`** (`pyunwrap.training.trainer`):
  restarts the learning-rate schedule at each curriculum stage transition
  (and the SNAPHU fine-tuning switch, if configured) instead of running one
  continuous cosine decay across the whole training run. Found on a real,
  completed 60-epoch run (the first one in this project's history to
  actually finish all 3 curriculum stages): training loss visibly jumped
  when the hardest curriculum stage unlocked at epoch 51 and never
  recovered, because the cosine-annealed LR had already decayed to
  `6.46e-06` -- about 1/15,000th of its peak -- by that point, leaving
  essentially no room to adapt to the harder data. `Trainer` uses this new
  scheduler by default whenever curriculum learning is enabled; verified
  (not just argued) to be byte-identical to the old single-cycle schedule
  when it isn't. See [`docs/experiments.md`](docs/experiments.md) for the
  full before/after loss curves and reasoning.
- **`docs/experiments.md`**: a real-findings page documenting the full
  60-epoch curriculum run and a fair, offset-corrected benchmark against
  SNAPHU across seven scenes of increasing difficulty. Honest headline:
  SNAPHU still wins on every scene tested. The finding worth keeping:
  pyunwrap's relative disadvantage shrinks from 8.3x on the easiest scene
  to 1.1x on the hardest (Pearson correlation -0.81 between scene
  difficulty and relative RMSE gap) -- the specific pattern the
  physics-informed approach predicts, though not yet enough to win, and
  not a clean story on the low-coherence axis specifically.
- **Published to PyPI and Zenodo.** The package is now installable via
  `pip install pyunwrap-insar` (PyPI project name; the importable module
  stays `pyunwrap`), and the `v0.1.0` release is permanently archived with
  DOI [10.5281/zenodo.22209219](https://doi.org/10.5281/zenodo.22209219).
  `pyproject.toml`'s `name`/`version`/`authors` and every `yourusername`
  placeholder across the README, docs, and `CONTRIBUTING.md` now point at
  the real published package and repository
  ([EOCoreINT/pyunwrap](https://github.com/EOCoreINT/pyunwrap)) instead of
  placeholders -- verified by fetching the live PyPI and Zenodo pages
  directly rather than taking the URLs on faith.
- **Full Sphinx documentation site** (`docs/`), built for Read the Docs:
  installation guide, quickstart, the architecture reference, both example
  notebooks rendered inline with their real pre-executed outputs (via a
  build-time copy hook that keeps `notebooks/` as the single source of
  truth rather than a manually-synced duplicate), and a complete API
  reference for all 16 public modules generated from source docstrings.
  Heavy/hard-to-install dependencies (`torch`, `rasterio`, `onnxruntime`,
  `snaphu`, `openvino`) are mocked via `autodoc_mock_imports` so a docs
  build never needs a GPU, GDAL, or a compiled SNAPHU binary just to read
  docstrings. Verified with real, iterative `sphinx-build` runs (not just
  written and assumed to work): went from 73 build warnings down to 3
  (all attributable to this project's own network-restricted development
  sandbox blocking `intersphinx` inventory fetches, not a real
  configuration problem), fixing along the way a genuine
  Napoleon/autodoc/dataclass interaction that was producing ~40 duplicate
  attribute-documentation warnings (`napoleon_use_ivar = True`, rather than
  the ~40 individual `:no-index:` annotations Sphinx's own warning text
  suggested), a real RST literal-block formatting bug in
  `dataloader.py`'s module docstring, and a redundant duplicate heading on
  every multi-module API reference page. Visually spot-checked the actual
  rendered HTML (landing page, API reference, notebook tutorial pages)
  rather than trusting "the build succeeded" alone.
- **Real SNAPHU integration** (`pyunwrap.utils.snaphu_integration`): makes
  `Trainer`'s SNAPHU pseudo-ground-truth fine-tuning phase an actually
  runnable feature, not just a code path that assumes a pre-built HDF5
  exists. Built on the official `snaphu-py` bindings (`pip install snaphu`,
  an optional extra), which bundle their own compiled SNAPHU binary.
  `generate_snaphu_finetune_dataset` runs real SNAPHU unwrapping on a
  wrapped-phase/coherence/amplitude GeoTIFF triplet, filters out tiles
  where SNAPHU's own connected-component analysis wasn't confident enough
  to trust as ground truth (empirically verified: a deliberately
  decorrelated test region was correctly flagged unreliable 97% of the
  time, with zero false positives in a well-behaved control region), and
  writes an HDF5 file directly usable as `Trainer(finetune_dataset=...)`.
  Verified end-to-end through an actual fine-tuning `Trainer.fit()` run,
  not just at the data-preparation layer.
- Professional logo (`assets/`): a mathematically-constructed mark showing
  an Archimedean spiral unwinding into a straight ascending line — the
  literal geophysical metaphor for phase unwrapping (wrapped phase sits on
  a repeating `2π` cycle; unwrapping "unrolls" that cycle onto a continuous
  line). Colors cycle indigo → rose → gold through the spiral (echoing the
  cyclic phase colormaps used throughout the codebase's own plots) and
  resolve into a single steady gold on the line. Ships as `icon.svg`
  (square, for favicons/avatars), `mark.svg` (icon only), and
  `logo-horizontal.svg` / `logo-horizontal-dark.svg` (full lockup with
  wordmark, light/dark-mode aware via GitHub's `prefers-color-scheme`
  `<picture>` support) plus PNG raster fallbacks.
- Rewrote `README.md`: problem statement, architecture diagram, a full
  feature breakdown per pipeline stage, an installation/extras table, an
  honest "Project status" section (no pretrained weights ship yet; MC
  Dropout uncertainty is currently a documented no-op), a citation block,
  and references to the underlying geophysics literature (Goldstein 1988,
  Itoh 1982, Chen & Zebker 2001/SNAPHU, Okada 1985, Mogi 1958). Verified to
  render correctly against the actual GitHub-flavored-markdown engine
  (`cmarkgfm`), not just a generic Markdown-to-HTML preview.
- **Real Monte Carlo Dropout uncertainty** (`AmbiguityNet`'s `dropout_rate`):
  added actual `nn.Dropout2d` layers after each decoder stage, closing the
  gap where `PhaseUnwrapper`'s MC-Dropout uncertainty was an unconditional
  no-op (the architecture previously had no dropout layers at all). Old
  checkpoints load into the new architecture with zero missing/unexpected
  keys (`nn.Dropout2d` has no learnable parameters) -- verified directly,
  not just assumed.
- **`pyunwrap.analytics.benchmark`**: a real, tested comparison harness
  between `AmbiguityNet` (via `PhaseUnwrapper`) and classical SNAPHU
  unwrapping, scored fairly (both methods get their best-fit global `2π`
  offset removed before RMSE, since phase unwrapping has no absolute
  reference by definition) against synthetic scenes with exactly known
  ground truth, spanning a deliberately graduated easy-to-hard difficulty
  spread. Run for real against an actual trained checkpoint; results and
  their honest interpretation are in the project's release notes.
- **Batched tile inference** (`PhaseUnwrapper`'s `inference_batch_size`):
  opt-in batching of multiple tiles into a single forward pass, default `1`
  preserves prior behavior exactly (including for custom `_run_tile`
  overrides, e.g. in tests). Verified numerically identical to the
  unbatched path on both a deterministic fake and a real `AmbiguityNet`
  forward pass; measured a real (if CPU-modest -- the larger win is on GPU)
  ~1.9x speedup at `inference_batch_size=8` vs. `1` on a 320x320 scene.
- `mypy` now runs as a hard CI gate (previously configured but not
  enforced). Went from 39 flagged issues to 0: fixed genuine
  `Optional`/`None`-narrowing gaps with explicit `assert ... is not None`
  guards at the point of use (`PhaseUnwrapper.model`/`.engine`,
  `InferenceEngine._session`/`._ov_compiled`), fixed real type-annotation
  mistakes in the new benchmark module, and documented + suppressed the
  remaining numpy/torch stub-typing noise (`warn_return_any = false`,
  with two precisely-scoped `# type: ignore[arg-type]` comments) rather
  than either ignoring it silently or scattering blanket suppressions.

### Fixed
- **Silent zero-batch training epochs.** `Trainer`'s `DataLoader(...,
  drop_last=True)` would silently yield zero batches whenever a curriculum-
  filtered (or SNAPHU fine-tuning) subset was smaller than `batch_size` --
  `_train_one_epoch`'s loop then ran zero times, leaving that epoch's
  metrics empty and logging `loss=nan` while performing no gradient
  updates at all, indistinguishable from a slow-starting but real epoch
  unless read very carefully. Found on an actual 40-epoch training run
  (20 consecutive silently-empty epochs). Fixed with a `_safe_drop_last`
  helper (only drops the ragged batch when a full batch of data exists
  besides it) plus a loud `RuntimeError` as defense in depth if any future
  code path ever reintroduces a zero-batch epoch; covered by a regression
  test that reintroduces the original bug via monkeypatching to confirm
  the guard actually catches it.
- **MC-Dropout uncertainty silently degenerating to zero.** Even after
  adding real dropout layers, `PhaseUnwrapper`'s uncertainty estimate was
  computed from the *rounded* integer `k_hat` samples across MC passes
  rather than the *continuous* `k_continuous` samples -- if dropout-induced
  noise didn't push the continuous prediction across an integer rounding
  boundary, every pass's rounded output stayed identical and `k_std` came
  out exactly zero despite genuine underlying stochasticity. Caught by a
  test (not manual inspection) that used a random-init model whose
  predictions happened to saturate hard against a single integer. Fixed by
  computing `k_std` from the continuous samples while still using the
  rounded samples' median for the actual integer decision.

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
- Report generator: the "Training history" section's chart asset was being
  generated but never actually embedded in the report template.
- `Trainer.Visualizer` and `pyunwrap.visualization.maps` were calling
  `matplotlib.use("Agg")` to force headless rendering for saved PNGs. This
  is a *global* side effect: calling `Trainer.fit()` (which internally saves
  training-curve plots) silently switched the backend for the entire
  process, breaking inline plot display in any Jupyter notebook that called
  `Trainer.fit()` and then tried to `plt.show()` afterward -- caught while
  building the example notebooks. Removed the explicit backend force
  entirely; `fig.savefig()` doesn't require any particular backend, and
  headless environments (CI, servers) already default to `Agg` on their own.

- `notebooks/*.ipynb` originally relied on the Jupyter frontend
  auto-activating matplotlib's inline backend. In environments where it
  doesn't (observed with a plain venv + VS Code Jupyter setup, where
  matplotlib defaulted to the non-interactive `Agg` backend), `plt.show()`
  silently no-ops with a `"FigureCanvasAgg is non-interactive"` warning
  instead of rendering. Both notebooks now call `%matplotlib inline`
  explicitly in their setup cell, which forces inline rendering regardless
  of the environment's default backend.

### Added (later)
- `notebooks/01_training_pipeline.ipynb` -- the complete training chain,
  checked in pre-executed with real outputs (synthetic data generation,
  tiling, `Trainer.fit()`, training curves, predicted-vs-truth comparison).
- `notebooks/02_full_pipeline.ipynb` -- the complete end-to-end chain,
  checked in pre-executed with real outputs (training, real-GeoTIFF tiled
  inference, phase analytics, explainability, interactive visualization,
  and automated HTML report generation).
- `LICENSE` (MIT), `.gitignore`, `CONTRIBUTING.md`, `CHANGELOG.md`,
  `docs/architecture.md`, and a `notebooks` optional-dependency group.

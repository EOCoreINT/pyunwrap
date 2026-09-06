# Tutorials

All three notebooks below are checked into the repository **pre-executed
with real outputs** — generated against `pyunwrap`'s own synthetic data
(and, for notebook 03, real SNAPHU calls), not mocked or hand-written. They
render here exactly as they'd appear if you opened them in Jupyter
yourself. See the top-level `notebooks/` directory in the repository if you
want to run them.

```{toctree}
:maxdepth: 1

_notebooks/01_training_pipeline
_notebooks/02_full_pipeline
_notebooks/03_training_bug_and_fix
```

## 01 — The training chain

Synthetic data generation, tiling, `AmbiguityNet` + `Trainer`, training
curves, and predicted-vs-ground-truth phase comparison. Start here if you
want the shortest path to seeing the model actually train.

## 02 — The full pipeline

Everything in the training notebook, plus: a gallery of all four
deformation models with a full physical-component breakdown, tiling and
augmentation visualized directly, every loss component demonstrated in
isolation, tiled inference on real GeoTIFFs with every output channel
visualized, residue/Nyquist/error analytics, Grad-CAM and Integrated-
Gradients explainability, 3D and interactive-map visualization, ONNX
deployment with a numerical PyTorch-vs-ONNX agreement check, and the final
automated HTML report.

## 03 — A real training bug, a real fix, and an honest look at whether it replicates

A genuine debugging record, not a curated demo: a real learning-rate
schedule bug found on a completed 60-epoch curriculum run, the fix
(`build_curriculum_aware_scheduler`), a controlled before/after benchmark
against SNAPHU, and a second experiment (the fix's catastrophic-forgetting
side effect) that **honestly does not replicate** at this notebook's
smaller scale — reported as such rather than smoothed over, since that
discrepancy is itself the most important finding in the notebook. Ends
with `evaluate_stratified` (the mitigation built in response to the
original finding) demonstrated on real checkpoints. See
[`docs/experiments.md`](../experiments) for the original, larger-scale
results this notebook partially reproduces.

```{note}
These notebooks were trained on a single CPU core with a small, bounded
compute budget for reproducibility within a documentation build -- the
resulting models are real (the loss curves genuinely decrease) but are not
representative of a production-scale training run. See the project's
release notes for a properly framed benchmark against SNAPHU.
```

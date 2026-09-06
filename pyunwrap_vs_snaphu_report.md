# Why `pyunwrap` Does Not Yet Beat SNAPHU — Except in One Case

**Prepared for**: scientific review and prioritization of next steps
**Basis**: two independent, real training-and-benchmark runs (not simulated or estimated)
**Bottom line**: across 7 benchmark scenes and 2 independent runs at different scales, `pyunwrap` beats SNAPHU in exactly one scene category — low-coherence decorrelation — and loses everywhere else, usually by a wide margin. This report lays out the evidence, the most likely causes, and a prioritized path forward.

---

## 1. Executive summary

`pyunwrap` is a physics-informed U-Net that predicts the integer phase-ambiguity map for InSAR phase unwrapping, rather than regressing unwrapped phase directly. The central design bet is that this approach should be *more* robust than classical statistical-cost unwrapping (SNAPHU) specifically in regimes where SNAPHU's own local-consistency assumptions break down: low coherence and steep deformation gradients.

Two independent benchmark runs — one on a 360-tile synthetic dataset, one on a differently-seeded 216-tile dataset — test this. The result is consistent and unambiguous:

- **`pyunwrap` loses to SNAPHU on 6 of 7 benchmark scenes, in both runs**, often by a factor of 2–9x in RMSE.
- **`pyunwrap` beats SNAPHU in exactly one scene category — low-coherence decorrelation — in both independent runs.** This is the only result that replicated cleanly across both runs, which is what makes it worth taking seriously rather than treating as noise.
- A second finding (that training through the hardest curriculum stage causes catastrophic forgetting on an easier scene) was clearly demonstrated in the 360-tile run but **did not replicate** in the 216-tile run — itself an important, honest data point about how sensitive these results are to training configuration.

None of this means the underlying idea is wrong. It means the current checkpoint is undertrained, under-tuned, and validated only on synthetic data — and the report below tries to separate "the approach doesn't work" from "this specific, resource-constrained execution of it hasn't worked yet," because the evidence supports the second, not the first.

---

## 2. The evidence

Both runs used the same fair scoring method: a best-fit global `2π` offset is removed from each method's output before computing RMSE against ground truth (phase unwrapping has no absolute reference, so this is necessary for a fair comparison, not a way of flattering either method). Both runs used the same 7 synthetic benchmark scenes, spanning easy to hard, and the same underlying `pyunwrap.analytics.benchmark.run_benchmark_suite` code.

### Run A — 360 training tiles (original session)

| Scene | % pixels violating Nyquist | RMSE ratio (`pyunwrap` / SNAPHU) | Winner |
|---|---:|---:|:---:|
| easy_bowl | 1.6% | 8.97 | SNAPHU |
| moderate_bowl | 7.5% | 2.33 | SNAPHU |
| realistic_default_mogi | 9.9% | 2.69 | SNAPHU |
| moderate_mogi | 13.4% | 1.81 | SNAPHU |
| easy_control | 16.8% | 2.65 | SNAPHU |
| realistic_default_okada | 34.8% | 1.00 | ~tie (SNAPHU, barely) |
| **low_coherence_control** | 35.1% | **0.84** | **pyunwrap** |

### Run B — 216 training tiles, different seed and scene composition (reproduction)

| Scene | % pixels violating Nyquist | RMSE ratio (`pyunwrap` / SNAPHU) | Winner |
|---|---:|---:|:---:|
| easy_bowl | 1.6% | 8.03 | SNAPHU |
| moderate_bowl | 7.5% | 1.91 | SNAPHU |
| realistic_default_mogi | 9.9% | 3.29 | SNAPHU |
| moderate_mogi | 13.4% | 1.89 | SNAPHU |
| easy_control | 16.8% | 1.46 | SNAPHU |
| realistic_default_okada | 34.8% | 1.09 | SNAPHU |
| **low_coherence_control** | 35.1% | **0.90** | **pyunwrap** |

*(A ratio below 1.0 means `pyunwrap` had lower RMSE. Full source data: `docs/_static/experiments/three_way_comparison.csv` for Run A, `notebooks/03_training_bug_and_fix.ipynb` for Run B.)*

### What's robust across both runs

- **`pyunwrap` loses on every scene except `low_coherence_control`, in both runs.** This is not a coincidence of one unlucky configuration.
- **`pyunwrap` wins on `low_coherence_control`, in both runs**, with a similar margin (RMSE ratio 0.84 and 0.90). This is the one result that replicated.
- **The steep-gradient scene (`realistic_default_okada`) is close to parity in both runs** (ratio 1.00 and 1.09) but does not consistently cross into a `pyunwrap` win.
- Everything else — easy and moderate scenes — is not close. SNAPHU wins by large, consistent margins.

---

## 3. Why the one win happens where it does

`low_coherence_control` is a scene with high decorrelation noise but *no deformation signal* — the phase noise is close to spatially unstructured. SNAPHU's cost function assumes that the true unwrapped phase is *locally smooth and internally consistent*; in a fully decorrelated region that assumption is violated almost everywhere at once, and SNAPHU's own connected-component reliability metric confirms this — it reports 0% confidence in its own output on this scene in both runs. A statistical-cost method has no local signal left to exploit once coherence collapses this far.

A learned model isn't relying on local consistency in the same way — it has an implicit, learned prior over what phase-ambiguity fields tend to look like, built from many training examples. In a regime where the classical method's core assumption fails completely, a model with *any* usable prior, however imperfect, has a structural chance to do better. This is also consistent with prior published work in this space (e.g., Sica et al., 2020, *IEEE GRSL*), which reports competitive-to-superior results for CNN-based ambiguity prediction specifically in low-coherence conditions, and does not claim superiority in easy, high-coherence regimes.

The steep-gradient scene sitting close to parity (but not reliably winning) is consistent with the same logic in a weaker form: SNAPHU's assumption (gradients under `π` per pixel) is violated on 35% of pixels there too, but not as totally as the coherence collapse in `low_coherence_control`.

**Working hypothesis**: `pyunwrap`'s comparative advantage is real but currently confined to the specific regime where SNAPHU's underlying statistical assumption is violated most completely. Elsewhere, SNAPHU is simply a very good, decades-refined algorithm, and beating it requires more than a plausible architecture — it requires enough training to actually learn a better prior than "guess based on limited examples."

---

## 4. Why it doesn't beat SNAPHU everywhere — candidate causes

None of these are mutually exclusive; all likely contribute to some degree.

1. **Severe undertraining relative to the problem.** Both runs trained for 60 epochs on 216–360 tiles, on a single CPU core, in minutes. Published CNN-based unwrapping work in this space typically trains on far larger, more diverse datasets. The model has seen a tiny fraction of the pattern diversity a production model would need.

2. **100% synthetic training data.** Every result above is against `pyunwrap`'s own synthetic generator. Real SAR imagery has sensor-specific speckle statistics, real atmospheric artifacts, and real decorrelation patterns that the synthetic generator approximates but does not reproduce exactly. No result here has been checked against real satellite data at all — this is arguably the single largest unaddressed gap, not a training-scale problem.

3. **A real, demonstrated training-dynamics problem (curriculum/LR-schedule interaction).** A learning-rate-restart mechanism was found to fix a real bug (the LR decaying to near-zero exactly when the hardest curriculum stage unlocked), and it measurably improved 6 of 7 benchmark scenes in Run A. But it also caused a 74% RMSE regression on an easier scene in that same run — a textbook catastrophic-forgetting trade-off. That effect did not reproduce as strongly in Run B, suggesting the training procedure itself is still unstable and configuration-sensitive, independent of how much data or compute is available. **This means some of the ceiling on current performance is a training-procedure problem, fixable without more compute.**

4. **No architecture or hyperparameter search.** A single U-Net/ResNet-34 configuration, default loss-term weights, one `k_max` value, and one dropout rate were used throughout. None of this has been tuned. It is very plausible that meaningful accuracy is being left on the table purely from configuration choices.

5. **Narrow synthetic training distribution.** The generator covers four deformation archetypes (Gaussian bowl, Okada fault, Mogi source, and a no-deformation control) at one fixed tile size and one sensor-geometry assumption. Generalization outside that narrow distribution — including to the benchmark's own harder scenes — is not guaranteed.

6. **SNAPHU is a very high bar.** It is the field's mature, heavily-optimized reference implementation. A young, minimally-tuned ML approach losing to it on most scenes is the expected outcome at this stage, not evidence the approach is flawed.

---

## 5. Recommended way forward, prioritized

**Priority 1 — Real data.** No result in this report has touched real SAR imagery. Before investing further in synthetic-data tuning, validate whether the one real finding (the low-coherence win) holds on real Sentinel-1 or similar data with genuine decorrelation. If it doesn't, the synthetic generator's decorrelation model is likely the thing to fix, not the network architecture.

**Priority 2 — Stabilize the training procedure before scaling it up.** The curriculum/LR-restart trade-off (item 3 above) should be resolved first, since more compute spent on top of an unstable training procedure will not reliably translate into better accuracy. Three concrete, already-scoped options (see `docs/experiments.md`):
   - Smaller LR-restart peaks for later curriculum stages, rather than a full restart to peak LR.
   - Curriculum replay — mix a minority of easier tiles into the hardest stage's batches, rather than an abrupt 100% switch.
   - Difficulty-stratified validation tracking is already implemented (`evaluate_stratified`) and should be used on every future run to catch this class of regression immediately rather than requiring a dedicated diagnostic study.

**Priority 3 — Scale up training on GPU with a larger, more diverse synthetic corpus**, once priority 2 is addressed. Both runs to date used well under an hour of CPU time; this is not representative of what the architecture can achieve with realistic training budgets.

**Priority 4 — A basic hyperparameter/architecture sweep** (loss term weights, `k_max`, dropout rate, encoder choice) once 1–3 make results stable enough for a sweep to be interpretable.

**Priority 5 — Consider a narrower, honest near-term goal.** Rather than pursuing across-the-board superiority over SNAPHU, the current evidence supports a more defensible, achievable target: a specialized low-coherence unwrapping assist, used either standalone in that regime or as one component of a regime-aware system that defers to SNAPHU elsewhere. This matches where the real, reproduced advantage currently is, rather than where the architecture was hoped to win.

---

## 6. Source data and traceability

- Run A (360 tiles): `docs/experiments.md`, Experiments 1–3; raw data in `docs/_static/experiments/`.
- Run B (216 tiles, reproduction): `notebooks/03_training_bug_and_fix.ipynb`, fully executed with real outputs.
- Benchmark code: `pyunwrap/analytics/benchmark.py`.
- Training code and the LR-restart fix: `pyunwrap/training/trainer.py`.

All numbers in this report are drawn directly from those sources; none are estimated or illustrative.

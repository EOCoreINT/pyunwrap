# Experiments & findings

This page documents real training runs and benchmark results against
SNAPHU, run against this project's own synthetic generator (not real
satellite data -- see {doc}`architecture` and the project's release notes
for why that's a real, acknowledged limitation, not a hidden one). Numbers
here are from actual executions, not illustrative estimates, and this page
is updated when a run produces a finding worth keeping rather than after
every run.

## Experiment 1: full 60-epoch, 3-stage curriculum run

**Setup**: `AmbiguityNet` (24.4M params, `k_max=10`, `dropout_rate=0.1`),
360 training tiles / 72 validation tiles (64x64, cycling through all four
deformation types), the full curriculum schedule (`use_curriculum=True`,
stages at epochs 1-20 / 21-50 / 51-60), AdamW, single CPU core.

### Finding: the LR schedule and curriculum stages fought each other

```{image} _static/experiments/full_training_curve.png
:alt: Training loss across the full 60-epoch curriculum run, showing a spike at each stage transition
:width: 100%
```

Training loss jumps at *both* curriculum stage transitions (epoch 21 and
epoch 51) -- expected, since harder data is genuinely harder. But look at
the recovery: after the epoch-21 jump, loss drops back down over the next
~15 epochs. After the epoch-51 jump (stage 3, full difficulty unlocked),
**it never recovers before training ends at epoch 60.**

The reason: `Trainer` originally ran a single cosine-annealed LR schedule
across the entire run, with no awareness of the curriculum's stage
boundaries. By epoch 51, the LR had decayed to `6.46e-06` -- roughly
1/15,000th of its peak (`1e-4`). The hardest data arrived exactly when the
model had the least room left to adapt to it.

**Fix**: `pyunwrap.training.trainer.build_curriculum_aware_scheduler` gives
each curriculum stage (and the SNAPHU fine-tuning phase, if configured) its
own independent warmup-then-cosine cycle, restarting the LR back toward
its peak at every stage boundary. It's on by default in `Trainer` whenever
curriculum learning is enabled, and is verified to degenerate to the exact
old single-cycle behavior when it's not (see `tests/test_trainer.py`'s
`TestCurriculumAwareScheduler` for the byte-exact equivalence check). This
was found and fixed *because* a full run was finally completed -- every
prior training run in this project's development stopped before reaching
epoch 51, so the bug was invisible until this run.

## Experiment 2: benchmark against SNAPHU, by difficulty

**Setup**: the checkpoint from Experiment 1, compared against SNAPHU via
`pyunwrap.analytics.benchmark.run_benchmark_suite` across 7 synthetic
scenes spanning easy to hard, scored with a best-fit global `2*pi` offset
removed from both methods before computing RMSE (phase unwrapping has no
absolute reference by definition, so scoring the raw un-aligned output
would penalize SNAPHU for a property that isn't a real limitation).

```{image} _static/experiments/benchmark_comparison.png
:alt: RMSE comparison between pyunwrap and SNAPHU across scenes of increasing difficulty
:width: 100%
```

**Honest headline: SNAPHU wins on accuracy on every single scene tested.**
This checkpoint does not beat SNAPHU anywhere in this benchmark.

**The finding worth reporting anyway**: the *ratio* of pyunwrap's RMSE to
SNAPHU's RMSE drops from 8.3x worse on the easiest scene to 1.1x worse on
the hardest one -- a Pearson correlation of **-0.81** between scene
difficulty (% of pixels violating the Nyquist condition) and pyunwrap's
relative disadvantage. That's the specific pattern the entire
physics-informed-ambiguity-prediction approach predicts: classical
unwrapping's advantage should erode as gradients get steeper. It isn't
enough to win yet, but it's the first real, quantified evidence in this
project that the relative gap moves in the theoretically-expected
direction as training scales up, rather than a single flattering
cherry-picked data point.

**One result complicates the clean story**: `low_coherence_control` (35%
Nyquist violations from decorrelation rather than deformation) sits
slightly *worse* than the difficulty trend would predict -- SNAPHU's own
reliability metric is `0.0` there too (it doesn't trust its own output
either), but it's still more accurate than pyunwrap on average. The
"gap narrows with difficulty" pattern looks cleaner for steep-gradient
difficulty than for low-coherence difficulty. Both are named as target
failure modes for classical unwrapping in this project's own
{doc}`README <index>`; this result suggests they may not respond
identically to more training, and that's worth tracking in future runs
rather than smoothing over.

Full numeric results: [`benchmark_results_60epoch.csv`](_static/experiments/benchmark_results_60epoch.csv).

## Experiment 3: does the LR-restart fix actually help? (a controlled retest)

Experiments 1 and 2 used a checkpoint trained *before* the LR-restart fix
existed. The obvious, necessary follow-up: retrain with identical
hyperparameters (same seed, same data, same everything except the
scheduler) and rerun the same benchmark, to isolate what the fix actually
changes.

```{image} _static/experiments/three_way_comparison.png
:alt: RMSE comparison across SNAPHU, the pre-fix checkpoint, and the fixed-scheduler checkpoint
:width: 100%
```

**The fix helps substantially on 6 of 7 scenes** (median RMSE improvement
+31.3%), including two results worth calling out specifically:
- **`low_coherence_control`: pyunwrap now beats SNAPHU outright** (6.17 vs
  7.36 rad) -- the first genuine win anywhere in this project's
  benchmarking history.
- **`realistic_default_okada`** (steepest-gradient scene): now at virtual
  parity with SNAPHU (13.77 vs 13.75 rad), versus 1.12x worse before.

**But `easy_control` got dramatically worse** -- 12.2 -> 21.3 rad, a 74%
regression. Full numbers: [`three_way_comparison.csv`](_static/experiments/three_way_comparison.csv).

### Follow-up: this is catastrophic forgetting, directly demonstrated

Hypothesis: restarting the LR to full peak for curriculum stage 3 gives the
model real capacity to learn the hardest data (exactly what fixed the
low-coherence and steep-gradient scenes above) -- but that same freedom to
move far from its current weights could also let it drift away from
whatever it had already learned about easier-scene distributions.

Tested directly, not just argued for: trained an identical model (same
seed, same data) with `total_epochs=50`, so curriculum stage 3 (which
starts at epoch 51) never triggers and the model never experiences that
restart. Evaluated on the exact same `easy_control` scene:

| Checkpoint | `easy_control` RMSE |
|---|---|
| Stage-2-only (never saw stage 3) | **2.44 rad** |
| Buggy scheduler, full 60 epochs | 12.23 rad |
| Fixed scheduler, full 60 epochs | 21.32 rad |

The model was performing *well* on `easy_control` before stage 3 -- better
than either full-length run, buggy or fixed. Stage 3's LR restart is what
degrades it. This is a real, controlled demonstration of catastrophic
forgetting, not a hypothesis left unverified: the same mechanism that
produces the low-coherence win above is the mechanism that breaks
`easy_control`.

### What this suggests for future work

- **Smaller restart peaks for later stages** -- decay the restart ceiling
  itself at each stage transition (e.g. stage 3 restarts to ~0.3-0.5x
  peak rather than 1.0x), trading some learning speed for less
  destructive drift. *Not yet implemented.*
- **Curriculum replay** -- mix a minority of easy/moderate tiles into stage
  3's batches instead of switching to 100% full-difficulty data abruptly.
  *Not yet implemented.*
- **Difficulty-stratified validation tracking** -- **Implemented**: see
  `pyunwrap.training.trainer.evaluate_stratified`, on by default. `Trainer`
  previously logged only one aggregate `val_rmse`; this regression would
  now be visible during training itself (as a divergence between
  `val/rmse_rad_easy` and `val/rmse_rad_hard` in the training curves)
  rather than requiring a separate downstream benchmark to discover, as it
  did here.
- **A regime-aware dispatcher instead of one model** -- route inference to
  a stage-2-tuned checkpoint for easy scenes and a stage-3-tuned one for
  hard scenes, using the same coherence/gradient stats the curriculum
  already computes, rather than requiring one model to be uniformly good
  across a difficulty spectrum where training dynamics trade one end
  against the other. *Not yet implemented.*

## What these experiments do and don't tell you

They tell you the training pipeline, curriculum learning, and physics
constraints all work correctly end to end on real (if CPU-bounded)
hardware -- that was genuinely uncertain before these runs, and several
real bugs (this page's LR-restart issue, plus others documented in
{doc}`changelog`) were only found by actually running things to completion.
They also tell you the core approach can, in a specific regime, genuinely
outperform SNAPHU on this benchmark (`low_coherence_control`) -- not a
theoretical claim, a measured one.

They do **not** tell you this approach is ready for real deformation
monitoring, or that any single trained checkpoint is reliably better than
SNAPHU across the board -- Experiment 3 shows the opposite is just as
real: the same training change that produced a win also produced a 74%
regression elsewhere. Every scene here is synthetic. Closing the gap to a
genuinely trustworthy tool needs real satellite data validation and materially more training
compute (GPU, not one CPU core) than has been applied so far -- see the
project status note on the {doc}`index` page.

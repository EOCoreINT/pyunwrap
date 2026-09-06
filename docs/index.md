# pyunwrap

**A hybrid, regime-aware InSAR phase unwrapper — classical where classical
is strong, learned where it isn't.**

`pyunwrap` unwraps Interferometric Synthetic Aperture Radar (InSAR) phase —
the core measurement behind satellite-based ground-deformation monitoring.
It is **not** a claim to replace classical unwrapping (SNAPHU) everywhere —
this project's own benchmarks (see {doc}`experiments`) show classical
unwrapping winning on most scenes tested, and a learned model winning
reliably in exactly one regime: low-coherence decorrelation, where
classical statistical-cost unwrapping's core assumption breaks down
hardest. `pyunwrap`'s goal, reframed directly from that evidence, is a
**hybrid, regime-aware unwrapper**
({class}`~pyunwrap.inference.hybrid.HybridUnwrapper`): route each region
of a scene to whichever method is actually good there.

The learned side of that route is a physics-informed U-Net that predicts
an **integer ambiguity map**, not the unwrapped phase itself. The
rewrapping identity

$$
\hat\phi = \psi + 2\pi \cdot \operatorname{round}(\hat{k}), \qquad \hat{k} \in \mathbb{Z}
$$

is enforced by construction, so the model is structurally incapable of
producing an output that contradicts the observed wrapped phase $\psi$ — it
can only be wrong about *how many* $2\pi$ cycles were missed, never about the
physics of wrapping. The hybrid router extends this same discipline: it
merges the classical and learned results in this integer ambiguity space,
never by averaging raw phase values, so the final output satisfies the
same wrap identity exactly regardless of how the two methods' outputs were
blended.

Every classical phase-unwrapping algorithm — branch-cut methods, minimum-cost
flow (the approach behind SNAPHU), Goldstein's algorithm — rests on the
assumption that the true phase gradient between adjacent pixels never
exceeds $\pi$ radians (the Nyquist/Itoh condition). Low coherence and steep
deformation gradients both break that assumption in practice; the learned
side of `pyunwrap` targets exactly that gap, without inheriting the failure
mode of naive deep-learning approaches that regress phase directly and have
no reason to respect $\operatorname{wrap}(\text{prediction}) = \psi$.

::::{grid} 2
:gutter: 3

:::{grid-item-card} 🚀 Quickstart
:link: quickstart
:link-type: doc
Install the package and run your first tiled unwrap in a few lines.
:::

:::{grid-item-card} 🧠 How it works
:link: architecture
:link-type: doc
The physics invariant every stage of the pipeline is built around.
:::

:::{grid-item-card} 🔀 Hybrid unwrapping
:link: api/inference
:link-type: doc
Route each region to classical or learned unwrapping, based on this
project's own benchmark evidence.
:::

:::{grid-item-card} 📓 Tutorials
:link: tutorials/index
:link-type: doc
Pre-executed, real-output notebooks covering the full pipeline.
:::

:::{grid-item-card} 📚 API reference
:link: api/index
:link-type: doc
Every public module, class, and function, generated from source.
:::
::::

## Why this exists

Traditional InSAR phase unwrapping is a mature field with a well-established
reference implementation (SNAPHU), but it has two well-known blind spots:

- **Low coherence** (vegetation, water, temporal decorrelation) injects
  near-random phase noise. Branch-cut and minimum-cost-flow methods don't
  just get the noisy pixel wrong — they propagate the error across the
  connected region downstream of it.
- **Steep deformation gradients** (earthquakes, volcanic inflation, mining
  subsidence) can produce genuine phase gradients exceeding $\pi$
  radians/pixel near the source, which no classical algorithm can recover
  without an external prior.

`pyunwrap` predicts the integer ambiguity map `k` instead of the unwrapped
phase directly, which makes the re-wrapping identity a mathematical
guarantee rather than a training objective the network might fail to learn.
See {doc}`architecture` for the full explanation, and {doc}`api/index` for
every module involved.

## Project status

Early-stage and actively developed. **The project's goal is deliberately
not "beat SNAPHU everywhere"** — see {doc}`experiments` for the benchmark
evidence that framing is based on. The pipeline — synthetic data
generation, real-data injection, curriculum training (sequential or
replay-based), tiled inference, the hybrid router, a real SNAPHU
integration, analytics, and reporting — is implemented and tested end to
end, but no pretrained weights ship yet, and no validation against real
satellite data has been performed (every result to date is against
`pyunwrap`'s own synthetic generator or a clearly-labeled real-*like* test
fixture, not genuine SAR data). See the [CHANGELOG](changelog) for exactly
what's landed so far, and
the project's release notes for an honest, quantified benchmark against
SNAPHU.

```{toctree}
:maxdepth: 2
:hidden:
:caption: Getting Started

installation
quickstart
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: How It Works

architecture
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Tutorials

tutorials/index
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Reference

experiments
api/index
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Project

contributing
changelog
GitHub <https://github.com/EOCoreINT/pyunwrap>
```

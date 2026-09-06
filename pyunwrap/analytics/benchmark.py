"""
pyunwrap.analytics.benchmark
===============================

Quantified comparison between `pyunwrap` (`AmbiguityNet` via
`PhaseUnwrapper`) and classical SNAPHU unwrapping, on data with **known
ground truth**.

Why synthetic data, not real Sentinel-1 scenes
------------------------------------------------
The honest scope of this module: it benchmarks against `pyunwrap`'s own
synthetic generator, not real satellite interferograms. That's a real
limitation, not a stylistic choice -- but it also means every comparison
here has *exact* ground truth to score against, which real-data benchmarks
usually don't (real InSAR "ground truth" is itself often another algorithm's
output, or GPS/leveling data at sparse points). Both methods are scored
identically: a best-fit global `2*pi` offset is removed from each result
before computing error (phase unwrapping has no absolute reference by
definition, so scoring on the raw, un-aligned output would penalize SNAPHU
for a property that isn't actually a limitation of the algorithm).

Both methods also get to use the same information: the same wrapped phase,
coherence, and amplitude inputs.
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np
import pandas as pd

from pyunwrap.analytics.phase_stats import gradient_analysis_summary
from pyunwrap.models.ambiguity_net import AmbiguityNet
from pyunwrap.synthetic.generator import InSARSyntheticGenerator, SyntheticSample
from pyunwrap.utils.snaphu_integration import _HAS_SNAPHU, unwrap_with_snaphu


def _write_sample_geotiffs(sample: SyntheticSample, out_dir) -> dict:
    """Write a `SyntheticSample`'s wrapped phase, coherence, and amplitude
    to temporary GeoTIFFs, for feeding into `PhaseUnwrapper`/
    `HybridUnwrapper`, which operate on file paths. Shared by
    `compare_against_snaphu` and `compare_hybrid_against_baselines` so both
    write files in exactly the same way.

    Args:
        sample: The `SyntheticSample` to write.
        out_dir: Directory to write into (typically a `TemporaryDirectory`).

    Returns:
        `{"wrapped": Path, "coherence": Path, "amplitude": Path}`.
    """
    import rasterio
    from rasterio.transform import from_origin

    size = sample.wrapped_phase.shape[0]
    transform = from_origin(500_000, 5_000_000, 20, 20)
    profile = {
        "driver": "GTiff",
        "height": size,
        "width": size,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:32633",
        "transform": transform,
    }
    paths = {}
    for name, arr in [
        ("wrapped", sample.wrapped_phase),
        ("coherence", sample.coherence),
        ("amplitude", sample.amplitude),
    ]:
        path = out_dir / f"{name}.tif"
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(arr.astype(np.float32), 1)
        paths[name] = path
    return paths


def _best_fit_offset_rmse(predicted: np.ndarray, true: np.ndarray) -> tuple[np.ndarray, float]:
    """Remove the best-fit global 2*pi*n offset from `predicted` relative to
    `true`, then compute RMSE. Fair scoring for methods with no absolute
    phase reference (see module docstring).

    Args:
        predicted: Predicted unwrapped phase, radians.
        true: Ground-truth unwrapped phase, radians, same shape.

    Returns:
        `(residual, rmse)`: the offset-corrected error map and its RMSE.
    """
    diff = predicted - true
    offset = np.round(diff.mean() / (2 * np.pi)) * 2 * np.pi
    residual = diff - offset
    rmse = float(np.sqrt(np.mean(residual**2)))
    return residual, rmse


@dataclasses.dataclass
class MethodResult:
    """One method's result on one scene.

    Attributes:
        rmse_rad: RMSE against ground truth, radians, after best-fit offset
            removal (see `_best_fit_offset_rmse`).
        pct_under_0p1_rad: Percentage of pixels with absolute error < 0.1 rad.
        reliable_fraction: For SNAPHU, the fraction of pixels its own
            connected-component analysis considers reliable. `1.0` for
            pyunwrap (no equivalent hard reliability gate; residue_prob is a
            continuous uncertainty signal, not a binary one -- see
            `AmbiguityNet`'s docstring).
        runtime_s: Wall-clock seconds for this method on this scene.
    """

    rmse_rad: float
    pct_under_0p1_rad: float
    reliable_fraction: float
    runtime_s: float


@dataclasses.dataclass
class BenchmarkResult:
    """Full comparison result for one scene."""

    scene_id: str
    deformation_type: str
    mean_coherence: float
    pct_nyquist_violations: float
    max_abs_ambiguity: float
    snaphu: MethodResult | None
    pyunwrap: MethodResult


def compare_against_snaphu(
    model: AmbiguityNet,
    sample: SyntheticSample,
    scene_id: str,
    deformation_type: str,
    device: str = "cpu",
    snaphu_nlooks: float = 4.0,
    snaphu_kwargs: dict | None = None,
) -> BenchmarkResult:
    """Run both SNAPHU and `pyunwrap` on the same synthetic sample and score
    both against its known ground truth.

    `pyunwrap` inference goes through the real, already-tested
    `PhaseUnwrapper.unwrap()` path (via temporary GeoTIFFs), rather than
    re-implementing normalization and the forward pass inline here --
    deliberately, to avoid a second, untested copy of that logic silently
    drifting from the tested one.

    Args:
        model: A (trained) `AmbiguityNet`.
        sample: A `SyntheticSample` with known `unwrapped_phase`.
        scene_id: Identifier for this scene, for the results table.
        deformation_type: Deformation type used to generate `sample`, for
            the results table.
        device: Device to run `pyunwrap` inference on.
        snaphu_nlooks: `nlooks` parameter forwarded to SNAPHU.
        snaphu_kwargs: Extra keyword arguments forwarded to
            `unwrap_with_snaphu`.

    Returns:
        A `BenchmarkResult`. `snaphu` is `None` if the `snaphu` package
        isn't installed (pyunwrap-only results are still returned).

    Raises:
        ValueError: If `sample`'s scene size is not a multiple of 32 (the
            same constraint `PhaseUnwrapper.unwrap` enforces -- see its
            docstring for why).
    """
    import tempfile
    from pathlib import Path

    import rasterio
    from rasterio.transform import from_origin

    from pyunwrap.inference.unwrapper import PhaseUnwrapper

    snaphu_kwargs = dict(snaphu_kwargs or {})
    grad_summary = gradient_analysis_summary(sample.unwrapped_phase)
    size = sample.wrapped_phase.shape[0]
    if size % 32 != 0:
        raise ValueError(
            f"Scene size {size} must be a multiple of 32 to run through "
            "PhaseUnwrapper.unwrap as a single tile (same constraint as "
            "production inference -- see PhaseUnwrapper.unwrap's docstring)."
        )

    # --- SNAPHU ---
    snaphu_result = None
    if _HAS_SNAPHU:
        t0 = time.time()
        snaphu_out = unwrap_with_snaphu(
            sample.wrapped_phase,
            sample.coherence,
            amplitude=sample.amplitude,
            nlooks=snaphu_nlooks,
            **snaphu_kwargs,
        )
        snaphu_runtime = time.time() - t0
        _residual, snaphu_rmse = _best_fit_offset_rmse(
            snaphu_out.unwrapped_phase, sample.unwrapped_phase
        )
        snaphu_pct_under = float(100.0 * np.mean(np.abs(_residual) < 0.1))
        snaphu_result = MethodResult(
            rmse_rad=snaphu_rmse,
            pct_under_0p1_rad=snaphu_pct_under,
            reliable_fraction=snaphu_out.reliable_fraction,
            runtime_s=snaphu_runtime,
        )

    # --- pyunwrap, via the real (tested) PhaseUnwrapper path ---
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        transform = from_origin(500_000, 5_000_000, 20, 20)
        profile = {
            "driver": "GTiff",
            "height": size,
            "width": size,
            "count": 1,
            "dtype": "float32",
            "crs": "EPSG:32633",
            "transform": transform,
        }
        paths = {}
        for name, arr in [
            ("wrapped", sample.wrapped_phase),
            ("coherence", sample.coherence),
            ("amplitude", sample.amplitude),
        ]:
            path = tmpdir / f"{name}.tif"
            with rasterio.open(path, "w", **profile) as dst:
                dst.write(arr.astype(np.float32), 1)
            paths[name] = path

        unwrapper = PhaseUnwrapper(model=model, device=device, mc_dropout_passes=1)
        t0 = time.time()
        result = unwrapper.unwrap(
            paths["wrapped"],
            paths["coherence"],
            paths["amplitude"],
            tile_size=size,
            overlap=0,
        )
        pyunwrap_runtime = time.time() - t0
        predicted = result.unwrapped_phase

    _residual, pyunwrap_rmse = _best_fit_offset_rmse(predicted, sample.unwrapped_phase)
    pyunwrap_pct_under = float(100.0 * np.mean(np.abs(_residual) < 0.1))
    pyunwrap_result = MethodResult(
        rmse_rad=pyunwrap_rmse,
        pct_under_0p1_rad=pyunwrap_pct_under,
        reliable_fraction=1.0,
        runtime_s=pyunwrap_runtime,
    )

    return BenchmarkResult(
        scene_id=scene_id,
        deformation_type=deformation_type,
        mean_coherence=float(sample.coherence.mean()),
        pct_nyquist_violations=grad_summary["pct_nyquist_violations"],
        max_abs_ambiguity=float(np.abs(sample.ambiguity).max()),
        snaphu=snaphu_result,
        pyunwrap=pyunwrap_result,
    )


def _default_benchmark_scenarios(size: int, seed: int) -> list[tuple[str, str, SyntheticSample]]:
    """Build the standard graduated easy-to-hard scene set shared by
    `run_benchmark_suite` and `run_hybrid_benchmark_suite`, so both draw
    from exactly the same scenes rather than two definitions that could
    silently drift apart.

    Args:
        size: Scene edge length, pixels.
        seed: Base seed; scene `i` uses `seed + i`.

    Returns:
        List of `(scene_id, deformation_type, SyntheticSample)` triples.
    """
    scenarios: list[tuple[str, str, dict]] = [
        (
            "easy_control",
            "none",
            {"base_coherence": 0.95, "atmosphere_amplitude_rad": 0.15, "ramp_amplitude_rad": 0.1},
        ),
        (
            "easy_bowl",
            "gaussian_bowl",
            {
                "base_coherence": 0.9,
                "atmosphere_amplitude_rad": 0.2,
                "ramp_amplitude_rad": 0.15,
                "deformation_kwargs": {"amplitude_m": 0.02, "sigma_m": 1000.0},
            },
        ),
        (
            "moderate_bowl",
            "gaussian_bowl",
            {"base_coherence": 0.75, "deformation_kwargs": {"amplitude_m": 0.06, "sigma_m": 700.0}},
        ),
        (
            "moderate_mogi",
            "mogi",
            {
                "base_coherence": 0.7,
                "deformation_kwargs": {"volume_change_m3": 5e5, "source_depth_m": 3000.0},
            },
        ),
        ("realistic_default_mogi", "mogi", {}),  # generator's own defaults: routinely hard
        (
            "realistic_default_okada",
            "okada",
            {
                "deformation_kwargs": {
                    "strike_deg": 25.0,
                    "dip_deg": 65.0,
                    "rake_deg": 90.0,
                    "slip_m": 0.5,
                    "length_m": 4000.0,
                    "width_m": 2000.0,
                    "depth_m": 5000.0,
                }
            },
        ),
        ("low_coherence_control", "none", {"base_coherence": 0.35}),
    ]

    samples = []
    for i, (scene_id, deformation_type, kwargs) in enumerate(scenarios):
        gen = InSARSyntheticGenerator(size=size, seed=seed + i)
        # deformation_type here is a plain `str` (from the scenarios list
        # above), while generate_sample's parameter is typed as a narrower
        # Literal for IDE/type-checking convenience elsewhere; every value
        # actually used above is one of the four valid literals, so this is
        # runtime-safe despite the static type mismatch.
        sample = gen.generate_sample(deformation_type=deformation_type, **kwargs)  # type: ignore[arg-type]
        samples.append((scene_id, deformation_type, sample))
    return samples


def run_benchmark_suite(
    model: AmbiguityNet,
    size: int = 128,
    device: str = "cpu",
    seed: int = 0,
    snaphu_nlooks: float = 4.0,
) -> pd.DataFrame:
    """Run `compare_against_snaphu` across a spread of scenes spanning easy
    to hard difficulty and all four deformation types, and tabulate results.

    The scene set is deliberately graduated: from a genuinely easy control
    (high coherence, no deformation) through to the generator's realistic
    defaults (which routinely include real Nyquist violations -- see
    `pyunwrap.synthetic.generator`'s module docstring), so the resulting
    table shows *where* each method's advantage actually shows up rather
    than a single aggregate number that hides it.

    Args:
        model: A (trained) `AmbiguityNet` to benchmark.
        size: Scene edge length, pixels.
        device: Device to run `pyunwrap` inference on.
        seed: Base seed for reproducibility.
        snaphu_nlooks: `nlooks` parameter forwarded to SNAPHU for every scene.

    Returns:
        A `pandas.DataFrame`, one row per scene, with columns for both
        methods' RMSE, reliable/accurate-pixel fractions, and runtime.
    """
    rows = []
    for scene_id, deformation_type, sample in _default_benchmark_scenarios(size, seed):
        result = compare_against_snaphu(
            model,
            sample,
            scene_id=scene_id,
            deformation_type=deformation_type,
            device=device,
            snaphu_nlooks=snaphu_nlooks,
        )
        row = {
            "scene_id": result.scene_id,
            "deformation_type": result.deformation_type,
            "mean_coherence": result.mean_coherence,
            "pct_nyquist_violations": result.pct_nyquist_violations,
            "max_abs_ambiguity": result.max_abs_ambiguity,
            "pyunwrap_rmse_rad": result.pyunwrap.rmse_rad,
            "pyunwrap_pct_under_0p1_rad": result.pyunwrap.pct_under_0p1_rad,
            "pyunwrap_runtime_s": result.pyunwrap.runtime_s,
        }
        if result.snaphu is not None:
            row.update(
                {
                    "snaphu_rmse_rad": result.snaphu.rmse_rad,
                    "snaphu_pct_under_0p1_rad": result.snaphu.pct_under_0p1_rad,
                    "snaphu_reliable_fraction": result.snaphu.reliable_fraction,
                    "snaphu_runtime_s": result.snaphu.runtime_s,
                }
            )
        rows.append(row)

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Hybrid benchmark: classical vs. pure-learned vs. hybrid
# --------------------------------------------------------------------------- #

#: A relative-RMSE margin below which two methods are considered tied
#: rather than one declared a winner -- see compare_hybrid_against_baselines.
_INCONCLUSIVE_MARGIN = 0.05


@dataclasses.dataclass
class HybridBenchmarkRow:
    """One scene's three-way comparison: classical alone, learned alone,
    and the hybrid router, all scored against the same known ground truth.

    Attributes:
        scene_id: Scene identifier.
        deformation_type: Deformation type used to generate the scene.
        mean_coherence: Scene's mean coherence.
        pct_nyquist_violations: Scene's difficulty, as elsewhere in this module.
        classical_rmse_rad: RMSE of the classical-only result (SNAPHU if
            available, else the scikit-image fallback -- see
            `pyunwrap.inference.hybrid.build_default_classical_adapter`).
        learned_rmse_rad: RMSE of the pure-learned (`PhaseUnwrapper`) result.
        hybrid_rmse_rad: RMSE of the `HybridUnwrapper` result.
        winner: `"classical"`, `"learned"`, `"hybrid"`, or `"inconclusive"`
            (the three RMSEs are all within `_INCONCLUSIVE_MARGIN` of each
            other, relative to the best one) -- see
            `compare_hybrid_against_baselines`'s docstring for exactly how
            this is decided. Deliberately not just "whichever has the
            lowest number": a 0.1% difference is not a meaningful win, and
            reporting it as one would overclaim precision this benchmark
            doesn't have.
    """

    scene_id: str
    deformation_type: str
    mean_coherence: float
    pct_nyquist_violations: float
    classical_rmse_rad: float | None
    learned_rmse_rad: float
    hybrid_rmse_rad: float
    winner: str


def compare_hybrid_against_baselines(
    model: AmbiguityNet,
    sample: SyntheticSample,
    scene_id: str,
    deformation_type: str,
    device: str = "cpu",
    snaphu_nlooks: float = 4.0,
    hybrid_config=None,
) -> HybridBenchmarkRow:
    """Compare classical-only, learned-only, and hybrid unwrapping on one
    scene with known ground truth.

    Uses the real, tested `PhaseUnwrapper` and `HybridUnwrapper` classes
    (via temporary GeoTIFFs, matching `compare_against_snaphu`'s approach)
    rather than any separate/duplicated inference logic.

    Args:
        model: A (trained) `AmbiguityNet`.
        sample: A `SyntheticSample` with known `unwrapped_phase`.
        scene_id: Scene identifier, for the results table.
        deformation_type: Deformation type, for the results table.
        device: Device for `pyunwrap` inference.
        snaphu_nlooks: Forwarded to the classical adapter if it's SNAPHU.
        hybrid_config: Optional `pyunwrap.inference.hybrid.HybridUnwrapperConfig`.

    Returns:
        A `HybridBenchmarkRow`.

    Raises:
        ValueError: If the scene size is not a multiple of 32 (see
            `compare_against_snaphu`'s docstring for why).
    """
    import tempfile
    from pathlib import Path

    from pyunwrap.inference.hybrid import HybridUnwrapper, SnaphuAdapter
    from pyunwrap.inference.unwrapper import PhaseUnwrapper

    grad_summary = gradient_analysis_summary(sample.unwrapped_phase)
    size = sample.wrapped_phase.shape[0]
    if size % 32 != 0:
        raise ValueError(
            f"Scene size {size} must be a multiple of 32 to run through "
            "PhaseUnwrapper.unwrap as a single tile."
        )

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        paths = _write_sample_geotiffs(sample, tmpdir)

        learned_unwrapper = PhaseUnwrapper(model=model, device=device, mc_dropout_passes=1)
        learned_result = learned_unwrapper.unwrap(
            paths["wrapped"], paths["coherence"], paths["amplitude"], tile_size=size, overlap=0
        )
        _residual, learned_rmse = _best_fit_offset_rmse(
            learned_result.unwrapped_phase, sample.unwrapped_phase
        )

        classical_rmse: float | None = None
        if _HAS_SNAPHU:
            classical_adapter = SnaphuAdapter(nlooks=snaphu_nlooks)
            classical_unwrapped, _reliability = classical_adapter.unwrap(
                sample.wrapped_phase, sample.coherence, sample.amplitude
            )
            _residual, classical_rmse = _best_fit_offset_rmse(
                classical_unwrapped, sample.unwrapped_phase
            )

        hybrid = HybridUnwrapper(learned_unwrapper, config=hybrid_config)
        hybrid_result = hybrid.unwrap(
            paths["wrapped"], paths["coherence"], paths["amplitude"], tile_size=size, overlap=0
        )
        _residual, hybrid_rmse = _best_fit_offset_rmse(
            hybrid_result.unwrapped_phase, sample.unwrapped_phase
        )

    winner = _decide_winner(classical_rmse, learned_rmse, hybrid_rmse)

    return HybridBenchmarkRow(
        scene_id=scene_id,
        deformation_type=deformation_type,
        mean_coherence=float(sample.coherence.mean()),
        pct_nyquist_violations=grad_summary["pct_nyquist_violations"],
        classical_rmse_rad=classical_rmse,
        learned_rmse_rad=learned_rmse,
        hybrid_rmse_rad=hybrid_rmse,
        winner=winner,
    )


def _decide_winner(classical_rmse: float | None, learned_rmse: float, hybrid_rmse: float) -> str:
    """Decide which method wins a scene, treating near-ties as
    `"inconclusive"` rather than manufacturing a false margin of precision.

    A method is the winner only if its RMSE is more than
    `_INCONCLUSIVE_MARGIN` (relatively) better than *both* other methods'
    best value; otherwise `"inconclusive"`.
    """
    candidates = {"learned": learned_rmse, "hybrid": hybrid_rmse}
    if classical_rmse is not None:
        candidates["classical"] = classical_rmse

    best_name = min(candidates, key=lambda k: candidates[k])
    best_value = candidates[best_name]
    others = [v for k, v in candidates.items() if k != best_name]
    if not others:
        return best_name  # only one candidate existed at all

    second_best = min(others)
    if best_value <= 0:
        return best_name
    relative_gap = (second_best - best_value) / best_value
    if relative_gap < _INCONCLUSIVE_MARGIN:
        return "inconclusive"
    return best_name


def run_hybrid_benchmark_suite(
    model: AmbiguityNet,
    size: int = 128,
    device: str = "cpu",
    seed: int = 0,
    snaphu_nlooks: float = 4.0,
    hybrid_config=None,
) -> pd.DataFrame:
    """Run `compare_hybrid_against_baselines` across the same graduated
    easy-to-hard scene set `run_benchmark_suite` uses, tabulating
    classical/learned/hybrid RMSE and a per-scene winner.

    Args:
        model: A (trained) `AmbiguityNet`.
        size: Scene edge length, pixels.
        device: Device for `pyunwrap` inference.
        seed: Base seed.
        snaphu_nlooks: Forwarded to SNAPHU if available.
        hybrid_config: Optional `HybridUnwrapperConfig`.

    Returns:
        A `pandas.DataFrame`, one row per scene.
    """
    rows = []
    for scene_id, deformation_type, sample in _default_benchmark_scenarios(size, seed):
        row = compare_hybrid_against_baselines(
            model,
            sample,
            scene_id=scene_id,
            deformation_type=deformation_type,
            device=device,
            snaphu_nlooks=snaphu_nlooks,
            hybrid_config=hybrid_config,
        )
        rows.append(dataclasses.asdict(row))
    return pd.DataFrame(rows)


def generate_regime_report(df: pd.DataFrame) -> str:
    """Produce a plain-text summary explicitly stating where each method
    wins, per this project's benchmarking requirement to never report only
    a single aggregate score and to never overclaim across-the-board
    superiority.

    Args:
        df: Output of `run_hybrid_benchmark_suite`.

    Returns:
        A human-readable multi-line report string.
    """
    lines = ["Regime-aware benchmark report", "=" * 30, ""]

    counts = df["winner"].value_counts()
    total = len(df)
    for outcome in ("classical", "learned", "hybrid", "inconclusive"):
        n = int(counts.get(outcome, 0))
        lines.append(f"{outcome.capitalize():14s}: {n}/{total} scenes")
    lines.append("")

    for outcome, label in [
        ("classical", "Classical (SNAPHU/adapter) wins"),
        ("learned", "Pure learned model wins"),
        ("hybrid", "Hybrid router wins"),
        (
            "inconclusive",
            f"Inconclusive (methods within {_INCONCLUSIVE_MARGIN:.0%} of each other)",
        ),
    ]:
        scenes = df[df["winner"] == outcome]["scene_id"].tolist()
        lines.append(f"{label}: {', '.join(scenes) if scenes else '(none)'}")

    lines.append("")
    lines.append(
        "This report intentionally does not claim any single method wins "
        "across the board -- see the per-scene breakdown above and "
        "docs/experiments.md for the evidence this project's own hybrid "
        "framing is based on."
    )
    return "\n".join(lines)

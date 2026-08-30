"""
tests/test_pipeline.py
=========================

Integration tests exercising multiple `pyunwrap` modules together:

- `TestTileMergingNoArtifacts`: the critical correctness property of
  `pyunwrap.inference.unwrapper`'s smart tile merging -- that splitting a
  scene into overlapping tiles and merging the results back together must
  reproduce exactly what a single whole-image pass would have produced.
  This is a stronger, more direct test of "no 2*pi boundary artifacts" than
  visual inspection: with a model whose output is a deterministic function
  of local pixel content only (not of tile position), tiled-and-merged
  inference and single-tile whole-image inference must agree everywhere,
  including at every former tile seam.
- `TestFullPipeline` (marked `@pytest.mark.slow`): the complete
  synthetic-generation -> preprocessing/tiling -> 1-epoch training ->
  tiled inference -> HTML report pipeline, run end-to-end on a small scene.

All randomness is seeded via `conftest.py`'s autouse fixture.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from pyunwrap.data.dataloader import InSARTileDataset
from pyunwrap.data.preprocessing import (
    NormalizedRasters,
    iter_tiles,
    normalize_amplitude,
    normalize_coherence,
    normalize_phase,
    save_tiles_hdf5,
)
from pyunwrap.inference.unwrapper import PhaseUnwrapper, TileInferenceResult
from pyunwrap.models.ambiguity_net import AmbiguityNet
from pyunwrap.synthetic.generator import InSARSyntheticGenerator
from pyunwrap.training.trainer import Trainer

# --------------------------------------------------------------------------- #
# Tile merging correctness (no 2*pi boundary artifacts)
# --------------------------------------------------------------------------- #


class _DeterministicPixelwiseUnwrapper(PhaseUnwrapper):
    """A `PhaseUnwrapper` whose "model" output is a pure, deterministic
    function of each tile's own pixel content (`k_hat = round(5 * coherence)`,
    full confidence everywhere). Since this function does not depend on tile
    position or neighboring tiles in any way, tiled-and-merged inference MUST
    reproduce bit-for-bit what a single whole-image "tile" pass would give;
    any discrepancy can only come from a bug in the merge/windowing logic
    itself (e.g. the Hanning-window boundary-zeroing bug found and fixed
    during development), not from anything model-related.
    """

    def __init__(self) -> None:
        self.backend = "deterministic-fake"

    def _run_tile(self, x: np.ndarray) -> TileInferenceResult:
        coherence_tile = x[1]
        k_hat = np.round(5.0 * coherence_tile)
        residue_prob = np.zeros_like(coherence_tile)
        return TileInferenceResult(
            k_hat=k_hat, residue_prob=residue_prob, k_std=np.zeros_like(coherence_tile)
        )


@pytest.fixture
def deterministic_scene_geotiffs(tmp_workdir, rng):
    """Write wrapped/coherence/amplitude GeoTIFFs for a 128x128 synthetic
    scene, returning their paths."""
    gen = InSARSyntheticGenerator(size=128, seed=11)
    sample = gen.generate_sample(deformation_type="mogi")

    transform = from_origin(500000, 5000000, 20, 20)
    profile = {
        "driver": "GTiff",
        "height": 128,
        "width": 128,
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
        path = tmp_workdir / f"{name}.tif"
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(arr.astype(np.float32), 1)
        paths[name] = path
    return paths


class TestTileMergingNoArtifacts:
    def test_tiled_and_merged_matches_whole_image_pass(self, deterministic_scene_geotiffs):
        """Core boundary-artifact regression test: a 128x128 scene processed
        as ONE 128x128 tile must give (numerically) the same merged result as
        the same scene processed as FOUR overlapping 64x64 tiles, since the
        underlying per-pixel "model" output never depends on tile position.
        """
        unwrapper = _DeterministicPixelwiseUnwrapper()

        whole_result = unwrapper.unwrap(
            deterministic_scene_geotiffs["wrapped"],
            deterministic_scene_geotiffs["coherence"],
            deterministic_scene_geotiffs["amplitude"],
            tile_size=128,
            overlap=0,
        )
        tiled_result = unwrapper.unwrap(
            deterministic_scene_geotiffs["wrapped"],
            deterministic_scene_geotiffs["coherence"],
            deterministic_scene_geotiffs["amplitude"],
            tile_size=64,
            overlap=32,
        )

        np.testing.assert_allclose(
            tiled_result.ambiguity_map,
            whole_result.ambiguity_map,
            atol=0,
            err_msg="Tiled-and-merged ambiguity map differs from the whole-image pass -- "
            "this indicates a tile-boundary merging artifact.",
        )
        np.testing.assert_allclose(
            tiled_result.unwrapped_phase, whole_result.unwrapped_phase, atol=1e-9
        )

    def test_no_spurious_jump_at_former_tile_seams(self, deterministic_scene_geotiffs):
        """Directly inspect the pixel columns/rows that correspond to former
        tile boundaries (at multiples of the 64px tile stride) and confirm
        the merged ambiguity map's gradient there is no larger than the
        gradient found elsewhere in the (position-independent) field --
        i.e. tiling introduced no seam-localized discontinuity."""
        unwrapper = _DeterministicPixelwiseUnwrapper()
        result = unwrapper.unwrap(
            deterministic_scene_geotiffs["wrapped"],
            deterministic_scene_geotiffs["coherence"],
            deterministic_scene_geotiffs["amplitude"],
            tile_size=64,
            overlap=32,
        )
        k = result.ambiguity_map
        grad_col = np.abs(np.diff(k, axis=1))

        seam_cols = [63, 64]  # the two columns straddling the interior tile boundary at col 64
        other_cols = [c for c in range(grad_col.shape[1]) if c not in seam_cols]

        seam_grad_mean = grad_col[:, seam_cols].mean()
        other_grad_mean = grad_col[:, other_cols].mean()
        # The seam should not be a systematic outlier relative to the rest of
        # the field (allow generous slack since the field itself has some
        # genuine local variation; the check is for a *systematic* seam
        # artifact, not zero difference).
        assert seam_grad_mean <= other_grad_mean + 0.5

    def test_scene_boundary_pixels_are_not_zeroed(self, deterministic_scene_geotiffs):
        """Regression test for the specific Hanning-window bug found during
        development: pixels on the true outer edge of the scene must reflect
        the tile's actual prediction, not collapse toward 0 from an
        improperly-tapered edge weight."""
        unwrapper = _DeterministicPixelwiseUnwrapper()
        result = unwrapper.unwrap(
            deterministic_scene_geotiffs["wrapped"],
            deterministic_scene_geotiffs["coherence"],
            deterministic_scene_geotiffs["amplitude"],
            tile_size=64,
            overlap=32,
        )
        # Compare the merged result's outer border directly against the
        # whole-image (untiled) pass, which has no boundary-weighting concerns.
        whole_result = unwrapper.unwrap(
            deterministic_scene_geotiffs["wrapped"],
            deterministic_scene_geotiffs["coherence"],
            deterministic_scene_geotiffs["amplitude"],
            tile_size=128,
            overlap=0,
        )
        np.testing.assert_array_equal(result.ambiguity_map[0, :], whole_result.ambiguity_map[0, :])
        np.testing.assert_array_equal(
            result.ambiguity_map[-1, :], whole_result.ambiguity_map[-1, :]
        )
        np.testing.assert_array_equal(result.ambiguity_map[:, 0], whole_result.ambiguity_map[:, 0])
        np.testing.assert_array_equal(
            result.ambiguity_map[:, -1], whole_result.ambiguity_map[:, -1]
        )


# --------------------------------------------------------------------------- #
# Full pipeline (slow / marked)
# --------------------------------------------------------------------------- #


@pytest.mark.slow
class TestFullPipeline:
    def test_synthetic_to_report_end_to_end(self, tmp_workdir):
        """Synthetic generation -> preprocessing/tiling -> 1-epoch training ->
        tiled inference -> HTML report generation, run on a small scene with
        a small model so the test completes quickly on CPU."""
        # --- 1. Synthetic data generation + tiling ---
        gen = InSARSyntheticGenerator(size=128, seed=21)
        all_tiles = []
        for i, deformation_type in enumerate(["gaussian_bowl", "mogi", "none"]):
            sample = gen.generate_sample(deformation_type=deformation_type)
            rasters = NormalizedRasters(
                wrapped_phase=normalize_phase(sample.wrapped_phase),
                coherence=normalize_coherence(sample.coherence),
                amplitude=normalize_amplitude(sample.amplitude),
            )
            all_tiles.extend(
                iter_tiles(rasters, true_unwrapped=sample.unwrapped_phase, tile_size=64, overlap=16)
            )

        train_h5 = tmp_workdir / "train.h5"
        val_h5 = tmp_workdir / "val.h5"
        save_tiles_hdf5(all_tiles[: int(len(all_tiles) * 0.8)], train_h5)
        save_tiles_hdf5(all_tiles[int(len(all_tiles) * 0.8) :], val_h5)
        assert train_h5.exists() and val_h5.exists()

        # --- 2. DataLoader ---
        train_ds = InSARTileDataset(train_h5, augment=True, require_ground_truth=True, seed=0)
        val_ds = InSARTileDataset(val_h5, augment=False, require_ground_truth=True, seed=1)
        assert len(train_ds) > 0 and len(val_ds) > 0

        # --- 3. Training (1 epoch, curriculum disabled for a deterministic
        #        small run -- curriculum staging is covered separately in
        #        test_trainer-style manual verification during development) ---
        model = AmbiguityNet(pretrained=False, k_max=10.0)
        trainer = Trainer(
            model=model,
            train_dataset=train_ds,
            val_dataset=val_ds,
            out_dir=tmp_workdir / "run",
            total_epochs=1,
            warmup_epochs=1,
            batch_size=2,
            num_workers=0,
            validate_every=1,
            device="cpu",
            use_curriculum=False,
        )
        trainer.fit()
        checkpoints = list((tmp_workdir / "run" / "checkpoints").glob("*.pt"))
        assert len(checkpoints) >= 1

        # --- 4. Tiled inference on a full scene ---
        eval_sample = gen.generate_sample(deformation_type="mogi")
        transform = from_origin(500000, 5000000, 20, 20)
        profile = {
            "driver": "GTiff",
            "height": 128,
            "width": 128,
            "count": 1,
            "dtype": "float32",
            "crs": "EPSG:32633",
            "transform": transform,
        }
        paths = {}
        for name, arr in [
            ("wrapped", eval_sample.wrapped_phase),
            ("coherence", eval_sample.coherence),
            ("amplitude", eval_sample.amplitude),
        ]:
            path = tmp_workdir / f"eval_{name}.tif"
            with rasterio.open(path, "w", **profile) as dst:
                dst.write(arr.astype(np.float32), 1)
            paths[name] = path

        unwrapper = PhaseUnwrapper(model=model, device="cpu", mc_dropout_passes=1)
        result = unwrapper.unwrap(
            paths["wrapped"],
            paths["coherence"],
            paths["amplitude"],
            tile_size=64,
            overlap=32,
        )
        assert result.unwrapped_phase.shape == (128, 128)
        # The core physics identity must hold regardless of model quality.
        reconstructed = result.wrapped_phase + 2 * math.pi * result.ambiguity_map
        np.testing.assert_allclose(reconstructed, result.unwrapped_phase, atol=1e-9)

        # --- 5. Report generation ---
        from pyunwrap.analytics.report_generator import generate_report

        report_path = generate_report(
            result,
            out_dir=tmp_workdir / "report",
            true_unwrapped=eval_sample.unwrapped_phase,
        )
        assert report_path.exists()
        html = report_path.read_text()
        assert "{{" not in html and "}}" not in html
        assert (tmp_workdir / "report" / "assets").exists()

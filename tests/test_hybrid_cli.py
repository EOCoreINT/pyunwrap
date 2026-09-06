"""
tests/test_hybrid_cli.py
===========================

Test for the `pyunwrap-unwrap` CLI entry point
(`pyunwrap.inference.hybrid.main`), verified via a real end-to-end
invocation, not just argument parsing.
"""

from __future__ import annotations

import numpy as np
import rasterio
import torch
from rasterio.transform import from_origin

from pyunwrap.models.ambiguity_net import AmbiguityNet
from pyunwrap.synthetic.generator import InSARSyntheticGenerator


class TestUnwrapCLI:
    def test_end_to_end_produces_valid_output(self, tmp_path, monkeypatch):
        from pyunwrap.inference.hybrid import main

        model = AmbiguityNet(pretrained=False, k_max=10.0)
        checkpoint_path = tmp_path / "model.pt"
        torch.save(model.state_dict(), checkpoint_path)

        gen = InSARSyntheticGenerator(size=64, seed=0)
        sample = gen.generate_sample(deformation_type="none", base_coherence=0.7)
        transform = from_origin(500_000, 5_000_000, 20, 20)
        profile = {
            "driver": "GTiff",
            "height": 64,
            "width": 64,
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
            path = tmp_path / f"{name}.tif"
            with rasterio.open(path, "w", **profile) as dst:
                dst.write(arr.astype(np.float32), 1)
            paths[name] = path

        output_path = tmp_path / "out.tif"
        monkeypatch.setattr(
            "sys.argv",
            [
                "pyunwrap-unwrap",
                "--wrapped-phase",
                str(paths["wrapped"]),
                "--coherence",
                str(paths["coherence"]),
                "--amplitude",
                str(paths["amplitude"]),
                "--checkpoint",
                str(checkpoint_path),
                "--output",
                str(output_path),
                "--tile-size",
                "64",
                "--overlap",
                "0",
                "--device",
                "cpu",
            ],
        )
        main()

        assert output_path.exists()
        with rasterio.open(output_path) as src:
            data = src.read(1)
        assert data.shape == (64, 64)
        assert np.isfinite(data).all()

    def test_no_hybrid_flag_disables_hybrid_mode(self, tmp_path, monkeypatch):
        from pyunwrap.inference.hybrid import build_argparser

        parser = build_argparser()
        args = parser.parse_args(
            [
                "--wrapped-phase",
                "a",
                "--coherence",
                "b",
                "--amplitude",
                "c",
                "--checkpoint",
                "d",
                "--output",
                "e",
                "--no-hybrid",
            ]
        )
        assert args.hybrid is False

    def test_hybrid_is_default(self):
        from pyunwrap.inference.hybrid import build_argparser

        parser = build_argparser()
        args = parser.parse_args(
            [
                "--wrapped-phase",
                "a",
                "--coherence",
                "b",
                "--amplitude",
                "c",
                "--checkpoint",
                "d",
                "--output",
                "e",
            ]
        )
        assert args.hybrid is True

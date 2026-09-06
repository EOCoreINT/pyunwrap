# Quickstart

This page walks through the smallest complete example of each major stage:
generating synthetic training data, training a model, and running tiled
inference on a real interferogram. For the full, pre-executed walkthrough
with real outputs at every step, see {doc}`tutorials/index`.

## 1. Generate a synthetic sample

```python
from pyunwrap.synthetic.generator import InSARSyntheticGenerator

gen = InSARSyntheticGenerator(size=256, seed=42)
sample = gen.generate_sample(deformation_type="mogi")

print(sample.wrapped_phase.shape)      # (256, 256)
print(sample.unwrapped_phase.min(), sample.unwrapped_phase.max())
```

`sample.ambiguity` is the ground-truth integer map satisfying
`sample.wrapped_phase + 2*pi*sample.ambiguity == sample.unwrapped_phase`
exactly — this identity is the invariant the whole package is built around
(see {doc}`architecture`).

## 2. Tile and train

```python
from pyunwrap.data.preprocessing import (
    NormalizedRasters, iter_tiles, save_tiles_hdf5,
    normalize_phase, normalize_coherence, normalize_amplitude,
)
from pyunwrap.data.dataloader import InSARTileDataset
from pyunwrap.models.ambiguity_net import AmbiguityNet
from pyunwrap.training.trainer import Trainer

rasters = NormalizedRasters(
    wrapped_phase=normalize_phase(sample.wrapped_phase),
    coherence=normalize_coherence(sample.coherence),
    amplitude=normalize_amplitude(sample.amplitude),
)
tiles = list(iter_tiles(rasters, true_unwrapped=sample.unwrapped_phase, tile_size=64, overlap=16))
save_tiles_hdf5(tiles, "train.h5")

train_ds = InSARTileDataset("train.h5", augment=True, require_ground_truth=True)
val_ds = InSARTileDataset("train.h5", augment=False, require_ground_truth=True)  # same file, for brevity only
model = AmbiguityNet(pretrained=False, k_max=10.0)

trainer = Trainer(
    model=model, train_dataset=train_ds, val_dataset=val_ds,
    out_dir="runs/quickstart", total_epochs=5, device="cpu",
)
trainer.fit()
```

A real run should use two genuinely separate held-out datasets, and the CLI
entry point instead:

```bash
pyunwrap-train \
  --train-hdf5 train_tiles.h5 --val-hdf5 val_tiles.h5 \
  --epochs 60 --warmup-epochs 5 \
  --out-dir runs/pyunwrap_v1
```

## 3. Tiled inference on a real interferogram

```python
from pyunwrap.inference.unwrapper import PhaseUnwrapper

unwrapper = PhaseUnwrapper(model=model, device="cpu")
result = unwrapper.unwrap(
    wrapped_phase_path="wrapped_phase.tif",
    coherence_path="coherence.tif",
    amplitude_path="amplitude.tif",
    tile_size=512, overlap=64,   # must be a multiple of 32 -- see PhaseUnwrapper.unwrap
    generate_report=True,
)
result.save_geotiff("unwrapped_output.tif")
```

`generate_report=True` writes a self-contained HTML report (residue maps,
gradient/Nyquist analysis, error distribution if ground truth is available)
next to the input file — see `pyunwrap.analytics.report_generator` in the
{doc}`api/index`.

## Next steps

- {doc}`architecture` — the physics invariant and how data flows through
  every module.
- {doc}`tutorials/index` — full, pre-executed notebooks covering the entire
  pipeline including SNAPHU fine-tuning, explainability, and benchmarking.
- {doc}`api/index` — the complete reference.

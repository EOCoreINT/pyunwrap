# Installation

## Requirements

`pyunwrap` targets **Python 3.10+**. The core install pulls in PyTorch,
rasterio, scikit-image, and the rest of the scientific stack listed in
`pyproject.toml`.

## From source

```bash
git clone https://github.com/EOCoreINT/pyunwrap.git
cd pyunwrap
pip install -e .
```

## Optional extras

Install only what you need, or everything at once:

| Extra | Adds | Use case |
|---|---|---|
| `dev` | `pytest`, `pytest-cov`, `black`, `ruff`, `mypy` | Development, testing, linting |
| `maps` | `folium`, `leafmap` | Interactive map visualization |
| `deploy` | `openvino`, `onnx`, `onnxconverter-common`, `onnxscript` | ONNX export, OpenVINO inference |
| `notebooks` | `jupyter`, `ipykernel` | Running the example notebooks yourself |
| `snaphu` | `snaphu` (official isce-framework bindings) | SNAPHU pseudo-ground-truth fine-tuning, benchmarking |
| `docs` | `sphinx`, `furo`, `myst-parser`, `nbsphinx`, ... | Building this documentation site locally |

```bash
pip install -e ".[dev,maps,deploy,notebooks,snaphu]"   # everything except docs
```

## A note on `rasterio` and GDAL

`rasterio` depends on GDAL, a compiled C library. The PyPI wheels bundle
GDAL for the most common platforms, so `pip install` alone is usually
sufficient. If you hit a build error mentioning GDAL headers, install GDAL
via your system package manager first (e.g. `apt install gdal-bin
libgdal-dev` on Debian/Ubuntu, or use conda-forge's `gdal` package), then
retry.

## SNAPHU

The `snaphu` extra installs the official [`snaphu-py`](https://github.com/isce-framework/snaphu-py)
Python bindings, which bundle their own compiled SNAPHU binary — no separate
system install is required, unlike some older SNAPHU wrapper packages.

## Verifying the install

```python
import pyunwrap
from pyunwrap.models.ambiguity_net import AmbiguityNet

model = AmbiguityNet(pretrained=False)
print(f"AmbiguityNet parameters: {sum(p.numel() for p in model.parameters()):,}")
```

If that prints a parameter count (24,436,258 for the default configuration)
without error, the install is working. Next: {doc}`quickstart`.

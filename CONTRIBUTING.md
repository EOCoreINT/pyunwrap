# Contributing to pyunwrap

Thanks for your interest in contributing! `pyunwrap` is a young project and
contributions of all sizes are welcome -- bug reports, documentation fixes,
new synthetic deformation models, additional analytics, or core architecture
improvements.

## Getting started

```bash
git clone https://github.com/yourusername/pyunwrap.git
cd pyunwrap
pip install -e ".[dev,maps,deploy]"
```

This installs `pyunwrap` in editable mode along with:
- `dev`: `pytest`, `pytest-cov`, `black`, `ruff`, `mypy`
- `maps`: `folium`, `leafmap` (for `pyunwrap.visualization.maps`)
- `deploy`: `openvino`, `onnx`, `onnxconverter-common`, `onnxscript` (for
  `pyunwrap.utils.deployment`'s ONNX export path)

## Running the test suite

```bash
# Fast tests only (recommended for local iteration)
pytest -m "not slow"

# Full suite, including the end-to-end synthetic -> train -> infer -> report
# integration test
pytest

# With coverage
pytest --cov=pyunwrap --cov-report=term-missing
```

Tests are organized as:
- `tests/test_synthetic.py` -- the synthetic InSAR data generator
- `tests/test_model.py` -- `AmbiguityNet` and `PhysicsInformedUnwrapLoss`
- `tests/test_physics.py` -- residue detection, Nyquist analysis, explainability
- `tests/test_pipeline.py` -- multi-module integration, including the
  tile-merging boundary-artifact regression tests (marked `@pytest.mark.slow`
  where they involve a full training epoch)

All tests seed their randomness (see `tests/conftest.py`'s autouse
`_deterministic_seeds` fixture); please keep new tests deterministic too.

## Code style

`pyunwrap` uses `black` (line length 100) and `ruff` for linting. Before
opening a PR:

```bash
black pyunwrap tests
ruff check pyunwrap tests
```

Both run in CI (`.github/workflows/tests.yml`) and must pass. `ruff` is
configured to ignore `BLE001`/`S110` (broad `except Exception`) project-wide,
since several modules deliberately use broad exception handling for graceful
degradation (see the comment in `pyproject.toml`'s `[tool.ruff.lint]` section
for the specific, documented sites) -- please follow the same pattern
(document *why* at the call site) if you add a new one, rather than
suppressing it ad hoc.

Type hints are expected on new public functions; `mypy` is configured in
`pyproject.toml` but not yet enforced in CI as a hard gate.

## Design principles to keep in mind

1. **The hard physics constraint is non-negotiable.** `AmbiguityNet` predicts
   the integer ambiguity map `k`, and `phi_hat = wrapped_phase + 2*pi*k` by
   construction -- never modify the architecture to regress phase directly,
   as that reintroduces exactly the failure mode this package exists to avoid.
2. **Never average phase directly across tile boundaries.** Always merge the
   integer ambiguity map (see `pyunwrap.inference.unwrapper`'s module
   docstring for why).
3. **Loss/metric components should do what their names say.** Two of
   `PhysicsInformedUnwrapLoss`'s four components are architecturally
   near-trivial by construction; if you touch `losses.py`, please preserve
   (or improve) the docstring's explanation of *why*, so future contributors
   don't mistake a trivial value for a bug -- or a bug for expected triviality.
4. **Prefer graceful degradation over hard failures in production paths.**
   `PhaseUnwrapper.unwrap(..., generate_report=True)` must always return a
   valid core result even if reporting fails; the ONNX/OpenVINO backend
   selection must fall back rather than crash when GPU execution is
   unavailable.

## Reporting bugs

Please include: the `pyunwrap` version (or commit hash), a minimal
reproduction, and -- for anything involving phase unwrapping correctness --
the actual vs. expected wrapped/unwrapped phase values, since "off by one
cycle" bugs are usually only diagnosable with concrete numbers.

## License

By contributing, you agree that your contributions will be licensed under
the project's [MIT License](LICENSE).

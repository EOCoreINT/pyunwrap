"""
tests/conftest.py
====================

Shared pytest fixtures for the `pyunwrap` test suite.

Every fixture that involves randomness seeds explicitly (`np.random.default_rng(seed)`
or `torch.manual_seed(seed)`), so the full suite is deterministic across runs --
required for CI reproducibility and for `test_pipeline.py`'s artifact-based checks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from pyunwrap.models.ambiguity_net import AmbiguityNet
from pyunwrap.synthetic.generator import InSARSyntheticGenerator

#: Fixed seed used throughout the suite for reproducibility.
SEED = 1234


@pytest.fixture(autouse=True)
def _deterministic_seeds():
    """Autouse fixture: reset global RNG state before every test, so test
    execution order never affects individual test outcomes.
    """
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    yield


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded, isolated NumPy `Generator` for tests that need explicit randomness."""
    return np.random.default_rng(SEED)


@pytest.fixture
def small_generator() -> InSARSyntheticGenerator:
    """A small (64x64) `InSARSyntheticGenerator` for fast unit tests."""
    return InSARSyntheticGenerator(size=64, seed=SEED)


@pytest.fixture
def synthetic_sample(small_generator: InSARSyntheticGenerator):
    """One generated synthetic sample (Mogi deformation) from `small_generator`."""
    return small_generator.generate_sample(deformation_type="mogi")


@pytest.fixture
def tiny_model() -> AmbiguityNet:
    """A `pretrained=False` `AmbiguityNet` (no network access needed), for fast
    forward/backward-pass tests. Always constructed in eval mode.
    """
    model = AmbiguityNet(pretrained=False, k_max=10.0)
    model.eval()
    return model


@pytest.fixture
def dummy_input() -> torch.Tensor:
    """A physically-plausible-range dummy input tensor, shape [2, 3, 64, 64]
    (batch=2, channels=[wrapped_phase in [-1,1], coherence in [0,1],
    amplitude in [0,1]]), for model forward-pass tests.
    """
    g = torch.Generator().manual_seed(SEED)
    x = torch.rand(2, 3, 64, 64, generator=g)
    x[:, 0] = x[:, 0] * 2 - 1  # wrapped phase channel -> [-1, 1]
    x[:, 1] = x[:, 1]  # coherence -> already [0, 1]
    x[:, 2] = x[:, 2]  # amplitude -> already [0, 1]
    return x


@pytest.fixture
def tmp_workdir(tmp_path: Path) -> Path:
    """A per-test temporary working directory (thin wrapper around pytest's
    built-in `tmp_path`, kept for naming consistency across the suite).
    """
    return tmp_path

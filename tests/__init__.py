"""Shared pytest fixtures for the EVM baseline tests.

Makes the repo root importable so ``import evm`` works when running pytest
from any directory, and provides small synthetic signals we can verify the
filters against analytically.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def synthetic_pulse_clip() -> np.ndarray:
    """A 32x32 RGB clip with a global 1 Hz brightness flicker on flat grey.

    The temporal signal is a pure sinusoid at exactly 1 Hz sampled at 30 fps
    for 3 s, so a 0.83-1.0 Hz bandpass should pass it almost untouched and a
    band outside should reject it.
    """
    fps = 30.0
    t = 90
    freq = 1.0  # Hz
    n = np.arange(t)
    flicker = 0.05 * np.sin(2 * np.pi * freq * n / fps)  # +/-5% amplitude
    base = 0.5
    intensity = (base + flicker).astype(np.float32)
    frame = intensity[:, None, None, None] * np.ones(
        (t, 32, 32, 3), dtype=np.float32
    )
    return frame

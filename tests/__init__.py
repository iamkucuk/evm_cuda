"""Shared pytest fixtures for the EVM baseline tests.

Small synthetic signals we can verify the filters against analytically.
``import vidmag`` resolves through the installed distribution — there is no
``sys.path`` bridge and no repository-root ``conftest.py`` any more, so a
checkout with nothing installed fails collection on purpose. Run
``make install-dev`` (or ``pip install -e ".[dev]"``) first.
"""

from __future__ import annotations

import numpy as np
import pytest


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

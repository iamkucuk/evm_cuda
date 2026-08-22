"""The pipelines assembled from primitives must equal the direct ones.

:mod:`vidmag.backend.generic` writes the four magnification pipelines once, in
terms of the operations any backend provides. :mod:`vidmag.cpu.magnify` writes the
same four directly in NumPy, and is the version checked against the reference
implementation from the original paper.

If the two disagree, every backend built on the generic version inherits the
disagreement, and the correctness argument for the whole project stops
transferring to new hardware. So they are compared here, on the same input,
with the same parameters, and the bar is tight: both compute in double
precision through the same helper functions, so the only differences allowed
are the last bits of floating-point rounding.
"""

from __future__ import annotations

import numpy as np
import pytest

import vidmag.backend.generic as generic
import vidmag.cpu.backend as cpu_backend
from vidmag.cpu import magnify as direct

OPS = cpu_backend.OPS
FPS = 30.0


def _clip(seed: int = 11, frames: int = 40, size: int = 32) -> np.ndarray:
    """A textured image that really moves, sub-pixel, at one cycle a second.

    The motion pipelines amplify *movement*, which they find as change in the
    detail bands of a pyramid. Adding a constant to every pixel — the obvious
    way to write a "changing" clip — is a brightness change, not a movement: it
    lands entirely in the coarsest band, the detail bands see nothing, and those
    pipelines return their input untouched. A comparison against an untouched
    input passes no matter what the code does, so the clip has to move.
    """
    rng = np.random.default_rng(seed)
    base = rng.integers(60, 190, (size + 4, size + 4, 3)).astype(np.float64)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float64)

    out = np.empty((frames, size, size, 3), dtype=np.uint8)
    for t in range(frames):
        shift = 0.8 * np.sin(2 * np.pi * 1.0 * t / FPS)
        sy, sx = ys + 2.0 + shift, xs + 2.0
        y0, x0 = np.floor(sy).astype(int), np.floor(sx).astype(int)
        wy = (sy - y0)[..., None]
        top = base[y0, x0] * (1 - wy) + base[y0 + 1, x0] * wy
        out[t] = np.clip(top, 0, 255).astype(np.uint8)
    return out


def _assert_reference_did_something(
    expected: np.ndarray, clip: np.ndarray, what: str
) -> None:
    """Guard against comparing two copies of the input.

    If the reference returns its input unchanged, the comparison below holds
    for any implementation at all, including one that does nothing.
    """
    moved = np.abs(expected.astype(np.int16) - clip.astype(np.int16)).max()
    assert moved > 0, (
        f"{what}: the reference returned its input unchanged on this clip, so "
        f"this comparison would pass for any implementation"
    )


def _assert_matches(got: np.ndarray, expected: np.ndarray, what: str) -> None:
    assert got.shape == expected.shape, f"{what}: {got.shape} != {expected.shape}"
    assert got.dtype == expected.dtype, f"{what}: {got.dtype} != {expected.dtype}"
    # Both sides quantise to 8 bits at the end, so a single step of difference
    # is rounding landing either side of a boundary. More than that is a real
    # divergence in the arithmetic.
    diff = np.abs(got.astype(np.int16) - expected.astype(np.int16))
    assert diff.max() <= 1, (
        f"{what}: largest difference {diff.max()} steps, "
        f"{int((diff > 1).sum())} values differ by more than one"
    )


def test_colour_pipeline_matches_the_direct_implementation():
    clip = _clip()
    params = dict(alpha=20.0, level=2, fl=0.5, fh=1.5, chrom_attenuation=1.0)
    got = generic.color_gdown_ideal_core(OPS, clip, FPS, **params)
    expected = direct.color_gdown_ideal_core(clip, FPS, **params)
    _assert_reference_did_something(expected, clip, "colour pipeline")
    _assert_matches(got, expected, "colour pipeline")


def test_recursive_motion_pipeline_matches_the_direct_implementation():
    clip = _clip()
    params = dict(alpha=10.0, lambda_c=16.0, r1=0.4, r2=0.05, chrom_attenuation=0.1)
    got = generic.motion_lpyr_iir_core(OPS, clip, FPS, **params)
    expected = direct.motion_lpyr_iir_core(clip, FPS, **params)
    _assert_reference_did_something(expected, clip, "recursive motion pipeline")
    _assert_matches(got, expected, "recursive motion pipeline")


def test_fourier_motion_pipeline_matches_the_direct_implementation():
    clip = _clip()
    params = dict(alpha=10.0, lambda_c=16.0, fl=0.5, fh=1.5, chrom_attenuation=0.0)
    got = generic.motion_lpyr_ideal_core(OPS, clip, FPS, **params)
    expected = direct.motion_lpyr_ideal_core(clip, FPS, **params)
    _assert_reference_did_something(expected, clip, "Fourier motion pipeline")
    _assert_matches(got, expected, "Fourier motion pipeline")


def test_butterworth_motion_pipeline_matches_the_direct_implementation():
    clip = _clip()
    params = dict(alpha=10.0, lambda_c=16.0, fl=0.5, fh=1.5, chrom_attenuation=0.0)
    got = generic.motion_lpyr_butter_core(OPS, clip, FPS, **params)
    expected = direct.motion_lpyr_butter_core(clip, FPS, **params)
    _assert_reference_did_something(expected, clip, "Butterworth motion pipeline")
    _assert_matches(got, expected, "Butterworth motion pipeline")


@pytest.mark.parametrize(
    "operation",
    [
        "from_numpy",
        "to_numpy",
        "bgr_u8_to_ntsc",
        "add_and_quantize",
        "blur_dn",
        "build_lpyr",
        "recon_lpyr",
        "upsample_bilinear",
        "ideal_bandpass",
        "butter_bandpass",
        "iir_bandpass",
        "apply_gain",
    ],
)
def test_the_numpy_backend_provides_every_operation(operation):
    """The list a new backend has to implement, written down and checked."""
    assert callable(getattr(OPS, operation, None)), (
        f"the reference operations are missing {operation!r}; a backend "
        f"cannot be conformant against an incomplete reference"
    )

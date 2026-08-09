"""The portable backend, checked against the NumPy reference.

This is the backend that runs on hardware the hand-written CUDA code cannot
reach: Apple, AMD and Intel graphics processors, and ordinary processors
through a software driver. One set of kernels in ``src/evm/opencl/kernels.cl``
is compiled by whatever driver is present, so what is verified here is not one
vendor's result but the source every vendor compiles.

These tests skip when no OpenCL driver is installed. Where they do run, they
compare against :mod:`evm.cpu.backend`, the same NumPy reference the CUDA tests
use, so a result is only accepted if it agrees with the implementation that was
checked against the original paper.

Tolerances are looser than the CUDA suite's. These kernels compute in single
precision against a double-precision reference and are written for portability
rather than to match one vendor's arithmetic, so a few parts in ten million is
the honest bar.
"""

from __future__ import annotations

import numpy as np
import pytest

from evm.backend import generic
from evm.cpu import backend as cpu_backend
from evm.cpu import magnify as direct
from evm.opencl import runtime as cl_runtime

_REASON = cl_runtime.unavailable_reason()
skip_no_opencl = pytest.mark.skipif(
    _REASON is not None, reason=f"OpenCL unavailable: {_REASON}"
)

CPU = cpu_backend.OPS
FPS = 30.0

# Single precision against double precision. Measured on an Apple M2 Max at
# around 6e-7 for the worst operation (the coarsest pyramid band); this leaves
# room for other vendors' rounding without accepting a real error.
TOL = 5e-6


@pytest.fixture(scope="module")
def ops():
    from evm.opencl.ops import OpenClOps

    return OpenClOps()


def _clip(seed: int = 11, frames: int = 40, size: int = 32) -> np.ndarray:
    """A textured image that really moves, sub-pixel, at one cycle a second.

    The motion pipelines amplify *movement*, which they detect as change in the
    detail bands of a pyramid. Adding a constant to every pixel — an obvious way
    to write a "changing" clip — is a brightness change, not a movement: it
    lands entirely in the coarsest band and the detail bands see nothing, so
    those pipelines return their input untouched and any comparison against
    them passes for the wrong reason. This shifts the image instead, by a
    fraction of a pixel, which is exactly what the method is built to find.
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


def _f32(seed: int = 3, frames: int = 40) -> np.ndarray:
    return np.random.default_rng(seed).random((frames, 16, 20, 3)).astype(np.float32)


def _err(got, expected) -> float:
    return float(
        np.abs(
            np.asarray(got, dtype=np.float64) - np.asarray(expected, dtype=np.float64)
        ).max()
    )


def _not_degenerate(a, what: str) -> None:
    assert np.abs(np.asarray(a)).max() > 1e-6, f"{what}: reference is all zeros"


# ---------------------------------------------------------------------------
# Availability is reported honestly
# ---------------------------------------------------------------------------


def test_unavailable_reason_names_what_is_missing():
    """A backend that cannot run must say which of two things to install."""
    reason = cl_runtime.unavailable_reason()
    if reason is None:
        assert cl_runtime.available()
    else:
        assert "pyopencl" in reason or "driver" in reason or "devices" in reason, (
            f"reason {reason!r} does not tell the reader what to do about it"
        )


@skip_no_opencl
def test_the_device_identifies_itself(ops):
    assert cl_runtime.device_name()


# ---------------------------------------------------------------------------
# Each operation against the reference
# ---------------------------------------------------------------------------


@skip_no_opencl
def test_colour_conversion(ops):
    frames = _clip()
    got = ops.to_numpy(ops.bgr_u8_to_ntsc(ops.from_numpy(frames)))
    expected = CPU.bgr_u8_to_ntsc(frames)
    _not_degenerate(expected, "colour conversion")
    assert _err(got, expected) < TOL


@skip_no_opencl
@pytest.mark.parametrize("levels", [1, 2, 3])
def test_blur_and_downsample(ops, levels):
    host = _f32()
    got = ops.to_numpy(ops.blur_dn(ops.from_numpy(host), levels))
    expected = CPU.blur_dn(host, levels)
    _not_degenerate(expected, "blur and downsample")
    assert got.shape == expected.shape
    assert _err(got, expected) < TOL


@skip_no_opencl
@pytest.mark.parametrize("levels", [2, 3])
def test_pyramid_bands(ops, levels):
    host = _f32()
    got = [ops.to_numpy(b) for b in ops.build_lpyr(ops.from_numpy(host), levels)]
    expected = CPU.build_lpyr(host, levels)
    assert len(got) == len(expected)
    for i, (g, e) in enumerate(zip(got, expected)):
        _not_degenerate(e, f"pyramid band {i}")
        assert g.shape == e.shape, f"band {i}: {g.shape} != {e.shape}"
        assert _err(g, e) < TOL, f"band {i}"


@skip_no_opencl
def test_pyramid_reconstruction(ops):
    host = _f32()
    got = ops.to_numpy(ops.recon_lpyr(ops.build_lpyr(ops.from_numpy(host), 3)))
    expected = CPU.recon_lpyr(CPU.build_lpyr(host, 3))
    _not_degenerate(expected, "pyramid reconstruction")
    assert _err(got, expected) < TOL


@skip_no_opencl
def test_fourier_bandpass(ops):
    """Done as a matrix multiply, so this also checks that shortcut is exact.

    Selecting frequency bins is a linear map, so it can be written as one
    matrix instead of a transform and its inverse. That removes the need for a
    vendor maths library, which is what lets these kernels run anywhere — but
    only if the shortcut really is equal to the transform, which is what this
    compares.
    """
    host = _f32()
    got = ops.to_numpy(ops.ideal_bandpass(ops.from_numpy(host), 0.5, 3.0, FPS))
    expected = CPU.ideal_bandpass(host.astype(np.float64), 0.5, 3.0, FPS)
    _not_degenerate(expected, "Fourier bandpass")
    assert _err(got, expected) < TOL


@skip_no_opencl
def test_butterworth_bandpass(ops):
    host = _f32()
    got = ops.to_numpy(ops.butter_bandpass(ops.from_numpy(host), 0.5, 3.0, FPS))
    expected = CPU.butter_bandpass(host.astype(np.float64), 0.5, 3.0, FPS)
    _not_degenerate(expected, "Butterworth bandpass")
    assert _err(got, expected) < TOL


@skip_no_opencl
def test_recursive_bandpass(ops):
    host = _f32()
    got = ops.to_numpy(ops.iir_bandpass(ops.from_numpy(host), 0.4, 0.05))
    expected = CPU.iir_bandpass(host.astype(np.float64), 0.4, 0.05)
    _not_degenerate(expected, "recursive bandpass")
    assert _err(got, expected) < TOL


@skip_no_opencl
def test_upsampling(ops):
    host = _f32()
    got = ops.to_numpy(ops.upsample_bilinear(ops.from_numpy(host), 32, 40))
    expected = CPU.upsample_bilinear(host, 32, 40)
    _not_degenerate(expected, "upsampling")
    assert got.shape == expected.shape
    assert _err(got, expected) < TOL


# ---------------------------------------------------------------------------
# The pipelines, end to end
# ---------------------------------------------------------------------------


@skip_no_opencl
@pytest.mark.parametrize(
    "name,core,reference,params",
    [
        (
            "colour",
            generic.color_gdown_ideal_core,
            direct.color_gdown_ideal_core,
            dict(alpha=20.0, level=2, fl=0.5, fh=1.5, chrom_attenuation=1.0),
        ),
        (
            "motion recursive",
            generic.motion_lpyr_iir_core,
            direct.motion_lpyr_iir_core,
            dict(alpha=10.0, lambda_c=16.0, r1=0.4, r2=0.05, chrom_attenuation=0.1),
        ),
        (
            "motion Fourier",
            generic.motion_lpyr_ideal_core,
            direct.motion_lpyr_ideal_core,
            dict(alpha=10.0, lambda_c=16.0, fl=0.5, fh=1.5, chrom_attenuation=0.0),
        ),
        (
            "motion Butterworth",
            generic.motion_lpyr_butter_core,
            direct.motion_lpyr_butter_core,
            dict(alpha=10.0, lambda_c=16.0, fl=0.5, fh=1.5, chrom_attenuation=0.0),
        ),
    ],
)
def test_whole_pipeline_matches_the_reference(ops, name, core, reference, params):
    """The output a user actually gets, compared frame by frame.

    Both sides finish by rounding to 8 bits, so a difference of one step is
    rounding landing either side of a boundary. Anything larger is a real
    divergence.
    """
    clip = _clip()
    got = core(ops, clip, FPS, **params)
    expected = reference(clip, FPS, **params)

    assert np.abs(expected.astype(np.int16) - clip.astype(np.int16)).max() > 0, (
        "the reference did nothing on this clip; the comparison is meaningless"
    )
    diff = np.abs(got.astype(np.int16) - expected.astype(np.int16))
    assert diff.max() <= 1, (
        f"{name}: largest difference {diff.max()} steps, "
        f"{int((diff > 1).sum())} values off by more than one"
    )


@skip_no_opencl
def test_the_backend_is_selectable_by_name():
    """What a user types must reach this backend and say that it did."""
    import evm

    clip = _clip(frames=24)
    out = evm.magnify(clip, preset="motion", fps=FPS, backend="opencl")
    assert out.shape == clip.shape
    assert out.dtype == np.uint8


def test_selecting_it_without_a_driver_explains_why():
    """Asking for a backend that cannot run must not fall back silently."""
    import evm
    from evm.backend.registry import BackendUnavailableError

    if cl_runtime.available():
        pytest.skip("OpenCL is available here, so there is no failure to check")

    with pytest.raises(BackendUnavailableError, match="opencl"):
        evm.magnify(_clip(frames=8), preset="motion", fps=FPS, backend="opencl")

"""Each public GPU operation against its NumPy counterpart.

The tests beside this one check the compiled kernels through the raw bindings.
These check the layer users actually touch: :mod:`evm.cuda.ops`, which wraps
those kernels so they take and return arrays that know their own shape. The
wrapping is where a layout shuffle can go wrong without the kernel being at
fault, so it needs its own comparison against :mod:`evm.cpu.ops`.

Tolerances come from ``tests/cuda/conftest.py`` rather than being written again
here: single-precision GPU arithmetic against double-precision NumPy has a
known, documented spread per operation, and there is one place that records it.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import TOL, abs_err, have_cuda, skip_no_cuda

if have_cuda:
    import evm.cpu.ops as cpu_ops
    from evm.cuda import ops as gpu_ops
    from evm.cuda.array import DeviceArray


T, H, W = 12, 24, 32


def _frames_u8(seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (T, H, W, 3), dtype=np.uint8)


def _frames_f32(seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((T, H, W, 3)).astype(np.float32)


def _not_degenerate(a: np.ndarray, what: str) -> None:
    """A comparison of two all-zero arrays proves nothing."""
    assert np.abs(a).max() > 1e-6, f"{what} produced an all-zero reference"


@skip_no_cuda
def test_colour_conversion_matches_the_reference():
    host = _frames_u8()
    expected = cpu_ops.bgr_u8_to_ntsc(host)
    got = gpu_ops.bgr_u8_to_ntsc(DeviceArray.from_numpy(host)).numpy()
    _not_degenerate(expected, "colour conversion")
    assert got.shape == expected.shape
    assert abs_err(got, expected) < TOL["color_cvt"]


@skip_no_cuda
@pytest.mark.parametrize("levels", [1, 2, 3])
def test_blur_and_downsample_matches_the_reference(levels):
    host = _frames_f32()
    expected = cpu_ops.blur_dn(host, levels)
    got = gpu_ops.blur_dn(DeviceArray.from_numpy(host), levels).numpy()
    _not_degenerate(expected, "blur_dn")
    assert got.shape == expected.shape, f"{got.shape} != {expected.shape}"
    assert abs_err(got, expected) < TOL["blur_dn"]


@skip_no_cuda
@pytest.mark.parametrize("levels", [2, 3])
def test_pyramid_bands_match_the_reference(levels):
    host = _frames_f32()
    expected = cpu_ops.build_lpyr(host, levels)
    got = [b.numpy() for b in gpu_ops.build_lpyr(DeviceArray.from_numpy(host), levels)]
    assert len(got) == len(expected)
    for i, (g, e) in enumerate(zip(got, expected)):
        _not_degenerate(e, f"pyramid band {i}")
        assert g.shape == e.shape, f"band {i}: {g.shape} != {e.shape}"
        assert abs_err(g, e) < TOL["corr_dn"], f"band {i}"


@skip_no_cuda
def test_pyramid_reconstruction_matches_the_reference():
    """The reconstruction step, compared rather than only round-tripped.

    A round trip can hide a matched pair of errors: a build that is wrong in
    one direction and a reconstruction wrong in the other still returns the
    input. Comparing the reconstruction itself against the NumPy version
    catches that.
    """
    host = _frames_f32()
    dev_bands = gpu_ops.build_lpyr(DeviceArray.from_numpy(host), 3)
    got = gpu_ops.recon_lpyr(dev_bands, H, W).numpy()

    cpu_bands = cpu_ops.build_lpyr(host, 3)
    expected = cpu_ops.recon_lpyr(cpu_bands, H, W)

    _not_degenerate(expected, "pyramid reconstruction")
    assert got.shape == expected.shape, f"{got.shape} != {expected.shape}"
    assert abs_err(got, expected) < TOL["up_conv"]


@skip_no_cuda
def test_pyramid_round_trip_returns_the_original():
    """Building a pyramid and summing it back must return what went in."""
    host = _frames_f32()
    bands = gpu_ops.build_lpyr(DeviceArray.from_numpy(host), 3)
    back = gpu_ops.recon_lpyr(bands, H, W).numpy()

    # recon returns one plane per channel: (channel*time, H, W).
    rebuilt = np.empty_like(host)
    for c in range(3):
        rebuilt[..., c] = back[c * T : (c + 1) * T]
    assert abs_err(rebuilt, host) < TOL["lpyr_roundtrip"]


@skip_no_cuda
def test_ideal_bandpass_matches_the_reference():
    host = _frames_f32()
    # 30 frames at 30 fps resolve 1 Hz steps; this band keeps several bins.
    expected = cpu_ops.ideal_bandpass(host.astype(np.float64), 0.5, 3.0, 30.0)
    got = gpu_ops.ideal_bandpass(DeviceArray.from_numpy(host), 0.5, 3.0, 30.0).numpy()
    _not_degenerate(expected, "ideal_bandpass")
    assert abs_err(got, expected) < TOL["ideal"]


@skip_no_cuda
def test_butterworth_bandpass_matches_the_reference():
    """This binding did not exist before the public operations layer.

    The kernel and its launcher were written long ago, but nothing exposed a
    device-resident version to Python, so this path had never been compared
    against the reference at all.
    """
    host = _frames_f32()
    expected = cpu_ops.butter_bandpass(host.astype(np.float64), 0.5, 3.0, 30.0)
    got = gpu_ops.butter_bandpass(DeviceArray.from_numpy(host), 0.5, 3.0, 30.0).numpy()
    _not_degenerate(expected, "butter_bandpass")
    assert abs_err(got, expected) < TOL["butter"]


@skip_no_cuda
def test_recursive_bandpass_matches_the_reference():
    host = _frames_f32()
    expected = cpu_ops.iir_bandpass(host.astype(np.float64), 0.4, 0.05)
    got = gpu_ops.iir_bandpass(DeviceArray.from_numpy(host), 0.4, 0.05).numpy()
    _not_degenerate(expected, "iir_bandpass")
    assert abs_err(got, expected) < TOL["iir"]


@skip_no_cuda
def test_channel_gain_matches_the_reference():
    host = _frames_f32()
    expected = cpu_ops.apply_gain(host.astype(np.float64), 50.0, 5.0, 5.0)
    got = gpu_ops.apply_gain(DeviceArray.from_numpy(host), 50.0, 5.0, 5.0).numpy()
    # A relative comparison, not the absolute tolerances used above. Those are
    # calibrated for values of order one; multiplying by a gain of 50 scales the
    # absolute error with it, while the accuracy of the multiply itself is
    # unchanged. The bound below is a few times single-precision resolution.
    assert np.allclose(got, expected, rtol=1e-6, atol=0), (
        f"largest relative difference "
        f"{np.max(np.abs(got - expected) / np.maximum(np.abs(expected), 1e-12)):.2e}"
    )


# ---------------------------------------------------------------------------
# The checking the raw bindings do not do
# ---------------------------------------------------------------------------


@skip_no_cuda
def test_operations_reject_a_wrong_dtype_instead_of_reading_raw_bytes():
    wrong = DeviceArray.from_numpy(_frames_u8())  # uint8, not float32
    with pytest.raises(TypeError, match="dtype"):
        gpu_ops.blur_dn(wrong, 1)


@skip_no_cuda
def test_operations_reject_a_wrong_shape():
    flat = DeviceArray.from_numpy(np.zeros(64, dtype=np.float32))
    with pytest.raises(ValueError, match="dimensions"):
        gpu_ops.iir_bandpass(flat, 0.4, 0.05)


@skip_no_cuda
def test_operations_reject_a_plain_numpy_array():
    """Passing host memory where a device array belongs must not be silent."""
    with pytest.raises(TypeError, match="DeviceArray"):
        gpu_ops.blur_dn(_frames_f32(), 1)


@skip_no_cuda
def test_a_chain_of_operations_stays_on_the_device():
    """The point of the layer: no copy back to the host between steps."""
    host = _frames_u8()
    ntsc = gpu_ops.bgr_u8_to_ntsc(DeviceArray.from_numpy(host))
    small = gpu_ops.blur_dn(ntsc, 2)
    band = gpu_ops.iir_bandpass(small, 0.4, 0.05)
    amplified = gpu_ops.apply_gain(band, 10.0, 1.0, 1.0)
    assert amplified.device == "cuda"
    assert amplified.shape == small.shape
    _not_degenerate(amplified.numpy(), "chained operations")

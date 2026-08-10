"""One conformance suite, run against every backend that can run here.

These are the backends that reach hardware the hand-written CUDA code cannot:
OpenCL for Apple, AMD and Intel graphics processors and for ordinary processors
through a software driver; Metal for Apple hardware specifically; Vulkan where
a driver is present. Each has its own kernels in its own language, and each is
compared here against :mod:`evm.cpu.backend` — the same NumPy reference the
CUDA tests use — so a result is only accepted if it agrees with the
implementation that was checked against the original paper.

Every test is parameterised over whichever backends are available on the
machine running it, and skips the rest with the reason. Adding a backend means
adding one entry to the table below, not another file: a suite that has to be
copied per backend is a suite that drifts per backend.

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
from evm.metal import runtime as metal_runtime
from evm.opencl import runtime as cl_runtime
from evm.vulkan import runtime as vk_runtime

CPU = cpu_backend.OPS
FPS = 30.0


def _backends():
    """Every portable backend, with why it cannot run if it cannot.

    Building the list at collection time means a machine with no graphics
    hardware still reports one skip per backend per test, naming the reason,
    rather than silently testing nothing.
    """
    entries = []
    for name, module, factory in (
        (
            "opencl",
            cl_runtime,
            lambda: __import__("evm.opencl.ops", fromlist=["OpenClOps"]).OpenClOps(),
        ),
        (
            "metal",
            metal_runtime,
            lambda: __import__("evm.metal.ops", fromlist=["MetalOps"]).MetalOps(),
        ),
        (
            "vulkan",
            vk_runtime,
            lambda: __import__("evm.vulkan.ops", fromlist=["VulkanOps"]).VulkanOps(),
        ),
    ):
        reason = module.unavailable_reason()
        entries.append(
            pytest.param(
                factory,
                id=name,
                marks=pytest.mark.skipif(
                    reason is not None, reason=f"{name} unavailable: {reason}"
                ),
            )
        )
    return entries


BACKENDS = _backends()

# Single precision against double precision. Measured on an Apple M2 Max at
# around 6e-7 for the worst operation (the coarsest pyramid band); this leaves
# room for other vendors' rounding without accepting a real error.
TOL = 5e-6


@pytest.fixture
def ops(request):
    """The operations for the backend this test run is parameterised on."""
    return request.param()


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


@pytest.mark.parametrize(
    "module", [cl_runtime, metal_runtime, vk_runtime], ids=["opencl", "metal", "vulkan"]
)
def test_unavailable_reason_names_what_is_missing(module):
    """A backend that cannot run must say what to do about it.

    The two ways it can be missing need different fixes — a Python package that
    can be installed, or a driver or piece of hardware that cannot — so the
    message has to distinguish them rather than reporting a bare failure.
    """
    reason = module.unavailable_reason()
    if reason is None:
        assert module.available()
    else:
        assert any(
            word in reason
            for word in ("install", "driver", "devices", "device", "macOS")
        ), f"reason {reason!r} does not tell the reader what to do about it"


@pytest.mark.parametrize("ops", BACKENDS, indirect=True)
def test_the_device_identifies_itself(ops):
    assert ops.name


# ---------------------------------------------------------------------------
# Each operation against the reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ops", BACKENDS, indirect=True)
def test_colour_conversion(ops):
    frames = _clip()
    got = ops.to_numpy(ops.bgr_u8_to_ntsc(ops.from_numpy(frames)))
    expected = CPU.bgr_u8_to_ntsc(frames)
    _not_degenerate(expected, "colour conversion")
    assert _err(got, expected) < TOL


@pytest.mark.parametrize("ops", BACKENDS, indirect=True)
@pytest.mark.parametrize("levels", [1, 2, 3])
def test_blur_and_downsample(ops, levels):
    host = _f32()
    got = ops.to_numpy(ops.blur_dn(ops.from_numpy(host), levels))
    expected = CPU.blur_dn(host, levels)
    _not_degenerate(expected, "blur and downsample")
    assert got.shape == expected.shape
    assert _err(got, expected) < TOL


@pytest.mark.parametrize("ops", BACKENDS, indirect=True)
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


@pytest.mark.parametrize("ops", BACKENDS, indirect=True)
def test_pyramid_reconstruction(ops):
    host = _f32()
    got = ops.to_numpy(ops.recon_lpyr(ops.build_lpyr(ops.from_numpy(host), 3)))
    expected = CPU.recon_lpyr(CPU.build_lpyr(host, 3))
    _not_degenerate(expected, "pyramid reconstruction")
    assert _err(got, expected) < TOL


@pytest.mark.parametrize("ops", BACKENDS, indirect=True)
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


@pytest.mark.parametrize("ops", BACKENDS, indirect=True)
def test_butterworth_bandpass(ops):
    host = _f32()
    got = ops.to_numpy(ops.butter_bandpass(ops.from_numpy(host), 0.5, 3.0, FPS))
    expected = CPU.butter_bandpass(host.astype(np.float64), 0.5, 3.0, FPS)
    _not_degenerate(expected, "Butterworth bandpass")
    assert _err(got, expected) < TOL


@pytest.mark.parametrize("ops", BACKENDS, indirect=True)
def test_recursive_bandpass(ops):
    host = _f32()
    got = ops.to_numpy(ops.iir_bandpass(ops.from_numpy(host), 0.4, 0.05))
    expected = CPU.iir_bandpass(host.astype(np.float64), 0.4, 0.05)
    _not_degenerate(expected, "recursive bandpass")
    assert _err(got, expected) < TOL


@pytest.mark.parametrize("ops", BACKENDS, indirect=True)
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


@pytest.mark.parametrize("ops", BACKENDS, indirect=True)
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


@pytest.mark.parametrize("name", ["opencl", "metal", "vulkan"])
def test_the_backend_is_selectable_by_name(name):
    """What a user types must reach that backend, or say why it cannot."""
    import evm
    from evm.backend.registry import BackendUnavailableError

    clip = _clip(frames=24)
    try:
        out = evm.magnify(clip, preset="motion", fps=FPS, backend=name)
    except BackendUnavailableError as exc:
        # Not a failure: this machine has no such device. What matters is that
        # it said so rather than quietly running somewhere else.
        assert name in str(exc)
        pytest.skip(f"{name} unavailable here: {exc}")
    assert out.shape == clip.shape
    assert out.dtype == np.uint8


@pytest.mark.parametrize(
    "name,module",
    [("opencl", cl_runtime), ("metal", metal_runtime), ("vulkan", vk_runtime)],
)
def test_selecting_it_without_a_driver_explains_why(name, module):
    """Asking for a backend that cannot run must not fall back silently."""
    import evm
    from evm.backend.registry import BackendUnavailableError

    if module.available():
        pytest.skip(f"{name} is available here, so there is no failure to check")

    with pytest.raises(BackendUnavailableError, match=name):
        evm.magnify(_clip(frames=8), preset="motion", fps=FPS, backend=name)

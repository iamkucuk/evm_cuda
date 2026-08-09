"""Tests for the ``evm.magnify`` facade (plan step 3.3) and the built-in
backend registrations (step 3.2).

What is pinned down here:

* **array in, array out, no filesystem.** The facade's whole point is that a
  clip already in memory never has to be written to a file first, so the decoder
  is sabotaged for the duration of those tests: if anything reached
  ``cv2.VideoCapture``, they fail.
* **plan decision D8.** The path functions drop the last ten frames because the
  MATLAB reference does; ``magnify()`` drops none unless told to. Both halves
  are asserted, including the byte-for-byte equality that proves ``drop_last=10``
  is the *only* difference between the two routes — for every preset, and for a
  clip handed in as a path *and* as the array that same file decodes to.
* **every preset actually magnifies.** Each row of ``evm.presets.PRESETS`` is
  run end to end at a frame rate where its band contains signal, and must move
  the picture. A preset that quietly does nothing would otherwise ship looking
  like a feature.
* **no silent fallback.** ``backend="cuda"`` on a machine without it raises and
  names the reason; ``precision="fp16"`` on a backend with no fp16 body raises
  rather than quietly computing in fp32. The CPU/GPU cliff is ~700x, so neither
  may ever happen by accident.
* **the choice is discoverable.** The selected backend is logged, and
  ``evm.backend.select()`` answers the same question before the run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import evm
from evm import backend
from evm.cpu import magnify as cpu_magnify
from evm.presets import PRESETS, Preset

from .test_golden import load_input, write_lossless_video

FIXTURES = Path(__file__).resolve().parent / "fixtures"

# Small, fast parameters for the two presets; the band brackets the fixture
# clip's 2 Hz oscillation exactly as tests/test_golden.py does.
PULSE_OVERRIDES = dict(alpha=10.0, level=2, fl=1.5, fh=2.5)


@pytest.fixture(scope="module")
def frames() -> np.ndarray:
    """The committed synthetic clip: (40, 32, 32, 3) uint8 BGR."""
    return load_input()[0]


@pytest.fixture(scope="module")
def retained(frames) -> np.ndarray:
    """The 30 frames the path functions keep — the window the clip was built for.

    ``tests/test_golden.py`` chose the fixture so its 2 Hz oscillation lands on
    exactly DFT bin 2 of *these* frames, where the ideal filter's brick wall
    passes it cleanly instead of straddling two bins. Running the preset tests
    on this window is what makes them sensitive rather than lucky.
    """
    return frames[: len(frames) - cpu_magnify.DROP_LAST]


@pytest.fixture
def no_video_decode(monkeypatch: pytest.MonkeyPatch):
    """Make any attempt to open a video file fail loudly."""
    import cv2

    def _boom(*a, **k):
        raise AssertionError("magnify() touched the filesystem decoder")

    monkeypatch.setattr(cv2, "VideoCapture", _boom)


def in_band_fps(preset: Preset, frames: np.ndarray) -> float | None:
    """The frame rate that puts ``frames``' oscillation mid-band for ``preset``.

    A preset whose band is quoted in Hz can only act on a clip carrying motion
    inside that band, and the frame rate is what relates the two: the fixture
    clip oscillates once every fixed number of *frames*, so declaring a rate is
    what decides its frequency in Hz. The rate is solved for rather than
    tabulated per preset, which would go stale the moment the fixture is
    regenerated: bin ``k`` of ``N`` frames sits at ``k * fps / N`` Hz, so
    ``fps = centre * N / k`` moves the clip's own bin to the middle of
    ``[fl, fh]``. That is 13.75 fps for ``pulse`` and 1230 fps for
    ``vibration``, whose preset docstring warns that its 72-92 Hz band needs a
    high-speed camera.

    Returns ``None`` for a preset with no ``fl``/``fh``: the r1/r2 IIR band runs
    on frame index and has no sampling rate to set, which is exactly the case
    ``magnify(fps=None)`` has to handle.
    """
    if "fl" not in preset.params:
        return None
    signal = frames.reshape(len(frames), -1).mean(axis=1).astype(np.float64)
    bin_index = int(np.abs(np.fft.rfft(signal - signal.mean())).argmax())
    assert bin_index > 0, "the fixture clip carries no temporal oscillation to band-pass"
    centre = (preset.params["fl"] + preset.params["fh"]) / 2.0
    return centre * len(frames) / bin_index


# ---------------------------------------------------------------------------
# Array in, array out
# ---------------------------------------------------------------------------


def test_pulse_preset_on_an_array_touches_no_files(frames, no_video_decode):
    out = evm.magnify(frames, preset="pulse", fps=30.0, **PULSE_OVERRIDES)
    assert out.shape == frames.shape
    assert out.dtype == np.uint8
    assert not np.array_equal(out, frames), "magnification did nothing"


def test_motion_preset_on_an_array_needs_no_fps(frames, no_video_decode):
    """The r1/r2 IIR band has no sampling rate, so the facade must not demand one."""
    out = evm.magnify(frames, preset="motion")
    assert out.shape == frames.shape and out.dtype == np.uint8
    assert not np.array_equal(out, frames)


def test_the_facade_calls_the_same_core_with_the_same_parameters(frames):
    """No reinterpretation between preset table and core.

    ``backend="cpu"`` because the reference side is a CPU core: on ``"auto"``
    this would silently become a cross-backend equality check on any machine
    with a GPU, and CUDA agrees with the oracle to about 1 LSB, not exactly.
    """
    direct = cpu_magnify.color_gdown_ideal_core(
        frames, 30.0, chrom_attenuation=1.0, **PULSE_OVERRIDES
    )
    viafacade = evm.magnify(
        frames, preset="pulse", backend="cpu", fps=30.0, **PULSE_OVERRIDES
    )
    assert np.array_equal(direct, viafacade)


def test_an_iterable_of_frames_is_accepted(frames, no_video_decode):
    out = evm.magnify(list(frames), preset="motion")
    assert np.array_equal(out, evm.magnify(frames, preset="motion"))


def test_overrides_beat_the_preset(frames, no_video_decode):
    weak = evm.magnify(frames, preset="motion", alpha=1.0)
    strong = evm.magnify(frames, preset="motion", alpha=40.0)
    ref = frames.astype(np.int16)
    assert (np.abs(strong.astype(np.int16) - ref).mean()
            > np.abs(weak.astype(np.int16) - ref).mean())


def test_an_unknown_override_names_itself(frames, no_video_decode):
    with pytest.raises(TypeError, match="lambda_c"):
        evm.magnify(frames, preset="pulse", fps=30.0, lambda_c=16.0)


# ---------------------------------------------------------------------------
# Every shipped preset, end to end
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset_name", sorted(PRESETS))
def test_every_preset_runs_and_moves_the_picture(preset_name, retained, no_video_decode):
    """A preset that quietly does nothing is worse than a missing one.

    It ships as a working feature and returns output indistinguishable from the
    input, so nobody finds out. That is one frame rate away for ``vibration``:
    below about 184 fps its 72 Hz lower cutoff is above Nyquist, the ideal
    bandpass passes nothing, and the render is the source clip. So every row of
    the table is run — a new preset joins this test by existing — at a rate
    where its band contains signal, and the result must differ from the input by
    more than rounding.
    """
    spec = PRESETS[preset_name]
    out = evm.magnify(retained, preset=preset_name, fps=in_band_fps(spec, retained))

    assert out.shape == retained.shape and out.dtype == np.uint8
    moved = float(np.abs(out.astype(np.int16) - retained.astype(np.int16)).mean())
    # Measured on this clip: motion 7.6, vibration 45.1, pulse 59.2 LSB of mean
    # change. The threshold sits far below the smallest of the three, so it
    # catches "did nothing" without pinning any pipeline's exact strength.
    assert moved > 1.0, (
        f"preset {preset_name!r} changed the clip by {moved:.3f} LSB on average: "
        "it is running but not magnifying"
    )


# ---------------------------------------------------------------------------
# Plan decision D8: who drops the last ten frames
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def source_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The committed clip as a losslessly-encoded file (see tests/test_golden.py)."""
    clip, fps = load_input()
    path = tmp_path_factory.mktemp("api") / "synthetic.mkv"
    write_lossless_video(path, clip, fps)
    return path


def test_array_input_keeps_every_frame_by_default(frames, no_video_decode):
    assert len(evm.magnify(frames, preset="motion")) == len(frames)


def test_path_input_keeps_every_frame_by_default(source_video, frames):
    assert len(evm.magnify(str(source_video), preset="motion")) == len(frames)


@pytest.mark.parametrize("form", ["path", "array"])
@pytest.mark.parametrize("preset_name", sorted(PRESETS))
def test_drop_last_10_reproduces_the_path_function_byte_for_byte(
    preset_name, form, source_video, tmp_path
):
    """Decision D8 in full: one clip, one parameter set, two doors.

    ``magnify(..., drop_last=10)`` has to *be* the legacy ``magnify_<stem>``
    call — whether the clip arrives as the path that function opens itself, or
    as the array that same file decodes to. That is the whole content of D8: the
    routes differ in which frames they keep and in nothing else, so passing the
    reference's ten back in must close the gap completely.

    Equality is exact (``array_equal``), not approximate, and has to be: both
    routes hand the same uint8 frames to the same float64 core, and the only
    arithmetic between the two results is the ``/255`` the path wrapper applies
    on its way out. Both sides are therefore exactly ``k/255`` in float32 and a
    tolerance here could only hide a real difference. Measured max difference,
    all six cases: 0.

    The shipped preset parameters are used rather than test-local stand-ins, so
    this also checks that every preset is expressible through both entry points.
    ``sampling_rate`` is pinned (to the rate that puts the clip's oscillation
    in-band) so the comparison is between two magnified clips: at the file's own
    30 fps the ``pulse`` band would sit between bins and both sides would agree
    on having done almost nothing.

    ``backend="cpu"`` is pinned because the four ``magnify_*`` functions *are*
    the CPU baseline: on ``"auto"`` this would silently become a CPU-versus-CUDA
    equality check on any machine with a GPU — which it is not, and which fails
    on the 3090 at 1 LSB for two of the three presets. That question belongs to
    ``tests/cuda/test_api_cuda.py``, under a measured tolerance.
    """
    spec = PRESETS[preset_name]
    clip = frames_of(source_video)
    kept = clip[: len(clip) - cpu_magnify.DROP_LAST]

    params = dict(spec.params)
    rate = in_band_fps(spec, kept)
    if rate is not None:
        params["sampling_rate"] = rate

    legacy = getattr(evm, f"magnify_{spec.pipeline}")(
        str(source_video), str(tmp_path / f"{preset_name}.mp4"), **params
    )
    viafacade = evm.magnify(
        str(source_video) if form == "path" else clip,
        preset=preset_name,
        backend="cpu",
        drop_last=cpu_magnify.DROP_LAST,
        **params,
    )

    assert len(viafacade) == len(kept)
    assert np.array_equal(viafacade.astype(np.float32) / 255.0, legacy)


def frames_of(path: Path) -> np.ndarray:
    import cv2

    cap = cv2.VideoCapture(str(path))
    got = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        got.append(fr)
    cap.release()
    return np.stack(got, axis=0)


def test_drop_last_applies_to_arrays_too(frames, no_video_decode):
    out = evm.magnify(frames, preset="motion", drop_last=10)
    assert len(out) == len(frames) - 10


def test_a_negative_drop_last_is_refused(frames, no_video_decode):
    with pytest.raises(ValueError, match="drop_last"):
        evm.magnify(frames, preset="motion", drop_last=-1)


# ---------------------------------------------------------------------------
# Writing a file
# ---------------------------------------------------------------------------


def test_out_writes_a_video_and_still_returns_the_frames(frames, tmp_path):
    dest = tmp_path / "written.mp4"
    out = evm.magnify(frames, preset="motion", fps=30.0, out=dest)
    assert dest.exists() and dest.stat().st_size > 0
    assert out.shape == frames.shape and out.dtype == np.uint8


def test_writing_without_a_frame_rate_is_refused(frames, tmp_path, no_video_decode):
    with pytest.raises(ValueError, match="fps"):
        evm.magnify(frames, preset="motion", out=tmp_path / "nope.mp4")


# ---------------------------------------------------------------------------
# Presets and precision
# ---------------------------------------------------------------------------


def test_a_missing_preset_lists_the_available_ones(frames, no_video_decode):
    with pytest.raises(ValueError) as exc:
        evm.magnify(frames)
    assert "pulse" in str(exc.value) and "motion" in str(exc.value)


def test_an_unknown_preset_lists_the_available_ones(frames, no_video_decode):
    with pytest.raises(KeyError, match="pulse"):
        evm.magnify(frames, preset="heartbeat")


def test_fps_is_required_when_the_band_is_in_hz(frames, no_video_decode):
    with pytest.raises(ValueError, match="fps="):
        evm.magnify(frames, preset="pulse", **PULSE_OVERRIDES)


def test_pinning_the_sampling_rate_removes_the_need_for_fps(frames, no_video_decode):
    out = evm.magnify(
        frames, preset="pulse", sampling_rate=30.0, **PULSE_OVERRIDES
    )
    assert out.shape == frames.shape


def test_fp16_on_a_backend_without_it_raises_rather_than_computing_fp32(
    frames, no_video_decode
):
    with pytest.raises(ValueError) as exc:
        evm.magnify(frames, preset="motion", backend="cpu", precision="fp16")
    msg = str(exc.value)
    assert "cpu" in msg and "fp16" in msg and "motion_lpyr_iir" in msg


def test_an_unknown_precision_is_refused(frames, no_video_decode):
    with pytest.raises(ValueError, match="precision"):
        evm.magnify(frames, preset="motion", precision="bf16")


# ---------------------------------------------------------------------------
# Backend selection: registered, honest, never silent
# ---------------------------------------------------------------------------


def test_importing_evm_registers_the_built_in_backends():
    names = [info.name for info in backend.list_backends()]
    # Preference order, not registration order: the hand-written CUDA code
    # first, then the portable OpenCL kernels, then the NumPy reference.
    assert names == ["cuda", "opencl", "cpu"], names
    caps = {info.name: info.capabilities for info in backend.list_backends()}
    assert caps["cpu"].dtypes == ("float64",)      # the oracle's working dtype
    assert caps["cuda"].dtypes == ("float32", "float16")
    assert caps["opencl"].dtypes == ("float32",)   # the portable kernels
    # None of them can stream yet; that is a later phase, and claiming it
    # before it exists would be a lie a caller could act on.
    assert not any(c.streaming for c in caps.values())


def test_the_cpu_backend_is_always_available():
    info = {i.name: i for i in backend.list_backends()}["cpu"]
    assert info.available and info.unavailable_reason is None


def test_asking_for_cuda_is_answered_honestly(frames, no_video_decode, caplog):
    """Either it runs on the GPU, or it says why it cannot. Never a quiet CPU run.

    Both halves are real assertions, so this test skips on no host: a machine
    with the extension proves the GPU path end to end, a machine without it
    proves the refusal.
    """
    from evm.cuda import have_cuda

    if have_cuda:
        with caplog.at_level("INFO", logger="evm"):
            out = evm.magnify(frames, preset="motion", backend="cuda")
        assert out.shape == frames.shape and out.dtype == np.uint8
        assert "backend='cuda'" in caplog.text
        return

    with pytest.raises(backend.BackendUnavailableError) as exc:
        evm.magnify(frames, preset="motion", backend="cuda")
    msg = str(exc.value)
    assert "cuda" in msg
    assert "_evm_cuda" in msg or "not importable" in msg

    # The message must carry the real cause, not a generic "unavailable": what
    # the reader needs is which of the two conditions failed, and it is quoted
    # from the probe rather than paraphrased, so there is one sentence about
    # this condition in the project and not two that can drift.
    reason = {i.name: i for i in backend.list_backends()}["cuda"].unavailable_reason
    assert reason and reason in msg
    assert "not built" in msg or "no CUDA device" in msg


def test_an_unknown_backend_name_lists_the_registered_ones(frames, no_video_decode):
    with pytest.raises(backend.UnknownBackendError) as exc:
        evm.magnify(frames, preset="motion", backend="vulkan")
    message = str(exc.value)
    assert "cpu" in message and "cuda" in message and "opencl" in message


def test_the_backend_actually_used_is_reported(frames, no_video_decode, caplog):
    """Discoverability: the run says which backend it used, and select() agrees."""
    with caplog.at_level("INFO", logger="evm"):
        evm.magnify(frames, preset="motion")
    assert "evm.magnify" in caplog.text
    chosen, _ = backend.select("auto")
    assert f"backend={chosen!r}" in caplog.text


def test_the_cpu_backend_satisfies_the_pipelines_protocol():
    """The CPU oracle is the protocol's reference implementation.

    ``isinstance`` is only meaningful for it: since Python 3.12 a
    runtime-checkable protocol resolves members with ``inspect.getattr_static``,
    which does not run a module's ``__getattr__`` — and ``evm.cuda`` resolves its
    cores through exactly that hook to stay lazy on GPU-less machines. The CUDA
    side is checked by name instead, below and on a real GPU.
    """
    _, impl = backend.select("cpu")
    assert isinstance(impl, backend.Pipelines)


#: The four pipelines this project ships, spelled out rather than read off the
#: protocol: the signature test below parameterises over whatever the protocol
#: happens to declare, so on its own it cannot notice a core going missing —
#: deleting one simply removes a case, and deleting all four leaves an empty
#: parameter set that pytest skips. This list is what makes that a failure.
PIPELINE_STEMS = (
    "color_gdown_ideal",
    "motion_lpyr_ideal",
    "motion_lpyr_butter",
    "motion_lpyr_iir",
)


def test_the_protocol_declares_exactly_the_four_pipelines():
    declared = sorted(m for m in vars(backend.Pipelines) if m.endswith("_core"))
    assert declared == sorted(f"{stem}_core" for stem in PIPELINE_STEMS)


@pytest.mark.parametrize("name", sorted(
    m for m in vars(backend.Pipelines) if m.endswith("_core")
))
def test_the_protocol_matches_the_cpu_cores_signature_for_signature(name):
    """The protocol describes the working code, not the other way round.

    Only names, kinds and defaults are compared: the protocol annotates frames
    as ``evm.backend.Array`` (so a future device array satisfies it) where the
    CPU core says ``np.ndarray``, which is the one intended difference.
    """
    import inspect

    declared = list(inspect.signature(getattr(backend.Pipelines, name)).parameters.values())
    assert declared[0].name == "self"
    actual = list(inspect.signature(getattr(cpu_magnify, name)).parameters.values())

    def shape(params):
        return [(p.name, p.kind, p.default) for p in params]

    assert shape(declared[1:]) == shape(actual)


CUDA_CORE_NAMES = (
    *(f"{stem}_core" for stem in PIPELINE_STEMS),
    "color_gdown_ideal_fp16_core",
    "motion_lpyr_iir_fp16_core",
)


def test_the_cuda_backend_exposes_the_four_cores_by_name():
    """Without importing it: the routing table in evm/cuda/__init__.py is the
    contract the facade relies on, so it is asserted as data."""
    import ast

    src = (Path(evm.__file__).parent / "cuda" / "__init__.py").read_text()
    names = {
        s.value for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.Set)
        for s in node.elts
        if isinstance(s, ast.Constant) and isinstance(s.value, str)
    }
    for name in CUDA_CORE_NAMES:
        assert name in names
    assert set(CUDA_CORE_NAMES) <= set(evm.cuda.__all__)


@pytest.mark.parametrize("name", CUDA_CORE_NAMES)
def test_each_cuda_core_name_is_actually_routed(name):
    """The set literals above are data; this checks the ``__getattr__`` that
    reads them, and it works with or without the compiled extension.

    ``evm/cuda/__init__.py`` resolves a routed name by importing ``batched`` or
    ``pipelines``, which need ``_evm_cuda``: without it the lookup fails with
    ``ImportError``. An *un*routed name falls through to the submodule-import
    fallback, which swallows the ImportError and raises ``AttributeError``. So
    the two outcomes tell routed from unrouted apart on a GPU-less machine,
    which the name-in-a-set-literal assertion above cannot: breaking the routing
    while leaving the literals intact left the whole suite green.
    """
    assert getattr(evm.cuda, "nonsense_core", None) is None  # the unrouted control
    with pytest.raises(AttributeError):
        getattr(evm.cuda, "nonsense_core")

    from evm.cuda import have_cuda

    if have_cuda:
        assert callable(getattr(evm.cuda, name))
        return
    with pytest.raises(ImportError):
        getattr(evm.cuda, name)

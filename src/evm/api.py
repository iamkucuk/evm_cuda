"""The one-line facade — :func:`magnify` — and the built-in backend registrations.

    import evm
    out = evm.magnify(frames, preset="motion", fps=30)      # array in, array out
    out = evm.magnify("baby.mp4", preset="motion", out="magnified.mp4")

Three things happen here and nowhere else: a preset name becomes parameters
(:mod:`evm.presets` holds the numbers), a backend name becomes an
implementation (:mod:`evm.backend` holds the registry), and a video — path,
array, or any iterable of frames — becomes a ``(T, H, W, 3)`` uint8 BGR stack.

How this differs from the four ``magnify_*`` functions
------------------------------------------------------

They stay exactly as they are: path in, file out, ``float32`` in [0, 1] back,
last ten frames dropped, MIT-reference parameters spelled out in full. They are
what the reference tests drive. :func:`magnify` is the convenience layer over
the same cores, and differs deliberately in three places:

* **``drop_last`` defaults to 0**, not 10 (plan decision D8). The MATLAB
  reference drops the last ten frames of every clip, and the path functions
  reproduce that because their job is to reproduce the reference. An array the
  caller assembled has no such convention — silently returning ten fewer frames
  than were passed in would be a bug, not a feature. ``drop_last=10``
  reproduces the reference behaviour, and applies to array input as well.
* **uint8 in, uint8 out.** The cores work in ``(T, H, W, 3)`` uint8 BGR — the
  shape OpenCV decodes into and the shape the encoder wants — so a round trip
  through :func:`magnify` neither scales nor re-quantises anything.
* **the backend is chosen and announced.** ``backend="auto"`` prefers native
  CUDA and falls back to the CPU baseline only because there is nothing else;
  the choice is logged at INFO on the ``evm`` logger, and
  ``evm.backend.select("auto")`` answers the same question ahead of time. An
  explicitly named backend that cannot run raises and says why. Nothing ever
  drops from GPU to CPU quietly: that cliff is roughly 700x.

The built-in registrations
--------------------------

``"cpu"`` and ``"cuda"`` are registered when this module is imported, i.e. by
``import evm``. Registration is data only — neither implementation is imported
until something selects it, so ``import evm`` on a machine with no GPU still
never touches the CUDA extension.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import os
from typing import Any, Iterable

import numpy as np

from . import backend as _backend
from . import presets as _presets
from .backend import Capabilities
from .cpu.magnify import _read_frames

__all__ = ["magnify"]

_log = logging.getLogger(__name__)

#: What ``magnify(precision=...)`` maps to on an implementation: the attribute
#: suffix of the core to call. A backend advertises a precision by *having* that
#: core; there is no flag to get out of step with the code.
_PRECISION_SUFFIX = {"fp32": "_core", "fp16": "_fp16_core"}

#: Passed as ``fps`` to a core that documents it as unused (the r1/r2 IIR runs
#: on frame index, so no sampling rate exists to pass). Never reaches arithmetic.
_FPS_UNUSED = 0.0


# ---------------------------------------------------------------------------
# The facade
# ---------------------------------------------------------------------------


def magnify(
    video: str | os.PathLike | np.ndarray | Iterable[np.ndarray],
    *,
    preset: str | None = None,
    backend: str = "auto",
    precision: str = "fp32",
    fps: float | None = None,
    out: str | os.PathLike | None = None,
    drop_last: int = 0,
    **overrides: Any,
) -> np.ndarray:
    """Magnify a clip with a named preset, on the best available backend.

    Args:
        video: a path to a video file, a ``(T, H, W, 3)`` uint8 BGR array, or
            any iterable of such frames (it is materialised into one array —
            the temporal filters need the whole clip).
        preset: which preset to run; see :data:`evm.presets.PRESETS` for the
            names, the numbers, and where each came from. Required.
        backend: ``"auto"`` (native CUDA if it can run, else the CPU baseline),
            or a registered name such as ``"cpu"`` or ``"cuda"``. A named
            backend that cannot run here raises
            :class:`evm.backend.BackendUnavailableError` with the reason; it is
            never quietly replaced by another.
        precision: ``"fp32"`` or ``"fp16"``. ``"fp16"`` exists on the CUDA
            backend for the pulse and motion-IIR pipelines only, and asking for
            it elsewhere raises rather than silently computing in fp32. The CPU
            oracle computes in float64 regardless — it is the reference, and its
            precision is not a knob.
        fps: the clip's frame rate. Read from the file for path input (pass it
            to override); required for array input whenever the pipeline's
            temporal band is in Hz, or whenever ``out`` is given, because the
            encoder needs a rate.
        out: write the result to this path as H.264 as well as returning it.
        drop_last: drop this many frames from the end before magnifying.
            Defaults to 0 — see the module docstring, plan decision D8. Pass 10
            to reproduce the MATLAB reference and the ``magnify_*`` functions.
        **overrides: individual preset parameters to replace, e.g.
            ``alpha=25``, ``fl=0.5``. A name the pipeline does not take raises
            ``TypeError`` from the core, naming it.

    Returns:
        The magnified clip, ``(T, H, W, 3)`` uint8 BGR — same shape and dtype as
        the input frames.

    Raises:
        ValueError: no preset, an unusable precision, or a missing frame rate.
        KeyError: an unknown preset name (the message lists the known ones).
        evm.backend.BackendError: the named backend is unknown or unavailable.
    """
    if preset is None:
        raise ValueError(
            "magnify() needs a preset, e.g. preset='pulse'; available: "
            f"{', '.join(sorted(_presets.PRESETS))}. Each one's parameters and "
            "provenance are in evm.presets.PRESETS."
        )
    spec = _presets.get(preset)
    params = {**spec.params, **overrides}

    frames, rate = _read_input(video, fps=fps, drop_last=drop_last)

    name, impl = _backend.select(backend)
    core = _resolve_core(impl, name, spec.pipeline, precision)
    rate = _resolve_rate(rate, core=core, params=params, stem=spec.pipeline,
                         writing=out is not None)

    _log.info(
        "evm.magnify: backend=%r pipeline=%r precision=%s frames=%d fps=%s "
        "preset=%r",
        name, spec.pipeline, precision, len(frames),
        rate if rate else "unused by this pipeline", preset,
    )

    result = core(frames, rate, **params)

    if out is not None:
        from .io.video import encode_video

        encode_video(result, out, rate)
    return result


# ---------------------------------------------------------------------------
# Input, rate and core resolution
# ---------------------------------------------------------------------------


def _read_input(
    video: str | os.PathLike | np.ndarray | Iterable[np.ndarray],
    *,
    fps: float | None,
    drop_last: int,
) -> tuple[np.ndarray, float | None]:
    """Turn any accepted input into ``(frames, fps_or_None)``.

    Frames are handed on unvalidated: the cores' own checker rejects a wrong
    shape or dtype with one message per backend, so there is no second opinion
    here to disagree with it.
    """
    if drop_last < 0:
        raise ValueError(f"drop_last must be >= 0; got {drop_last}")

    if isinstance(video, (str, os.PathLike)):
        decoded, file_fps = _read_frames(video, drop_last=drop_last)
        if not decoded:
            raise ValueError(f"no frames decoded from {os.fspath(video)!r}")
        return np.stack(decoded, axis=0), float(fps) if fps is not None else file_fps

    frames = video if isinstance(video, np.ndarray) else np.stack(list(video), axis=0)
    if drop_last and len(frames) > drop_last:
        frames = frames[: len(frames) - drop_last]
    return frames, float(fps) if fps is not None else None


def _resolve_rate(
    rate: float | None, *, core: Any, params: dict, stem: str, writing: bool
) -> float:
    """The frame rate to hand the core, or a loud explanation of why it is needed."""
    if rate is not None:
        return rate

    # A core that takes ``sampling_rate`` interprets fl/fh in Hz against it, and
    # defaults it to fps. Asked from the signature rather than a second table,
    # so a new pipeline cannot fall out of step with this check.
    takes_rate = "sampling_rate" in inspect.signature(core).parameters
    if takes_rate and "sampling_rate" not in params:
        raise ValueError(
            f"fps= is required: the {stem} pipeline measures its fl/fh band in "
            "Hz against the frame rate, and frames passed as an array carry "
            "none. Pass fps=..., or pin the filter with sampling_rate=..."
        )
    if writing:
        raise ValueError(
            "fps= is required to write a video: out= was given but the frames "
            "carry no frame rate."
        )
    return _FPS_UNUSED


def _resolve_core(impl: Any, backend_name: str, stem: str, precision: str) -> Any:
    """Find ``<stem>_core`` (or the fp16 twin) on the selected implementation."""
    suffix = _PRECISION_SUFFIX.get(precision)
    if suffix is None:
        raise ValueError(
            f"unknown precision {precision!r}; choose one of "
            f"{', '.join(sorted(_PRECISION_SUFFIX))}"
        )
    core = getattr(impl, stem + suffix, None)
    if core is None:
        have = [p for p, s in _PRECISION_SUFFIX.items() if hasattr(impl, stem + s)]
        raise ValueError(
            f"backend {backend_name!r} has no {precision} implementation of the "
            f"{stem} pipeline (no {stem + suffix!r}); it offers "
            f"{', '.join(have) if have else 'no precision of this pipeline'}. "
            "Falling back would mean computing in a precision you did not ask "
            "for, so it raises instead."
        )
    return core


# ---------------------------------------------------------------------------
# Built-in backend registrations (plan section 3c, step 3.5)
# ---------------------------------------------------------------------------


def _probe_cpu() -> str | None:
    """Always available: NumPy, SciPy and OpenCV are hard requirements, and this
    module could not have been imported without them."""
    return None


def _probe_cuda() -> str | None:
    """The truthful reason the CUDA backend can or cannot run here.

    The wording comes from ``evm.cuda.require_cuda()`` rather than being written
    again here: that is the sentence every script and test already prints when
    the extension is missing, and two explanations of one condition would drift.
    """
    try:
        runtime = importlib.import_module("evm.cuda.runtime")
    except Exception as exc:  # the wrapper package itself failed to import
        return f"evm.cuda could not be imported: {exc!r}"
    try:
        runtime.require_cuda()
    except RuntimeError as exc:
        return str(exc)
    return None


def _register_builtin_backends() -> None:
    """Register ``"cpu"`` and ``"cuda"``. Imports neither."""
    _backend.register(
        "cpu",
        load=lambda: importlib.import_module("evm.cpu.magnify"),
        probe=_probe_cpu,
        # The oracle computes in float64 throughout (evm/cpu/magnify.py builds
        # its NTSC frames as float64); it has no other precision.
        capabilities=Capabilities(dtypes=("float64",), fft=True, streaming=False),
    )
    _backend.register(
        "cuda",
        # The package, not one of its modules: its __getattr__ routes each
        # pipeline to whichever of batched/pipelines implements it fastest.
        load=lambda: importlib.import_module("evm.cuda"),
        probe=_probe_cuda,
        # fp32 and fp16 bodies both exist (batched.py); cuFFT provides the exact
        # ideal filter. Streaming is Phase 7 and is not claimed before it lands.
        capabilities=Capabilities(
            dtypes=("float32", "float16"), fft=True, streaming=False
        ),
    )


_register_builtin_backends()

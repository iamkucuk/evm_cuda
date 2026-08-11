"""Eulerian Video Magnification — Python baseline (MIT reference port).

Four entry points matching the four MATLAB amplification functions in
EVM_Matlab-1.1.zip:

* ``magnify_color_gdown_ideal`` — ``amplify_spatial_Gdown_temporal_ideal``
* ``magnify_motion_lpyr_ideal``  — ``amplify_spatial_lpyr_temporal_ideal``
* ``magnify_motion_lpyr_butter`` — ``amplify_spatial_lpyr_temporal_butter``
* ``magnify_motion_lpyr_iir``    — ``amplify_spatial_lpyr_temporal_iir``

The spatial (binom5 + reflect1 corrDn/upConv) and temporal (ideal / 1st-order
Butterworth / direct r1-r2 IIR) kernels reproduce matlabPyrTools and the
reference bandpassing bit-for-bit, so this package is the correctness oracle
for every other backend — NVIDIA, OpenCL, Apple and Vulkan alike.

For a one-liner, :func:`evm.magnify` takes a path, an array, or any iterable of
frames, a preset name, and picks a backend::

    out = evm.magnify(frames, preset="motion", fps=30)   # numpy in, numpy out
    evm.magnify("baby.mp4", preset="motion", out="magnified.mp4")

It differs from the four functions above in three documented ways (dtype,
``drop_last``, backend selection) — see :func:`evm.api.magnify`.

Subpackages:

* :mod:`evm.cpu` — the NumPy baseline above (``pyramids``, ``filters``,
  ``magnify``). Everything it exports is re-exported here.
* :mod:`evm.io` — video decode/encode and the RGB<->YIQ conversions.
* :mod:`evm.presets` — the frozen preset table :data:`evm.PRESETS`.
* :mod:`evm.backend` — the backend protocols and registry. ``"cpu"`` and
  ``"cuda"`` are registered by importing this package;
  ``evm.backend.list_backends()`` says which can run here and why not.
* :mod:`evm.cuda` — the CUDA port. **Resolved lazily**: importing ``evm`` on a
  machine with no GPU and no compiled extension never touches it. Ask for
  ``evm.cuda`` (or ``import evm.cuda``) and you get the wrapper package, whose
  ``have_cuda`` / ``require_cuda`` report the extension's state.
"""

from importlib.metadata import (
    PackageNotFoundError as _PackageNotFoundError,
    version as _version,
)
# Underscore-aliased like the two names above: ``__dir__`` below returns
# ``globals()``, so a bare ``Any`` would show up in ``dir(evm)``.
from typing import Any as _Any

from .io import load_video, save_video, VideoInfo, rgb_to_yiq, yiq_to_rgb
from .cpu import (
    BINOM5,
    BINOM5_SUM1,
    blur_dn,
    blur_dn_clr,
    build_lpyr,
    recon_lpyr,
    laplacian_pyramid_channels,
    reconstruct_from_channels,
    max_pyr_ht,
    ideal_bandpass,
    butter_bandpass,
    iir_bandpass,
    figure6_alpha_schedule,
    magnify_color_gdown_ideal,
    magnify_motion_lpyr_ideal,
    magnify_motion_lpyr_butter,
    magnify_motion_lpyr_iir,
    magnify_phase,
    DROP_LAST,
    EXAGGERATION_FACTOR,
)
# Imported last: evm.api registers the built-in backends and needs evm.cpu.
# Importing it also makes ``evm.backend`` and ``evm.presets`` reachable as
# attributes of this package.
from .api import magnify
from .presets import PRESETS, Preset

try:
    __version__ = _version("evm-magnify")
except _PackageNotFoundError:
    # Running straight from a source checkout (not pip-installed). Say so
    # rather than inventing a number that could be mistaken for a release.
    __version__ = "0.0.0+unknown"


def __getattr__(name: str) -> _Any:
    # ``evm.cuda`` is resolved on first access, not at import time, so a
    # CPU-only machine importing ``evm`` never loads the CUDA wrapper.
    if name == "cuda":
        import importlib

        return importlib.import_module(f"{__name__}.cuda")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    # PEP 562: keep the lazily-resolved ``cuda`` visible to dir()/completion.
    return sorted([*globals(), "cuda"])


__all__ = [
    "magnify",
    "PRESETS",
    "Preset",
    "backend",
    "presets",
    "load_video",
    "save_video",
    "VideoInfo",
    "rgb_to_yiq",
    "yiq_to_rgb",
    "BINOM5",
    "BINOM5_SUM1",
    "blur_dn",
    "blur_dn_clr",
    "build_lpyr",
    "recon_lpyr",
    "laplacian_pyramid_channels",
    "reconstruct_from_channels",
    "max_pyr_ht",
    "ideal_bandpass",
    "butter_bandpass",
    "iir_bandpass",
    "figure6_alpha_schedule",
    "magnify_color_gdown_ideal",
    "magnify_motion_lpyr_ideal",
    "magnify_motion_lpyr_butter",
    "magnify_motion_lpyr_iir",
    "magnify_phase",
    "DROP_LAST",
    "EXAGGERATION_FACTOR",
]

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
for the CUDA port.

Subpackages:

* :mod:`evm.cpu` — the NumPy baseline above (``pyramids``, ``filters``,
  ``magnify``). Everything it exports is re-exported here.
* :mod:`evm.io` — video decode/encode and the RGB<->YIQ conversions.
* :mod:`evm.cuda` — the CUDA port. **Resolved lazily**: importing ``evm`` on a
  machine with no GPU and no compiled extension never touches it. Ask for
  ``evm.cuda`` (or ``import evm.cuda``) and you get the wrapper package, whose
  ``have_cuda`` / ``require_cuda`` report the extension's state.
"""

from importlib.metadata import (
    PackageNotFoundError as _PackageNotFoundError,
    version as _version,
)

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
    DROP_LAST,
    EXAGGERATION_FACTOR,
)

try:
    __version__ = _version("evm-cuda")
except _PackageNotFoundError:
    # Running straight from a source checkout (not pip-installed). Say so
    # rather than inventing a number that could be mistaken for a release.
    __version__ = "0.0.0+unknown"


def __getattr__(name: str):
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
    "DROP_LAST",
    "EXAGGERATION_FACTOR",
]

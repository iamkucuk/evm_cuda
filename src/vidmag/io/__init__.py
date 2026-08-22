"""Video I/O: decode to arrays, encode arrays back to playable H.264.

* :mod:`vidmag.io.video` — whole-clip load/save plus the RGB<->YIQ (NTSC) color
  conversions the pipelines run in.
* :mod:`vidmag.io.h264` — the single H.264 (``avc1`` / ``yuv420p`` / faststart)
  encoder, shared by the CPU baseline and the CUDA port so both write
  byte-identical container layouts. It used to live in a third top-level
  package, ``shared/``, for exactly that reason; the package split is gone but
  the single encode implementation is not.
"""

from .video import (
    VideoInfo,
    encode_video,
    load_video,
    rgb_to_yiq,
    save_video,
    yiq_to_rgb,
)
from .h264 import encode_h264

__all__ = [
    "VideoInfo",
    "encode_video",
    "encode_h264",
    "load_video",
    "rgb_to_yiq",
    "save_video",
    "yiq_to_rgb",
]

"""Shared H.264 video encoder used by both the CPU baseline (``evm``) and the
CUDA port (``evm_cuda``).

Both packages need to write browser/VSCode-playable H.264 (``avc1``,
``yuv420p``, ``+faststart``) from a ``(T, H, W, 3)`` uint8 BGR frame array.
Keeping the encoder here — in a module neither package owns — means the two
packages don't have to import each other (they're deliberately decoupled:
``evm`` is the standalone Python baseline, ``evm_cuda`` is the CUDA port) yet
the encode logic lives exactly once.
"""
from __future__ import annotations

from pathlib import Path


def encode_h264(
    frames_uint8,
    path: str | Path,
    fps: float,
    *,
    codec: str = "libx264",
) -> None:
    """Write a ``(T, H, W, 3)`` uint8 BGR frame array to an H.264 MP4.

    Encodes directly to H.264 ``yuv420p`` with a faststart (``moov`` before
    ``mdat``) atom via PyAV, so the output plays in browsers (Colab's HTML5
    ``<video>``), VSCode, and QuickTime — not just VLC. Single pass, no temp
    file, no external binary.
    """
    import av
    from fractions import Fraction
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    t, h, w, _ = frames_uint8.shape

    # PyAV expects a rational (not float) framerate; limit_denominator maps
    # common floats like 29.97 to 30000/1001 cleanly.
    rate = Fraction(fps).limit_denominator(1_000_000)

    with av.open(
        str(path), mode="w", options={"movflags": "+faststart"}
    ) as container:
        stream = container.add_stream(codec, rate=rate)
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv420p"
        stream.options = {"preset": "veryfast", "crf": "18"}
        for i in range(t):
            frame = av.VideoFrame.from_ndarray(frames_uint8[i], format="bgr24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():  # flush the encoder
            container.mux(packet)

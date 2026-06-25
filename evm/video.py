"""Video I/O helpers.

Loads a video into a single float32 array of shape ``(T, H, W, C)`` with values
in ``[0, 1]`` and writes one back out. Keeping the whole clip in memory is fine
for the baseline (the EVM temporal filters need random access to all frames
anyway) and makes the algorithm easy to read; the CUDA port will stream frames
through device memory instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class VideoInfo:
    """Metadata needed to reconstruct an output file from a float array."""

    fps: float
    width: int
    height: int
    frame_count: int
    is_color: bool


def load_video(path: str | Path) -> tuple[np.ndarray, VideoInfo]:
    """Load a video as a float32 array of shape ``(T, H, W, C)`` in ``[0, 1]``.

    Color videos come back as 3-channel BGR (OpenCV's native order, so we can
    hand frames straight back to ``save_video`` without permuting channels).
    Grayscale videos come back as ``(T, H, W, 1)`` so downstream code can assume
    a trailing channel axis unconditionally.
    """
    path = str(path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path!r}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.ndim == 2:
            frame = frame[:, :, None]
        frames.append(frame)
    cap.release()

    if not frames:
        raise ValueError(f"No frames decoded from {path!r}")

    stack = np.stack(frames, axis=0).astype(np.float32) / 255.0
    info = VideoInfo(
        fps=float(fps),
        width=width or int(stack.shape[2]),
        height=height or int(stack.shape[1]),
        frame_count=int(stack.shape[0]),
        is_color=stack.shape[3] == 3,
    )
    return stack, info


def encode_video(
    frames_uint8: np.ndarray,
    path: str | Path,
    fps: float,
    *,
    codec: str = "libx264",
) -> None:
    """Write a ``(T, H, W, 3)`` uint8 BGR frame array to an H.264 MP4.

    This is the single encoder implementation shared by every writer in the
    package. It encodes directly to H.264 ``yuv420p`` with a faststart
    (``moov`` before ``mdat``) atom via PyAV, so the output plays in browsers
    (Colab's HTML5 <video>), VSCode, and QuickTime — not just VLC. Single
    pass, no temp file, no external binary.
    """
    import av
    from fractions import Fraction

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


def save_video(
    frames: np.ndarray,
    path: str | Path,
    fps: float,
    *,
    codec: str = "libx264",
) -> None:
    """Write a float32 array in ``[0, 1]`` back to an H.264 MP4.

    ``frames`` is ``(T, H, W, C)`` with C in {1, 3}. Single-channel arrays are
    broadcast to 3 channels for the encoder. Values are clipped to the valid
    range and converted to ``uint8`` (BGR), then handed to :func:`encode_video`.
    """
    if frames.ndim != 4:
        raise ValueError(f"Expected (T,H,W,C); got shape {frames.shape!r}")

    c = frames.shape[3]
    clipped = np.clip(frames, 0.0, 1.0)
    scaled = np.round(clipped * 255.0).astype(np.uint8)
    if c == 1:
        # H.264/yuv420p needs a 3-channel source; replicate grayscale to BGR.
        scaled = np.repeat(scaled, 3, axis=3)

    encode_video(scaled, path, fps, codec=codec)


def rgb_to_yiq(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB float in ``[0, 1]`` to YIQ, matching MATLAB ``rgb2ntsc``.

    Uses the exact transform matrix documented for MATLAB's ``rgb2ntsc`` so the
    luminance/chrominance split is identical to the MIT reference. The row order
    is ``(Y, I, Q)``. Input shape ``(..., 3)``.
    """
    m = np.array(
        [
            [0.299, 0.587, 0.114],
            [-0.168736, -0.331264, 0.5],
            [0.5, -0.418688, -0.081312],
        ],
        dtype=np.float32,
    )
    return rgb @ m.T


def yiq_to_rgb(yiq: np.ndarray) -> np.ndarray:
    """Inverse of :func:`rgb_to_yiq`, matching MATLAB ``ntsc2rgb``."""
    m = np.array(
        [
            [1.0, -1.21889419e-06, 1.40199959],
            [1.0, -3.44135678e-01, -7.14136156e-01],
            [1.0, 1.77200007, 4.06298063e-07],
        ],
        dtype=np.float32,
    )
    return yiq @ m.T

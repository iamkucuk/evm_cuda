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


def _h264_via_ffmpeg(src, dst, fps) -> bool:
    """Transcode ``src`` to H.264 ``yuv420p`` +faststart at ``dst``.

    Browsers (Colab's HTML5 <video>), VSCode, and most non-VLC players need
    H.264; OpenCV's pip wheel can't encode it (no libx264 backend), so we let
    OpenCV write the raw frames and hand the transcode to ffmpeg. Returns True
    on success, False if ffmpeg/libx264 is unavailable (caller falls back).
    """
    import shutil
    import subprocess

    if not shutil.which("ffmpeg"):
        return False
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src),
        "-r", str(fps),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(dst),
    ]
    try:
        subprocess.run(
            cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except Exception:
        return False
    return True


def save_video(
    frames: np.ndarray,
    path: str | Path,
    fps: float,
    *,
    codec: str = "mp4v",
) -> None:
    """Write a float32 array in ``[0, 1]`` back to a video file.

    ``frames`` is ``(T, H, W, C)`` with C in {1, 3}. Single-channel arrays are
    squeezed to 2-D for the writer. Values are clipped to the valid range and
    converted to ``uint8``.

    The frames are staged to a temporary file with OpenCV (``codec``), then
    transcoded to H.264 ``yuv420p`` +faststart via ffmpeg so the result plays
    in browsers (Colab) and VSCode, not just VLC. If ffmpeg/libx264 is absent
    the staged mp4v file is kept as a fallback (still readable by OpenCV/VLC).
    """
    import os
    import tempfile

    path = Path(path)
    if frames.ndim != 4:
        raise ValueError(f"Expected (T,H,W,C); got shape {frames.shape!r}")

    t, h, w, c = frames.shape
    path.parent.mkdir(parents=True, exist_ok=True)

    suffix = os.path.splitext(str(path))[1] or ".mp4"
    fd, tmp_name = tempfile.mkstemp(suffix=suffix, dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(
            str(tmp_path), fourcc, float(fps), (w, h), isColor=(c == 3))
        if not writer.isOpened():
            raise RuntimeError(
                f"VideoWriter failed to open for {path!r} (codec={codec!r}, "
                f"size={w}x{h}, fps={fps})"
            )

        try:
            clipped = np.clip(frames, 0.0, 1.0)
            scaled = np.round(clipped * 255.0).astype(np.uint8)
            for i in range(t):
                frame = scaled[i]
                if c == 1:
                    frame = frame[:, :, 0]
                writer.write(frame)
        finally:
            writer.release()

        if _h264_via_ffmpeg(tmp_path, path, fps):
            return
        if path.exists() or path == tmp_path:
            return
        tmp_path.replace(path)
    finally:
        if tmp_path.exists() and tmp_path != path:
            tmp_path.unlink(missing_ok=True)


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

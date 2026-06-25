"""Regression tests for the H.264 video encoder.

Guarantees that the output writers produce browser/VSCode-playable H.264
(``avc1``, ``yuv420p``) with a faststart (``moov`` before ``mdat``) atom, so
the result plays in Colab's HTML5 <video> and VSCode — not just VLC. This
guards against silently regressing back to OpenCV's ``mp4v`` (MPEG-4 Part 2).

Reads the written file back with PyAV rather than shelling out to ``ffprobe``,
so the test has no system-binary dependency.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evm.video import encode_video, save_video  # noqa: E402


def _codec_and_pixfmt(path: Path) -> tuple[str, str]:
    """Return (codec_name, pix_fmt) of the first video stream, via PyAV."""
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        codec = stream.codec_context.name
        pix_fmt = stream.codec_context.pix_fmt
    return codec, pix_fmt


def _is_faststart(path: Path) -> bool:
    """True if the moov atom precedes the mdat atom (web-playable layout)."""
    data = path.read_bytes()
    moov, mdat = data.find(b"moov"), data.find(b"mdat")
    return 0 < moov < mdat


def test_encode_video_produces_h264_yuv420p_faststart(tmp_path: Path) -> None:
    """The uint8 writer (used by magnify.py + batched.py) emits H.264."""
    frames = (
        np.random.RandomState(0).randint(0, 256, size=(12, 64, 80, 3), dtype=np.uint8)
    )
    out = tmp_path / "uint8.mp4"
    encode_video(frames, out, 30.0)

    codec, pix_fmt = _codec_and_pixfmt(out)
    assert codec == "h264", f"expected h264, got {codec}"
    assert pix_fmt == "yuv420p", f"expected yuv420p, got {pix_fmt}"
    assert _is_faststart(out), "moov atom is not before mdat (no faststart)"


def test_save_video_produces_h264_yuv420p_faststart(tmp_path: Path) -> None:
    """The float [0,1] writer emits H.264 (exercises the scaling + gray path)."""
    frames = (
        np.random.RandomState(1).rand(12, 64, 80, 3).astype(np.float32) * 0.5 + 0.2
    )
    out = tmp_path / "float.mp4"
    save_video(frames, out, 29.97)

    codec, pix_fmt = _codec_and_pixfmt(out)
    assert codec == "h264", f"expected h264, got {codec}"
    assert pix_fmt == "yuv420p", f"expected yuv420p, got {pix_fmt}"
    assert _is_faststart(out), "moov atom is not before mdat (no faststart)"


def test_save_video_grayscale_broadcast(tmp_path: Path) -> None:
    """Single-channel input is broadcast to 3 channels and still encodes H.264."""
    frames = (
        np.random.RandomState(2).rand(8, 48, 48, 1).astype(np.float32) * 0.6
    )
    out = tmp_path / "gray.mp4"
    save_video(frames, out, 30.0)

    codec, _ = _codec_and_pixfmt(out)
    assert codec == "h264", f"expected h264, got {codec}"

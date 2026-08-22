"""Reading frames from a camera, a file, or a network stream.

A thin wrapper over OpenCV's capture, existing for two reasons. It closes the
device even when the loop it feeds raises, which matters for a camera because
an unreleased one stays busy until the process exits. And it makes the choice
between "keep up with the source" and "process every frame" explicit, because
the right answer differs: a live camera should drop frames rather than fall
behind, while a file should not drop any.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

__all__ = ["FrameSource", "open_source"]


class FrameSource:
    """Frames from one source, closed when you are finished with it."""

    def __init__(self, capture: Any, description: str) -> None:
        self._capture = capture
        self.description = description

    @property
    def fps(self) -> float:
        """The source's frame rate, or 30 if it does not report a usable one.

        Cameras frequently report 0. The fallback is stated here rather than
        hidden because the frame rate is what turns a filter band in cycles per
        second into something meaningful, so a wrong one produces a wrong
        result quietly.
        """
        import cv2

        rate = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)
        return rate if rate > 0 else 30.0

    @property
    def size(self) -> tuple[int, int]:
        """(height, width) of the frames this source produces."""
        import cv2

        return (
            int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        )

    def frames(self) -> Iterator[np.ndarray]:
        """Yield frames until the source ends or is interrupted."""
        while True:
            ok, frame = self._capture.read()
            if not ok:
                return
            yield frame

    def close(self) -> None:
        self._capture.release()

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def open_source(source: str | int) -> FrameSource:
    """Open a camera, a video file, or a network stream.

    Args:
        source: a camera index such as ``0``, a path to a video file, or a
            stream address such as ``rtsp://...``. A string of digits is taken
            as a camera index, since that is how it arrives from a command line.

    Raises:
        RuntimeError: the source could not be opened, naming what was tried.
    """
    import cv2

    target: str | int = source
    if isinstance(source, str) and source.isdigit():
        target = int(source)

    capture = cv2.VideoCapture(target)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"could not open {source!r}. For a camera, check the index and "
            f"that nothing else is using it; for a file, check the path; for a "
            f"network stream, check the address and that it is reachable."
        )
    kind = (
        "camera"
        if isinstance(target, int)
        else "stream"
        if "://" in str(target)
        else "file"
    )
    return FrameSource(capture, f"{kind} {source!r}")

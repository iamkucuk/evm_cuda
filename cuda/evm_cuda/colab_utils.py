"""Small display helpers for notebooks (Colab / Jupyter).

Keeps the ``<video>``/base64 plumbing out of notebook cells so they stay
readable. ``show_video`` embeds an H.264 clip as an inline HTML5 ``<video>``
that plays in the browser.
"""
from __future__ import annotations

import os
from pathlib import Path

from IPython.display import HTML, display
from base64 import b64encode


def show_video(path: str | Path, label: str = "") -> None:
    """Embed a video file as an inline, looping HTML5 ``<video>`` element.

    The clip must be browser-playable (H.264 / yuv420p); this repo's writers
    produce exactly that. If the file is missing, a message is printed instead.
    """
    if not os.path.exists(path):
        print(f"  {label or path}: file not found")
        return
    data = Path(path).read_bytes()
    url = "data:video/mp4;base64," + b64encode(data).decode()
    display(HTML(
        f'<h4>{label}</h4>'
        f'<video width=480 controls loop>'
        f'<source src="{url}" type="video/mp4"></video>'))

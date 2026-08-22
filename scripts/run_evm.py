#!/usr/bin/env python3
"""DEPRECATED — use ``vidmag magnify`` instead.

This script is now a forwarder. ``vidmag magnify`` (:mod:`vidmag._cli`) takes
every argument this file ever took — same names, same defaults, same four
``--mode`` values — so the arguments are handed over untouched and the output is
what it always was::

    python scripts/run_evm.py data/face.mp4 out.mp4 --mode color --alpha 50 ...
    vidmag magnify       data/face.mp4 out.mp4 --mode color --alpha 50 ...

The new command adds ``--preset`` (``pulse`` reproduces the first line above
from :data:`vidmag.presets.PRESETS`) and ``--backend``, and it is on PATH after
``pip install .`` — no repository checkout and no ``scripts/`` on ``sys.path``.
"""

from __future__ import annotations

import sys
import warnings

from vidmag._cli import main


def _forward(argv: list[str] | None = None) -> int:
    warnings.warn(
        "scripts/run_evm.py is deprecated; use `vidmag magnify` "
        "(same arguments, installed on PATH by `pip install .`).",
        DeprecationWarning,
        stacklevel=2,
    )
    return main(["magnify", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    raise SystemExit(_forward())

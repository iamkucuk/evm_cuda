#!/usr/bin/env python3
"""Regenerate the golden end-to-end fixtures in ``tests/fixtures/``.

Writes five files:

* ``golden_input.npz`` — the synthetic source clip (uint8 BGR frames + fps),
  built here from a fixed seed so the committed data, not the RNG, is what the
  tests depend on.
* ``golden_<case>.npz`` — the uint8 output of each of the four public
  ``magnify_*`` pipelines for that clip, one file per case.

The clip is 40 frames of 32x32 (the pipelines drop the last 10, leaving 30) at
30 fps, carrying two things the pipelines must react to: a 2 Hz horizontal
translation of a striped texture (motion) and a 2 Hz global brightness
oscillation (color). 2 Hz over the 30 retained frames at 30 fps is exactly DFT
bin 2, so the ideal filter's brick-wall mask passes it cleanly. Per-channel
gains make I and Q non-zero, so the chrominance path is exercised too. A small
static seeded texture breaks the symmetry of the analytic pattern so every
pyramid band carries signal.

The case list and the lossless writer are imported from ``tests/test_golden.py``
— that file is the source of truth for what each fixture means, and importing it
here makes drift between generator and test impossible.

Usage::

    python scripts/dev/make_golden_fixtures.py            # rewrite in place
    python scripts/dev/make_golden_fixtures.py --out DIR  # write elsewhere
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.test_golden import CASES, FIXTURES, write_lossless_video  # noqa: E402

SEED = 20260809
N_FRAMES = 40
HEIGHT = 32
WIDTH = 32
FPS = 30.0
OSC_HZ = 2.0

# Kept small enough that the amplified output never hits the 0/1 clipping rails
# (checked below): a saturated fixture would be insensitive to changes.
SHIFT_PX = 1.2          # peak horizontal translation of the stripes
BRIGHTNESS = 0.02       # peak global brightness swing
CHANNEL_GAIN = np.array([1.0, 0.75, 0.5])  # R, G, B -> non-zero I and Q


def make_clip() -> np.ndarray:
    """Build the synthetic clip as ``(T, H, W, 3)`` uint8 BGR."""
    rng = np.random.default_rng(SEED)
    texture = rng.normal(0.0, 0.02, size=(HEIGHT, WIDTH, 3))

    y, x = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float64)
    phase = 2.0 * np.pi * OSC_HZ * np.arange(N_FRAMES) / FPS

    rgb = np.empty((N_FRAMES, HEIGHT, WIDTH, 3), dtype=np.float64)
    for i in range(N_FRAMES):
        stripes = (
            np.sin(2 * np.pi * (x + SHIFT_PX * np.sin(phase[i])) / 8.0)
            * np.cos(2 * np.pi * y / 11.0)
        )
        rgb[i] = (
            0.45
            + 0.18 * stripes[..., None] * CHANNEL_GAIN
            + BRIGHTNESS * np.sin(phase[i]) * CHANNEL_GAIN
            + texture
        )

    u8 = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
    return u8[..., ::-1].copy()  # RGB -> BGR, the order cv2 hands the pipelines


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--out",
        type=Path,
        default=FIXTURES,
        help=f"Destination directory (default: {FIXTURES}).",
    )
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    frames = make_clip()
    np.savez_compressed(args.out / "golden_input.npz", frames=frames, fps=FPS)
    print(f"[golden] input: {frames.shape} uint8 @ {FPS} fps", file=sys.stderr)

    # The pipelines read a path, so the clip has to hit disk. FFV1 keeps the
    # decode bit-exact; the test asserts that separately before comparing.
    src = args.out / "_source.mkv"
    write_lossless_video(src, frames, FPS)
    try:
        for name in sorted(CASES):
            fn, kwargs = CASES[name]
            out = fn(str(src), str(args.out / f"_{name}.mp4"), **kwargs)

            u8 = np.round(out * 255.0).astype(np.uint8)
            # The test compares u8/255 against the pipeline's float32 return
            # value, so that quantisation has to be exactly reversible.
            if not np.array_equal(u8.astype(np.float32) / 255.0, out):
                raise AssertionError(
                    f"{name}: uint8 round-trip is not exact; the fixture would "
                    "not reproduce the pipeline output"
                )
            np.savez_compressed(args.out / f"golden_{name}.npz", output=u8)

            ref = frames[: out.shape[0]].astype(np.float32) / 255.0
            saturated = float(((out <= 0.0) | (out >= 1.0)).mean())
            print(
                f"[golden] {name}: {out.shape} "
                f"max|out-in|={np.abs(out - ref).max():.4f} "
                f"saturated={saturated:.4f} "
                f"{(args.out / f'golden_{name}.npz').stat().st_size} bytes",
                file=sys.stderr,
            )
    finally:
        src.unlink(missing_ok=True)
        for name in CASES:
            (args.out / f"_{name}.mp4").unlink(missing_ok=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

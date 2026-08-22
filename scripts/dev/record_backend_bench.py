"""Regenerate the cross-backend timing record for the machine it runs on.

`docs/performance.md` and `docs/concepts/backends.md` both carry a table
comparing every available backend on one machine. Those numbers were previously
copied by hand from console output, and two sessions' figures ended up quoted
side by side in different files without either being marked superseded. Running
this produces the whole table in one session, so every row shares a method and
a date.

    python scripts/dev/record_backend_bench.py --machine "Apple M2 Max" \
        --date 2026-08-18 --out benches/backends_apple_m2_max.json

Timing covers magnification only: frames go in as an array and come out as an
array, so decoding and encoding the video are excluded. Each backend gets one
warm-up run first, which is what keeps a driver's one-off kernel compilation
out of the figures.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

CLIPS = [("colour", "data/face.mp4", "pulse"), ("motion", "data/baby.mp4", "motion")]


def read_bgr_u8(path: str) -> tuple[Any, float]:
    """All frames as 8-bit blue-green-red, and the frame rate.

    The same thing `vidmag.cuda._common.read_frames` does, including dropping
    the trailing frames the pipelines drop, but without importing the NVIDIA
    extension — so this script runs on a machine that has no NVIDIA card, which
    is the whole point of a cross-backend comparison.
    """
    import cv2
    import numpy as np

    from vidmag.cpu.magnify import DROP_LAST

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {path!r}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if len(frames) > DROP_LAST:
        frames = frames[: len(frames) - DROP_LAST]
    return np.asarray(frames), float(fps)


def measure(backend: str, clip: str, preset: str, repeats: int) -> dict:
    """Time one backend on one clip, array in and array out."""
    import vidmag

    frames, fps = read_bgr_u8(clip)
    vidmag.magnify(frames, preset=preset, fps=fps, backend=backend)  # warm-up
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        vidmag.magnify(frames, preset=preset, fps=fps, backend=backend)
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return {
        "median_ms": times[len(times) // 2],
        "min_ms": times[0],
        "max_ms": times[-1],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--machine", required=True, help='e.g. "Apple M2 Max".')
    ap.add_argument("--date", required=True, help="YYYY-MM-DD.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument(
        "--cpu-repeats",
        type=int,
        default=2,
        help="The NumPy reference is slow; time it fewer times.",
    )
    ap.add_argument("--one", nargs=3, metavar=("BACKEND", "CLIP", "PRESET"))
    args = ap.parse_args()

    if args.one:
        b_, c_, p_ = args.one
        n = args.cpu_repeats if b_ == "cpu" else args.repeats
        print(json.dumps(measure(b_, c_, p_, n)))
        return 0

    from vidmag.backend import registry

    available = [i.name for i in registry.list_backends() if not i.unavailable_reason]
    print(f"available here: {', '.join(available)}", file=sys.stderr)

    results: dict[str, dict[str, dict]] = {}
    for backend in available:
        results[backend] = {}
        for label, clip, preset in CLIPS:
            print(f"  {backend} / {label} ...", file=sys.stderr, flush=True)
            # Own process per backend: the device memory pools of two backends in
            # one process compete, which is how a PyTorch figure nine times too
            # slow was once recorded.
            cmd = [
                sys.executable,
                __file__,
                "--machine",
                args.machine,
                "--date",
                args.date,
                "--out",
                args.out,
                "--repeats",
                str(args.repeats),
                "--cpu-repeats",
                str(args.cpu_repeats),
                "--one",
                backend,
                clip,
                preset,
            ]
            p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
            if p.returncode != 0:
                results[backend][label] = {"failed": p.stderr.strip()[-300:]}
                continue
            results[backend][label] = json.loads(p.stdout.strip().splitlines()[-1])

    doc = {
        "machine": args.machine,
        "date": args.date,
        "method": (
            "array in, array out, magnification only (decode and encode "
            f"excluded); one warm-up then the median of {args.repeats} "
            f"({args.cpu_repeats} for the NumPy reference); one process per "
            "backend. Regenerated by scripts/dev/record_backend_bench.py."
        ),
        "clips": {label: clip for label, clip, _ in CLIPS},
        "results": results,
    }
    Path(args.out).write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Regenerate the stored benchmark record for one NVIDIA card.

The files in `benches/bench_*.json` are cited as the evidence behind every
timing in `README.md` and `docs/performance.md`. Until this script existed they
were assembled by hand, and the motion rows drifted out of date when the motion
path was optimised while the file was not re-run. Running this replaces the
whole file in one session, so every number in it shares one date and one method.

    python scripts/dev/record_gpu_bench.py --gpu "RTX 3090" --out benches/bench_rtx3090.json

Each configuration is timed in its own process. That is not a stylistic choice:
the device memory pool holds freed blocks, so a configuration measured after
another in the same process competes with memory the pool has not returned.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# (key, pipeline, clip, precision). `pipeline` is the benchmark harness's own
# name, not the preset name: the harness covers the two device-resident
# pipelines, colour-gdown-ideal and motion-lpyr-iir.
CONFIGS = [
    ("color_face_fp32", "color", "data/face.mp4", "fp32"),
    ("color_face_fp16", "color", "data/face.mp4", "fp16"),
    ("color_baby_fp32", "color", "data/baby.mp4", "fp32"),
    ("color_baby_fp16", "color", "data/baby.mp4", "fp16"),
    ("motion_baby_fp32", "motion", "data/baby.mp4", "fp32"),
    ("motion_baby_fp16", "motion", "data/baby.mp4", "fp16"),
]

PRESET_FOR = {"color": "pulse", "motion": "motion"}


def measure_one(pipeline: str, clip: str, precision: str, n_iter: int) -> dict:
    """Time one configuration in this process and return it as plain data."""
    from vidmag.cuda import benchmark
    from vidmag.presets import PRESETS

    params: dict[str, object] = dict(PRESETS[PRESET_FOR[pipeline]].params)
    params["vid"] = clip
    r = benchmark.run(pipeline, precision, params, n_iter=n_iter)
    if not r.measured:
        return {"skipped": r.notes}
    return {
        "compute_ms": r.compute_ms,
        "transfer_ms": r.transfer_ms,
        "total_ms": r.total_ms,
        "stages": {
            s.name: {"median": s.median_ms, "min": s.min_ms, "max": s.max_ms}
            for s in r.stages
        },
    }


def wall_clock_ms(pipeline: str, clip: str, precision: str, repeats: int) -> float:
    """What a caller of `vidmag.magnify()` actually waits, array in and out.

    Different from the stage sums above, which exclude the entry point's own
    argument handling and allocation. Reported separately for that reason.
    """
    import time

    import numpy as np

    import vidmag
    from vidmag.cuda._common import read_frames

    frame_list, fps = read_frames(clip)
    frames = np.asarray(frame_list)
    preset = PRESET_FOR[pipeline]
    vidmag.magnify(frames, preset=preset, fps=fps, backend="cuda", precision=precision)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        vidmag.magnify(
            frames, preset=preset, fps=fps, backend="cuda", precision=precision
        )
        times.append((time.perf_counter() - t0) * 1e3)
    return sorted(times)[len(times) // 2]


def cpu_baseline_ms(pipeline: str, clip: str, repeats: int = 3) -> float:
    """The NumPy reference on the same machine, which every ratio divides by.

    Timed more than once on purpose: this is the denominator of every speed-up
    ratio the documentation quotes, and single runs of it were observed to vary
    by about 7 percent.
    """
    import time

    import numpy as np

    import vidmag
    from vidmag.cuda._common import read_frames

    frame_list, fps = read_frames(clip)
    frames = np.asarray(frame_list)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        vidmag.magnify(frames, preset=PRESET_FOR[pipeline], fps=fps, backend="cpu")
        times.append((time.perf_counter() - t0) * 1e3)
    return sorted(times)[len(times) // 2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu", required=True, help='Card name, e.g. "RTX 3090".')
    ap.add_argument("--out", required=True, help="JSON file to write.")
    ap.add_argument("--date", required=True, help="Measurement date, YYYY-MM-DD.")
    ap.add_argument("--iters", type=int, default=7)
    # Internal: the parent re-invokes this script once per configuration.
    ap.add_argument("--one", nargs=3, metavar=("PIPELINE", "CLIP", "PRECISION"))
    ap.add_argument("--wall", nargs=3, metavar=("PIPELINE", "CLIP", "PRECISION"))
    ap.add_argument("--cpu", nargs=2, metavar=("PIPELINE", "CLIP"))
    args = ap.parse_args()

    if args.one:
        p_, c_, pr_ = args.one
        print(json.dumps(measure_one(p_, c_, pr_, args.iters)))
        return 0
    if args.wall:
        p_, c_, pr_ = args.wall
        print(json.dumps({"wall_ms": wall_clock_ms(p_, c_, pr_, 5)}))
        return 0
    if args.cpu:
        p_, c_ = args.cpu
        print(json.dumps({"cpu_ms": cpu_baseline_ms(p_, c_)}))
        return 0

    def child(flag: str, *rest: str) -> dict:
        cmd = [
            sys.executable,
            __file__,
            "--gpu",
            args.gpu,
            "--out",
            args.out,
            "--date",
            args.date,
            "--iters",
            str(args.iters),
            flag,
            *rest,
        ]
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
        if p.returncode != 0:
            raise SystemExit(f"child failed for {rest}:\n{p.stderr[-2000:]}")
        return json.loads(p.stdout.strip().splitlines()[-1])

    results = {}
    for key, pipeline, clip, precision in CONFIGS:
        print(f"  timing {key} ...", file=sys.stderr, flush=True)
        results[key] = child("--one", pipeline, clip, precision)

    wall = {}
    for key, pipeline, clip, precision in CONFIGS:
        if pipeline == "motion" or key == "color_face_fp32":
            print(f"  wall clock {key} ...", file=sys.stderr, flush=True)
            wall[key] = child("--wall", pipeline, clip, precision)["wall_ms"]

    cpu = {}
    for pipeline, clip, key in (
        ("color", "data/face.mp4", "color_face"),
        ("motion", "data/baby.mp4", "motion_baby"),
    ):
        print(f"  cpu baseline {key} ...", file=sys.stderr, flush=True)
        cpu[key] = child("--cpu", pipeline, clip)["cpu_ms"]

    doc = {
        "gpu": args.gpu,
        "date": args.date,
        "n_iter": args.iters,
        "method": (
            "fresh process per config; 1 warmup + median of "
            f"{args.iters}; device synchronised after every stage. "
            "Regenerated by scripts/dev/record_gpu_bench.py."
        ),
        "results": results,
        "wall_clock_ms": wall,
        "cpu_baseline_ms": cpu,
        "note": (
            "compute_ms sums the kernel stages; transfer_ms sums the H2D and "
            "D2H stages; total_ms is the sum of both. wall_clock_ms is what a "
            "caller of vidmag.magnify() waits with the clip already in memory, "
            "which is larger than total_ms because it includes argument "
            "handling and output allocation. cpu_baseline_ms is the NumPy "
            "reference on this same machine and is the denominator of every "
            "ratio quoted from this file."
        ),
    }
    Path(args.out).write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

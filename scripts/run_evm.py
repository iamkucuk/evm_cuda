#!/usr/bin/env python3
"""CLI front-end for the MIT-faithful EVM baseline.

Mirrors the four ``amplify_spatial_*`` MATLAB functions. Parameters follow the
reference naming and defaults match ``reproduceResults.m`` where applicable.

Examples (all verified against the MIT sample outputs)::

    # Color / pulse (face.mp4) — reproduceResults.m face call
    python scripts/run_evm.py data/face.mp4 output/face_color.mp4 \\
        --mode color --alpha 50 --level 4 --fl 0.8333 --fh 1.0 --chromatt 1

    # Motion / IIR (baby.mp4) — reproduceResults.m baby call
    python scripts/run_evm.py data/baby.mp4 output/baby_motion.mp4 \\
        --mode iir --alpha 10 --lambda-c 16 --r1 0.4 --r2 0.05 --chromatt 0.1

    # Motion / ideal LPyr (guitar.mp4 E-string)
    python scripts/run_evm.py data/guitar.mp4 output/guitar_motion.mp4 \\
        --mode motion --alpha 50 --lambda-c 10 --fl 72 --fh 92 --chromatt 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from evm import (  # noqa: E402
    magnify_color_gdown_ideal,
    magnify_motion_lpyr_butter,
    magnify_motion_lpyr_iir,
    magnify_motion_lpyr_ideal,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Eulerian Video Magnification baseline (MIT-faithful)."
    )
    p.add_argument("input", help="Input video path.")
    p.add_argument("output", help="Output video path (.mp4).")
    p.add_argument(
        "--mode",
        choices=["color", "motion", "iir", "butter"],
        required=True,
        help=(
            "color = Gdown+ideal; motion = LPyr+ideal; "
            "iir = LPyr+IIR(r1,r2); butter = LPyr+1st-order Butterworth."
        ),
    )
    p.add_argument("--alpha", type=float, required=True, help="Magnification factor.")
    p.add_argument(
        "--level",
        type=int,
        default=4,
        help="Gaussian pyramid level (color mode only). Default 4.",
    )
    p.add_argument(
        "--lambda-c",
        type=float,
        default=16.0,
        help="lambda_c for the Figure-6 per-level alpha schedule (motion modes).",
    )
    p.add_argument(
        "--fl", type=float, default=0.83, help="Lower cutoff (Hz, for ideal/butter)."
    )
    p.add_argument(
        "--fh", type=float, default=1.0, help="Upper cutoff (Hz, for ideal/butter)."
    )
    p.add_argument(
        "--r1", type=float, default=0.4, help="IIR high cutoff coefficient (iir mode)."
    )
    p.add_argument(
        "--r2", type=float, default=0.05, help="IIR low cutoff coefficient (iir mode)."
    )
    p.add_argument(
        "--chromatt", type=float, default=1.0, help="Chrominance attenuation."
    )
    p.add_argument(
        "--sampling-rate",
        type=float,
        default=None,
        help="Override sampling rate (Hz); defaults to the input fps.",
    )
    p.add_argument(
        "--exaggeration-factor",
        type=float,
        default=2.0,
        help="Figure-6 exaggeration (MIT hardcodes 2). Default 2.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    kwargs = dict(alpha=args.alpha, chrom_attenuation=args.chromatt)
    if args.sampling_rate is not None:
        kwargs["sampling_rate"] = args.sampling_rate
    if args.mode in {"motion", "iir", "butter"}:
        kwargs["lambda_c"] = args.lambda_c
        kwargs["exaggeration_factor"] = args.exaggeration_factor

    print(f"[evm] {args.mode} magnify on {args.input}", file=sys.stderr)
    if args.mode == "color":
        out = magnify_color_gdown_ideal(
            args.input, args.output, level=args.level, fl=args.fl, fh=args.fh, **kwargs
        )
    elif args.mode == "motion":
        out = magnify_motion_lpyr_ideal(
            args.input, args.output, fl=args.fl, fh=args.fh, **kwargs
        )
    elif args.mode == "butter":
        out = magnify_motion_lpyr_butter(
            args.input, args.output, fl=args.fl, fh=args.fh, **kwargs
        )
    elif args.mode == "iir":
        out = magnify_motion_lpyr_iir(
            args.input, args.output, r1=args.r1, r2=args.r2, **kwargs
        )
    else:
        raise AssertionError("unreachable")

    print(
        f"[evm] wrote {out.shape[0]} frames @ {out.shape[1]}x{out.shape[2]} "
        f"-> {args.output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

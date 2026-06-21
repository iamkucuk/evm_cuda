#!/usr/bin/env python3
"""Fetch MIT EVM sample videos into ``data/``.

The canonical source is the authors' project page
<https://people.csail.mit.edu/mrub/evm/>, which hosts the same clips used in
the SIGGRAPH 2012 paper (face/baby = pulse, wrist = pulse, shadow/camera/guitar
= motion). The full list is defined in :data:`SAMPLES`; pass names on the
command line to fetch a subset.

Usage::

    python scripts/download_samples.py              # all defaults
    python scripts/download_samples.py face baby    # just these two
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

BASE = "https://people.csail.mit.edu/mrub/evm/video"

# name -> (filename, typical use). URLs verified Jun 2026.
SAMPLES: dict[str, tuple[str, str]] = {
    "face": ("face.mp4", "color / pulse"),
    "face2": ("face2.mp4", "color / pulse"),
    "baby": ("baby.mp4", "color / pulse"),
    "baby2": ("baby2.mp4", "color / pulse"),
    "wrist": ("wrist.mp4", "color / pulse"),
    "shadow": ("shadow.mp4", "motion"),
    "camera": ("camera.mp4", "motion"),
    "guitar": ("guitar.mp4", "motion"),
    "subway": ("subway.mp4", "motion"),
}

# MIT's own rendered result videos, used as ground truth for the integration
# tests in tests/test_against_mit_reference.py. Each entry maps a local
# filename to the (filename) on the MIT server.
REFERENCE_OUTPUTS: dict[str, str] = {
    "face_mit_ref.mp4": "face-ideal-from-0.83333-to-1-alpha-50-level-4-chromAtn-1.mp4",
    "baby_mit_ref.mp4": "baby-iir-r1-0.4-r2-0.05-alpha-10-lambda_c-16-chromAtn-0.1.mp4",
}


def download(name: str, dest_dir: Path, *, timeout: float = 60.0) -> Path:
    if name not in SAMPLES:
        raise KeyError(
            f"unknown sample {name!r}; choose from {sorted(SAMPLES)}"
        )
    filename, _use = SAMPLES[name]
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[samples] {name}: already present at {dest}", file=sys.stderr)
        return dest

    url = f"{BASE}/{filename}"
    print(f"[samples] {name}: fetching {url}", file=sys.stderr)
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        dest.write_bytes(r.content)
    print(
        f"[samples] {name}: wrote {dest.stat().st_size} bytes to {dest}",
        file=sys.stderr,
    )
    return dest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "samples",
        nargs="*",
        help="Which samples to fetch (default: all). "
        f"Choices: {sorted(SAMPLES)}",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Destination directory (default: <repo>/data).",
    )
    p.add_argument(
        "--with-references",
        action="store_true",
        help="Also fetch MIT's own rendered outputs for the integration tests.",
    )
    args = p.parse_args(argv)

    if args.with_references:
        for local, remote in REFERENCE_OUTPUTS.items():
            try:
                dest = args.out / local
                if dest.exists() and dest.stat().st_size > 0:
                    print(f"[samples] {local}: already present", file=sys.stderr)
                    continue
                url = f"{BASE}/{remote}"
                print(f"[samples] fetching {url}", file=sys.stderr)
                r = requests.get(url, timeout=60.0)
                r.raise_for_status()
                args.out.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(r.content)
            except Exception as e:  # noqa: BLE001
                print(f"[samples] {local}: FAILED {e}", file=sys.stderr)

    names = args.samples or sorted(SAMPLES)
    for n in names:
        try:
            download(n, args.out)
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f"[samples] {n}: FAILED {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

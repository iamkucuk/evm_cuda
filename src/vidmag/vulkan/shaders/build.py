#!/usr/bin/env python3
"""Compile every ``.comp`` shader in this directory to the ``.spv`` beside it.

The compiled shaders are committed, so that installing this project needs no
shader compiler. This script regenerates them, and exists because the flags
matter and were otherwise written down nowhere: they had to be recovered once
by compiling a shader every way and comparing the result byte for byte against
the committed file.

No optimisation flag. ``glslc -O`` roughly halves the binary and measured no
faster on the two upsample shaders, which are the hottest here, so the plain
form is kept — it is what every committed ``.spv`` already matches, and a
byte-identical rebuild is a useful property in itself.

    python src/vidmag/vulkan/shaders/build.py [--check]

``--check`` rebuilds into a temporary directory and reports any ``.spv`` that
does not match its source, without writing anything.
"""

from __future__ import annotations

import argparse
import filecmp
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent


def compile_one(source: Path, output: Path) -> None:
    subprocess.run(["glslc", str(source), "-o", str(output)], check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report stale .spv files instead of rewriting them",
    )
    args = parser.parse_args()

    if shutil.which("glslc") is None:
        print(
            "glslc not found. It ships with the Vulkan SDK, and on macOS with "
            "'brew install shaderc'.",
            file=sys.stderr,
        )
        return 2

    sources = sorted(HERE.glob("*.comp"))
    if not sources:
        print(f"no .comp shaders in {HERE}", file=sys.stderr)
        return 2

    stale = []
    with tempfile.TemporaryDirectory() as tmp:
        for source in sources:
            target = source.with_suffix(".spv")
            if args.check:
                fresh = Path(tmp) / target.name
                compile_one(source, fresh)
                if not target.exists() or not filecmp.cmp(target, fresh, shallow=False):
                    stale.append(target.name)
            else:
                compile_one(source, target)
                print(f"  {source.name} -> {target.name}")

    if args.check:
        if stale:
            print("stale compiled shaders: " + ", ".join(stale), file=sys.stderr)
            return 1
        print(f"all {len(sources)} compiled shaders match their source")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

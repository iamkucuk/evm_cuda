"""Build the EVM CUDA extension via CMake.

We deliberately do NOT use setuptools' built-in CUDA support (build_ext with
a custom compiler). Instead, setup.py shells out to CMake, which handles
nvcc invocation, multi-arch -gencode flags, and CUDA::cufft linking. The
resulting `_evm_cuda.so` is dropped into `cuda/evm_cuda/` next to the
Python wrapper package.

Usage::

    pip install -e cuda/         # editable install, builds the .so
    python cuda/setup.py build_ext --inplace

The build is a no-op on machines without nvcc; tests/cuda/conftest.py skips
the CUDA suite if the module fails to import.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_ext import build_ext as _build_ext

HERE = Path(__file__).resolve().parent
BUILD_DIR = HERE / "build"


class cmake_build_ext(_build_ext):
    """Run CMake to build the extension, then skip setuptools' own build."""

    def run(self) -> None:
        self._run_cmake()
        # The .so is already placed at LIBRARY_OUTPUT_DIRECTORY by CMake;
        # nothing for setuptools to copy.

    def _run_cmake(self) -> None:
        BUILD_DIR.mkdir(parents=True, exist_ok=True)
        cfg = "Debug" if self.debug else "Release"
        cmake = os.environ.get("CMAKE_COMMAND", "cmake")
        generator_args: list[str] = []
        # Use the Ninja generator when available — much faster than Make.
        try:
            subprocess.run([cmake, "--version"], check=True,
                           capture_output=True)
            ninja_check = subprocess.run(["ninja", "--version"],
                                         capture_output=True)
            if ninja_check.returncode == 0:
                generator_args = ["-G", "Ninja"]
        except FileNotFoundError:
            pass

        configure = [
            cmake, "-S", str(HERE), "-B", str(BUILD_DIR),
            *generator_args,
            f"-DCMAKE_BUILD_TYPE={cfg}",
        ]
        build = [cmake, "--build", str(BUILD_DIR), "--config", cfg,
                 "-j", str(os.cpu_count() or 4)]

        for cmd in (configure, build):
            print(f"[evm_cuda] $ {' '.join(cmd)}", file=sys.stderr)
            subprocess.run(cmd, check=True)


setup(
    name="evm-cuda",
    version="0.1.0",
    description="CUDA-accelerated Eulerian Video Magnification (kernels).",
    packages=["evm_cuda"],
    package_dir={"evm_cuda": "evm_cuda"},
    python_requires=">=3.9",
    cmdclass={"build_ext": cmake_build_ext},
    # The .so is produced by CMake into evm_cuda/, so declare it as package
    # data. setuptools won't try to compile it.
    package_data={"evm_cuda": ["*.so", "*.pyd", "*.dylib"]},
    zip_safe=False,
)

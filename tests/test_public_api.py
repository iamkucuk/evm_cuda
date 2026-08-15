"""The set of names this project promises to keep working.

Version numbers only mean something if there is a written-down list of what
they cover. This is that list. A name added here is a promise; a name removed
here is a breaking change and needs a major version. The test exists so that
neither can happen by accident — deleting an export, or exposing an internal
by forgetting an underscore, fails right here.

Nothing in this file needs a GPU. The compiled extension is absent on most
machines that will run these tests, and the names below must be importable
anyway; that is the point of resolving the GPU package lazily.
"""

from __future__ import annotations

import importlib

import pytest

import evm

# Everything importable as ``from evm import <name>``. Grouped by what it is
# for, because the grouping is the actual documentation of the surface.
EXPECTED_ROOT = {
    # The one-line entry point and the table of named parameter sets.
    "magnify",
    "PRESETS",
    "Preset",
    "presets",
    # The four pipelines, named as the reference paper names them. These take
    # a file path and write a file.
    "magnify_color_gdown_ideal",
    "magnify_motion_lpyr_ideal",
    "magnify_motion_lpyr_butter",
    "magnify_motion_lpyr_iir",
    # The 2013 phase-based method, which amplifies motion by changing phase
    # rather than by scaling image detail.
    "magnify_phase",
    # Video reading and writing.
    "load_video",
    "save_video",
    "VideoInfo",
    "rgb_to_yiq",
    "yiq_to_rgb",
    # Spatial building blocks.
    "BINOM5",
    "BINOM5_SUM1",
    "blur_dn",
    "blur_dn_clr",
    "build_lpyr",
    "recon_lpyr",
    "laplacian_pyramid_channels",
    "reconstruct_from_channels",
    "max_pyr_ht",
    # Temporal building blocks.
    "ideal_bandpass",
    "butter_bandpass",
    "iir_bandpass",
    # The per-level amplification schedule from the reference paper's Figure 6,
    # needed by anyone assembling a motion pipeline of their own.
    "figure6_alpha_schedule",
    # Constants that define agreement with the reference implementation.
    "DROP_LAST",
    "EXAGGERATION_FACTOR",
    # Subpackages, and the module the entry point is defined in. `api` is bound
    # as a side effect of importing `magnify` from it; it is listed here
    # because that makes `evm.api.magnify` a path callers can rely on, which is
    # a smaller promise than pretending the module is private while leaving it
    # perfectly reachable.
    "backend",
    "cpu",
    "io",
    "cuda",
    "opencl",
    "metal",
    "vulkan",
    "torch_backend",
    "api",
    # Live magnification. Bound as an attribute once anything imports it, and
    # documented as `from evm.stream import MotionStream`, so it is a path
    # callers may rely on.
    "stream",
    # Metadata.
    "__version__",
}


def test_every_promised_name_is_importable():
    missing = sorted(n for n in EXPECTED_ROOT if not hasattr(evm, n))
    assert not missing, (
        f"names this project promises are missing: {missing}. Removing one is "
        f"a breaking change and needs a major version bump."
    )


def test_nothing_extra_leaked_into_the_public_surface():
    """An accidental export is a promise nobody meant to make."""
    actual = {n for n in dir(evm) if not n.startswith("_")} | {"__version__"}
    unexpected = sorted(actual - EXPECTED_ROOT)
    assert not unexpected, (
        f"unexpected public names: {unexpected}. Either add them to "
        f"EXPECTED_ROOT deliberately, or give them a leading underscore."
    )


def test_all_matches_what_is_actually_exported():
    declared = set(evm.__all__)
    missing = sorted(n for n in declared if not hasattr(evm, n))
    assert not missing, f"__all__ names things that do not exist: {missing}"


def test_importing_evm_does_not_load_the_gpu_extension():
    """Importing the package must not touch CUDA on a machine without it."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", "import sys, evm; print('evm.cuda' in sys.modules)"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False", (
        "importing evm eagerly imported evm.cuda; on a machine with no GPU "
        "that turns a working install into an import error"
    )


@pytest.mark.parametrize(
    "module",
    [
        "evm.backend",
        "evm.cpu",
        "evm.cpu.ops",
        "evm.io",
        "evm.presets",
    ],
)
def test_supporting_modules_import_without_a_gpu(module):
    importlib.import_module(module)


def test_the_gpu_operations_module_is_named_even_without_a_gpu():
    """``evm.cuda.ops`` must be a promised name regardless of hardware.

    It cannot be imported without the compiled extension, but the name has to
    be part of the documented surface either way, or callers cannot write code
    against it on a machine where they only run the tests.
    """
    from evm import cuda

    assert "ops" in cuda.__all__
    assert "DeviceArray" in cuda.__all__

"""The facade on real hardware: an accelerator must agree with the CPU oracle.

Plan step 3.8, and the first instance of the shared conformance suite described
in `docs/dev/PLAN.md` section 3c. ``tests/test_api.py`` pins the facade's
contract on the CPU baseline; this file pins the part only a GPU can answer —
that choosing a different backend changes the speed and nothing else. A backend
that fails here may not appear in the support matrix (section 3c, consequence 4).

**Adding a backend.** Append one line to :data:`ACCELERATORS`:
``pytest.param("vulkan", marks=skip_no_vulkan)``. Everything else in the parity
test is written against the facade, not against CUDA. The fp16 section stays
CUDA-specific on purpose — it reads the CUDA package's own ``__all__``, and a
future backend adds its own case if it grows a lower-precision body.

**Tolerance.** ``TOL["end_to_end_rmse"]`` is imported from
``tests/cuda/conftest.py``; that table is append-only and lives in exactly one
place. It is quoted on the [0, 1] float scale the four path functions return, so
the uint8 frames the facade hands back are divided by 255 before comparison —
1e-2 there is 2.55 LSB of RMSE.

**Vacuity.** A preset that did nothing would make both backends return the input
and agree perfectly, proving nothing at all. Every comparison therefore first
asserts that the oracle actually magnified the clip.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import vidmag
from vidmag import backend
from vidmag.cpu.magnify import DROP_LAST
from vidmag.presets import PRESETS, Preset

from conftest import TOL, skip_no_cuda  # tests/cuda/ imports its conftest by name

#: The accelerators held to the CPU oracle, each carrying the mark that reports
#: why it cannot run on this host. One line per backend, by design.
ACCELERATORS = [pytest.param("cuda", marks=skip_no_cuda)]

#: Which pipelines the CUDA backend has an fp16 body for — read from the
#: package's own ``__all__`` rather than retyped here. A backend advertises a
#: precision by *having* ``<stem>_fp16_core`` (``vidmag/api.py:_resolve_core``), so
#: this set cannot drift from the code the way a capability flag could.
CUDA_FP16_STEMS = {
    name[: -len("_fp16_core")]
    for name in vidmag.cuda.__all__
    if name.endswith("_fp16_core")
}

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "golden_input.npz"


@pytest.fixture(scope="module")
def clip() -> np.ndarray:
    """The committed synthetic clip, trimmed to the window it was designed for.

    Loaded straight from the ``.npz`` rather than through ``tests.test_golden``:
    ``tests/cuda/`` is not a package — its modules import ``conftest`` by bare
    name — so it cannot import the ``tests`` package portably.
    """
    with np.load(FIXTURE) as f:
        frames = f["frames"]
    return frames[: len(frames) - DROP_LAST]


def in_band_fps(preset: Preset, frames: np.ndarray) -> float | None:
    """The frame rate that puts ``frames``' oscillation mid-band for ``preset``.

    Same derivation as ``tests/test_api.py``, duplicated for the import reason in
    :func:`clip`: bin ``k`` of ``N`` frames sits at ``k * fps / N`` Hz, so
    ``fps = centre * N / k`` moves the clip's own oscillation to the middle of
    ``[fl, fh]``. ``None`` for a preset with no ``fl``/``fh`` — the r1/r2 IIR
    band runs on frame index and has no rate to set.
    """
    if "fl" not in preset.params:
        return None
    signal = frames.reshape(len(frames), -1).mean(axis=1).astype(np.float64)
    bin_index = int(np.abs(np.fft.rfft(signal - signal.mean())).argmax())
    assert bin_index > 0, "the fixture clip carries no temporal oscillation to band-pass"
    centre = (preset.params["fl"] + preset.params["fh"]) / 2.0
    return centre * len(frames) / bin_index


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    """RMSE of two uint8 clips, on the [0, 1] scale ``TOL`` is quoted in."""
    diff = (a.astype(np.float64) - b.astype(np.float64)) / 255.0
    return float(np.sqrt((diff**2).mean()))


def _mean_abs_lsb(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


# ---------------------------------------------------------------------------
# Cross-backend parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset_name", sorted(PRESETS))
@pytest.mark.parametrize("backend_name", ACCELERATORS)
def test_backend_agrees_with_the_cpu_oracle(backend_name, preset_name, clip):
    """The same clip through two backends must be the same video.

    This is the claim the whole registry rests on: ``backend=`` picks how the
    arithmetic is executed, never what it computes. The CPU baseline is the
    oracle (it is what the MIT-reference tests measure), and the accelerator is
    held to ``TOL["end_to_end_rmse"]`` against it — the same tolerance the
    pipeline-level CUDA tests in this directory already use, imported, not
    restated.
    """
    spec = PRESETS[preset_name]
    fps = in_band_fps(spec, clip)

    oracle = vidmag.magnify(clip, preset=preset_name, backend="cpu", fps=fps)
    try:
        got = vidmag.magnify(clip, preset=preset_name, backend=backend_name, fps=fps)
    except ValueError as exc:
        # Not every pipeline exists on every backend — the phase-based one is
        # currently written only for the processor. Refusing is the correct
        # behaviour, and is asserted on its own below; what must not happen is
        # the backend quietly computing something else, which it did not.
        if "no fp32 implementation" in str(exc):
            pytest.skip(
                f"{backend_name} has no {spec.pipeline!r} pipeline: {exc}")
        raise

    assert got.shape == oracle.shape and got.dtype == oracle.dtype
    assert _mean_abs_lsb(oracle, clip) > 1.0, (
        f"preset {preset_name!r} did not magnify on the oracle, so agreeing with "
        "it would prove nothing"
    )
    rmse = _rmse(got, oracle)
    assert rmse < TOL["end_to_end_rmse"], (
        f"{backend_name} vs cpu on preset {preset_name!r}: RMSE {rmse:.5f} "
        f"exceeds TOL['end_to_end_rmse'] = {TOL['end_to_end_rmse']}"
    )


@skip_no_cuda
@pytest.mark.parametrize("preset_name", sorted(PRESETS))
def test_cuda_fp16_matches_the_oracle_where_it_exists_and_refuses_where_it_does_not(
    preset_name, clip
):
    """fp16 is opt-in per pipeline, and the opt-out is loud.

    Two assertions in one test because they are two halves of one rule: a
    backend advertises fp16 by *having* the core. Where the body exists, the
    cheaper precision still has to land inside the same end-to-end tolerance —
    "faster" may not quietly mean "different". Where it does not, asking for
    fp16 must raise, not compute in fp32 and hand back a result the caller will
    record as fp16.
    """
    spec = PRESETS[preset_name]
    fps = in_band_fps(spec, clip)

    if spec.pipeline not in CUDA_FP16_STEMS:
        with pytest.raises(ValueError, match="fp16"):
            vidmag.magnify(
                clip, preset=preset_name, backend="cuda", precision="fp16", fps=fps
            )
        return

    oracle = vidmag.magnify(clip, preset=preset_name, backend="cpu", fps=fps)
    got = vidmag.magnify(
        clip, preset=preset_name, backend="cuda", precision="fp16", fps=fps
    )
    assert _mean_abs_lsb(oracle, clip) > 1.0, "the oracle did not magnify"
    rmse = _rmse(got, oracle)
    assert rmse < TOL["end_to_end_rmse"], (
        f"cuda fp16 vs cpu on preset {preset_name!r}: RMSE {rmse:.5f} exceeds "
        f"TOL['end_to_end_rmse'] = {TOL['end_to_end_rmse']}"
    )


# ---------------------------------------------------------------------------
# The choice is made loudly, on the machine where it matters
# ---------------------------------------------------------------------------


@skip_no_cuda
def test_auto_picks_the_gpu_here_and_says_so(clip, caplog):
    """On a machine with a working extension, ``auto`` must not land on the CPU.

    The fall back it must never make silently is roughly 700x; the only way a
    caller can tell is the reported name and the INFO line, so both are checked
    on the host where the difference is real.
    """
    name, _ = backend.select("auto")
    assert name == "cuda", f"auto chose {name!r} on a machine with CUDA available"

    with caplog.at_level("INFO", logger="vidmag"):
        vidmag.magnify(clip, preset="motion")
    assert "backend='cuda'" in caplog.text

"""Tests for ``vidmag`` (:mod:`vidmag._cli`) and the ``run_evm.py`` shim.

The console script and the shim were shipped with no automated coverage at all:
every claim about them — that the shim forwards untouched, that ``--mode``
still carries ``run_evm.py``'s defaults, that a flag the pipeline cannot take
raises instead of being dropped — rested on one-off manual runs. This file
pins the behaviour that has to survive, without decoding a single video:
:func:`vidmag.backend.select` is replaced by a recorder, so what is asserted is
*which pipeline function was called with which arguments*.

The one thing it deliberately does not re-check is byte-level output: that is
``tests/test_golden.py``'s job, and the CLI reaches those pipelines through the
same ``magnify_<stem>`` functions the golden fixtures already hold fixed.
"""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pytest

from vidmag import _cli
from vidmag.presets import PRESETS

REPO = Path(__file__).resolve().parents[1]


class Recorder:
    """Stands in for a backend implementation: records the call, decodes nothing."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def __getattr__(self, name: str):
        if not name.startswith("magnify_"):
            raise AttributeError(name)

        def fn(vid_path, out_path, **kwargs):
            self.calls.append((name, (vid_path, out_path), kwargs))
            return np.zeros((3, 4, 5, 3), np.float32)

        # The CLI filters parameters by the real function's signature, so the
        # stand-in has to carry it — otherwise this test would pass on a
        # pipeline that no longer accepts what the CLI sends it.
        import vidmag.cpu.magnify as cpu

        # Setting __signature__ is how inspect.signature is redirected at
        # run time; the type system has no way to express it.
        fn.__signature__ = __import__("inspect").signature(  # type: ignore[attr-defined]
            getattr(cpu, name)
        )
        return fn


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    import vidmag.backend

    rec = Recorder()
    monkeypatch.setattr(vidmag.backend, "select", lambda name: ("cpu", rec))
    return rec


def run(argv: list[str]) -> int:
    return _cli.main(argv)


# ---------------------------------------------------------------------------
# Which pipeline, with which numbers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset_name", sorted(PRESETS))
def test_a_preset_runs_its_pipeline_with_the_tables_own_numbers(preset_name, recorder):
    """No second copy of the preset numbers in the CLI: the table is the source."""
    assert run(["magnify", "in.mp4", "out.mp4", "--preset", preset_name]) == 0

    spec = PRESETS[preset_name]
    (name, paths, kwargs), = recorder.calls
    assert name == f"magnify_{spec.pipeline}"
    assert paths == ("in.mp4", "out.mp4")
    assert kwargs == dict(spec.params)


def test_mode_color_reproduces_run_evm_pys_defaults(recorder):
    """``run_evm.py --mode color --alpha 50`` used to call this exact function
    with these exact numbers; the shim promises it still does."""
    assert run(["magnify", "in.mp4", "out.mp4", "--mode", "color", "--alpha", "50"]) == 0
    (name, _, kwargs), = recorder.calls
    assert name == "magnify_color_gdown_ideal"
    assert kwargs == dict(alpha=50.0, level=4, fl=0.83, fh=1.0, chrom_attenuation=1.0)


def test_mode_iir_reproduces_run_evm_pys_defaults(recorder):
    assert run(["magnify", "in.mp4", "out.mp4", "--mode", "iir", "--alpha", "10"]) == 0
    (name, _, kwargs), = recorder.calls
    assert name == "magnify_motion_lpyr_iir"
    assert kwargs == dict(
        alpha=10.0, lambda_c=16.0, r1=0.4, r2=0.05,
        chrom_attenuation=1.0, exaggeration_factor=2.0,
    )


def test_butter_is_reachable_and_no_preset_covers_it(recorder):
    """The reason the CLI drives ``magnify_<stem>`` rather than ``vidmag.magnify``:
    this pipeline has no preset, and ``run_evm.py --mode butter`` always ran it."""
    assert "motion_lpyr_butter" not in {s.pipeline for s in PRESETS.values()}
    assert run(["magnify", "in.mp4", "out.mp4", "--mode", "butter",
                "--alpha", "50", "--lambda-c", "10", "--fl", "72", "--fh", "92"]) == 0
    (name, _, kwargs), = recorder.calls
    assert name == "magnify_motion_lpyr_butter"
    assert kwargs["fl"] == 72.0 and kwargs["fh"] == 92.0 and kwargs["lambda_c"] == 10.0


def test_a_default_the_pipeline_cannot_take_is_dropped_silently(recorder):
    """``level`` means nothing to a motion pipeline, and the caller did not ask
    for it — dropping it is what ``run_evm.py`` did by hand."""
    assert run(["magnify", "in.mp4", "out.mp4", "--mode", "iir", "--alpha", "1"]) == 0
    (_, _, kwargs), = recorder.calls
    assert "level" not in kwargs


# ---------------------------------------------------------------------------
# Refusals: loud, named, non-zero
# ---------------------------------------------------------------------------


def test_a_typed_flag_the_pipeline_cannot_take_raises_and_names_it(recorder, capsys):
    with pytest.raises(SystemExit) as exc:
        run(["magnify", "in.mp4", "out.mp4", "--mode", "iir", "--alpha", "1",
             "--level", "4"])
    assert "level" in str(exc.value) and "motion_lpyr_iir" in str(exc.value)
    assert not recorder.calls, "the pipeline ran despite an inapplicable flag"


def test_mode_without_alpha_is_refused(recorder):
    with pytest.raises(SystemExit) as exc:
        run(["magnify", "in.mp4", "out.mp4", "--mode", "iir"])
    assert exc.value.code == 2  # argparse usage error, as run_evm.py gave


def test_preset_and_mode_are_mutually_exclusive(recorder):
    with pytest.raises(SystemExit) as exc:
        run(["magnify", "in.mp4", "out.mp4", "--preset", "pulse", "--mode", "iir"])
    assert exc.value.code == 2


def test_neither_preset_nor_mode_is_refused(recorder):
    with pytest.raises(SystemExit) as exc:
        run(["magnify", "in.mp4", "out.mp4"])
    assert exc.value.code == 2


def test_an_unknown_backend_exits_naming_the_registered_ones(capsys):
    """Not through the recorder: this is the real registry answering."""
    with pytest.raises(SystemExit) as exc:
        run(["magnify", "in.mp4", "out.mp4", "--preset", "pulse",
             "--backend", "nonesuch"])
    msg = str(exc.value)
    for expected in ("nonesuch", "cpu", "cuda", "opencl", "metal", "vulkan"):
        assert expected in msg, f"{expected} missing from {msg!r}"


def test_the_chosen_backend_is_always_announced(recorder, capsys):
    run(["magnify", "in.mp4", "out.mp4", "--preset", "pulse"])
    err = capsys.readouterr().err
    assert "backend=cpu" in err and "pipeline=color_gdown_ideal" in err


# ---------------------------------------------------------------------------
# The scripts/run_evm.py shim
# ---------------------------------------------------------------------------


def load_shim():
    path = REPO / "scripts" / "run_evm.py"
    spec = importlib.util.spec_from_file_location("run_evm_shim", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_shim_warns_and_forwards_its_arguments_untouched(recorder):
    """The Makefile still calls this file, so its argv contract is load-bearing."""
    shim = load_shim()
    argv = ["in.mp4", "out.mp4", "--mode", "color", "--alpha", "50",
            "--level", "4", "--fl", "0.8333", "--fh", "1.0", "--chromatt", "1"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert shim._forward(argv) == 0
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)

    (name, paths, kwargs), = recorder.calls
    assert name == "magnify_color_gdown_ideal"
    assert paths == ("in.mp4", "out.mp4")
    assert kwargs == dict(alpha=50.0, level=4, fl=0.8333, fh=1.0,
                          chrom_attenuation=1.0)


# ---------------------------------------------------------------------------
# A backend that implements only the array-in/array-out core
#
# The portable backends (metal, vulkan, opencl, torch) implement `<stem>_core`
# but not the file-in/file-out `magnify_<stem>`. Before the fallback below
# existed, `vidmag magnify` selected one of them and then died with
# "backend 'metal' has no color_gdown_ideal pipeline" — so the command failed
# on every machine whose fastest backend was one of those (every Mac). The CLI
# must instead read the file, run the core, and write the file itself.
# ---------------------------------------------------------------------------


class CoreOnlyRecorder:
    """A backend with only the array core, like metal/vulkan/opencl/torch."""

    def __init__(self) -> None:
        self.core_calls: list[tuple[str, float, dict]] = []

    def __getattr__(self, name: str):
        if name.startswith("magnify_"):
            # The whole point: no path function on this backend.
            raise AttributeError(name)
        if not name.endswith("_core"):
            raise AttributeError(name)

        def core(frames, fps, **kwargs):
            self.core_calls.append((name, fps, kwargs))
            return np.zeros((3, 4, 5, 3), np.uint8)

        import inspect

        from vidmag.backend import generic

        core.__signature__ = inspect.signature(getattr(generic, name))
        return core


def test_a_core_only_backend_reads_runs_the_core_and_writes(monkeypatch, tmp_path, capsys):
    """A backend with only `<stem>_core` still produces a file from the CLI.

    The CLI does the decode and encode itself, using the same reader and the
    same encoder the path-function backends use, so the container is identical.
    """
    import vidmag.backend
    import vidmag.cpu.magnify as cpu

    rec = CoreOnlyRecorder()
    monkeypatch.setattr(vidmag.backend, "select", lambda name: ("metal", rec))

    # No real video is decoded or encoded: stand in for the shared reader and
    # writer, and assert they were driven with the core's output.
    monkeypatch.setattr(
        cpu, "_read_frames",
        lambda path, **k: ([np.zeros((4, 5, 3), np.uint8)] * 3, 30.0),
    )
    written: dict = {}
    monkeypatch.setattr(
        cpu, "_write",
        lambda out_path, frames, fps: written.update(
            out=str(out_path), n=int(frames.shape[0]), fps=fps
        ),
    )

    inp, out = tmp_path / "in.mp4", tmp_path / "out.mp4"
    inp.write_bytes(b"")
    rc = run(["magnify", str(inp), str(out), "--preset", "pulse"])

    assert rc == 0
    assert rec.core_calls, "the core was never called"
    name, fps, kwargs = rec.core_calls[0]
    assert name == "color_gdown_ideal_core"
    assert fps == 30.0
    # The core is handed exactly the preset's own numbers, nothing invented.
    assert kwargs == dict(PRESETS["pulse"].params)
    assert written["out"] == str(out), "the shared encoder was not driven with the output path"
    assert "backend=metal" in capsys.readouterr().err

"""``vidmag`` — the command-line front end (`docs/dev/PLAN.md` step 3.6).

Three subcommands::

    vidmag magnify data/face.mp4 output/face.mp4 --preset pulse
    vidmag download face baby --with-references
    vidmag bench data/face.mp4 --preset pulse --precision all

``magnify`` is the main one. Its flags are exactly the set
``scripts/run_evm.py`` has always taken — same names, same meanings, same
per-mode defaults — plus ``--preset`` and ``--backend``. That is why
``scripts/run_evm.py`` could be reduced to a five-line forwarder: every
argument it accepted is still accepted here, so it hands its argv over
untouched and its users keep working.

Two ways to say which pipeline to run, exactly one required:

``--preset NAME``
    A row of :data:`vidmag.presets.PRESETS`: pipeline *and* numbers, from the one
    table that also feeds :func:`vidmag.magnify` and the docs.
``--mode {color,motion,iir,butter}``
    The pipeline only, with ``run_evm.py``'s defaults for everything else.
    ``--alpha`` is then required, as it always was. This is also the only way
    to reach ``motion_lpyr_butter``, which no preset covers.

Either way the individual flags override what the preset or the defaults
supply, and a flag the chosen pipeline does not accept is an error naming it —
it is never dropped on the floor.

Why this drives ``magnify_<stem>(in_path, out_path, ...)`` rather than
:func:`vidmag.magnify`
-------------------------------------------------------------------------

Both exist on every backend and both call the same cores. The path functions
are the right ones for a file-in/file-out tool for two reasons:

1. **Plan decision D8.** A path in means reference semantics, and that
   includes dropping the last ten frames. ``make run-color`` and every
   ``run_evm.py`` invocation depend on it — an ideal bandpass over 301 frames
   is not the same filter as one over 291, so the change would be silent and
   numeric, not cosmetic. :func:`vidmag.magnify` is the array API and defaults to
   dropping none, deliberately.
2. **All four pipelines are reachable.** :func:`vidmag.magnify` selects its
   pipeline through a preset, and no preset covers ``motion_lpyr_butter``;
   ``run_evm.py --mode butter`` has always worked and still does.

The backend still comes from the registry (:func:`vidmag.backend.select`), so
``--backend`` behaves as it does everywhere else: ``auto`` prefers native CUDA,
a named backend that cannot run here says why, and the resolved name is printed
on every run.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any

from . import backend as _backend
from . import presets as _presets
from .cpu.magnify import EXAGGERATION_FACTOR

__all__ = ["build_parser", "main"]

#: ``--mode`` value -> pipeline stem. The four modes ``scripts/run_evm.py``
#: has always offered; the stems are what ``magnify_<stem>`` is named after.
_MODE_STEM = {
    "color": "color_gdown_ideal",
    "motion": "motion_lpyr_ideal",
    "butter": "motion_lpyr_butter",
    "iir": "motion_lpyr_iir",
}

#: What ``--mode`` supplies for the flags the caller leaves out. These are
#: ``scripts/run_evm.py``'s argparse defaults, moved here when that file became
#: a shim, so a forwarded invocation runs the identical numbers it used to.
#: Values the chosen pipeline does not take are dropped (its signature decides);
#: a value the *caller* typed is never dropped, it raises.
_MODE_DEFAULTS: dict[str, float | int] = {
    "level": 4,
    "lambda_c": 16.0,
    "fl": 0.83,
    "fh": 1.0,
    "r1": 0.4,
    "r2": 0.05,
    "chrom_attenuation": 1.0,
    # Imported, not retyped: the reference's hardcoded 2 lives in
    # vidmag/cpu/magnify.py and is what every pipeline defaults to.
    "exaggeration_factor": EXAGGERATION_FACTOR,
}

#: The pipeline parameters ``magnify`` exposes, by the keyword the pipeline
#: functions use. Every one defaults to ``None`` in the parser, so "the caller
#: typed this" and "this is a default" stay distinguishable.
_PARAM_DESTS = (
    "alpha",
    "level",
    "lambda_c",
    "fl",
    "fh",
    "r1",
    "r2",
    "chrom_attenuation",
    "sampling_rate",
    "exaggeration_factor",
)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """The full ``vidmag`` parser, subcommands included."""
    p = argparse.ArgumentParser(
        prog="vidmag",
        description=(
            "Eulerian Video Magnification: reveal subtle colour and motion "
            "changes in video (MIT SIGGRAPH 2012), on your processor or "
            "any graphics hardware present."
        ),
    )
    sub = p.add_subparsers(dest="command", required=True, metavar="{magnify,download,bench}")

    _add_magnify(sub)
    _add_download(sub)
    _add_bench(sub)
    return p


def _add_magnify(sub: argparse._SubParsersAction) -> None:
    m = sub.add_parser(
        "magnify",
        help="Run a pipeline on a video file.",
        description=(
            "Run one magnification pipeline on a video file. Pick the pipeline "
            "with --preset (name, numbers and all) or with --mode (pipeline "
            "only, plus --alpha); individual flags override either."
        ),
        epilog=(
            "examples:\n"
            "  vidmag magnify data/face.mp4 out.mp4 --preset pulse\n"
            "  vidmag magnify data/baby.mp4 out.mp4 --preset motion "
            "--alpha 20\n"
            "  vidmag magnify data/guitar.mp4 out.mp4 --mode butter "
            "--alpha 50 --lambda-c 10 --fl 72 --fh 92\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    m.add_argument("input", help="Input video path.")
    m.add_argument("output", help="Output video path (.mp4).")

    what = m.add_mutually_exclusive_group(required=True)
    what.add_argument(
        "--preset",
        choices=sorted(_presets.PRESETS),
        help="Named parameter set: "
        + "; ".join(
            f"{n} = {s.pipeline}" for n, s in sorted(_presets.PRESETS.items())
        )
        + ". See vidmag.presets.PRESETS for the numbers and their provenance.",
    )
    what.add_argument(
        "--mode",
        choices=sorted(_MODE_STEM),
        help=(
            "Pipeline without a preset (--alpha then required): "
            "color = Gdown+ideal; motion = LPyr+ideal; iir = LPyr+IIR(r1,r2); "
            "butter = LPyr+1st-order Butterworth."
        ),
    )

    m.add_argument("--alpha", type=float, help="Magnification factor.")
    m.add_argument(
        "--level", type=int, help="Gaussian pyramid level (color mode only)."
    )
    m.add_argument(
        "--lambda-c",
        type=float,
        dest="lambda_c",
        help="lambda_c for the Figure-6 per-level alpha schedule (motion modes).",
    )
    m.add_argument("--fl", type=float, help="Lower cutoff (Hz, for ideal/butter).")
    m.add_argument("--fh", type=float, help="Upper cutoff (Hz, for ideal/butter).")
    m.add_argument("--r1", type=float, help="IIR high cutoff coefficient (iir mode).")
    m.add_argument("--r2", type=float, help="IIR low cutoff coefficient (iir mode).")
    m.add_argument(
        "--chromatt",
        type=float,
        dest="chrom_attenuation",
        help="Chrominance attenuation.",
    )
    m.add_argument(
        "--sampling-rate",
        type=float,
        dest="sampling_rate",
        help="Override sampling rate (Hz); defaults to the input fps.",
    )
    m.add_argument(
        "--exaggeration-factor",
        type=float,
        dest="exaggeration_factor",
        help="Figure-6 exaggeration (MIT hardcodes 2). Default 2.",
    )
    m.add_argument(
        "--backend",
        default="auto",
        help=(
            "auto (native CUDA if it can run here, else the CPU baseline), or a "
            "registered name such as cpu or cuda. Never falls back silently; "
            "the chosen name is printed. Default: auto."
        ),
    )
    # The subparser goes in the namespace so a handler's parser.error()
    # prints `vidmag magnify`'s usage, not the top-level one.
    m.set_defaults(func=_cmd_magnify, parser=m)


def _add_download(sub: argparse._SubParsersAction) -> None:
    d = sub.add_parser(
        "download",
        help="Fetch the MIT sample videos into data/.",
        description=(
            "Fetch the MIT EVM sample clips. Thin wrapper over "
            "scripts/download_samples.py, which owns the URLs and the file "
            "list; this needs a source checkout, since scripts/ is not part of "
            "the installed package."
        ),
    )
    d.add_argument(
        "samples",
        nargs="*",
        help="Which samples to fetch (default: all). Names come from "
        "scripts/download_samples.py:SAMPLES.",
    )
    d.add_argument("--out", type=Path, help="Destination directory (default: <repo>/data).")
    d.add_argument(
        "--with-references",
        action="store_true",
        help="Also fetch MIT's own rendered outputs for the integration tests.",
    )
    d.set_defaults(func=_cmd_download, parser=d)


def _add_bench(sub: argparse._SubParsersAction) -> None:
    b = sub.add_parser(
        "bench",
        help="Per-stage GPU benchmark (needs CUDA).",
        description=(
            "Time one pipeline per stage with the benchmark harness "
            "(vidmag.cuda.benchmark): one warmup, then --iters timed runs, median "
            "reported, device synchronised after every stage."
        ),
    )
    b.add_argument("video", help="Clip to benchmark.")
    b.add_argument(
        "--preset",
        choices=sorted(_presets.PRESETS),
        required=True,
        help="Which preset's pipeline and parameters to time. The harness "
        "covers the two device-resident pipelines only; the others say so.",
    )
    b.add_argument(
        "--precision",
        choices=["fp32", "fp16", "all"],
        default="fp32",
        help="Precision to time; 'all' times both and prints the comparison "
        "table. Default: fp32.",
    )
    b.add_argument(
        "--iters", type=int, default=5, help="Timed runs per config. Default: 5."
    )
    b.add_argument(
        "--out",
        help="Also render the result here (written during the first timed run).",
    )
    b.set_defaults(func=_cmd_bench, parser=b)

    s = sub.add_parser(
        "stream",
        help="Magnify a camera, file or network stream as it arrives.",
        description=(
            "Amplify motion frame by frame, using only what has already been "
            "seen. The result is identical to magnifying the whole clip at "
            "once; the difference is that this needs no future frames, so it "
            "can run on a live feed. Only the motion pipeline can work this "
            "way: selecting frequencies with a Fourier transform needs all of "
            "time at once."
        ),
    )
    s.add_argument(
        "source",
        help="Camera index (e.g. 0), a video file, or a stream address.",
    )
    s.add_argument(
        "--out", help="Write the magnified frames here as a video file."
    )
    s.add_argument(
        "--display",
        action="store_true",
        help="Show the result in a window. Press q to stop.",
    )
    s.add_argument("--alpha", type=float, default=10.0,
                   help="How much to amplify. Default: 10.")
    s.add_argument("--lambda-c", type=float, default=16.0, dest="lambda_c",
                   help="Spatial cutoff in pixels; raise if the output "
                        "shimmers. Default: 16.")
    s.add_argument("--r1", type=float, default=0.4,
                   help="Faster decay rate. Default: 0.4.")
    s.add_argument("--r2", type=float, default=0.05,
                   help="Slower decay rate. Default: 0.05.")
    s.add_argument("--chrom-attenuation", type=float, default=0.1,
                   dest="chrom_attenuation",
                   help="How much colour to amplify relative to brightness. "
                        "Default: 0.1.")
    s.add_argument("--backend", default="cpu",
                   help="Which backend to use. Default: cpu, which is measured "
                        "to be fastest for one frame at a time — the graphics "
                        "backends pay a launch cost per frame that does not "
                        "shrink with the frame size, and they win on whole "
                        "clips rather than live feeds.")
    s.add_argument("--max-frames", type=int, default=None, dest="max_frames",
                   help="Stop after this many frames.")
    s.set_defaults(func=_cmd_stream, parser=s)


# ---------------------------------------------------------------------------
# magnify
# ---------------------------------------------------------------------------


def _cmd_magnify(args: argparse.Namespace) -> int:
    # Everything the caller actually typed. Absent flags are None, so preset
    # numbers and defaults can fill the gaps without being second-guessed.
    typed = {k: getattr(args, k) for k in _PARAM_DESTS if getattr(args, k) is not None}

    if args.preset is not None:
        spec = _presets.get(args.preset)
        stem = spec.pipeline
        params = {**spec.params, **typed}
        # A preset's own numbers are as deliberate as a typed flag: if the
        # pipeline cannot take one, that is a broken preset, not a default.
        must_apply = params
    else:
        stem = _MODE_STEM[args.mode]
        params = {**_MODE_DEFAULTS, **typed}
        must_apply = typed
        if "alpha" not in typed:
            args.parser.error(f"--alpha is required with --mode {args.mode}")

    name, impl = _backend.select(args.backend)
    # Two ways a backend can offer a pipeline. `magnify_<stem>` reads the file
    # and writes the file (the processor and NVIDIA backends). `<stem>_core`
    # takes an array and returns an array (every backend, via the generic
    # binder); the portable backends — metal, vulkan, opencl, torch — have only
    # this one. Prefer the path function where it exists; otherwise do the file
    # I/O here so the command still runs on those backends instead of dying.
    fn = getattr(impl, "magnify_" + stem, None)
    core = getattr(impl, stem + "_core", None)
    # Whichever of the two this backend has, preferring the path function. One
    # variable rather than two, so that refusing None here is what tells a type
    # checker the rest of this function has something callable.
    pipeline = fn if fn is not None else core
    if pipeline is None:
        raise SystemExit(
            f"error: backend {name!r} has no {stem} pipeline "
            f"(no 'magnify_{stem}' and no '{stem}_core'). Pick another "
            "--backend, or run a pipeline it does implement."
        )

    # The pipeline's own signature decides which parameters apply — no second
    # table here to fall out of step with it. Defaults may be dropped silently
    # (level means nothing to a motion pipeline); a typed flag never is. The
    # path function and the core name their parameters the same way, minus the
    # frames and sampling rate the core reads from the file, so either answers
    # "which parameters does this pipeline take".
    accepted = set(inspect.signature(pipeline).parameters)
    refused = sorted(k for k in must_apply if k not in accepted)
    if refused:
        internal = {"vid_path", "out_path", "frames_bgr_u8", "fps", "on_stage"}
        raise SystemExit(
            f"error: the {stem} pipeline takes no "
            f"{', '.join(repr(k) for k in refused)}; it takes "
            f"{', '.join(sorted(accepted - internal))}."
        )
    params = {k: v for k, v in params.items() if k in accepted}

    chosen = args.preset or args.mode
    print(
        f"[vidmag] backend={name} (requested {args.backend}) pipeline={stem} "
        f"{'preset' if args.preset else 'mode'}={chosen}",
        file=sys.stderr,
    )
    if fn is not None:
        out = fn(args.input, args.output, **params)
    else:
        # Array-core-only backend: read the clip and write the result here,
        # with the same reader (which drops the reference's trailing frames)
        # and the same H.264 encoder the path functions use, so the container
        # is byte-identical to a path-function backend's output.
        import numpy as np

        from .cpu.magnify import _read_frames, _write

        frame_list, fps = _read_frames(args.input)
        # `pipeline` is the core here: this branch runs only when there is no
        # path function, which is the case that made it the core above.
        out = pipeline(np.stack(frame_list, axis=0), fps, **params)
        _write(args.output, out, fps)
    print(
        f"[vidmag] wrote {out.shape[0]} frames @ {out.shape[1]}x{out.shape[2]} "
        f"-> {args.output}",
        file=sys.stderr,
    )
    return 0


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------


def _cmd_download(args: argparse.Namespace) -> int:
    mod = _load_download_samples()
    argv = list(args.samples)
    if args.out is not None:
        argv += ["--out", str(args.out)]
    if args.with_references:
        argv.append("--with-references")
    return mod.main(argv)


def _load_download_samples() -> Any:
    """Import ``scripts/download_samples.py`` from the checkout, by path.

    That script owns the sample list, the MIT URLs and the skip-if-present
    rule, and copying any of it here would give this repository two answers to
    "where do the clips come from". It is not part of the installed package
    (``wheel.packages`` is ``src/vidmag`` and ``src/evm_cuda``), so two named
    places are tried and both are named again if neither has it: beside the
    package (a source checkout or an editable install) and under the working
    directory (a wheel install, run from a checkout — which is where the clips
    are wanted anyway, since they land in ``<repo>/data``).
    """
    candidates = (
        Path(__file__).resolve().parents[2] / "scripts" / "download_samples.py",
        Path.cwd() / "scripts" / "download_samples.py",
    )
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        tried = "\n".join(f"  {p}" for p in candidates)
        raise SystemExit(
            "error: scripts/download_samples.py not found. Looked in:\n"
            f"{tried}\n"
            "`vidmag download` runs that script, which does not ship in "
            "the wheel; clone "
            "https://github.com/iamkucuk/eulerian-video-magnification-cuda and "
            "run this from the checkout, or fetch the clips by hand from "
            "https://people.csail.mit.edu/mrub/evm/."
        )
    spec = importlib.util.spec_from_file_location("vidmag._download_samples", path)
    if spec is None or spec.loader is None:
        # Unreachable for the .py file `p.is_file()` just confirmed, but both
        # are Optional in the import machinery, and an unloadable spec must say
        # so rather than surface as an AttributeError on None two lines down.
        raise SystemExit(f"error: {path} exists but could not be loaded as a module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# bench
# ---------------------------------------------------------------------------


def _cmd_bench(args: argparse.Namespace) -> int:
    try:
        from .cuda import benchmark
    except ImportError as exc:
        # runtime.require_cuda() is the one sentence this project uses to
        # explain a missing extension; ask it rather than writing a second.
        from .cuda import runtime

        try:
            runtime.require_cuda()
        except RuntimeError as why:
            raise SystemExit(f"error: {why}") from exc
        raise

    spec = _presets.get(args.preset)
    # benchmark._PIPELINES is the harness's own map of (config key, precision)
    # -> the function it times; reading the stems out of it keeps this list
    # from drifting when the harness grows a pipeline.
    keys = {
        fname[len("magnify_"):]: key
        for (key, precision), fname in benchmark._PIPELINES.items()
        if precision == "fp32"
    }
    if spec.pipeline not in keys:
        raise SystemExit(
            f"error: the benchmark harness does not cover {spec.pipeline} "
            f"(preset {args.preset!r}); it covers "
            f"{', '.join(sorted(keys))}."
        )

    precisions = ["fp32", "fp16"] if args.precision == "all" else [args.precision]
    params = {"vid": args.video, **spec.params}
    results = []
    for precision in precisions:
        print(
            f"[vidmag] bench {keys[spec.pipeline]} {precision} on {args.video} "
            f"({args.iters} timed runs)",
            file=sys.stderr,
        )
        result = benchmark.run(
            keys[spec.pipeline],
            precision,
            params,
            out_path=args.out,
            n_iter=args.iters,
        )
        print(result)
        results.append(result)

    if len(results) > 1:
        print()
        print(benchmark.summarize(results, n_iter=args.iters))
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _cmd_stream(args: argparse.Namespace) -> int:
    """Magnify a live source frame by frame, reporting how fast it kept up."""
    import time

    from .io.capture import open_source
    from .stream import MotionStream

    source = open_source(args.source)
    height, width = source.size
    if height <= 0 or width <= 0:
        source.close()
        raise SystemExit(
            f"error: {source.description} did not report a frame size; it may "
            f"not be delivering video"
        )

    stream = MotionStream(
        height, width, alpha=args.alpha, lambda_c=args.lambda_c,
        r1=args.r1, r2=args.r2, chrom_attenuation=args.chrom_attenuation,
        backend=args.backend,
    )
    print(f"[vidmag] {source.description}: {width}x{height} at "
          f"{source.fps:g} frames/s, backend={stream.backend_name}")

    writer = None
    durations: list[float] = []
    try:
        for index, frame in enumerate(source.frames()):
            if args.max_frames is not None and index >= args.max_frames:
                break
            started = time.perf_counter()
            magnified = stream.push(frame)
            durations.append(time.perf_counter() - started)

            if args.out:
                if writer is None:
                    import cv2
                    writer = cv2.VideoWriter(
                        # Spelled through VideoWriter because the
                        # module-level alias is absent from OpenCV's
                        # type information.
                        args.out, cv2.VideoWriter.fourcc(*"mp4v"),
                        source.fps, (width, height))
                writer.write(magnified)
            if args.display:
                import cv2
                cv2.imshow("vidmag", magnified)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print()
    finally:
        source.close()
        if writer is not None:
            writer.release()
        if args.display:
            import cv2
            cv2.destroyAllWindows()

    if durations:
        ordered = sorted(durations)
        median = ordered[len(ordered) // 2] * 1000
        worst = ordered[int(len(ordered) * 0.95)] * 1000
        # Both numbers, because an average hides the stalls that are what a
        # viewer actually notices on a live feed.
        print(f"[vidmag] {len(durations)} frames: {median:.1f} ms typical, "
              f"{worst:.1f} ms at the 95th percentile "
              f"({1000 / median:.1f} frames/s sustained)")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run ``vidmag``. Returns the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except _backend.BackendError as exc:
        # The registry's message already names the backend and the reason; a
        # traceback would add noise, not information. Exit code stays non-zero.
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":  # pragma: no cover - `python -m vidmag._cli`
    raise SystemExit(main())

"""Fair, methodology-clean benchmarking for the EVM CUDA pipelines.

Owns the *measurement* so the notebook (and CLI profilers) own only the
*question*. A single ``run()`` call executes one pipeline/precision config
with the same warmup + sync + median discipline for every configuration, and
returns a structured ``BenchResult`` (per-stage timings + the GPU it ran on).

Design rules
------------
* **No duplication.** ``run()`` calls the real ``magnify_*`` entry points
  (the same code the application uses); per-stage timing comes from an
  ``on_stage`` hook those functions already expose. ``benchmark`` never
  re-implements a pipeline stage.
* **Fair by construction.** Every config is measured identically: one warmup
  run (excluded), ``n_iter`` timed runs, median reported, a
  ``cudaDeviceSynchronize`` after every stage.
* **No crash-then-catch.** VRAM is checked up front; an OOM-risky config
  returns a ``BenchResult`` with ``notes='skipped (insufficient VRAM)'``
  rather than bringing down the runtime.
"""
from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import _evm_cuda
from . import batched


# ---------------------------------------------------------------------------
# GPU introspection
# ---------------------------------------------------------------------------

def gpu_name() -> str:
    """The CUDA device name (e.g. 'NVIDIA A100-SXM4-80GB'), or 'unknown'."""
    import subprocess
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=10)
        return r.stdout.strip().splitlines()[0]
    except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return "unknown"


def gpu_free_bytes() -> int:
    """Free GPU memory in bytes (0 if the device/introspection is unavailable)."""
    # gpu_mem_info raises RuntimeError when no CUDA device is present; that's the
    # expected "unavailable" case. Anything else is a real error and propagates.
    try:
        free_b, _ = _evm_cuda.gpu_mem_info()
        return int(free_b)
    except RuntimeError:
        return 0


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _sync() -> None:
    _evm_cuda.device_synchronize()


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class StageReport:
    name: str
    median_ms: float
    min_ms: float
    max_ms: float
    pct_of_total: float

    def __str__(self) -> str:
        return (f"  {self.name:<22s} {self.median_ms:>8.1f} ms  "
                f"({self.min_ms:.1f}-{self.max_ms:.1f})  {self.pct_of_total:5.1f}%")


@dataclass
class BenchResult:
    pipeline: str            # "color" | "motion"
    precision: str           # "fp32" | "fp16"
    stages: list[StageReport] = field(default_factory=list)
    total_ms: float = 0.0
    gpu: str = ""
    output_path: str | None = None
    notes: str = ""          # non-empty => not measured (e.g. 'skipped (OOM)')

    @property
    def measured(self) -> bool:
        return not self.notes and bool(self.stages)

    def __str__(self) -> str:
        head = f"{self.pipeline} {self.precision.upper()} on {self.gpu}"
        if self.notes:
            return f"{head}: {self.notes}"
        lines = [head, "-" * 52]
        lines += [str(s) for s in self.stages]
        lines.append("-" * 52)
        lines.append(f"  {'TOTAL':<22s} {self.total_ms:>8.1f} ms")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# The stage-capture hook (the contract ``magnify_*`` fulfills)
# ---------------------------------------------------------------------------

class _StageRecorder:
    """Collects per-stage wall-clock times across one ``run()`` call.

    ``magnify_*`` call ``on_stage(name, fn)`` for each stage; the recorder
    syncs, times, and returns the stage's result."""
    def __init__(self) -> None:
        self.stages: dict[str, list[float]] = {}

    def __call__(self, name: str, fn: Callable[[], object]) -> object:
        _sync()
        t0 = time.perf_counter()
        result = fn()
        _sync()
        self.stages.setdefault(name, []).append(time.perf_counter() - t0)
        return result


# ---------------------------------------------------------------------------
# VRAM budget estimates (rough — used only to pre-skip hopeless configs)
# ---------------------------------------------------------------------------

def _estimated_peak_vram(pipeline: str, precision: str, n: int, h: int, w: int) -> int:
    """A conservative byte estimate of peak VRAM for one config."""
    per = 2 if precision == "fp16" else 4
    if pipeline == "color":
        level = 4
        hl, wl = h, w
        for _ in range(level):
            hl, wl = (hl + 1) // 2, (wl + 1) // 2
        # ntsc + planar + gdown + filt + out (overlaps approximated)
        return int((n * h * w * 3 * per * 2.0) + (n * 3 * hl * wl * 4) + (n * h * w * 3))
    # motion: band data dominates
    levels = 1; hh, ww = h, w
    while hh >= 5 and ww >= 5:
        levels += 1; hh = (hh + 1) // 2; ww = (ww + 1) // 2
    band = 0; ch, cw = h, w
    for _ in range(levels):
        band += ch * cw * n * 3; ch, cw = (ch + 1) // 2, (cw + 1) // 2
    return int(band * per * 1.8 + n * h * w * 3)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

_PIPELINES = {
    ("color", "fp32"): "magnify_color_gdown_ideal",
    ("color", "fp16"): "magnify_color_gdown_ideal_fp16",
    ("motion", "fp32"): "magnify_motion_lpyr_iir",
    ("motion", "fp16"): "magnify_motion_lpyr_iir_fp16",
}


def run(
    pipeline: str,
    precision: str,
    params: dict,
    *,
    out_path: str | Path | None = None,
    n_iter: int = 5,
) -> BenchResult:
    """Benchmark one (pipeline, precision) configuration.

    ``params`` are the keyword arguments forwarded to the underlying
    ``magnify_*`` function (``vid``/``out`` keys are handled here). Returns a
    :class:`BenchResult`; if VRAM is too tight the config is *skipped* (not
    crashed) with ``notes='skipped (insufficient VRAM)'``.

    The first timed call also writes the output video to ``out_path`` (if
    given), so a single ``run()`` both measures and renders.
    """
    key = (pipeline, precision)
    if key not in _PIPELINES:
        raise ValueError(
            f"unknown config {key!r}; choose pipeline in {{color, motion}} "
            f"and precision in {{fp32, fp16}}")
    fn = getattr(batched, _PIPELINES[key])

    vid = params["vid"]
    call_params = {k: v for k, v in params.items() if k not in ("vid", "out")}

    # Probe clip dimensions for the VRAM check (reads metadata only).
    frames, fps = batched._read_frames(vid)
    n = len(frames); h, w = frames[0].shape[:2]
    del frames

    need = _estimated_peak_vram(pipeline, precision, n, h, w)
    free = gpu_free_bytes()
    result = BenchResult(pipeline=pipeline, precision=precision, gpu=gpu_name(),
                         output_path=str(out_path) if out_path else None)
    # 1.15x safety margin on the (rough) estimate.
    if free and need * 1.15 > free:
        result.notes = (f"skipped (insufficient VRAM: need ~{need/1e9:.1f} GB, "
                        f"have {free/1e9:.1f} GB free)")
        return result

    target = str(out_path) if out_path else ""

    def _one_call(recorder: _StageRecorder | None, *, write: bool):
        # Empty target => magnify_* skips _write and returns the float array only.
        out = target if write else ""
        return fn(vid, out, on_stage=recorder, **call_params)

    recorder = _StageRecorder()
    try:
        _one_call(None, write=False)            # warmup (excluded, no file)
        _one_call(recorder, write=True)         # iter 1 — also renders the video
        for _ in range(n_iter - 1):             # remaining timed iters (no write)
            _one_call(recorder, write=False)
    except MemoryError as e:                    # GPU/CPU OOM — report, don't crash
        gc.collect(); _sync()
        result.notes = f"skipped (out of memory: {e})"
        return result
    except RuntimeError as e:                    # CUDA runtime errors (incl. OOM)
        if "out of memory" in str(e).lower():
            gc.collect(); _sync()
            result.notes = f"skipped (out of memory)"
            return result
        raise                                    # genuine CUDA errors must surface

    gc.collect(); _sync()

    # Collapse per-stage lists into median/min/max.
    names = list(recorder.stages.keys())
    medians = {k: _median(recorder.stages[k]) for k in names}
    total = sum(medians.values())
    for k in names:
        vals = recorder.stages[k]
        result.stages.append(StageReport(
            name=k, median_ms=medians[k] * 1000,
            min_ms=min(vals) * 1000, max_ms=max(vals) * 1000,
            pct_of_total=(medians[k] / total * 100) if total else 0.0))
    result.total_ms = total * 1000
    return result


def summarize(results: list[BenchResult], *, n_iter: int | None = None) -> str:
    """A one-table FP32-vs-FP16 comparison across both pipelines.

    ``n_iter`` (if given) is reflected in the methodology footnote so the
    printed iteration count can't drift from how the runs were actually made.
    """
    gpu = next((r.gpu for r in results if r.gpu), "unknown")
    by = {(r.pipeline, r.precision): r for r in results}
    lines = [f"GPU: {gpu}", "",
             f"{'Pipeline':<12s} {'FP32':>12s} {'FP16':>12s} {'FP16/FP32':>10s}",
             "-" * 48]
    for pipe in ("color", "motion"):
        fp32 = by.get((pipe, "fp32"))
        fp16 = by.get((pipe, "fp16"))
        t32 = fp32.total_ms if fp32 and fp32.measured else 0
        t16 = fp16.total_ms if fp16 and fp16.measured else 0
        s32 = f"{t32:.0f} ms" if t32 else (fp32.notes if fp32 else "-")
        s16 = f"{t16:.0f} ms" if t16 else (fp16.notes if fp16 else "-")
        ratio = f"{t16/t32:.2f}x" if t32 and t16 else "-"
        lines.append(f"{pipe:<12s} {s32:>12s} {s16:>12s} {ratio:>10s}")
    lines.append("")
    iter_part = f"median of {n_iter} timed runs" if n_iter else "median of timed runs"
    lines.append(f"Methodology: 1 warmup + {iter_part}; "
                 "cudaDeviceSynchronize after every stage; video I/O excluded.")
    return "\n".join(lines)

#!/usr/bin/env python3
"""Direct regression smoke-test (no pytest) for TRUBA compute nodes.

Imports the refactored modules and runs the critical checks that pytest's
config discovery is choking on under the LUSTRE/palamut environment. Prints
PASS/FAIL per check and exits non-zero on any failure. Output is flushed
line-by-line so partial results survive a crash.
"""
from __future__ import annotations
import sys, os, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "cuda"))

def log(msg):
    print(msg, flush=True)

FAILS = []

def check(name, fn):
    try:
        fn()
        log(f"  PASS  {name}")
    except Exception:
        log(f"  FAIL  {name}")
        traceback.print_exc()
        FAILS.append(name)

# 1. _common imports + figure6 correctness
def t_common():
    from evm_cuda._common import figure6_alpha_schedule, read_frames
    sched = figure6_alpha_schedule(9, 10, 16, 544, 960)
    assert len(sched) == 9
    assert sched[0] == 0.0 and sched[-1] == 0.0  # boundary levels zeroed
check("1. _common.figure6_alpha_schedule", t_common)

# 2. batched imports + magnify_color exists (the _common import chain)
def t_batched_import():
    import evm_cuda
    from evm_cuda import batched
    assert hasattr(batched, "magnify_color_gdown_ideal")
    assert hasattr(batched, "magnify_motion_lpyr_iir_fp16")
check("2. batched import + magnify_* present", t_batched_import)

# 3. pipelines imports + the 2 unique variants present
def t_pipelines_import():
    from evm_cuda import pipelines
    assert hasattr(pipelines, "magnify_motion_lpyr_ideal")
    assert hasattr(pipelines, "magnify_motion_lpyr_butter")
check("3. pipelines import + unique variants present", t_pipelines_import)

# 4. benchmark imports + run signature
def t_benchmark_import():
    from evm_cuda import benchmark
    assert hasattr(benchmark, "run")
    assert hasattr(benchmark, "summarize")
    assert hasattr(benchmark, "gpu_name")
check("4. benchmark import", t_benchmark_import)

# 5. __init__ routing: optimized fns come from batched
def t_routing():
    import evm_cuda
    from evm_cuda import batched
    color = evm_cuda.magnify_color_gdown_ideal
    assert color is batched.magnify_color_gdown_ideal, "color not routed to batched"
    motion = evm_cuda.magnify_motion_lpyr_iir
    assert motion is batched.magnify_motion_lpyr_iir, "motion-iir not routed to batched"
check("5. __init__ routes optimized fns to batched", t_routing)

# 6. End-to-end correctness: color pipeline vs CPU oracle (RMSE < 0.01)
DATA = ROOT / "data"
def t_e2e_color():
    if not (DATA / "face.mp4").exists():
        log("    (skipped: data/face.mp4 absent)")
        return
    import numpy as np
    import evm
    from evm_cuda import batched
    py = evm.magnify_color_gdown_ideal(
        str(DATA / "face.mp4"), "",
        alpha=50, level=4, fl=50/60, fh=60/60, chrom_attenuation=1.0, sampling_rate=30.0)
    cu = batched.magnify_color_gdown_ideal(
        str(DATA / "face.mp4"), "",
        alpha=50, level=4, fl=50/60, fh=60/60, chrom_attenuation=1.0, sampling_rate=30.0)
    rmse = float(np.sqrt(((py - cu) ** 2).mean()))
    log(f"    color RMSE(cuda, cpu) = {rmse:.5f}")
    assert rmse < 0.01, f"RMSE {rmse} exceeds 0.01 tolerance"
check("6. color pipeline RMSE < 0.01 vs CPU oracle", t_e2e_color)

# 7. benchmark.run produces a measured result (exercises on_stage hook)
def t_benchmark_run():
    if not (DATA / "face.mp4").exists():
        log("    (skipped: data/face.mp4 absent)")
        return
    from evm_cuda import benchmark
    r = benchmark.run("color", "fp32",
                      dict(vid=str(DATA / "face.mp4"), alpha=50, level=4,
                           fl=50/60, fh=60/60, chrom_attenuation=1.0,
                           sampling_rate=30.0), n_iter=2)
    assert r.measured, f"not measured: {r.notes}"
    assert r.total_ms > 0
    assert any(s.name.startswith("1)") for s in r.stages)
    log(f"    color FP32 total = {r.total_ms:.1f} ms ({len(r.stages)} stages)")
check("7. benchmark.run measures color FP32", t_benchmark_run)

log("")
log("=" * 50)
if FAILS:
    log(f"RESULT: {len(FAILS)} FAILED -> {FAILS}")
    sys.exit(1)
log("RESULT: ALL CHECKS PASSED")

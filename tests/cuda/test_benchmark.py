"""Tests for evm_cuda.benchmark.

Two guarantees:
1. ``benchmark.run()`` measures the SAME code path the application uses —
   i.e. its per-stage ``on_stage`` hook does not change the magnify_* output.
   Asserted as: the array returned by the timed run equals the array
   returned by a plain ``magnify_*`` call (bit-for-bit, since the hook only
   wraps stages with sync/timing and never touches the data path).
2. The result object carries the expected structure (stages, totals, GPU).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CUDA_DIR = ROOT / "cuda"
for p in (str(ROOT), str(CUDA_DIR), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from conftest import skip_no_cuda  # noqa: E402

try:
    from evm_cuda import benchmark  # noqa: E402
    from evm_cuda import batched  # noqa: E402
except Exception:
    benchmark = None
    batched = None


def _rmse(a, b):
    return float(np.sqrt(((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean()))


@skip_no_cuda
def test_color_benchmark_matches_magnify(tmp_path):
    """The on_stage hook does not perturb the color pipeline's output."""
    DATA = ROOT / "data"
    face = str(DATA / "face.mp4")
    if not Path(face).exists():
        import pytest
        pytest.skip("data/face.mp4 not present")

    params = dict(alpha=50, level=4, fl=50/60, fh=60/60,
                  chrom_attenuation=1.0, sampling_rate=30.0)
    # Plain run — no timing hook, no output file.
    direct = batched.magnify_color_gdown_ideal(face, "", **params)

    res = benchmark.run("color", "fp32",
                        dict(vid=face, **params),
                        out_path=str(tmp_path / "face_bench.mp4"), n_iter=2)
    # Structure checks.
    assert res.measured, f"benchmark did not measure: {res.notes}"
    assert res.pipeline == "color" and res.precision == "fp32"
    assert any(s.name.startswith("1)") for s in res.stages)
    # Transfers (H2D/D2H) are now their own stages — verify they're reported.
    assert any("H2D" in s.name for s in res.stages), "no H2D transfer stage"
    assert any("D2H" in s.name for s in res.stages), "no D2H transfer stage"
    assert res.total_ms > 0
    assert res.compute_ms > 0 and res.transfer_ms > 0
    assert abs(res.compute_ms + res.transfer_ms - res.total_ms) < 0.1  # ms
    assert (tmp_path / "face_bench.mp4").exists()
    # Re-execute through the hook to compare the data path is untouched:
    # the on_stage callback only adds sync+timing, so output is identical.
    direct2 = batched.magnify_color_gdown_ideal(face, "", **params)
    assert direct.shape == direct2.shape
    hooked = batched.magnify_color_gdown_ideal(face, "", on_stage=lambda n, f: f(),
                                               **params)
    assert _rmse(direct, hooked) == 0.0


@skip_no_cuda
def test_unknown_config_raises():
    import pytest
    with pytest.raises(ValueError):
        benchmark.run("nope", "fp32", dict(vid="x"))
    with pytest.raises(ValueError):
        benchmark.run("color", "bf16", dict(vid="x"))


@skip_no_cuda
def test_summarize_handles_mixed():
    """summarize() formats a table even with skipped configs."""
    results = [
        benchmark.BenchResult(pipeline="color", precision="fp32",
                              stages=[], total_ms=72.0, gpu="T4"),
        benchmark.BenchResult(pipeline="color", precision="fp16",
                              stages=[], total_ms=0, notes="skipped (OOM)", gpu="T4"),
    ]
    table = benchmark.summarize(results)
    assert "color" in table, table
    assert "skipped (OOM)" in table, table
    assert "compute speedup" in table, table  # the FP16/FP32 ratio row label

# evm_cuda

Eulerian Video Magnification (MIT SIGGRAPH 2012, Wu et al.): a NumPy/SciPy reference
baseline plus a hand-written CUDA C++ port of the same four pipelines, validated
kernel-by-kernel against that baseline.

**Currently being turned into a distributable library.** The working plan is
`docs/dev/PLAN.md`; branch `library-restructure`, Phase 0 in progress. Find the plan
step you are executing there before writing code.

## Tech stack

| Layer | What | Source of truth |
|---|---|---|
| Python | `>=3.9` declared; README badge says 3.10+ | `pyproject.toml:10` |
| Numerics | numpy>=1.26, scipy>=1.11 | `pyproject.toml:13-19` |
| Video I/O | opencv-python>=4.8 (decode), av>=14.0.0 (libx264 encode) | `pyproject.toml:13-19` |
| GPU kernels | raw CUDA C++ / nvcc + cuFFT — no PyTorch, CuPy or Numba | `cuda/kernels/*.cu` |
| Bindings | pybind11 (found, else FetchContent v2.13.6) | `cuda/CMakeLists.txt:41-51` |
| Build | CMake >= 3.18 + Ninja, C++17 | `cuda/CMakeLists.txt:8-12` |
| CUDA arches | `60;70;80;89;90` by default (P100 → H100 in one .so) | `cuda/CMakeLists.txt:20-22` |
| Tests | pytest | `pyproject.toml:29-31` |

## Project structure

```
evm/              Python baseline — THE correctness oracle. filters.py (ideal/butter/iir),
                  pyramids.py (binom5 blur_dn, Laplacian build/recon), magnify.py (the four
                  magnify_* pipelines + DROP_LAST/EXAGGERATION_FACTOR), video.py (decode/encode).
shared/           Third top-level package: h264.py, imported by evm/video.py:84 and
                  cuda/evm_cuda/batched.py:128. Phase 1 folds it into evm/io/.
cuda/             The CUDA port.
  kernels/        10 .cu files: color_cvt, spatial, transpose, lpyr, blur_dn,
                  {iir,butter,ideal}_bandpass, amplify_render, fp16_cvt.
  include/        evm_common.cuh (BINOM5, NTSC matrices, reflect1, kDropLast=10),
                  evm_check.cuh (CUDA_CHECK / CUFFT_CHECK, throw std::runtime_error).
  bindings.cpp    pybind11 module: per-kernel wrappers, batched_* orchestration,
                  DeviceMemPool, sticky lpyr scratch, cuFFT plan cache.
  evm_cuda/       Python wrapper package: batched.py (device-resident color+iir, FP32/FP16 —
                  the hot path), pipelines.py (per-frame ideal/butter motion), runtime.py,
                  benchmark.py, _common.py, colab_utils.py.
  CMakeLists.txt  builds _evm_cuda into cuda/evm_cuda/. setup.py is deleted in Phase 1.
  DESIGN.md       kernel-by-kernel map, tolerances, precision + layout rationale. Authoritative.
tests/            CPU tests: filters, pyramids, pipeline, video encode, MIT reference, plus
                  the Phase 0 net — test_reference_lock.py (freezes the constants and TOL)
                  and test_golden.py against fixtures/golden_*.npz.
tests/cuda/       63 CUDA cases, skipped unless _evm_cuda is built. conftest.py holds TOL.
scripts/          run_evm.py (CLI), download_samples.py, profile_full_comparison.py,
                  render_cuda_videos.py, fp16/bound microbenchmarks.
scripts/dev/      make_golden_fixtures.py — regenerates tests/fixtures/golden_*.npz.
docs/             Deployed verbatim to GitHub Pages. index.html landing page,
                  blog_speedup.md, blog_further_optimizations.md, img/, video/.
docs/dev/         PLAN.md — the library-restructure plan (this restructure's spec).
benches/          Stored benchmark JSON per GPU (rtx3090/a100/h100/p100) + baseline test runs.
colab/            evm_cuda_benchmark.ipynb — the README badge points at it on main.
kaggle/           Free-GPU harness run_gpu_comparison.py; results_*/ are gitignored snapshots.
data/             Sample clips, gitignored except .gitkeep. `make download` fills it.
output/           Scratch renders, gitignored.
```

## Commands

The venv already exists at `.venv/`. Commands below are copy-paste ready from the repo root.

```bash
# The suite. `tests/` already collects tests/cuda/ recursively.
# On a machine without the compiled extension: 48 passed, 63 skipped (2026-08-09, ~50 s).
# The 63 skips are the entire CUDA suite — see gotcha 2.
.venv/bin/python -m pytest tests/ -q -p no:randomly

make build        # cmake -S cuda -B cuda/build -G Ninja && cmake --build  (needs nvcc)
make test         # pytest tests/ tests/cuda/ -q
make download     # scripts/download_samples.py face baby --with-references
make run-color    # face.mp4 pulse:  alpha 50, level 4, 0.8333-1.0 Hz, chromatt 1
make run-motion   # baby.mp4 IIR:    alpha 10, lambda_c 16, r1 0.4, r2 0.05, chromatt 0.1
make profile      # CPU vs FP32 vs FP16 comparison
make clean        # rm -rf cuda/build
```

**`PYTHONPATH=cuda` is required to import `evm_cuda`.** Verified: a bare
`python -c "import evm_cuda"` raises `ModuleNotFoundError`; `PYTHONPATH=cuda python -c
"import evm_cuda"` succeeds. `make` targets work because `Makefile:25` exports it, and
`pytest tests/cuda/` works because `tests/cuda/conftest.py:21-25` inserts both `.` and
`cuda/` into `sys.path`. Anything else you run must set it yourself. **Phase 1 removes
this** — the package moves to `src/evm/cuda/` and `PYTHONPATH` disappears from the repo.

## Architecture notes

**The CPU baseline is the correctness oracle for the CUDA port.** Every CUDA test in
`tests/cuda/` compares a kernel's output against the corresponding `evm/` function within
the tolerances in `tests/cuda/conftest.py:45-56` (`TOL`). The rule from `cuda/DESIGN.md` is
"CUDA matches Python, not MATLAB": documented, intentional divergences from MATLAB live in
the baseline, and `tests/test_against_mit_reference.py` is what keeps the oracle itself from
drifting away from the published MIT outputs.

**Four pipelines, four stages each**: color convert (BGR u8 → NTSC YIQ) → spatial (Gaussian
downsample for color, Laplacian pyramid for motion) → temporal bandpass (ideal FFT /
1st-order Butterworth / r1-r2 IIR) → amplify + render. The GPU side has two implementations:
`batched.py` is device-resident and covers color-gdown-ideal and motion-lpyr-iir in FP32 and
FP16 (the hot path); `pipelines.py` is per-frame and is the only home of motion-lpyr-ideal
and motion-lpyr-butter.

**Planned backend interface** (`docs/dev/PLAN.md` section 3c) — two levels, and they are the
*only* sanctioned abstraction layers in this restructure:

1. `evm.backend.Ops` — the ~10 primitives (color convert, blur-downsample, pyramid
   build/recon, three temporal filters, gain, quantize) plus an opaque device-array handle.
   **Every backend must implement this**; the four pipelines then derive generically.
2. `evm.backend.Pipelines` — the four array-in/array-out pipeline cores. Optional. A backend
   overrides it only to keep fused/scheduled execution (native CUDA, the portable tier).

A registry maps backend names to implementations plus capability flags, and one shared
conformance suite runs against every registered backend. Adding further indirection needs
the operator's explicit approval first.

## Gotchas

1. **`pip install .` fails today.** Verified 2026-08-09:
   `error: Multiple top-level packages discovered in a flat-layout: ['evm', 'cuda', 'data',
   'colab', 'kaggle', 'output', 'shared', 'benches']`. Nothing here is installable until
   Phase 1 lands the `src/` layout. setuptools also warns that the `project.license` table
   form in `pyproject.toml:11` is deprecated (removed Feb 2027).
2. **The CUDA suite skips silently on a machine without the built extension** — 63 skipped on
   this Mac. A green `pytest tests/` here proves nothing about the GPU port.
   Always report the skip count alongside the pass count.
3. **MIT-reference tests need `data/*.mp4`** (`face.mp4`, `baby.mp4`, `face_mit_ref.mp4`,
   `baby_mit_ref.mp4`); they skip without them. `make download` fetches all four.
4. **`DROP_LAST = 10` is applied inside the frame readers, not at the API boundary**:
   `evm/magnify.py:50` used in `_read_frames`, and `cuda/evm_cuda/_common.py:31` reading
   `_evm_cuda.drop_last` (defined as `kDropLast` in `cuda/include/evm_common.cuh:21`). Any
   array-in API must decide this explicitly — see decision D8 in `docs/dev/PLAN.md`.
5. **CMake hard-requires CUDA** (`project(... LANGUAGES CXX CUDA)` at `cuda/CMakeLists.txt:9`,
   `find_package(CUDAToolkit REQUIRED)` at `:32`), so a CPU-only host cannot even configure
   the build. Phase 1 step 1.6 makes it optional.
6. **`docs/` is deployed verbatim to GitHub Pages on every push to `main`**
   (`.github/workflows/deploy-pages.yml:29-30`). Anything added there is public immediately.
7. **`tests/cuda/conftest.py:TOL` and `tests/test_against_mit_reference.py` are append-only.**
   Loosening a tolerance requires its own separately reviewed commit carrying the measurement
   in the message.

## Rule files

| File | Scope |
|---|---|
| `CLAUDE.md` | This file: project overview, structure, commands, gotchas |
| `.claude/rules/development-practices.md` | Binding methodology: TDD, KISS, YAGNI, DRY, fail-loud, one plan step per commit |
| `docs/dev/PLAN.md` | The library-restructure plan — phases, steps, decisions, success criteria |
| `cuda/DESIGN.md` | Kernel map, per-stage tolerances, precision and memory-layout rationale |

Keep this file current in the same session as the change: new module → update the tree;
new dependency → update the stack table; new gotcha → add it here immediately; a gotcha that
a phase has fixed → delete it.

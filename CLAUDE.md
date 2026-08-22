# vidmag

Eulerian Video Magnification (MIT SIGGRAPH 2012, Wu et al.), plus the phase-based
follow-up (SIGGRAPH 2013): a NumPy/SciPy reference baseline and six backends that
compute the same four pipelines — NVIDIA (hand-written CUDA C++), OpenCL, Apple's
graphics interface, Vulkan, PyTorch, and the baseline itself. Every backend is
validated against the baseline, which is the correctness oracle for all of them.

**One word names all three things: `pip install vidmag`, `import vidmag`,
and `vidmag` at the terminal.** It was called `evm-cuda` until 2026-08-11 and
`evm-magnify` until 2026-08-18; neither had been published, so both renames cost
nothing. `vidmag` is free on PyPI under every spelling PyPI treats as
equivalent, checked 2026-08-18.

The import root could not stay `evm`, and this was tested rather than assumed.
The PyPI distribution named `evm` is the Extreme Value Machine, and it installs
a top-level module spelled `EVM`. On macOS and Windows, whose filesystems ignore
case, `EVM/` and `evm/` are one directory, so whichever package installs second
overwrites the other's `__init__.py`. Measured both orders on this Mac: install
theirs first and ours is not importable at all; install ours first and
`import evm` silently returns their package. Do not reintroduce `evm` as a
module name.

**Two facts about `vidmag` that were known when it was chosen, so nobody
re-discovers them and thinks they are a mistake.** `vidmag` is also the original
authors' own shorthand — MIT's project page is `people.csail.mit.edu/mrub/vidmag/`,
which this repository links to from `README.md` and
`docs/getting-started/first-result.md`. And `rgov/vidmag` on GitHub is a 25-star
MATLAB project in this same field. Both were raised and the name was chosen
anyway; see decision D1 in `docs/dev/PLAN.md`.

Only the NVIDIA backend needs a compiled extension; `pip install .` succeeds
with no NVIDIA compiler present.

**Three similar-looking names are NOT the package and must not be renamed with
it.** The C++ namespace `evm::` and the headers `evm_check.cuh`,
`evm_common.cuh`, `evm_dlpack.cuh` name the *method* — Eulerian Video
Magnification — which is not changing; likewise "EVM" in prose. The conda
environment on the GPU machine is called `evm-cuda`. The MIT dataset URL
`people.csail.mit.edu/mrub/evm/` and the Kaggle kernel slugs
`furkankucuk/evm-cuda-*` name resources owned by other people. The compiled
extension, by contrast, *is* part of the package and is called
`_vidmag_cuda`, because it is imported as `vidmag.cuda._vidmag_cuda`.

**The library restructure is essentially complete** on branch `library-restructure`.
The plan is `docs/dev/PLAN.md`. If you are executing a plan step, find it before
writing code; if you are doing performance work, read Rule 7 in
`.claude/rules/development-practices.md` first — it governs which backend leads.

## Tech stack

| Layer | What | Source of truth |
|---|---|---|
| Python | `>=3.9` declared; README badge says 3.10+ | `pyproject.toml:15` |
| Numerics | numpy>=1.26, scipy>=1.11 | `pyproject.toml:54-60` |
| Video I/O | opencv-python>=4.8 (decode), av>=14.0.0 (libx264 encode) | `pyproject.toml:54-60` |
| NVIDIA kernels | raw CUDA C++ / nvcc + cuFFT — no PyTorch, CuPy or Numba | `src/vidmag/cuda/kernels/*.cu` |
| Other backends | OpenCL (pyopencl), Apple (PyObjC), Vulkan (vulkan + MoltenVK on macOS), PyTorch (torch); each an optional extra | `pyproject.toml` extras |
| Bindings | pybind11 (found, else FetchContent v2.13.6) | `src/vidmag/cuda/CMakeLists.txt:126-136` |
| Build backend | scikit-build-core, `cmake.source-dir = "src/vidmag/cuda"` | `pyproject.toml` |
| Build | CMake >= 3.24 + Ninja, C++17; **CUDA optional** | `src/vidmag/cuda/CMakeLists.txt:15, 41-62` |
| CUDA arches | `native` by default, `VIDMAG_CUDA_ARCHS=all` → `60;70;80;89;90` | `src/vidmag/cuda/CMakeLists.txt:80-98` |
| Tests | pytest + pytest-cov (reported, never gated — no machine runs every backend) | `pyproject.toml` |

## Project structure

```
src/vidmag/     The installed package. `import vidmag` re-exports the processor
                  baseline at the root; `vidmag.cuda` resolves lazily (PEP 562), so
                  importing `vidmag` on a machine with no NVIDIA card never touches the
                  compiled extension.
  api.py          `magnify()` — the one entry point. Takes a path, an array, or any
                  iterable of frames; picks a backend and says which.
  presets.py      The named parameter sets (pulse, motion, motion_phase, vibration).
  _cli.py         The `vidmag` terminal command.
  stream.py       MotionStream — live magnification over a running capture.
  notebook.py     show_video — the Colab/Jupyter display helper.
  backend/        The backend interface and registry. ops.py (the ~12 primitives every
                  backend implements), pipelines.py (the optional fused override),
                  generic.py (the four pipelines derived from the primitives alone),
                  registry.py (name -> implementation + capability flags).
  cpu/            Python baseline — THE correctness oracle for every other backend.
                  filters.py (ideal/butter/iir), pyramids.py (binom5 blur_dn, Laplacian
                  build/recon), magnify.py (the four magnify_* pipelines +
                  DROP_LAST/EXAGGERATION_FACTOR), csp.py + phase_magnify.py (the 2013
                  phase-based method), ops.py + backend.py (its Ops implementation).
  cuda/           NVIDIA wrapper package: batched.py (device-resident colour+iir,
                  FP32/FP16 — the hot path), pipelines.py (per-frame ideal/butter
                  motion), runtime.py (`have_cuda`, `require_cuda`), benchmark.py,
                  array.py + ops.py (DeviceArray, DLPack, its Ops implementation),
                  _common.py. CMake writes `_vidmag_cuda*.so` into this directory.
                  The CUDA sources live here too, like every other backend's:
    kernels/      10 .cu files: color_cvt, spatial, transpose, lpyr, blur_dn,
                  {iir,butter,ideal}_bandpass, amplify_render, fp16_cvt.
    include/      evm_common.cuh (BINOM5, NTSC matrices, reflect1, kDropLast=10),
                  evm_check.cuh (CUDA_CHECK / CUFFT_CHECK, throw std::runtime_error),
                  evm_dlpack.cuh (zero-copy export to PyTorch/CuPy).
    bindings.cpp  pybind11 module: per-kernel wrappers, batched_* orchestration,
                  DeviceMemPool, sticky lpyr scratch, cuFFT plan cache.
    CMakeLists.txt  CUDA-optional; driven by scikit-build-core from pyproject.toml.
    DESIGN.md     kernel-by-kernel map, tolerances, precision + layout rationale.
                  Authoritative for the NVIDIA backend.
  opencl/         OpenCL backend: kernels.cl, runtime.py, array.py, ops.py.
  torch_backend/  PyTorch backend: runtime.py + ops.py, no kernels of its own.
                  Optional extra; nothing imports torch unless it is asked for.
  metal/          Apple graphics backend: kernels.metal + the same three modules.
  vulkan/         Vulkan backend: shaders/*.comp with committed *.spv beside them
                  (shaders/build.py regenerates, `--check` verifies), + the same three.
  io/             video.py (decode/encode + RGB<->YIQ), h264.py (the single libx264
                  encoder every backend calls, so they write byte-identical
                  containers), capture.py (camera/file capture for streaming).
tests/            402 tests collected in total. Processor tests (filters, pyramids,
                  pipeline, video encode, MIT reference), the public-API surface lock,
                  the cross-backend conformance suite, plus the Phase 0 net —
                  test_reference_lock.py (freezes the constants and TOL) and
                  test_golden.py against fixtures/golden_*.npz.
tests/cuda/       98 of those 402, skipped unless _vidmag_cuda is built. conftest.py holds TOL.
scripts/          The tools: run_evm.py, download_samples.py,
                  profile_full_comparison.py, render_cuda_videos.py.
scripts/dev/      make_golden_fixtures.py (regenerates tests/fixtures/golden_*.npz),
                  verify_install.sh (the packaging check — see Commands), and the two
                  benchmark recorders every published timing cites:
                  record_gpu_bench.py -> benches/bench_<card>.json (needs an NVIDIA
                  card) and record_backend_bench.py -> benches/backends_<machine>.json
                  (every backend present). Before these existed the numbers were
                  copied by hand and drifted; regenerate rather than edit.
scripts/cloud/    Benchmark harnesses for someone else's GPU: colab_benchmark.ipynb
                  (the README badge points at it on main) and kaggle/.
scripts/experiments/  One-off measurements whose answers are recorded in
                  docs/internals/. Kept so those claims can be re-checked; nothing
                  runs them. See the README there.
docs/             A mkdocs-material site, built with `mkdocs build --strict` and
                  published twice from one configuration: GitHub Pages on every
                  push to the default branch (`.github/workflows/deploy-pages.yml`),
                  and Read the Docs on every pull request and tag
                  (`.readthedocs.yaml`), which is what gives per-version pages.
                  index.md, getting-started/, concepts/ (incl. backends.md),
                  recipes/, comparison.md, performance.md, stability.md,
                  img/, video/.
docs/internals/   design.md and the two written accounts of the optimisation work,
                  blog_speedup.md and blog_further_optimizations.md. Both carry dated
                  notes saying which later changes superseded them; bodies are a
                  historical record and are not rewritten.
docs/dev/         PLAN.md — the library-restructure plan. packaging-notes.md — dated
                  findings from executing it; a historical record, not current
                  instructions. Also gpu-runner.md and release-checklist.md.
benches/          Stored benchmark JSON per GPU (rtx3090/a100/h100/p100) + baseline test
                  runs + kaggle_runs/ console logs.
data/             Sample clips, gitignored except .gitkeep. `make download` fills it.
output/           Scratch renders, gitignored.
```

## Commands

The venv already exists at `.venv/`. Tests import the *installed* package (there is no
`pythonpath` entry in `pyproject.toml`), so `make install-dev` has to have run once.

```bash
make install-dev  # pip install -e ".[dev,cuda-build]" — the one-time bootstrap
make build        # pip install -e . --no-build-isolation; recompiles _vidmag_cuda if nvcc is there
make test         # pytest tests/ tests/cuda/ -q
make download     # scripts/download_samples.py face baby --with-references
make run-color    # face.mp4 pulse:  alpha 50, level 4, 0.8333-1.0 Hz, chromatt 1
make run-motion   # baby.mp4 IIR:    alpha 10, lambda_c 16, r1 0.4, r2 0.05, chromatt 0.1
make profile      # CPU vs FP32 vs FP16 comparison
make clean        # rm -f the in-tree src/vidmag/cuda/_vidmag_cuda*.so (no cuda/build any more:
                  #   scikit-build-core configures CMake in a temp dir)

# The suite. `tests/` already collects tests/cuda/ recursively.
# On a machine without the NVIDIA extension: 300 passed, 102 skipped (2026-08-18, ~75 s).
# Most of those skips are the whole NVIDIA suite — see gotcha 1.
.venv/bin/python -m pytest tests/ -q -p no:randomly

# The packaging check: throwaway venv outside the repo, plain `pip install .`, imports run
# from a neutral cwd, then the suite. This is what judges any change to pyproject.toml,
# src/vidmag/cuda/CMakeLists.txt or the Makefile.
bash scripts/dev/verify_install.sh
```

**`pip install .` works on every machine, and `PYTHONPATH` is gone from the repo.**
`src/vidmag/cuda/CMakeLists.txt` runs `check_language(CUDA)` before anything else: with no nvcc it
prints a "NO CUDA COMPILER FOUND" banner, defines no target, and the install still
succeeds as a CPU-only package. Verified on this Mac —

```
vidmag.cuda.have_cuda = False
require_cuda -> RuntimeError vidmag.cuda._vidmag_cuda not importable; the extension was not built …
```

With nvcc present the same command compiles `_vidmag_cuda` for the local GPU into
`src/vidmag/cuda/` (`VIDMAG_CUDA_ARCHS=all` builds the portable `60;70;80;89;90` set instead).
`VIDMAG_CUDA_REQUIRE=1` turns a missing nvcc into a hard build failure.

## Architecture notes

**The processor baseline is the correctness oracle for every other backend.** Each NVIDIA
test in `tests/cuda/` compares a kernel's output against the corresponding `vidmag.cpu`
function within the tolerances in `tests/cuda/conftest.py` (`TOL`); the OpenCL, Apple and
Vulkan backends are held to the same baseline through the shared conformance suite. The
rule from `src/vidmag/cuda/DESIGN.md` is
"CUDA matches Python, not MATLAB": documented, intentional divergences from MATLAB live in
the baseline, and `tests/test_against_mit_reference.py` is what keeps the oracle itself from
drifting away from the published MIT outputs.

**Four pipelines, four stages each**: color convert (BGR u8 → NTSC YIQ) → spatial (Gaussian
downsample for color, Laplacian pyramid for motion) → temporal bandpass (ideal FFT /
1st-order Butterworth / r1-r2 IIR) → amplify + render. The GPU side has two implementations:
`batched.py` is device-resident and covers color-gdown-ideal and motion-lpyr-iir in FP32 and
FP16 (the hot path); `pipelines.py` is per-frame and is the only home of motion-lpyr-ideal
and motion-lpyr-butter.

**Every backend's kernels live in that backend's directory**, NVIDIA included:
`src/vidmag/cuda/kernels/*.cu` beside `src/vidmag/opencl/kernels.cl`,
`src/vidmag/metal/kernels.metal` and `src/vidmag/vulkan/shaders/*.comp`. One place to look,
whichever backend you are working on. The NVIDIA sources used to sit in a top-level
`cuda/` directory; they moved on 2026-08-11.

They are still different in *kind* from the others, and the packaging says so rather
than the directory layout: nvcc compiles the `.cu` files at install time and nothing
reads them afterwards, so `wheel.exclude` in `pyproject.toml` keeps them out of the
wheel while `sdist.include` keeps them in the source distribution, which is what a
source build needs. The other backends' kernels are compiled by their drivers at run
time, so those 30 files do ship. Re-verified after the 2026-08-18 rename, by building
both distributions and listing them: the wheel has 83 entries, all under `vidmag/`,
with zero `.cu`, `.cuh`, `.cpp` or `CMakeLists.txt` among them and all 30 runtime-compiled
kernels present; the sdist carries all 15 of those build files (10 `.cu`, 3 `.cuh`,
`bindings.cpp`, `CMakeLists.txt`).

**The NVIDIA GPU code is the primary optimisation target; the other backends follow it.**
Performance work starts in `src/vidmag/cuda/kernels/*.cu`. Once a change there is accepted, check whether
it applies to each other backend and carry it over if it does — OpenCL
`src/vidmag/opencl/kernels.cl`, Metal `src/vidmag/metal/kernels.metal`, Vulkan
`src/vidmag/vulkan/shaders/*.comp`, Python `src/vidmag/cpu/`. Say which backends it reached and
which it does not apply to. Rule 7 of `.claude/rules/development-practices.md` is binding here.

**Planned backend interface** (`docs/dev/PLAN.md` section 3c) — two levels, and they are the
*only* sanctioned abstraction layers in this restructure:

1. `vidmag.backend.Ops` — the ~10 primitives (color convert, blur-downsample, pyramid
   build/recon, three temporal filters, gain, quantize) plus an opaque device-array handle.
   **Every backend must implement this**; the four pipelines then derive generically.
2. `vidmag.backend.Pipelines` — the four array-in/array-out pipeline cores. Optional. A backend
   overrides it only to keep fused/scheduled execution (native CUDA, the portable tier).

A registry maps backend names to implementations plus capability flags, and one shared
conformance suite runs against every registered backend. Adding further indirection needs
the operator's explicit approval first.

## Gotchas

1. **The NVIDIA suite skips silently on a machine without the built extension** — 98 of the
   402 tests, and a green `pytest tests/` here proves nothing about the NVIDIA backend.
   The reverse also holds: the NVIDIA host has no Apple, OpenCL or Vulkan driver, so those
   backends read as untested there. No single machine covers this library. Both halves,
   same commit, 2026-08-18: this Mac gives 300 passed / 102 skipped, the RTX 3090 host
   gives 345 passed / 57 skipped, and neither run covers what the other skips.
   Always report the skip count alongside the pass count.
2. **A bare source checkout is not importable.** The packages live under `src/`, and
   `pyproject.toml` sets no `pythonpath`, so `python -c "import vidmag"` from the repo root
   fails until `make install-dev` has run. That is deliberate: it makes the test suite a
   real check on the packaging.
3. **MIT-reference tests need `data/*.mp4`** (`face.mp4`, `baby.mp4`, `face_mit_ref.mp4`,
   `baby_mit_ref.mp4`); they skip without them. `make download` fetches all four.
4. **`vidmag.cuda` resolves lazily** (PEP 562). Importing `vidmag` on a machine with no
   NVIDIA card never touches the compiled extension; asking for `vidmag.cuda` there raises
   an `AttributeError` carrying the reason, because `vidmag.cuda.__getattr__` turns the
   extension's `ImportError` into one (`src/vidmag/cuda/__init__.py:61-66`).
5. **`DROP_LAST = 10` is applied inside the frame readers, not at the API boundary**:
   `src/vidmag/cpu/magnify.py:50` used in `_read_frames` (`:121`), and
   `src/vidmag/cuda/_common.py:31` reading `_vidmag_cuda.drop_last` (defined as `kDropLast` in
   `src/vidmag/cuda/include/evm_common.cuh:21`). Any array-in API must decide this explicitly — see
   decision D8 in `docs/dev/PLAN.md`.
6. **`docs/` is built and published to GitHub Pages on every push to `main`** —
   `mkdocs build --strict`, publishing the built `site/` directory
   (`.github/workflows/deploy-pages.yml`). `--strict` means a broken internal link fails
   the build, and anything placed under `docs/` becomes public on merge.
7. **`tests/cuda/conftest.py:TOL` and `tests/test_against_mit_reference.py` are append-only.**
   Loosening a tolerance requires its own separately reviewed commit carrying the measurement
   in the message.

## Rule files

| File | Scope |
|------|-------|
| `CLAUDE.md` | This file: project overview, structure, commands, gotchas |
| `.claude/rules/development-practices.md` | Binding methodology: TDD, KISS, YAGNI, DRY, fail-loud, one plan step per commit |
| `docs/dev/PLAN.md` | The library-restructure plan — phases, steps, decisions, success criteria |
| `src/vidmag/cuda/DESIGN.md` | Kernel map, per-stage tolerances, precision and memory-layout rationale |

Keep this file current in the same session as the change: new module → update the tree;
new dependency → update the stack table; new gotcha → add it here immediately; a gotcha that
a phase has fixed → delete it.

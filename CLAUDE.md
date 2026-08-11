# evm

Eulerian Video Magnification (MIT SIGGRAPH 2012, Wu et al.), plus the phase-based
follow-up (SIGGRAPH 2013): a NumPy/SciPy reference baseline and five backends that
compute the same four pipelines — NVIDIA (hand-written CUDA C++), OpenCL, Apple's
graphics interface, Vulkan, and the baseline itself. Every backend is validated
against the baseline, which is the correctness oracle for all of them.

The distribution installs as `evm-magnify` and imports as `evm`. It was called
`evm-cuda` until 2026-08-11; the rename cost nothing because it had never been
published. `evm` alone was unavailable — that name belongs to an unrelated project on
PyPI, the Extreme Value Machine — and `evm-magnify` matches the terminal command this
package installs. Only the NVIDIA backend needs a compiled extension; `pip install .`
succeeds with no NVIDIA compiler present.

Three similar-looking names are NOT the distribution and must not be renamed with it:
the compiled extension `_evm_cuda`, the deprecated shim package `src/evm_cuda/`, and
the conda environment called `evm-cuda` on the GPU machine.

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
| NVIDIA kernels | raw CUDA C++ / nvcc + cuFFT — no PyTorch, CuPy or Numba | `cuda/kernels/*.cu` |
| Other backends | OpenCL (pyopencl), Apple (PyObjC), Vulkan (vulkan + MoltenVK on macOS); each an optional extra | `pyproject.toml` extras |
| Bindings | pybind11 (found, else FetchContent v2.13.6) | `cuda/CMakeLists.txt:126-136` |
| Build backend | scikit-build-core, `cmake.source-dir = "cuda"` | `pyproject.toml:6-8, 83-95` |
| Build | CMake >= 3.24 + Ninja, C++17; **CUDA optional** | `cuda/CMakeLists.txt:15, 41-62` |
| CUDA arches | `native` by default, `EVM_CUDA_ARCHS=all` → `60;70;80;89;90` | `cuda/CMakeLists.txt:80-98` |
| Tests | pytest + pytest-cov (reported, never gated — no machine runs every backend) | `pyproject.toml` |

## Project structure

```
src/evm/          The installed package. `import evm` re-exports the processor
                  baseline at the root; `evm.cuda` resolves lazily (PEP 562), so
                  importing `evm` on a machine with no NVIDIA card never touches the
                  compiled extension.
  api.py          `magnify()` — the one entry point. Takes a path, an array, or any
                  iterable of frames; picks a backend and says which.
  presets.py      The named parameter sets (pulse, motion, motion_phase, vibration).
  _cli.py         The `evm-magnify` terminal command.
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
                  _common.py. CMake writes `_evm_cuda*.so` into this directory.
  opencl/         OpenCL backend: kernels.cl, runtime.py, array.py, ops.py.
  metal/          Apple graphics backend: kernels.metal + the same three modules.
  vulkan/         Vulkan backend: shaders/*.comp with committed *.spv beside them
                  (shaders/build.py regenerates, `--check` verifies), + the same three.
  io/             video.py (decode/encode + RGB<->YIQ), h264.py (the single libx264
                  encoder every backend calls, so they write byte-identical
                  containers), capture.py (camera/file capture for streaming).
src/evm_cuda/     Deprecated shim forwarding to `evm.cuda`; warns on import. See gotcha 4.
cuda/             NVIDIA CUDA sources. No Python package lives here any more.
  kernels/        10 .cu files: color_cvt, spatial, transpose, lpyr, blur_dn,
                  {iir,butter,ideal}_bandpass, amplify_render, fp16_cvt.
  include/        evm_common.cuh (BINOM5, NTSC matrices, reflect1, kDropLast=10),
                  evm_check.cuh (CUDA_CHECK / CUFFT_CHECK, throw std::runtime_error),
                  evm_dlpack.cuh (zero-copy export to PyTorch/CuPy).
  bindings.cpp    pybind11 module: per-kernel wrappers, batched_* orchestration,
                  DeviceMemPool, sticky lpyr scratch, cuFFT plan cache.
  CMakeLists.txt  CUDA-optional; driven by scikit-build-core from the root pyproject.toml.
  DESIGN.md       kernel-by-kernel map, tolerances, precision + layout rationale.
                  Authoritative for the NVIDIA backend.
tests/            380 tests collected in total. Processor tests (filters, pyramids,
                  pipeline, video encode, MIT reference), the public-API surface lock,
                  the cross-backend conformance suite, plus the Phase 0 net —
                  test_reference_lock.py (freezes the constants and TOL) and
                  test_golden.py against fixtures/golden_*.npz.
tests/cuda/       98 of those 380, skipped unless _evm_cuda is built. conftest.py holds TOL.
scripts/          run_evm.py, download_samples.py, profile_full_comparison.py,
                  render_cuda_videos.py, fp16/bound microbenchmarks.
scripts/dev/      make_golden_fixtures.py (regenerates tests/fixtures/golden_*.npz) and
                  verify_install.sh (the packaging check — see Commands).
docs/             A mkdocs-material site, built with `mkdocs build --strict` and
                  published from the built output. index.md, getting-started/,
                  concepts/ (incl. backends.md), recipes/, comparison.md,
                  performance.md, stability.md, img/, video/.
docs/internals/   design.md and the two written accounts of the optimisation work,
                  blog_speedup.md and blog_further_optimizations.md. Both carry dated
                  notes saying which later changes superseded them; bodies are a
                  historical record and are not rewritten.
docs/dev/         PLAN.md — the library-restructure plan. packaging-notes.md — dated
                  findings from executing it; a historical record, not current
                  instructions. Also gpu-runner.md and release-checklist.md.
benches/          Stored benchmark JSON per GPU (rtx3090/a100/h100/p100) + baseline test
                  runs + kaggle_runs/ console logs.
colab/            evm_cuda_benchmark.ipynb — the README badge points at it on main.
kaggle/           Free-GPU harness run_gpu_comparison.py; results_*/ are gitignored.
data/             Sample clips, gitignored except .gitkeep. `make download` fills it.
output/           Scratch renders, gitignored.
```

## Commands

The venv already exists at `.venv/`. Tests import the *installed* package (there is no
`pythonpath` entry in `pyproject.toml`), so `make install-dev` has to have run once.

```bash
make install-dev  # pip install -e ".[dev,cuda-build]" — the one-time bootstrap
make build        # pip install -e . --no-build-isolation; recompiles _evm_cuda if nvcc is there
make test         # pytest tests/ tests/cuda/ -q
make download     # scripts/download_samples.py face baby --with-references
make run-color    # face.mp4 pulse:  alpha 50, level 4, 0.8333-1.0 Hz, chromatt 1
make run-motion   # baby.mp4 IIR:    alpha 10, lambda_c 16, r1 0.4, r2 0.05, chromatt 0.1
make profile      # CPU vs FP32 vs FP16 comparison
make clean        # rm -f the in-tree src/evm/cuda/_evm_cuda*.so (no cuda/build any more:
                  #   scikit-build-core configures CMake in a temp dir)

# The suite. `tests/` already collects tests/cuda/ recursively.
# On a machine without the NVIDIA extension: 279 passed, 101 skipped (2026-08-11, ~43 s).
# Most of those skips are the whole NVIDIA suite — see gotcha 1.
.venv/bin/python -m pytest tests/ -q -p no:randomly

# The packaging check: throwaway venv outside the repo, plain `pip install .`, imports run
# from a neutral cwd, then the suite. This is what judges any change to pyproject.toml,
# cuda/CMakeLists.txt or the Makefile.
bash scripts/dev/verify_install.sh
```

**`pip install .` works on every machine, and `PYTHONPATH` is gone from the repo.**
`cuda/CMakeLists.txt` runs `check_language(CUDA)` before anything else: with no nvcc it
prints a "NO CUDA COMPILER FOUND" banner, defines no target, and the install still
succeeds as a CPU-only package. Verified on this Mac —

```
evm.cuda.have_cuda = False
require_cuda -> RuntimeError evm.cuda._evm_cuda not importable; the extension was not built …
```

With nvcc present the same command compiles `_evm_cuda` for the local GPU into
`src/evm/cuda/` (`EVM_CUDA_ARCHS=all` builds the portable `60;70;80;89;90` set instead).
`EVM_CUDA_REQUIRE=1` turns a missing nvcc into a hard build failure.

## Architecture notes

**The processor baseline is the correctness oracle for every other backend.** Each NVIDIA
test in `tests/cuda/` compares a kernel's output against the corresponding `evm.cpu`
function within the tolerances in `tests/cuda/conftest.py` (`TOL`); the OpenCL, Apple and
Vulkan backends are held to the same baseline through the shared conformance suite. The
rule from `cuda/DESIGN.md` is
"CUDA matches Python, not MATLAB": documented, intentional divergences from MATLAB live in
the baseline, and `tests/test_against_mit_reference.py` is what keeps the oracle itself from
drifting away from the published MIT outputs.

**Four pipelines, four stages each**: color convert (BGR u8 → NTSC YIQ) → spatial (Gaussian
downsample for color, Laplacian pyramid for motion) → temporal bandpass (ideal FFT /
1st-order Butterworth / r1-r2 IIR) → amplify + render. The GPU side has two implementations:
`batched.py` is device-resident and covers color-gdown-ideal and motion-lpyr-iir in FP32 and
FP16 (the hot path); `pipelines.py` is per-frame and is the only home of motion-lpyr-ideal
and motion-lpyr-butter.

**Why the NVIDIA sources sit in `cuda/` and every other backend's sit inside the
package.** It looks inconsistent and is not. The OpenCL, Metal and Vulkan kernels are
read at *run* time by a driver, so they have to ship in the wheel — 30 such files do.
The CUDA sources are compiled by nvcc at *install* time, so none of them ship: a built
wheel contains zero `.cu` files. Putting them under `src/evm/cuda/` would place
build-only inputs inside the installed package and need an exclusion rule to undo. The
split follows what the files are for. Moving them is not a tidying job.

**The NVIDIA GPU code is the primary optimisation target; the other backends follow it.**
Performance work starts in `cuda/kernels/*.cu`. Once a change there is accepted, check whether
it applies to each other backend and carry it over if it does — OpenCL
`src/evm/opencl/kernels.cl`, Metal `src/evm/metal/kernels.metal`, Vulkan
`src/evm/vulkan/shaders/*.comp`, Python `src/evm/cpu/`. Say which backends it reached and
which it does not apply to. Rule 7 of `.claude/rules/development-practices.md` is binding here.

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

1. **The NVIDIA suite skips silently on a machine without the built extension** — 98 of the
   380 tests, and a green `pytest tests/` here proves nothing about the NVIDIA backend.
   The reverse also holds: the NVIDIA host has no Apple, OpenCL or Vulkan driver, so those
   backends read as untested there. No single machine covers this library.
   Always report the skip count alongside the pass count.
2. **A bare source checkout is not importable.** The packages live under `src/`, and
   `pyproject.toml` sets no `pythonpath`, so `python -c "import evm"` from the repo root
   fails until `make install-dev` has run. That is deliberate: it makes the test suite a
   real check on the packaging.
3. **MIT-reference tests need `data/*.mp4`** (`face.mp4`, `baby.mp4`, `face_mit_ref.mp4`,
   `baby_mit_ref.mp4`); they skip without them. `make download` fetches all four.
4. **The `evm_cuda` shim forwards attributes, not submodule imports.** `import evm_cuda` and
   `evm_cuda.have_cuda` work; `import evm_cuda.benchmark` raises `ModuleNotFoundError`. It is
   deliberate — aliasing submodules into `sys.modules` would load them a second time under a
   second name and give `DeviceMemPool` two independent sets of state. Fix callers, not the
   shim: write `from evm.cuda import benchmark`. Submodule *attributes* (`evm_cuda.benchmark`,
   `evm_cuda.batched`) resolve only where `_evm_cuda` is built; on this Mac they raise
   `AttributeError`, because `evm.cuda.__getattr__` turns the extension's `ImportError` into
   one (`src/evm/cuda/__init__.py:61-66`).
5. **`DROP_LAST = 10` is applied inside the frame readers, not at the API boundary**:
   `src/evm/cpu/magnify.py:50` used in `_read_frames` (`:121`), and
   `src/evm/cuda/_common.py:31` reading `_evm_cuda.drop_last` (defined as `kDropLast` in
   `cuda/include/evm_common.cuh:21`). Any array-in API must decide this explicitly — see
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
| `cuda/DESIGN.md` | Kernel map, per-stage tolerances, precision and memory-layout rationale |

Keep this file current in the same session as the change: new module → update the tree;
new dependency → update the stack table; new gotcha → add it here immediately; a gotcha that
a phase has fixed → delete it.

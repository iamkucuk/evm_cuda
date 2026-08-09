# Packaging and environment notes

Findings recorded while executing Phase 0 of `docs/dev/PLAN.md` on branch
`library-restructure`. Every entry is the output of a command that was actually run;
the command is given so it can be re-run.

## 1. `pip install .` fails today (plan step 0.1)

Date: 2026-08-09. Host: macOS 26.5.2 arm64, Python 3.13 in a fresh `venv`.

```bash
python3 -m venv /tmp/v01 && /tmp/v01/bin/pip install /path/to/evm_cuda
```

Result — the build backend never even reaches the CUDA extension:

```
error: Multiple top-level packages discovered in a flat-layout:
['evm', 'cuda', 'data', 'colab', 'kaggle', 'output', 'shared', 'benches'].
```

`pyproject.toml` declares no `[tool.setuptools]` package configuration, so setuptools
auto-discovery scans the repository root, finds eight importable-looking directories, and
refuses to guess. This confirms the suspicion recorded in the plan and is worse than
expected: the plan predicted two conflicting top-level packages (`evm`, `shared`), the real
count is eight.

Second finding from the same run — a deprecation that will become a hard error:

```
Please use a simple string containing a SPDX expression for `project.license`.
By 2027-Feb-18, you need to update your project and remove deprecated calls
or your builds will no longer be supported.
```

`pyproject.toml` uses the table form `license = { text = "..." }`. It must become an SPDX
string. This is handled in Phase 1 step 1.10 together with the rest of the rewrite.

## 2. Baseline test run (plan step 0.2)

Host: macOS 26.5.2 arm64, Python 3.14.5, repository `.venv`.

```bash
.venv/bin/python -m pytest tests/ -q -p no:randomly
```

Result: **32 passed, 63 skipped**. Full output saved to
`benches/baseline_tests_2026-08-09_macos.txt`.

All 63 skips share one cause: the compiled `_evm_cuda` extension is not built on macOS, so
everything under `tests/cuda/` skips. The tests that compare output against the MIT MATLAB
reference (`tests/test_against_mit_reference.py`) **do** run on this host, because `data/`
contains `face.mp4`, `baby.mp4`, `face_mit_ref.mp4` and `baby_mit_ref.mp4`. Those files are
not in version control, so the same tests will skip on a continuous-integration runner —
which is why the dependency-free golden fixtures (plan step 0.4) are needed before any
restructuring begins.

## 3. GPU development server `osiris` (plan step 0.7)

Reached with `ssh osiris` (Tailscale SSH; a browser authentication check may appear on a
new session).

| Property | Value |
|---|---|
| Machine | Windows desktop `DESKTOP-4LO089U` running WSL2, distribution "Pengwin" (Debian-based) |
| Kernel | 6.6.87.2-microsoft-standard-WSL2 |
| GPU | NVIDIA GeForce RTX 3090, 24 GB, driver 591.86, reports CUDA 13.1 capability |
| GPU tools | `nvidia-smi` exists only at `/usr/lib/wsl/lib/nvidia-smi`, not on `PATH` |
| Free disk | 537 GB |

### Constraint: no administrator rights

`sudo` requires a password that this environment does not have, and the machine has **no
system compiler at all** (`/usr/bin/gcc` does not exist). System package installation is
therefore impossible here.

### Solution: an isolated userspace environment

The machine already has a per-user conda installation at `/home/furkan/miniconda3` with
several research environments (`wan-lora`, `causvid`, `freqtrade` and others). Those belong
to unrelated work and were **not** modified. A dedicated environment was created instead:

```bash
conda create -y -n evm-cuda -c conda-forge -c nvidia \
    python=3.12 cmake ninja gxx_linux-64 cuda-toolkit=12.9 numpy scipy
```

Installed and verified:

| Tool | Version | Path |
|---|---|---|
| `nvcc` | 12.9.86 | `/home/furkan/miniconda3/envs/evm-cuda/bin/nvcc` |
| `cmake` | 4.4.2 | same directory |
| `ninja` | present | same directory |
| C++ compiler | `x86_64-conda-linux-gnu-g++` | same directory |

CUDA 12.9 was chosen to match the toolkit already proven to work in an existing environment
on this machine; the driver supports up to 13.1, so it is well within range.

Python packages added to that environment with `pip`: `pybind11`, `pytest`,
`opencv-python-headless`, `av`, `requests`.

### Working copy on the server

The repository is copied to `~/evm_cuda_dev` with `rsync` (excluding `.git`, `.venv`,
`output/`, `kaggle/results_*`, build directories). Pushing the development branch to GitHub
was deliberately avoided; the copy is local and disposable.

To refresh it after local changes:

```bash
rsync -az --delete --exclude '.venv' --exclude '.git' --exclude 'output' \
  --exclude 'kaggle/results_*' --exclude '__pycache__' --exclude '.pytest_cache' \
  --exclude 'cuda/build' ~/Documents/projects/evm_cuda/ osiris:~/evm_cuda_dev/
```

### Defect found while building: the GPU architecture list is silently ignored

Building on the server with the documented command failed:

```
cuda/kernels/spatial.cu(918): error: identifier "__hfma2" is undefined
```

`__hfma2` is a packed half-precision multiply-add intrinsic that requires compute
capability 5.3 or newer. Compiling the same file by hand for every intended architecture
(`sm_60`, `sm_70`, `sm_80`, `sm_86`, `sm_89`, `sm_90`) produced **zero errors**, so the
source is fine. Inspecting the command CMake actually generated showed why:

```
--generate-code=arch=compute_52,code=[compute_52,sm_52]
```

It was building for compute capability **5.2**, where that intrinsic does not exist.

Cause, in `cuda/CMakeLists.txt`:

```cmake
project(evm_cuda LANGUAGES CXX CUDA)      # line 9
...
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)  # line 20
    set(CMAKE_CUDA_ARCHITECTURES 60 70 80 89 90)
endif()
```

The `project()` call on line 9 enables the CUDA language, and CMake sets
`CMAKE_CUDA_ARCHITECTURES` to the compiler's own default (5.2 for this toolkit) as part of
that step. By the time line 20 runs, the variable is already defined, so the guard never
fires and the intended architecture list is discarded without any message.

Confirmed by passing the value explicitly, which compiles and links cleanly:

```bash
cmake -S cuda -B cuda/build -DCMAKE_BUILD_TYPE=Release -G Ninja \
      -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build cuda/build -j     # 7 seconds, 12/12 targets
```

Consequences worth noting:
- Anyone building through plain CMake, without passing the architecture explicitly, either
  fails to compile (newer toolkits) or silently gets a binary tuned for 2014-era hardware.
- The fix belongs to Phase 1 step 1.8, which already rewrites this block: set the value
  before `project()`, or use `CMAKE_CUDA_ARCHITECTURES_DEFAULT`, and default to `native`
  for source installs.
- Build cost is far lower than the plan assumed: a single-architecture build of all ten
  CUDA files plus the bindings takes about 7 seconds on this machine, so the multi-architecture
  build is a minor concern rather than a serious install-time problem.

### Full test suite on the GPU (plan step 0.2, second host)

```bash
PYTHONPATH=~/evm_cuda_dev/cuda python -m pytest tests/ -q -p no:randomly
```

Result: **111 passed, 0 skipped** in 169 seconds (after Phase 0 added its tests). Saved to
`benches/baseline_tests_2026-08-09_osiris_rtx3090.txt`.

**A correction worth recording, because the first number here was wrong.** The first run on
this machine reported "102 passed" and was initially written down as the pre-change GPU
baseline. It was not. The working copy had been copied to the server while the Phase 0
agents were still creating files, so it already contained `tests/test_reference_lock.py`
(7 cases, written at 13:10) but not yet `tests/test_golden.py` (written at 13:13):
95 original cases + 7 = 102. Once both files were present, both hosts collect an identical
111 cases, with identical per-file counts — there is no host-dependent collection at all,
which was the other thing the wrong number seemed to suggest.

The honest comparison is therefore:

| Host | Result | Meaning |
|---|---|---|
| macOS, no CUDA toolchain | 48 passed, 63 skipped | the 63 skips are the entire `tests/cuda/` suite |
| `osiris`, RTX 3090 | 111 passed, 0 skipped | everything runs |

The pre-Phase-0 count of 32 passed + 63 skipped, recorded on macOS, is the authoritative
record of where this work started.

### Caveat for benchmark numbers

This GPU is reached through the WSL2 virtualization layer, so host-to-device and
device-to-host transfer timings can differ from a native Linux host. Correctness testing is
unaffected. Any benchmark figure measured here must state the platform.

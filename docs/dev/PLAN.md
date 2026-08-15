> **This is the working plan for the library restructure, committed into the repo so
> every implementing session can find the step it is executing.**
>
> - Branch: `library-restructure`.
> - Status as of 2026-08-10: **phases 0 through 9 are done**, with the
>   exceptions recorded below. Phase 8 (visibility) is prepared but consists
>   almost entirely of actions only the project owner can take; see
>   `docs/dev/release-checklist.md`.
> - Departures from this plan, each recorded where it happened: the portable
>   backends are hand-written kernels in each interface's own language rather
>   than one source compiled by Halide, chosen after testing rather than from
>   the recollection the plan rested on (phase 4V); and the phase-based method
>   (phase 9) is checked against constructed motion rather than the authors'
>   published output, which is not among the files this project can fetch.
> - Section 3d (execution methodology) is binding and is enforced through
>   `CLAUDE.md` and `.claude/rules/development-practices.md`, which sessions load
>   automatically. This file is the reference for *what* to build; those two are the
>   reference for *how*.
> - The text below is the planner agent's document, copied unchanged.

---

# Implementation Plan: making `evm-cuda` the reference library for Eulerian Video Magnification

> Produced by the planner agent on 2026-08-09 from a full read of the repository at
> `/Users/furkan/Documents/projects/evm_cuda`. Status: AWAITING OPERATOR CONFIRMATION.
> No code has been written.
>
> **Revision 2 (same day):** added (a) the verified GPU dev server `osiris` and its
> provisioning step, and (b) a multi-backend strategy (decision D9, Phase 4P) covering
> AMD/ROCm, Apple/Metal, Intel, OpenCL and Vulkan — what is supported, through what
> mechanism, and what is explicitly out of scope.
>
> **Revision 3 (same day, operator override):** direct OpenCL and Vulkan support moved
> from "out of scope" to IN scope, per the operator's requirement that any future GPU or
> accelerator shipping a standard driver must be able to run the library. Backend strategy
> restructured into three tiers; new decision D10 (implementation route for the native
> portable tier) and new Phase 4V added.
>
> **Revision 4 (same day, operator override):** priorities flipped — native backends are
> the PRIMARY multi-vendor path (the single-source portable tier now also targets Metal
> natively, so Apple support does not depend on PyTorch). The PyTorch backend is demoted
> to an optional, explicitly-installed extra (it was already optional-install; what changed
> is its role: convenience/interop backend and cross-check, not the coverage mechanism).
> Phase 4V now precedes Phase 4P in priority; Phase 4P may slip past 1.0 without harm.
>
> **Revision 5 (same day, operator design input):** backends are organized behind explicit
> interfaces (operator's proposal), refined into a two-level design — an ops-level protocol
> a backend must implement, from which all four pipelines derive automatically, plus an
> optional pipeline-level override so fused/scheduled implementations (native CUDA, Halide)
> keep their speed. New section 3c; one shared conformance test suite runs against every
> registered backend.
>
> **Revision 6 (same day, operator requirement):** execution methodology added (section 3d):
> every agent/session implementing this plan is bound to test-driven development, KISS,
> YAGNI, DRY, fail-loud, and step-sized verified commits. Enforcement mechanism: the rules
> are committed INTO the repository (root `CLAUDE.md` + `.claude/rules/`) in new Phase 0
> step 0.8, because implementing sessions load those files automatically — a plan document
> alone does not bind them.

## 1. Requirements, restated

1. **Turn a research port into a library.** Today the repo is a correct, fast, well-benchmarked *implementation*. It is not consumable: nothing is pip-installable end to end, the GPU package is reachable only through a `PYTHONPATH` hack, and the only API takes a file path and writes a file.
2. **One install, two backends.** `pip install evm-cuda` must succeed on any machine. CPU (NumPy) always works; the CUDA extension compiles when `nvcc` is present, and its absence is reported loudly and legibly, never silently.
3. **Two API tiers.** A one-line facade (`magnify(video, preset="pulse")`, NumPy in / NumPy out, optional keep-on-GPU) *and* first-class components (pyramids, blur-downsample, three temporal filters, color conversion, video I/O) on both backends. GPU components must chain without host round-trips, which requires a public device-array type with DLPack so PyTorch/CuPy interop is free.
4. **Semver on the public surface only.** `evm.*` and the public GPU ops are covered; the compiled `_evm_cuda` module stays private with zero compatibility promise.
5. **Streaming as the flagship differentiator.** Webcam/RTSP in, magnified frames out at camera rate. The motion IIR path is already causal; the color path's FFT filter is offline-only and needs a causal variant.
6. **Phase-based magnification (SIGGRAPH 2013) later**, to own the category rather than one algorithm.
7. **Documentation and examples as a first-class deliverable**, organised by user task (pulse/vitals, vibration/modal, general motion), not by module.
8. **Standard OSS engineering**: CI, lint/format/types, changelog, contributing, citation, release automation.
9. **Visibility**: package index, repository topics, README positioning, announcements, and a Zenodo record for citation (a Journal of Open Source Software paper is ruled out by the licence decision).
10. **Run on non-NVIDIA GPUs and on future GPUs/accelerators, through NATIVE backends** (operator request, revisions 2–4): AMD, Apple and Intel hardware must be usable via native code paths (no framework dependency), and OpenCL and Vulkan must be supported directly, so that any future device shipping a standard driver runs the library without new code from us. PyTorch is permitted only as an optional, explicitly-installed extra backend. Resolved by decision D9 (three backend tiers, natives first), decision D10 (implementation route), Phase 4V (primary) and Phase 4P (optional).
11. **Non-negotiable invariant**: the MATLAB-reference correctness tests never weaken. Every phase ends with the suite green.

## 2. What is actually in the repo (verified by the planner agent)

| Fact | Evidence |
|---|---|
| CPU baseline is 4 modules, path-in/file-out API | `evm/magnify.py` (all four `magnify_*` take `vid_path`, call `_read_frames`, call `_write`) |
| `DROP_LAST = 10` is applied inside the reader, not the API | `evm/magnify.py:50`, `evm/magnify.py:134`; CUDA mirror at `cuda/evm_cuda/_common.py:31` |
| There is a **third** top-level package, `shared/` | `shared/h264.py`, imported by `evm/video.py:84` and `cuda/evm_cuda/batched.py:128` |
| `pyproject.toml` declares no packages; flat layout with `evm` + `shared` both present | no `[tool.setuptools]` section |
| `evm-cuda` has **never been installed** in the repo venv | site-packages contains only third-party dists |
| GPU package importable only via `PYTHONPATH=cuda` | `Makefile:25`; `tests/cuda/conftest.py:21-25`; Colab notebook; `kaggle/run_gpu_comparison.py:156-157` |
| CMake hard-fails without CUDA | `cuda/CMakeLists.txt:9` `LANGUAGES CXX CUDA`; `:32` `find_package(CUDAToolkit REQUIRED)` |
| CMake asks for full `Development` Python component | `cuda/CMakeLists.txt:38` — scikit-build-core wants `Development.Module` |
| Batched bindings take raw `uintptr_t`, no shape/dtype validation | `cuda/bindings.cpp:919` onward |
| Device memory is pool-backed, freed only at process exit | `DeviceMemPool` at `cuda/bindings.cpp:357-396`; `DeviceBuffer` dtor returns blocks to pool (`:408-411`) |
| `DeviceBuffer` Python wrapper is private and byte-oriented | `cuda/evm_cuda/batched.py:36-70` |
| **A batched Butterworth binding does not exist** | launcher `launch_butter_bandpass` declared at `cuda/bindings.cpp:94-97`, no `m.def` |
| GitHub Pages deploys the raw `docs/` directory | `.github/workflows/deploy-pages.yml:29-30`; `docs/index.html` is a hand-written marketing page |
| The only CI is the Pages deploy | `.github/workflows/` has one file |
| 69 test functions across 15 files (→ 92 collected cases) | `rg -c '^def test_'` over `tests/` |
| MIT-reference tests **skip unless `data/*.mp4` exists** | `tests/test_against_mit_reference.py:38-41, 60-63` |
| CPU end-to-end tests assert behaviour (swing ratios), not golden values | `tests/test_pipeline.py:106, 121, 134` |
| Stale docstring references non-existent `scripts/download_mit_outputs.py` | `tests/test_against_mit_reference.py:5` |
| Tolerance table centralised | `tests/cuda/conftest.py:45-56` (`TOL`) |
| Tests import a private class directly | `tests/cuda/test_device_buffer.py:27` |
| License is BSD-3 **Non-Commercial** with mandatory-citation clause | `LICENSE:1-25` |

### Two findings that change the plan's shape

- **`pip install .` is almost certainly broken today, not merely incomplete.** Setuptools flat-layout auto-discovery raises "Multiple top-level packages discovered in a flat-layout: ['evm', 'shared']" when no explicit package config exists. Nothing in the repo has ever exercised the install path. The planner could not execute commands — Phase 0 Step 1 is to run `pip install .` in a clean venv and record the actual failure. Do not treat the inference as fact.
- **JOSS is blocked by the current license.** JOSS requires an OSI-approved license. `BSD-3-Clause-NonCommercial` is not OSI-approved (no NC variant is), and PyPI has no classifier for it.

### The GPU dev server `osiris` (probed over SSH, 2026-08-09)

| Fact | Value |
|---|---|
| Reachability | `ssh osiris` works (Tailscale SSH; each new session may show a one-time browser auth check) |
| Machine | Windows desktop `DESKTOP-4LO089U` running WSL2, distro "Pengwin" (Debian-based), kernel 6.6.87.2-microsoft-standard-WSL2 |
| GPU | NVIDIA GeForce RTX 3090, 24 GB, Windows driver 591.86, passed through to WSL2 (`/usr/lib/wsl/lib/libcuda.so` present; `nvidia-smi` only at `/usr/lib/wsl/lib/nvidia-smi`, not on PATH) |
| Toolchain | Python 3.13.5 and git present. **No nvcc, no cmake, no ninja, no gcc/g++** — the CUDA extension cannot be compiled there until provisioned (new step 0.7) |
| Disk | 537 GB free |
| Caveat | Benchmark numbers measured under WSL2 can differ from native Linux (host-device transfers cross the WSL virtualization layer). Correctness testing is unaffected. Published benchmark claims should state the platform. |

This machine replaces every earlier "the 3090" reference in this plan: Phase 0 baselines, Phase 1 GPU install verification, and the self-hosted GPU CI candidate in D6 all run on `osiris`.

## 3. Decisions needed before Phase 1 (D1–D8)

| # | Decision | Options | Recommendation |
|---|---|---|---|
| **D1** | Import namespace (today `evm`, `evm_cuda`, `shared` are three unrelated top-level packages) | (a) single root `evm` with `evm.cpu`/`evm.cuda`/`evm.io`; (b) single root `evm_cuda`; (c) rename entirely (`vidmag`, `pyvidmag`) | **(a)**, plus a thin `evm_cuda` shim so existing notebooks/tests/snippets keep working. **Superseded in part on 2026-08-11:** the distribution name did not stay `evm-cuda`. By then the project had five backends, not one, and the name said otherwise; it had never been published, so the rename cost nothing. It installs as `evm-magnify`, matching the terminal command it provides. Plain `evm` was never available — it belongs to an unrelated project on PyPI, the Extreme Value Machine. The import root is still `evm`, which is what option (a) was actually about. Caveat as recorded: "EVM" means Ethereum Virtual Machine to most search traffic; `evm-magnify` reads on the distribution name rather than fixing that. |
| **D2** | License | keep BSD-3-NC; relicense BSD-3/Apache-2.0; dual-track | **DECIDED by the operator on 2026-08-09: the non-commercial restriction stays.** The licence was amended to state the permitted uses explicitly rather than relicensed: research (including research inside a company), teaching, personal and evaluation use, and inclusion in open-source software distributed at no charge under a licence permitting those same uses. Selling it, or building it into a product or service run for commercial advantage, still needs written permission. Consequences that follow and are now settled: no Journal of Open Source Software submission (it requires an OSI-approved licence, and no non-commercial variant is one), no standard licence classifier on the package index, and no adoption by companies shipping products. The realistic ceiling is the default library for research and education. Earlier recommendation, superseded: **relicense to BSD-3-Clause or Apache-2.0** if "most widely known" is the real goal. NC blocks JOSS, PyPI classification, corporate mirrors, downstream inclusion. Keep citation *request* in `CITATION.cff`/README, not as license condition. MIT method patent is orthogonal — add a NOTICE paragraph stating it plainly. Operator's call; if NC stays, delete the JOSS step. |
| **D3** | Docs tool | mkdocs-material + mkdocstrings; Sphinx + MyST | **mkdocs-material + mkdocstrings[python]** — every existing doc is already Markdown. |
| **D4** | Docs hosting + existing landing page | (a) mkdocs builds to `site/`, landing page becomes Material custom-template home; (b) keep `index.html` at `/`, docs at `/docs/`; (c) Read the Docs | **(a)**. One deploy, one URL, search across everything. `docs/index.html` → `docs/overrides/home.html`; media → `docs/assets/`. |
| **D5** | CLI | console_script `evm`; `evm-magnify`; keep as script | **`evm-magnify`** (`evm` collides with Ethereum tooling), new `src/evm/_cli.py`, subcommands `magnify`/`stream`/`bench`/`download`; `scripts/run_evm.py` becomes deprecation shim. |
| **D6** | GPU CI (hosted runners have no NVIDIA GPU) | self-hosted runner on `osiris`; manual Colab/Kaggle gate; cloud GPU runner on release tags | **Self-hosted runner inside WSL2 on `osiris`** (the RTX 3090 machine probed above), `workflow_dispatch` + `on: release`; keep `kaggle/run_gpu_comparison.py` as cross-GPU spot check; publish GPU-run attestation JSON into `benches/`. Caveat: the runner dies whenever the Windows host shuts WSL down; if that proves flaky, fall back to a cloud GPU runner on release tags only. |
| **D7** | Wheels | sdist only; cibuildwheel CUDA wheels | **sdist only for 0.2–0.4** with a documented why (fat multi-arch wheels, PyPI size limits). Revisit at 1.0. |
| **D8** | `drop_last=10` semantics in the new array API | | **Split**: path-based functions keep `drop_last=10` default; array-based `magnify()` defaults to `drop_last=0` and documents why. |
| **D9** | Multi-backend strategy (operator asked for OpenCL, Metal, ROCm, Vulkan, any future GPUs/accelerators, natives first, torch optional) | (a) hand-write one backend per API; (b) three tiers with the single-source native portable tier as the primary multi-vendor path and PyTorch as an optional extra; (c) PyTorch tier as primary coverage | **(b)** — fixed by operator overrides in revisions 3–4 (revision 3 added the native portable tier; revision 4 made it primary and demoted torch to optional). Full reasoning, coverage table and honest limits in section 3b. |
| **D10** | Implementation route for the portable native tier (OpenCL + Vulkan) | (a) Halide single-source (also yields Metal/D3D12/WebGPU targets); (b) OpenCL C kernels + `clspv`-compiled SPIR-V for Vulkan; (c) WebGPU/WGSL + separate OpenCL; (d) SYCL | **(a) Halide, with (b) as fallback**, decided by the time-boxed feasibility spike in Phase 4V step 1 — Halide's Vulkan-target maturity is unverified training-data knowledge and must be proven before committing. |

## 3b. Backend strategy (D9 + D10, full reasoning) — three tiers

**Why not one hand-written backend per API.** The CUDA backend is ~10 hand-optimized kernel files plus cuFFT, tuned over months (FP16 half2 paths, shared-memory layouts, a device memory pool, batched launches). OpenCL, Metal and Vulkan each use a different kernel language and have no cuFFT; hand-writing each backend means re-implementing and re-validating all of it per API, then owning the maintenance forever — multiple person-months per backend for a single-developer project. The operator's future-proofing requirement is met instead by writing the portable kernels **once** in a tool that compiles the same source to every target API (decision D10).

**The three tiers (revision 4: natives first, PyTorch last and optional):**

| Tier | What it is | Covers | Speed expectation |
|---|---|---|---|
| 1 — Native CUDA | The existing hand-tuned kernels, unchanged | NVIDIA GPUs | Fastest (the published numbers) |
| 2 — Portable native kernels (Phase 4V, **primary multi-vendor path**) | Kernels written once in a single-source tool, compiled to **OpenCL, Vulkan, and Metal** (D3D12 from the same source at no extra authoring cost) | AMD, Apple, Intel today; any device, present or future, that ships an OpenCL or Vulkan driver | Below tier 1; measured in-phase, never claimed in advance |
| 3 — PyTorch ops (Phase 4P, **optional extra, explicitly installed**) | Four pipelines implemented once on `torch` tensor operations. Role: convenience for users already inside torch pipelines, and an independent cross-check implementation. The torch dependency is large (hundreds of MB to gigabytes installed), which is why it can never be the coverage mechanism | NVIDIA (CUDA), AMD (ROCm builds), Apple (MPS), Intel (XPU), CPU — all redundantly to tiers 1–2 | Between CPU baseline and tier 1; measured in-phase |

**D10 — implementation route for tier 3.** Options: (a) **Halide** — a language for image-processing pipelines; one pipeline definition, per-target schedules; official compile targets include CUDA, OpenCL, Metal, Vulkan, D3D12 and WebGPU; well suited to this workload (separable convolutions, pyramids, elementwise passes). (b) Kernels in OpenCL C, compiled to SPIR-V for Vulkan with `clspv`. (c) WebGPU/WGSL plus a separate OpenCL implementation. (d) SYCL — rejected: no Vulkan or Metal target. **Recommendation: (a) Halide, with (b) as the fallback**, decided by a time-boxed feasibility spike (Phase 4V step 1) — my information on the maturity of Halide's Vulkan target is from training data and must be verified before committing, so the spike comes first.

**Removing the FFT obstacle for tier 3 (this is what makes it feasible).** The ideal temporal filter keeps only the DFT frequency bins inside [fl, fh]. When the kept band is narrow — it is, for the shipped presets — computing those few bins directly as two small matrix multiplications is *mathematically identical* to FFT → mask → inverse FFT (same linear map), at cost proportional to band width instead of a full FFT. Consequence: tier 3 needs **no FFT library at all**. The IIR and Butterworth filters are per-pixel recursions and port directly. Cost grows with band width; the docs state this and offer the offline CPU-FFT path for unusually wide bands.

**Hardware coverage after all tiers:**

| User's hardware | Primary (native) path | Optional torch path | Test hardware we have |
|---|---|---|---|
| NVIDIA GPU | Tier 1 hand-tuned CUDA; tier 2 also runs | torch-CUDA | `osiris` (RTX 3090; OpenCL/Vulkan testing on its Windows side — WSL2 passes through CUDA only) |
| AMD GPU | Tier 2 native kernels via Vulkan or OpenCL (a native HIP port of the CUDA kernels stays a deferred option if AMD demand appears) | torch-ROCm | None — ships "expected to work, unverified" until a result is contributed |
| Apple Silicon | Tier 2 native kernels via the **Metal** compile target (Vulkan via MoltenVK as second route) | torch-MPS | This development Mac |
| Intel GPU | Tier 2 native kernels via Vulkan or OpenCL | torch-XPU | None — "unverified" label |
| **Any future GPU/accelerator with an OpenCL or Vulkan driver** | Tier 2, without new code from us | — | CPU driver implementations in CI (see below) |
| CPU only | NumPy baseline; tier 2 on CPU drivers | torch-CPU | This Mac + CI runners |

**CI story for tier 3 (no GPU needed):** OpenCL has a CPU implementation (PoCL) and Vulkan has a CPU implementation (Mesa lavapipe); both install on hosted CI runners. The portable kernels therefore run in CI on every commit, on real drivers, without any GPU — real-GPU runs on the Mac and `osiris` remain the release gate.

**The honest limit, stated plainly:** an accelerator that exposes *neither* OpenCL, nor Vulkan, nor a PyTorch device backend (for example NPUs reachable only through vendor-specific graph compilers) cannot run a general-purpose pixel pipeline by any approach; supporting such a device would be its own dedicated port. New hardware becomes supported through three mechanisms: it ships an OpenCL/Vulkan driver (tier 3 runs as-is), PyTorch adds a device backend (tier 2 runs), or the D10 tool adds a compile target (tier 3 recompiles).

**Rules all non-tier-1 backends must obey:**
1. Optional dependencies only (`pip install evm-cuda[portable]`, `[torch]`); `import evm` always succeeds without them. **PyTorch is never imported unless the user explicitly selects `backend="torch"` or installed the `[torch]` extra and asked for auto-selection to consider it.**
2. Facade: `backend="auto"|"cpu"|"cuda"|"opencl"|"vulkan"|"metal"|"torch"` plus `device=`; automatic selection prefers tier 1 (native CUDA) → tier 2 (native portable, device-appropriate target) → torch only if installed → CPU, and the choice is always printed, never silent.
3. Reference-grade correctness claims (MATLAB-comparison tests and their tolerance table) remain the CPU baseline and native CUDA only; tiers 2 and 3 get their own parity tests against the CPU baseline with separate, documented tolerance tiers (vendor floating-point behavior differs: TF32 defaults, MPS precision, per-driver rounding).
4. Known open items validated at the start of the relevant phase, not assumed: Halide Vulkan and Metal target maturity and the access route to `osiris`'s Windows side (Phase 4V step 1); `torch.fft` on MPS (Phase 4P step 1).

## 3c. Backend architecture: two-level interface + registry (operator design, revision 5)

The operator's requirement: define interfaces once, and each backend fills them in; the facade selects a backend and the same calling code runs unchanged. Adopted, with one refinement to protect performance.

**Why one flat interface is not enough.** If the only contract is "implement these ~10 operations" and the pipelines are generic loops over those operations, then every backend pays one dispatch + one memory round-trip per operation. The hand-tuned CUDA path is fast precisely because it does the opposite — fused kernels (e.g. the fused downsample in `cuda/bindings.cpp`), collapsed launches, buffers reused from a pool across stages. A Halide backend has the same property: it compiles the whole pipeline with a schedule, not one op at a time. Forcing those through op-by-op execution would discard the library's main selling point.

**The two levels:**

| Level | Contract | Who implements it | Who calls it |
|---|---|---|---|
| **Ops protocol** (`evm.backend.Ops`) | The ~10 primitive operations: color convert, blur-downsample, pyramid build/reconstruct, the three temporal filters, gain/attenuation, quantize — plus an opaque backend array handle (shape/dtype/device, `to_numpy`, DLPack where the API supports it) | **Every backend, mandatorily.** This is the minimum a new backend must provide | Component users composing their own pipelines; the default pipeline implementations |
| **Pipeline protocol** (`evm.backend.Pipelines`) | The four `magnify_*` cores (array in, array out) | **Nobody is required to.** A generic default implementation, written once against the Ops protocol, is inherited by every backend for free. A backend *may* override any pipeline with a fused implementation | The facade (`evm.magnify`) |

Consequences, stated plainly:
1. **A new backend = implement the ops, get all four pipelines automatically.** This is what makes "any future accelerator" cheap to add.
2. **The native CUDA tier and the Halide tier override the pipeline level** with their fused/scheduled implementations, so the interface costs them nothing.
3. **A registry** (`evm.backend.register`, entry-point discoverable) maps names (`"cuda"`, `"opencl"`, `"vulkan"`, `"metal"`, `"torch"`, `"cpu"`) to implementations plus capability flags (dtypes supported, FFT available, streaming-capable). `backend="auto"` walks the registry in the fixed preference order from section 3b; unavailable backends report *why* (missing extra, no driver, no device).
4. **One conformance suite instead of per-tier test files**: `tests/backend_conformance/` is parameterized over every registered backend and runs the same parity checks against the CPU baseline, reading per-backend tolerance tables. The per-phase parity suites named in Phases 4P.5 and 4V.5 are instances of this one suite, not separate codebases. A backend that passes conformance may be listed in the support matrix; one that does not, may not.
5. The Phase 3 refactor (splitting each pipeline into an array-in/array-out `_core`) is the first concrete step of this design: those `_core` signatures *are* the Pipeline protocol. The CPU baseline becomes the reference implementation of both protocols.

## 3d. Execution methodology — binding on every implementing agent/session (operator requirement, revision 6)

These rules govern HOW every phase is implemented. They are committed into the repository in Phase 0 step 0.8 (root `CLAUDE.md` + `.claude/rules/development-practices.md`), so every future session loads them automatically. The plan file itself is also committed into the repo (`docs/dev/PLAN.md`) so agents can find the step they are executing.

**1. Test-driven development (TDD).**
- Every step begins by writing, or pointing at, the failing check named in that phase's success criteria — a parity test, a golden fixture, the fresh-venv install script — and ends when that check passes with the whole suite green. No implementation file is created before the check that will judge it exists.
- This is why Phase 0 (golden fixtures, constant-lock test, recorded baselines) precedes everything: it is the safety net the later red-green cycles run inside.
- Applies to non-Python work too: a CMake change is "tested" by the scripted fresh-venv install (`dev/verify_install.sh`); a docs change by `mkdocs build --strict` plus snippet execution.

**2. KISS — keep each piece simple.**
- The only sanctioned abstraction layers are the two protocols of section 3c (Ops, Pipelines) and the registry. Adding any further indirection (base classes, config objects, plugin hooks) requires the operator's explicit approval first.
- Prefer a plain function to a class, a frozen dict to a config system, a copy-pasteable example to a helper framework.

**3. YAGNI — build only what the current step's success criteria require.**
- No speculative parameters, no "while I'm here" generality, no code for a later phase smuggled into an earlier one. If an agent believes the plan misses something, it stops and surfaces the gap to the operator instead of silently building extra (expectation-alignment rule).

**4. DRY — one source of truth, with one named exception.**
- Single-sourced by design: the version string, the preset table, the tolerance tables, the conformance suite, the backend registry, the reference constants. New duplication requires a stated reason in the commit message.
- **The named exception:** the FP32/FP16 native pipeline bodies may remain duplicated when merging them risks numeric drift (README accuracy claims are load-bearing). Correctness outranks DRY; the exception must be commented at the site.

**5. Fail-loud, verify, land small.**
- No silent fallbacks, no silently skipped tests: backend selection is always printed; GPU-skipped test counts are reported, never hidden.
- A step is "done" only when its named verification command has actually run and its output is recorded — claims without run output are not accepted (verification-before-completion).
- Each plan step lands as its own commit with the full suite green; the commit message names the plan step (e.g. "Phase 1 step 1.6: ..."), so the history maps one-to-one onto this document. Unrelated refactoring in the same commit is not permitted (surgical-changes rule).

## 4. Invariants enforced in every phase

- `pytest tests/ -q` green on CPU-only hosts, and full suite green on the 3090, at the end of **every** phase.
- `tests/test_against_mit_reference.py` and `tests/cuda/conftest.py:TOL` are append-only. Loosening a tolerance requires a separate, individually reviewed commit with a measurement in the message.
- A new `tests/test_reference_lock.py` (Phase 0) freezes `DROP_LAST`, `EXAGGERATION_FACTOR`, `BINOM5`, `BINOM5_SUM1`, and the `TOL` dict against literal values.

## Phase 0 — Ground truth and safety net
**Complexity: Low. Effort: 1–1.5 days. Blocks: everything.**

| # | Step | Files |
|---|---|---|
| 0.1 | Fresh venv: run `pip install .` and `pip install -e .`; capture the exact error. Do this first; Phase 1 is scoped by the answer | — |
| 0.2 | Record green baseline test runs (Mac CPU; 3090 full), output committed to `benches/baseline_tests_<date>.txt` | `benches/` |
| 0.3 | Add `tests/test_reference_lock.py` freezing constants + `TOL` | new |
| 0.4 | Golden end-to-end fixtures needing no downloads: 24×32×32 synthetic clip through all four pipelines, `allclose(atol=1e-6)` forever. Highest-value item: MIT-reference tests skip without `data/`, so CI currently proves nothing end-to-end | `tests/fixtures/golden_*.npz`, `tests/test_golden.py` |
| 0.5 | Fix stale docstring pointing at non-existent `scripts/download_mit_outputs.py` | `tests/test_against_mit_reference.py:5` |
| 0.6 | **DONE — preserved first, then deleted (operator approved 2026-08-09).** All unique content was rescued before removal: five orphan branches `rescue/kaggle-{results_baseline,results_full,results_gpu,results_profile,results_tests}` hold each snapshot's full source tree, including edits that had been left uncommitted inside the nested clones. The four Kaggle console logs, which record measurements on hardware not otherwise available, were copied to `benches/kaggle_runs/`. Verified before deleting: every file in every snapshot is either present on a rescue branch, byte-identical to a copy already in the repository (the four sample videos), or a regenerable build artifact, cache, or generated output video. 178 MB reclaimed; `kaggle/` is now 24 KB. Original blocking reason, for the record: the snapshots were not stale copies — each held its own nested git repository, and `kaggle/results_tests/` held five source files that existed in **no other location** — `cuda/bindings.cpp`, `cuda/evm_cuda/batched.py`, `cuda/kernels/amplify_render.cu`, `cuda/DESIGN.md`, `docs/blog_speedup.md` — committed on a branch `feature/kernel-optimization-A` at commit `f382d21` which is absent from the parent repository. `kaggle/results_full/` and `kaggle/results_profile/` each hold three more (`AGENTS.md`, `CLAUDE.md`, `HANDOFF.md`), and `kaggle/results_gpu/` has ten uncommitted modifications plus an untracked results file. Deleting would destroy unrecoverable work. 178 MB, 766 files. Options for the operator: (a) leave in place, (b) extract the unique commits into real branches of the parent repository first, then delete, (c) archive the five directories outside the repository, then delete. | `kaggle/` |
| 0.7 | Provision `osiris` (verified present: Python 3.13, git, RTX 3090 via WSL2 passthrough; verified absent: nvcc, cmake, ninja, gcc): install build-essential + cmake + ninja + the CUDA toolkit WSL build, clone the repo, run the current manual build, run the full test suite. This is also the dry run proving a stranger's fresh-machine setup path | `osiris` over SSH; findings recorded in `docs/dev/packaging-notes.md` |
| 0.8 | Commit this plan into the repo as `docs/dev/PLAN.md`; write the root `CLAUDE.md` (100–200 lines per the operator's authoring guide: project description, structure, copy-paste commands, gotchas, rule-file reference table) and `.claude/rules/development-practices.md` encoding section 3d (TDD, KISS, YAGNI, DRY + exception, fail-loud, step-sized commits). From this point on, every implementing session loads the methodology automatically | new `CLAUDE.md`, `.claude/rules/development-practices.md`, `docs/dev/PLAN.md` |

**Success criteria:** CPU suite passes with zero skips among golden tests; editing `TOL` makes the lock test fail; the actual `pip install .` outcome recorded in `docs/dev/packaging-notes.md`.

## Phase 1 — One installable package
**Complexity: High. Effort: 5–7 days. Depends on: 0. Blocks: 2, 3, 4, 6, 7. Riskiest phase.**

Target layout:
```
pyproject.toml                    # scikit-build-core, single dist "evm-cuda"
src/evm/__init__.py               # facade + public re-exports + __version__
src/evm/cpu/{__init__,pyramids,filters,magnify}.py     <- from evm/
src/evm/io/{__init__,video,h264,capture}.py            <- from evm/video.py + shared/h264.py
src/evm/cuda/{__init__,batched,pipelines,runtime,_common,benchmark}.py  <- from cuda/evm_cuda/
src/evm/cuda/_evm_cuda.*.so       # installed here by CMake
src/evm/notebook.py               <- from cuda/evm_cuda/colab_utils.py
src/evm/_cli.py                   # new (D5)
src/evm_cuda/__init__.py          # back-compat shim -> evm.cuda
cuda/{CMakeLists.txt,bindings.cpp,kernels/,include/}    # unchanged location
```
`shared/` and `cuda/setup.py` are deleted.

| # | Step | Risk |
|---|---|---|
| 1.1 | Move CPU modules to `src/evm/cpu/`, rewrite intra-package imports | Low |
| 1.2 | Fold `shared/h264.py` into `src/evm/io/h264.py`; update the two call sites | Low |
| 1.3 | Move GPU wrapper package under `src/evm/cuda/`; keep `from . import _evm_cuda` so `bindings.cpp` needs no change | Medium |
| 1.4 | `src/evm/__init__.py`: re-export CPU components, lazily expose `evm.cuda` (importing `evm` on a CPU box never touches CUDA) | Medium |
| 1.5 | `src/evm_cuda/__init__.py` back-compat shim + `DeprecationWarning` | Low |
| 1.6 | **Make CMake CUDA-optional**: `check_language(CUDA)`; if absent, no target + prominent message; `EVM_CUDA_REQUIRE=1` to hard-fail | **High** |
| 1.7 | `find_package(Python3 ... Development.Module)`, drop NumPy requirement (verify bindings use pybind11 numpy header only) | Medium |
| 1.8 | Default `CMAKE_CUDA_ARCHITECTURES=native` for source installs; `60;70;80;89;90` behind `EVM_CUDA_ARCHS=all` for release/bench. **Also fixes a confirmed defect found in Phase 0:** the architecture list in `cuda/CMakeLists.txt` is silently discarded today, because `project(... LANGUAGES CUDA)` on line 9 already defines `CMAKE_CUDA_ARCHITECTURES` (to the compiler default, 5.2) before the `if(NOT DEFINED ...)` guard on line 20 can set it. The result is a build for compute capability 5.2, which fails to compile the packed half-precision intrinsic `__hfma2` in `cuda/kernels/spatial.cu:918`. Set the value before `project()`. See `docs/dev/packaging-notes.md`. Measured build cost is small — about 7 seconds for one architecture on an RTX 3090 — so the multi-architecture concern is minor | Medium |
| 1.9 | Retarget extension output to installed `evm/cuda/` | Medium |
| 1.10 | Rewrite `pyproject.toml`: scikit-build-core, `cmake.source-dir="cuda"`, extras `[cuda-build]`/`[dev]`/`[docs]`/`[stream]`, classifiers, keywords, URLs | Medium |
| 1.11 | Delete `cuda/setup.py`; `requirements.txt` → `pip install -e ".[dev]"` | Low |
| 1.12 | Strip `PYTHONPATH` from Makefile; `make build` = `pip install -e . --no-build-isolation -v` | Medium |
| 1.13 | Strip `sys.path` surgery from all test files and `scripts/run_evm.py` | Low |
| 1.14 | Update Colab notebook (`!pip install -e .` replacing manual CMake cells) and Kaggle harness. **Colab badge points at `main` — a broken notebook is publicly visible immediately** | **High** |
| 1.15 | `dev/verify_install.sh`: fresh venv → install → import checks → pytest | Low |

**Success criteria:** CPU-only host fresh-venv install prints version and `have_cuda=False` with a loud, named error from `require_cuda()`; 3090 install passes the full 92-test baseline; `PYTHONPATH` gone everywhere; install-from-sdist works (proves sdist carries `cuda/kernels/*.cu`); Colab notebook runs top-to-bottom on a fresh T4.

## Phase 2 — CI and quality gates
**Complexity: Low–Medium. Effort: 2–3 days. Depends: 1. Parallel with: 3.**

1. `ci.yml`: ubuntu/macos/windows × Python 3.10–3.13, no-nvcc install + CPU suite (the permanent regression guard for the CUDA-optional build).
2. `lint.yml`: ruff check/format, mypy on `src/evm` (strict only on public surface initially).
3. `.pre-commit-config.yaml` incl. clang-format on `cuda/**/*.{cu,cuh,cpp}`.
4. `gpu.yml` on self-hosted 3090 (`workflow_dispatch` + release): full CUDA suite + benchmark JSON to `benches/`.
5. Build-from-sdist smoke job; coverage badge.

**Success criteria:** deleting a line of `filters.py` turns CI red in ≤5 min; a no-nvcc runner still installs and passes; dispatched GPU run commits `benches/*.json` with matching SHA.

## Phase 3 — Facade API and CLI
**Complexity: Medium. Effort: 4–5 days. Depends: 1. Parallel with: 2.**

1. Split each CPU pipeline into `_core(frames, fps, ...) -> ndarray` + thin path wrapper preserving today's exact signatures (`evm/magnify.py:144-403`); same for the two batched GPU pipelines + FP16 twins (`on_stage` hook must survive — benchmark harness and tests drive it). These `_core` signatures are the Pipeline protocol of section 3c; the CPU baseline becomes its reference implementation.
2. Facade `src/evm/api.py`: `magnify(video, *, preset="pulse"|"motion", fps=None, backend="auto"|"cpu"|"cuda", precision="fp32"|"fp16", out=None, keep_on_device=False, **overrides)`; `video` accepts path/ndarray/iterable-of-frames.
3. `src/evm/presets.py`: frozen preset table — `pulse` (α=50, level=4, 50/60–1.0 Hz, chromAtn=1), `motion` (α=10, λc=16, r1=0.4, r2=0.05, chromAtn=0.1), plus `vibration`, `breathing`.
4. Backend selection through the section-3c registry: new `src/evm/backend/` package holding the two protocol definitions (Ops, Pipelines), `register()`, and capability flags; **never silently fall back to CPU** (a ~700x perf cliff must be loud), and an unavailable backend reports *why* (missing extra, no driver, no device).
5. `evm-magnify` console script (`magnify`/`download`/`bench`); cross-backend parity test within existing `end_to_end_rmse` tolerance; array/path equivalence test encoding D8.

**Success criteria:** array-in/array-out magnify works with no files touched; CLI reproduces `make run-color` within 1 LSB; legacy names still importable and passing golden fixtures.

## Phase 4 — Public GPU component API (`DeviceArray` + ops + DLPack)
**Complexity: High. Effort: 8–11 days. Depends: 1, 3. Blocks: 7 partially.**

Correctness risk concentrates here: today `batched_*` takes bare `uintptr_t` with no checking.

1. **Fix ownership first**: pooled buffer becomes `shared_ptr`-owned. Today `DeviceBuffer::~DeviceBuffer` returns the block to the pool immediately (`cuda/bindings.cpp:408-411`) — a DLPack consumer holding the pointer would get a block the pool hands to someone else. **Use-after-free waiting to happen; must land before any pointer escapes.**
2. Public `DeviceArray`: `.shape`, `.dtype` (f32/f16/u8), `.ptr`, `.nbytes`, `.device`, `.numpy()`, `.from_numpy()`, `__repr__`, `__len__`, axis-0 slicing (`src/evm/cuda/array.py`).
3. DLPack producer (`__dlpack__`, `__dlpack_device__`, kDLCUDA, legacy default stream only in 0.x) and consumer (`from_dlpack` borrowing torch/CuPy tensors without copy).
4. Migrate `batched.py` off private `DeviceBuffer`/`ptr_at` offsets onto `DeviceArray` views; keep `tests/cuda/test_device_buffer.py` passing.
5. Public ops: `src/evm/cuda/ops.py` (shape/dtype-checked `rgb_to_yiq`, `blur_dn`, `build_lpyr`, `recon_lpyr`, `ideal_bandpass`, `iir_bandpass`, `upsample_add_quantize`) — this is the native-CUDA implementation of the section-3c Ops protocol, while the fused batched pipelines remain its Pipeline-protocol override; **add the missing `batched_butter_bandpass` binding** (~15 lines, launcher exists); mirror CPU namespace `src/evm/cpu/ops.py`; op-level CPU/GPU parity tests under existing `TOL`; zero-copy interop test asserting `data_ptr()` identity; `__all__` freeze test.

**Success criteria:** GPU pyramid round-trip within `TOL["lpyr_roundtrip"]=1e-5`; torch sees identical `data_ptr`; the lifetime regression test (delete DeviceArray, allocate 2 GB, torch tensor still reads original values) passes; full suite green, no tolerance changed.

## Phase 4P — OPTIONAL PyTorch backend — DONE 2026-08-11 (findings: `torch-backend-notes.md`)
**Status after revision 4: optional and deprioritized — may land any time after Phase 3, including after 1.0, without blocking anything. Not the coverage mechanism for any vendor (Phase 4V is). Complexity: Medium-High. Effort: 8–12 days. Depends on: Phase 3 (the array-in/array-out pipeline cores define the contract it implements). Benefits from Phase 4 (DLPack allows zero-copy handoff between the native CUDA path and torch) but does not require it.**

| # | Step | Files |
|---|---|---|
| 4P.1 | **Validation before implementation** (per the project's validate-first rule): on this Mac, inline-test that torch MPS runs the needed op set (conv2d/pad/indexing for pyramids, elementwise for IIR, `torch.fft.rfft`/`irfft` along time). Record which ops work; pick the FFT fallback (CPU FFT or IIR/Butterworth-only) if needed. Same probe for torch-CUDA on `osiris` | findings into `docs/dev/torch-backend-notes.md` |
| 4P.2 | Implement the spatial components once in torch: binom5 blur-downsample, Laplacian pyramid build/reconstruct, matching the CPU baseline's border behavior (reflect1) exactly — border handling is where ports silently diverge | new `src/evm/torch_backend/pyramids.py` |
| 4P.3 | Implement the three temporal filters in torch (ideal FFT, first-order Butterworth, r1/r2 IIR), plus color conversion and the gain/attenuation steps | new `src/evm/torch_backend/filters.py`, `ops.py` |
| 4P.4 | Implement the section-3c Ops protocol on torch tensors and register as `"torch"` (the generic default pipelines then come free; override only if profiling justifies it); `device=` wired through the facade; auto-selection follows the fixed section-3b order (native tiers first, torch only if installed), selection always printed | `src/evm/torch_backend/`, `src/evm/api.py` |
| 4P.5 | Conformance: runs the shared parameterized suite of section 3c (`tests/backend_conformance/`) with its own tolerance table (separate from the native-CUDA `TOL` table — vendor floating-point behavior differs; disable TF32 in tests for determinism); runs on CI as torch-CPU (every commit), on this Mac as MPS, on `osiris` as torch-CUDA | `tests/backend_conformance/` |
| 4P.6 | Docs: a backend-support matrix page stating per vendor: mechanism, verified-on-real-hardware yes/no, measured speed. AMD/Intel ship labeled "expected to work, unverified — results welcome" with an issue template for contributed results. OpenCL/Vulkan listed as out of scope with the reason | `docs/backends.md` |

**Success criteria**
- `evm.magnify(arr, preset="motion", backend="torch", device="mps")` produces output passing the parity tolerance on this Mac.
- Same call with `device="cuda"` passes on `osiris`.
- `import evm` works and all CPU tests pass in an environment with no torch installed.
- CI runs the torch-CPU parity suite green on every commit.
- `docs/backends.md` published with honest verified/unverified labels.

## Phase 4V — Native portable kernel backend — DONE via OpenCL (2026-08-10)

**Outcome, and how it differs from what was planned.** The decision between
Halide and hand-written kernels (decision D10) was settled by testing rather
than by argument: macOS still ships OpenCL, it reaches the Apple graphics
processor, and a kernel run through it agreed with NumPy on the first attempt.
OpenCL covers Apple, AMD, Intel and NVIDIA from one source, needs no extra
build system because programs are compiled at run time by the driver, and is
close enough to CUDA C that porting the existing kernels was mechanical. Halide
was not used.

**Vulkan and Metal are implemented.** An earlier revision of this file recorded
a decision not to build them, on the reasoning that every device targeted was
already reachable through OpenCL. That reasoning answered the wrong question:
the requirement was that hardware appearing *later* should work, and OpenCL is
deprecated on Apple, absent on Android, and not what new hardware ships with.
Narrowing a stated requirement was not a decision this plan was entitled to
make. Both backends now exist, both pass the same conformance suite against the
NumPy reference, and all three portable backends are measured in
`docs/concepts/backends.md`.

Original plan text follows.

## Phase 4V (original) — OpenCL + Vulkan + Metal — PRIMARY multi-vendor path
**Complexity: High. Effort: 20–30 days. Depends on: Phase 3 (implements the same array-core contract). Runs as soon after Phase 3 as capacity allows; does not wait for Phase 4P (which is optional). Parallel with: Phases 4, 5, 6, 7. Cross-checks: the CPU baseline is the correctness oracle; native CUDA is the second reference on NVIDIA hardware.**

| # | Step | Detail |
|---|---|---|
| 4V.1 | **Feasibility spike, time-boxed to 3 days** (decides D10): implement one component — binom5 blur-downsample with reflect borders — in Halide; compile to OpenCL, Vulkan **and Metal**; run on (i) CI-style CPU drivers (PoCL for OpenCL, Mesa lavapipe for Vulkan) on a Linux box/container, (ii) this Mac (native Metal target, plus Vulkan via MoltenVK), (iii) `osiris` Windows side with native NVIDIA OpenCL + Vulkan — the access route to the Windows side (Windows Python over SSH, or WSL interop) is itself verified here. **Abort criterion:** if the Halide Vulkan or Metal target fails the spike, switch to route (b): OpenCL C kernels + `clspv` SPIR-V (Metal then reachable via MoltenVK only), and re-estimate before continuing | findings into `docs/dev/portable-backend-notes.md` |
| 4V.2 | Implement the spatial components once in the chosen tool: blur-downsample, Laplacian pyramid build/reconstruct (reflect1 borders exactly matching the CPU baseline — border handling is where ports silently diverge), color conversion, gain/attenuation, quantize | new `src/evm/portable/` |
| 4V.3 | Temporal filters with no FFT dependency: ideal bandpass as the two-matrix-multiplication band projection (exact; see section 3b), IIR and Butterworth as per-pixel recursions serial in time | `src/evm/portable/filters` |
| 4V.4 | Implement the section-3c Ops protocol from the tier's kernels, and override the Pipeline protocol with whole-pipeline scheduled compilations (op-by-op execution would forfeit the fusion benefit); register as `"opencl"` / `"vulkan"` / `"metal"` with capability flags; device enumeration printed; decide ahead-of-time-compiled pipeline libraries vs JIT in the spike | `src/evm/portable/`, `src/evm/api.py` |
| 4V.5 | Conformance: this backend runs the shared parameterized suite of section 3c (`tests/backend_conformance/`) with its own documented tolerance table — every commit on hosted CI via PoCL + lavapipe (no GPU needed); release gate on real GPUs (Mac native Metal + MoltenVK, `osiris` Windows side). Ideal-bandpass band projection additionally unit-tested against `evm.cpu.filters.ideal_bandpass` to 1e-5 on all shipped presets | `tests/backend_conformance/` |
| 4V.6 | Docs: backend matrix updated; a "how new hardware becomes supported" page stating the three mechanisms and the honest limit (devices with no standard driver are unreachable by any general approach) | `docs/backends.md` |

**Success criteria**
- The same synthetic-clip parity suite passes under five executions: OpenCL-on-PoCL (CI), Vulkan-on-lavapipe (CI), native Metal on this Mac, Vulkan-on-MoltenVK (this Mac), OpenCL and Vulkan on the RTX 3090 (`osiris`, Windows side).
- `evm.magnify(arr, preset="pulse", backend="vulkan")` — and `backend="metal"` on the Mac — produce output passing the parity tolerance, end to end, with zero NVIDIA-specific and zero PyTorch code in the path.
- The portable tier has no FFT library dependency and no PyTorch dependency.
- CI runs the portable parity suite on every commit without any GPU.

## Phase 5 — Documentation site
**Complexity: Medium. Effort: 6–8 days. Depends: 3+4 for API reference; conceptual pages can start after 1. Parallel with: 6.**

1. `mkdocs.yml` + Material + mkdocstrings; four-quadrant nav (Tutorial / How-to / Reference / Explanation); Pages workflow builds `site/`.
2. Landing page preserved as Material template override; media → `docs/assets/`.
3. Getting-started (install CPU/CUDA/Colab, 5-line first result); **three task recipes** with downloadable samples: pulse/vitals (face/baby/wrist), vibration/modal (guitar), motion (baby/shadow/camera); concepts pages (4-stage pipeline, choosing α/λc/band, why edge bands are zeroed, pitfalls).
4. API reference from existing docstrings; performance page importing `benches/bench_*.json` with the honest caveats; comparison page vs PyEVM/eulerian-magnification/euler_vid_mag/MATLAB across maintained/GPU/streaming/correctness-tested/components; gallery; three runnable example notebooks incl. a DLPack+torch one.
5. `cuda/DESIGN.md` + two blog posts under `docs/internals/` with redirect stubs; doc CI: `mkdocs build --strict` + executing every Python snippet on CPU-only runners; README top rewrite (positioning, install, 5-line example, comparison table).

**Success criteria:** strict build, zero broken links; every doc snippet executes in CI; a newcomer gets a magnified `face.mp4` from a cold machine in under 10 minutes; old Pages URLs still resolve.

## Phase 6 — Release engineering
**Complexity: Low. Effort: 2–3 days. Depends: 1, 2. Parallel with: 5.**

1. Semver + documented two-tier stability promise (`docs/stability.md`).
2. `CHANGELOG.md` (Keep a Changelog) backfilled; single-source version via `importlib.metadata`.
3. `CITATION.cff` + Zenodo DOI; align README BibTeX.
4. `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, issue/PR templates (incl. GPU-result report with nvidia-smi + benches JSON).
5. `release.yml`: tag → sdist → TestPyPI → PyPI (trusted publishing), GPU suite as required prior job; apply D2 license decision; cut **0.2.0**.

**Success criteria:** `pip install evm-cuda` from real PyPI works CPU-only and compiles on the 3090; `evm.__version__` matches the tag.

## Phase 7 — Live streaming (flagship)
**Complexity: High. Effort: 8–12 days. Depends: 3, 4.**

1. New CUDA kernel `iir_step`: one time-step of the r1/r2 recursion with externally held `y1`,`y2` state (recursion at `evm/filters.py:117-120`, state init `y1=y2=x[0]` — must match exactly). `butter_step` equivalent.
2. `MotionStream`: all device buffers allocated once at construction; `push(frame)->frame`; reuses batched pyramid ops with `n=1`.
3. **Streaming-vs-batch equivalence test — the correctness anchor**: T frames through `MotionStream` equals the offline IIR pipeline to near-bit-exactness. CPU `MotionStream` mirror so streaming logic is CI-testable without GPU.
4. **Color streaming**: ideal FFT filter is non-causal and cannot stream. Ship `ColorStream(filter="butter")` (causal default) and `filter="ideal_window"` (sliding FFT ring buffer, latency = window/2). Document loudly this is not the MIT reference algorithm; reference tests never route through it.
5. Capture/sink layer (`cv2.VideoCapture`: webcam/file/RTSP, frame-drop policy); `evm-magnify stream --source 0 --preset motion --display`; latency/throughput harness (p50/p95); docs page + webcam demo GIF.

**Success criteria:** ≥30 fps at 720p on the 3090 with published p95 latency; `max|stream − batch| < 1e-5` over 120 frames; zero device allocation after the first 3 frames (via `gpu_mem_info()`); offline reference tests untouched.

## Phase 8 — Visibility
**Complexity: Low. Effort: 3–5 days spread out. Depends: 5, 6.**

1. PyPI page polish; GitHub topics (`eulerian-video-magnification`, `motion-magnification`, `photoplethysmography`, `remote-ppg`, `vibration-analysis`, `cuda`, ...).
2. README positioning with comparison table above the fold.
3. Announcements: r/computervision, Hacker News (lead with the streaming demo, not the speedup), X/LinkedIn threads, rPPG/pyVHR community.
4. PRs/issues on awesome-lists and on the stale PyEVM/eulerian-magnification repos offering this as a maintained successor.
5. **No Journal of Open Source Software submission.** Removed by the D2 decision of 2026-08-09: that journal requires an OSI-approved licence, and the non-commercial restriction stays. Use a Zenodo record for a citable identifier instead. Courtesy note to the MIT authors still applies.

**Success criteria:** rendered package-index page; first search result for "eulerian video magnification python" within a quarter; a Zenodo record minted so the work is citable.

## Phase 9 — Phase-based motion magnification (SIGGRAPH 2013)
**Complexity: Very high. Effort: 20–30 days. Depends: 1, 3, 4. Independent of 5–8.**

1. CPU reference first: complex steerable pyramid (`src/evm/cpu/csp.py`), frequency-domain construction.
2. Validate against the authors' published `phaseAmplify` MATLAB outputs before any kernel is written; extend the sample downloader with the 2013 reference clips. **If the RMSE bar (<0.05) is not reached, stop and ship without it.**
3. Phase extraction, temporal filtering of unwrapped phase, amplitude-weighted blur, reconstruction (`src/evm/cpu/phase_magnify.py`).
4. CUDA kernels: cuFFT per frame per orientation (plan-cache entry alongside existing one), filter application, `atan2` phase, weighted blur, recon.
5. Facade wiring (`method="phase"`), docs (when phase beats linear), gallery side-by-side; cut **1.0.0**.

**Success criteria:** CPU matches published reference within RMSE <0.05 (the bar already used for the linear method); CUDA matches CPU within a locked new `TOL` entry; gallery shows artifact reduction vs linear at α=25.

## 5. Dependency graph and timeline

```
Phase 0 → Phase 1 (bottleneck)
   Phase 1 → {Phase 2 ∥ Phase 3}
   Phase 3 → Phase 4 → {Phase 7, Phase 9}
   Phase 3 → Phase 4V (native OpenCL/Vulkan/Metal — primary) ∥ {Phase 5, Phase 6, Phase 7}
   Phase 3 → Phase 4P (optional torch backend — any time, even post-1.0)
   Phase 1 → Phase 5 (concept pages) ; API ref needs 3+4
   Phase 1,2 → Phase 6 ∥ Phase 5
   Phase 5,6 → Phase 8
```

- Credible **0.2.0** release (0→1→3→5/6): **4–5 focused weeks**
- Flagship **0.3.0** with streaming (+4, +7): **8–9 weeks**
- Native portable tier (Phase 4V) adds **20–30 days** and lands as 0.4.x; the optional torch backend (Phase 4P, **8–12 days**) floats freely and may land after 1.0
- **1.0** with phase-based and the native backend tiers: **19–26 weeks** (torch backend not on the critical path)

## 6. Risk register

| Risk | Severity | Mitigation |
|---|---|---|
| scikit-build-core migration breaks Colab/Kaggle/Makefile simultaneously; Colab badge points at `main` | High | Branch; verify all three flows pre-merge; `evm_cuda` shim; fresh-venv install script as required CI job; consider pointing badge at a tag during migration |
| `pip install .` fails when nvcc absent (CMake hard-requires CUDA today) | High | `check_language(CUDA)` guard + `EVM_CUDA_REQUIRE=1`; permanent no-CUDA CI job |
| No GPU in hosted CI — the CUDA suite can't gate PRs | High | Self-hosted 3090 runner; golden CPU fixtures so end-to-end behaviour is CI-verified; GPU suite required on release tags; attestation JSON |
| NC license caps adoption, blocks JOSS | High | D2 decision; if NC stays, delete JOSS and set expectations (ceiling = academic citation) |
| DLPack over pool-backed memory → use-after-free | Med-High | `shared_ptr` ownership (4.1) + lifetime regression test; never ship export without it |
| Array-API refactor silently changes `drop_last` semantics | Med-High | D8 split; explicit equivalence test; Phase 0 golden fixtures |
| Benchmark harness coupled to pipeline internals (`on_stage`) | Medium | Keep hook unchanged through 3–4; document as private; revisit after 1.0 |
| FP16 paths diverge under refactor (four near-duplicate bodies; README accuracy claims load-bearing) | Medium | Unify behind `dtype` param only if FP16-vs-FP32 RMSE tests stay green; else keep duplication |
| Multi-arch build makes install unbearably slow | Medium | `native` default; `all` only for release/bench |
| Docs migration breaks Pages URLs | Medium | Redirect stubs; `mkdocs build --strict` in CI |
| Streaming color filter is not the reference algorithm | Medium | Separate names, loud docs, reference tests never routed through it |
| Phase-based is a research project, not a feature | Medium | CPU-first validation against published outputs before any kernel; abort criterion |
| torch MPS lacks needed ops (historically incomplete `torch.fft`) | Medium | Step 4P.1 probes the exact op set on this Mac before any implementation; FFT-on-CPU or IIR-only fallback chosen from evidence |
| AMD/Intel paths ship untested (no such hardware available) | Medium | Honest "unverified" labels in `docs/backends.md` + an issue template for contributed results; never claim verified support |
| Torch-backend numerics drift from the reference (TF32 defaults, MPS precision) | Medium | Separate documented tolerance tier; TF32 disabled in parity tests; reference-grade claims stay CPU + native CUDA only |
| GPU CI runner lives in WSL2 on a desktop (`osiris`) and dies when the host sleeps or reboots | Low-Medium | Runner is release-gate only, not per-PR; cloud GPU fallback documented in D6 |
| Halide's Vulkan compile target is not mature enough (unverified training-data knowledge) | Medium-High | Phase 4V step 1 is a 3-day time-boxed spike with a named fallback route (OpenCL C + `clspv` SPIR-V) and a re-estimate checkpoint before any further investment |
| Per-driver behavior differences across OpenCL/Vulkan implementations (rounding, workgroup limits, mobile drivers) | Medium | CPU driver implementations (PoCL, lavapipe) as the always-on CI floor; real-device results labeled per driver in `docs/backends.md`; parity tolerances documented per tier |
| Wide temporal bands make the FFT-free band-projection filter expensive (cost grows with band width) | Low-Medium | Shipped presets are narrow-band; docs state the cost model; offline CPU-FFT path remains for wide bands |
| OpenCL/Vulkan testing impossible from WSL2 (only CUDA is passed through); macOS OpenCL is deprecated | Medium | Test on `osiris`'s Windows side (access route verified in the 4V.1 spike); Mac's tier-3 path is Vulkan via MoltenVK, not OpenCL |

## 7. Definition of done

- [ ] `pip install evm-cuda` works everywhere without CUDA (CPU pipeline) and compiles with CUDA (GPU pipeline).
- [ ] `evm.magnify(array, preset="pulse")` with zero file I/O; `evm-magnify` on PATH.
- [ ] GPU components chain with zero host copies; DLPack interop verified by pointer identity.
- [ ] `evm-magnify stream --source 0` ≥30 fps at 720p.
- [ ] `PYTHONPATH` appears nowhere in the repo.
- [ ] CI green on every commit; GPU suite required on every release.
- [ ] Docs build `--strict`; every snippet executes in CI; three task recipes.
- [ ] `CHANGELOG.md`, `CITATION.cff`, `CONTRIBUTING.md`, semver policy, PyPI releases in place.
- [ ] MIT-reference tests and `TOL` unchanged from the Phase 0 lock; golden fixtures pass on every commit.
- [ ] `backend="opencl"`, `backend="vulkan"` and `backend="metal"` (native tier, no PyTorch involved) pass the parity suite on CPU drivers (PoCL, lavapipe) in hosted CI on every commit, and on real GPUs (Mac natively via Metal and via MoltenVK, RTX 3090 on `osiris`'s Windows side) at release; a new device with a standard driver runs the library with zero new code.
- [ ] PyTorch appears only in the optional `[torch]` extra; `import evm` and every native backend work on a machine where torch was never installed. (The torch backend itself is an optional deliverable, allowed to land after 1.0.)
- [ ] Adding a backend requires implementing only the Ops protocol (section 3c); the generic default pipelines plus the shared conformance suite then run against it with no further wiring. Verified by the conformance suite discovering backends through the registry alone.
- [ ] The execution methodology (section 3d) lives in the repository (`CLAUDE.md`, `.claude/rules/development-practices.md`, `docs/dev/PLAN.md`); the git history maps one-to-one onto plan steps with a green suite at every commit.

## Files touched most

- `pyproject.toml` — Phases 1, 2, 6
- `cuda/CMakeLists.txt` — Phase 1 (lines 9, 20-22, 32, 38, 69-72, 81-83)
- `cuda/bindings.cpp` — Phases 4 (lines 357-414, 863-915), 7
- `cuda/evm_cuda/batched.py` — Phases 1, 3, 4
- `evm/magnify.py` — Phase 3 (split at lines 144, 217, 349, 382)
- `evm/video.py` + `shared/h264.py` — Phase 1 merge
- `Makefile` — Phase 1 (lines 25, 37-41)
- `tests/cuda/conftest.py` — Phases 0, 1
- `colab/evm_cuda_benchmark.ipynb`, `kaggle/run_gpu_comparison.py` — Phase 1
- `.github/workflows/deploy-pages.yml` — Phase 5
- `LICENSE`, `README.md` — Phase 6 (pending D2)

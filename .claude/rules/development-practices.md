# Development Practices (binding)

These rules govern **how** every step of `docs/dev/PLAN.md` is implemented. They are
section 3d of that plan, committed here so every session loads them. They are not advice.

## 1. Test-driven — the check comes before the thing it checks

- Start each plan step by writing, or pointing at, the failing check named in that phase's
  success criteria. No implementation file is created before that check exists.
- The check is a real command, not a claim. Examples in this repo:
  `.venv/bin/python -m pytest tests/ -q -p no:randomly` (300 passed, 102 skipped on 2026-08-18),
  `pytest tests/cuda/ -q` on a GPU host, `scripts/dev/verify_install.sh` for packaging work.
- Non-Python work gets a check too: a `src/vidmag/cuda/CMakeLists.txt` change is tested by a scripted
  fresh-venv install, a docs change by `mkdocs build --strict` plus running its snippets.
- Phase 0 exists to build this net (golden fixtures, `tests/test_reference_lock.py`, recorded
  baselines in `benches/`). Later red-green cycles run inside it.

## 2. KISS — two sanctioned abstraction layers, no third

- The only abstractions this restructure may introduce are `vidmag.backend.Ops` and
  `vidmag.backend.Pipelines` (plan section 3c) plus the backend registry. Base classes, config
  objects, plugin hooks or dispatch layers beyond those need the operator's approval first.
- Prefer a plain function to a class, a frozen dict to a config system, a copy-pasteable
  example to a helper framework. `vidmag/` is written that way — match it.

## 3. YAGNI — build exactly the current step

- No speculative parameters, no "while I'm here" generality, no Phase 4 code smuggled into
  Phase 1. If the plan looks wrong or incomplete, stop and surface the gap to the operator
  instead of quietly building extra.
- Touch only the files your step names. No reformatting or refactoring of neighbouring code.

## 4. DRY — one source of truth, one named exception

- Single-sourced by design: the version string, the preset table, `tests/cuda/conftest.py:TOL`,
  the conformance suite, the backend registry, and the reference constants
  (`DROP_LAST`, `EXAGGERATION_FACTOR`, `BINOM5`, `BINOM5_SUM1`). Import them; never retype a
  literal that can be imported. New duplication needs a stated reason in the commit message.
- **The named exception:** the FP32/FP16 pipeline bodies in `src/vidmag/cuda/batched.py` and the
  templated kernels may stay duplicated where merging them risks numeric drift — the README's
  accuracy claims (motion FP16 vs FP32 RMSE 0.00140) are load-bearing. Correctness outranks
  DRY. Comment the exception at the site.

## 5. Fail loud

- No silent fallbacks and no silently skipped work. Backend selection is always printed; a
  missing backend reports *why* (missing extra, no driver, no device). A ~700x CPU/GPU cliff
  must never be reached by accident.
- No bare `except`. Every CUDA runtime call is wrapped in `CUDA_CHECK`
  (`src/vidmag/cuda/include/evm_check.cuh`), which throws `std::runtime_error` so pybind11 surfaces it as
  a catchable Python exception — "a silent error can never propagate into a tolerance
  failure". Keep that posture in Python: raise, don't degrade.
- Report skip counts, never just passes. On this Mac the honest line is
  "300 passed, 102 skipped" — those skips include the whole NVIDIA suite and prove
  nothing about it. No single machine runs every backend.
- `tests/cuda/conftest.py:TOL` and `tests/test_against_mit_reference.py` are append-only.
  Loosening a tolerance is its own separately reviewed commit carrying the measurement.

## 6. Verify, then claim; land one plan step per commit

- A step is done only when its named command has actually run and its output is recorded in
  the report. "Should work" and "looks right" are not results.
- One plan step = one commit, full suite green, the step named in the message
  (e.g. `Phase 1 step 1.6: make CUDA optional in CMake`), so history maps one-to-one onto
  `docs/dev/PLAN.md`. Unrelated changes in that commit are not permitted.
- Do not commit unless the operator asked for it; the orchestrating session commits.

## 7. The NVIDIA GPU code leads; the other backends follow it

- Performance work starts in the NVIDIA GPU code — `src/vidmag/cuda/kernels/*.cu` and `src/vidmag/cuda/bindings.cpp`.
  That is what this project exists to make fast, and it is where a measurement is worth taking.
- **A change to the NVIDIA code is not finished when it lands.** Once it is accepted, check
  whether the same change applies to every other backend, and apply it where it does:

  | Backend | Where its kernels live |
  |---|---|
  | OpenCL | `src/vidmag/opencl/kernels.cl` |
  | Apple Metal | `src/vidmag/metal/kernels.metal` |
  | Vulkan | `src/vidmag/vulkan/shaders/*.comp` |
  | PyTorch | `src/vidmag/torch_backend/ops.py` (tensor operations, no kernels) |
  | Processor (Python) | `src/vidmag/cpu/pyramids.py`, `src/vidmag/cpu/filters.py` |

- State which backends the change was carried to and which it does not apply to, with the
  reason. "Does not apply, because that backend has no separate upsample kernel" is a complete
  answer; saying nothing is not.
- Carry it in its own commit, separate from the NVIDIA change, so a numeric regression on one
  backend cannot be mistaken for a fault in the original.

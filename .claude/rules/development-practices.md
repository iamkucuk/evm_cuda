# Development Practices (binding)

These rules govern **how** every step of `docs/dev/PLAN.md` is implemented. They are
section 3d of that plan, committed here so every session loads them. They are not advice.

## 1. Test-driven — the check comes before the thing it checks

- Start each plan step by writing, or pointing at, the failing check named in that phase's
  success criteria. No implementation file is created before that check exists.
- The check is a real command, not a claim. Examples in this repo:
  `.venv/bin/python -m pytest tests/ -q -p no:randomly` (48 passed, 63 skipped on 2026-08-09),
  `pytest tests/cuda/ -q` on a GPU host, `dev/verify_install.sh` for packaging work.
- Non-Python work gets a check too: a `cuda/CMakeLists.txt` change is tested by a scripted
  fresh-venv install, a docs change by `mkdocs build --strict` plus running its snippets.
- Phase 0 exists to build this net (golden fixtures, `tests/test_reference_lock.py`, recorded
  baselines in `benches/`). Later red-green cycles run inside it.

## 2. KISS — two sanctioned abstraction layers, no third

- The only abstractions this restructure may introduce are `evm.backend.Ops` and
  `evm.backend.Pipelines` (plan section 3c) plus the backend registry. Base classes, config
  objects, plugin hooks or dispatch layers beyond those need the operator's approval first.
- Prefer a plain function to a class, a frozen dict to a config system, a copy-pasteable
  example to a helper framework. `evm/` is written that way — match it.

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
- **The named exception:** the FP32/FP16 pipeline bodies in `cuda/evm_cuda/batched.py` and the
  templated kernels may stay duplicated where merging them risks numeric drift — the README's
  accuracy claims (motion FP16 vs FP32 RMSE 0.00232) are load-bearing. Correctness outranks
  DRY. Comment the exception at the site.

## 5. Fail loud

- No silent fallbacks and no silently skipped work. Backend selection is always printed; a
  missing backend reports *why* (missing extra, no driver, no device). A ~700x CPU/GPU cliff
  must never be reached by accident.
- No bare `except`. Every CUDA runtime call is wrapped in `CUDA_CHECK`
  (`cuda/include/evm_check.cuh`), which throws `std::runtime_error` so pybind11 surfaces it as
  a catchable Python exception — "a silent error can never propagate into a tolerance
  failure". Keep that posture in Python: raise, don't degrade.
- Report skip counts, never just passes. On this Mac the honest line is
  "48 passed, 63 skipped" — the 63 are the whole CUDA suite and prove nothing.
- `tests/cuda/conftest.py:TOL` and `tests/test_against_mit_reference.py` are append-only.
  Loosening a tolerance is its own separately reviewed commit carrying the measurement.

## 6. Verify, then claim; land one plan step per commit

- A step is done only when its named command has actually run and its output is recorded in
  the report. "Should work" and "looks right" are not results.
- One plan step = one commit, full suite green, the step named in the message
  (e.g. `Phase 1 step 1.6: make CUDA optional in CMake`), so history maps one-to-one onto
  `docs/dev/PLAN.md`. Unrelated changes in that commit are not permitted.
- Do not commit unless the operator asked for it; the orchestrating session commits.

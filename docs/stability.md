# What this project promises not to break

A version number only means something if there is a written-down list of what
it covers. This is that list.

## Covered by the version number

These are checked by `tests/test_public_api.py`, so removing one fails
immediately rather than being discovered by whoever depended on it.

- Everything importable as `from evm import ...`: `magnify`, the four
  `magnify_*` pipeline functions, the presets, video reading and writing, the
  building blocks, and the reference constants.
- `evm.backend`: the operations protocol, the pipelines protocol, and the
  registry functions for selecting and registering a backend.
- `evm.cpu.ops` and `evm.cuda.ops`: the building blocks, with the same names and
  argument order on every backend.
- `evm.cuda.DeviceArray`: its shape, dtype, transfer methods, and its support
  for sharing memory with other array libraries.
- The `evm-magnify` command's subcommands and their options.
- The names in `evm.presets.PRESETS`. Their *values* may be corrected if one is
  found to disagree with the reference implementation; such a change is
  described in the changelog with the measurement behind it.

## Not covered

Anything below may change in any release.

- `evm.cuda._evm_cuda`, the compiled module. Its function names, signatures and
  memory layouts are internal. Use `evm.cuda.ops` instead.
- Anything whose name begins with an underscore.
- `evm.cuda.batched` and `evm.cuda.pipelines` internals, including the
  `on_stage` hook. It exists for the benchmark harness.
- Exact numerical output. Fixing a defect changes results, and correctness comes
  first; see below.
- The tolerances in `tests/cuda/conftest.py`.

## Numerical results

The project's purpose is agreement with the method as published. If a
difference from the reference implementation is found, it is fixed, and output
changes as a result. Such a change is always described in the changelog,
including what was wrong and how the fix was verified.

Tolerances are treated as ratchets: they may be tightened freely, and loosening
one requires its own separately reviewed change carrying the measurement that
justifies it.

## Versions

This project follows semantic versioning.

- **Major**: something in the covered list was removed or changed
  incompatibly.
- **Minor**: something was added, or numerical output changed because a defect
  was fixed.
- **Patch**: fixes that change no covered behaviour.

Before version 1.0, the covered surface may still change between minor
versions. Each such change is listed in the changelog.

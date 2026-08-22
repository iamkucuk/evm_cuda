# What this project promises not to break

A version number means something only if there is a written list of what it
covers. This is that list.

## Covered by the version number

Checked by `tests/test_public_api.py`, so removing one fails immediately rather
than being discovered by whoever depended on it.

- Everything importable as `from vidmag import ...`: `magnify`, the four
  `magnify_*` pipeline functions, the presets, video reading and writing, the
  building blocks, and the reference constants.
- `vidmag.backend`: the operations protocol, the pipelines protocol, and the
  registry functions for selecting and registering a backend.
- `vidmag.cpu.ops` and `vidmag.cuda.ops`: the building blocks, same names and
  argument order on every backend.
- `vidmag.cuda.DeviceArray`: its shape, dtype, transfer methods, and its support
  for sharing memory with other array libraries.
- The `vidmag` command's subcommands and their options.
- The names in `vidmag.presets.PRESETS`. Their *values* may be corrected if one
  is found to disagree with the reference, described in the changelog with the
  measurement behind it.

## Not covered — may change in any release

- `vidmag.cuda._vidmag_cuda`, the compiled module. Its function names,
  signatures and memory layouts are internal; use `vidmag.cuda.ops` instead.
- Anything whose name begins with an underscore.
- `vidmag.cuda.batched` and `vidmag.cuda.pipelines` internals, including the
  `on_stage` hook, which exists for the benchmark harness.
- Exact numerical output. Fixing a defect changes results, and correctness comes
  first.
- The tolerances in `tests/cuda/conftest.py`.

## Numerical results

The project's purpose is agreement with the method as published. If a difference
from the reference is found, it is fixed, and output changes as a result — always
described in the changelog, including what was wrong and how the fix was
verified. Tolerances are ratchets: they may be tightened freely, and loosening
one requires its own separately reviewed change carrying the justifying
measurement.

## Versions

Semantic versioning:

- **Major** — something in the covered list was removed or changed incompatibly.
- **Minor** — something was added, or numerical output changed because a defect
  was fixed.
- **Patch** — fixes that change no covered behaviour.

Before version 1.0 the covered surface may still change between minor versions.
Each such change is listed in the changelog.

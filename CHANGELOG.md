# Changelog

Notable changes to this project. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the version
numbers follow [semantic versioning](https://semver.org/) under the policy in
[what this project promises not to break](https://iamkucuk.github.io/eulerian-video-magnification-cuda/stability/).

## Unreleased

### Added

- `evm.magnify()`: one entry point taking a file path, an array of frames, or
  any iterable of frames, with named presets for common jobs and an
  `evm-magnify` command that does the same from a terminal.
- A portable backend written in OpenCL, running on Apple, AMD and Intel
  graphics processors as well as NVIDIA. On an Apple M2 Max, which previously
  had no acceleration at all, the colour pipeline runs 28 times faster than on
  the processor cores and the motion pipeline 19 times faster.
- A backend interface and registry. Implementing about a dozen primitive
  operations gives a new device all four pipelines, and the choice of backend
  is always reported and never silently changed.
- The building blocks as public API on every backend: pyramids, the three
  temporal filters, colour conversion and gain, under the same names
  everywhere.
- `evm.cuda.DeviceArray`: a GPU array that knows its own shape and dtype and
  can hand its memory to PyTorch or CuPy without copying.
- A documentation site organised by task, with every example executed by the
  test suite.
- Continuous integration across operating systems and Python versions, a
  linter, a type checker, and a graphics-processor suite that runs before a
  release.

### Fixed

- The build silently ignored its own list of graphics architectures, because
  the value was set after the point at which it is read. Every build targeted
  hardware from 2014 and failed to compile a half-precision instruction on a
  current toolkit.
- `pip install .` did not work at all: the project had eight top-level
  directories and the packaging tool refused to guess which were packages.
- The frequency filter could silently do nothing. When the requested band is
  narrower than a short clip can resolve, it selects no frequency at all and
  returns its input unchanged. It now says so, with the number of frames that
  would be needed. This also revealed two tests that were passing for no
  reason, including one comparing two all-zero arrays.
- The device-resident Butterworth filter had a kernel and a launcher but was
  never exposed, so that path had never been compared against the reference.
- The pyramid reconstruction in the NumPy operations module called its helper
  with the wrong arguments, in code no test exercised.

### Changed

- The package moved to a `src/` layout under one root package, with
  `evm.cpu`, `evm.io` and `evm.cuda` as subpackages. Importing `evm_cuda` still
  works and warns.
- The licence now states its permitted uses explicitly — research, teaching,
  personal use, and inclusion in freely distributed open-source software —
  rather than leaving them to be inferred. The non-commercial restriction is
  unchanged.

## 0.1.0

The original research implementation: a NumPy version of the method checked
against the original authors' published output, and a CUDA port of the same
four pipelines checked against that NumPy version.

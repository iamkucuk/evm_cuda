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

- The motion pipeline runs 1.9 times faster in 32-bit and 2.3 times faster in
  16-bit on NVIDIA hardware, measured on an RTX 3090 against the same code with
  the changes removed: computation time falls from 76.7 ms to 40.5 ms in
  32-bit and from 60.9 ms to 27.0 ms in 16-bit. Three causes. The temporal
  filter kept its running state in 64-bit
  floating point, which this class of card executes at a sixty-fourth of
  single-precision speed; the filter is a pair of decaying averages, which do
  not accumulate error the way the original reasoning assumed, and 32-bit state
  differs from 64-bit by at most 4.023e-07 against an allowed 1e-5. Separately,
  building and reconstructing an image pyramid wrote a full-resolution
  intermediate result and immediately read it back to add or subtract a
  detail level; that combination now happens inside the write that was already
  taking place. Third, the two kernels that enlarge an image staged their input
  in fast on-chip memory, which cost more than it saved: an enlarged pixel reads
  only two or three inputs and neighbouring threads overlap, so the ordinary
  cache already served them. Removing the staging and giving each thread four
  output pixels took those two kernels from 55% of what the card's memory can
  sustain to between 92% and 96%, with bit-identical output. The colour pipeline
  is unchanged by all three, as it has no image pyramid.
- Agreement between 16-bit and 32-bit output improved as a side effect, since
  the intermediate result is no longer rounded to half precision and read back:
  motion RMSE falls from 0.00232 to 0.00199, with the largest single-level
  difference unchanged at 5.
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

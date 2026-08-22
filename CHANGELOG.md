# Changelog

Notable changes to this project. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the version
numbers follow [semantic versioning](https://semver.org/) under the policy in
[what this project promises not to break](https://iamkucuk.github.io/eulerian-video-magnification-cuda/stability/).

## 0.2.0 — 2026-08-22

### Added

- `vidmag.magnify()`: one entry point taking a file path, an array of frames, or
  any iterable of frames, with named presets for common jobs and a
  `vidmag` command that does the same from a terminal.
- A portable backend written in OpenCL, running on Apple, AMD and Intel
  graphics processors as well as NVIDIA. On an Apple M2 Max, which previously
  had no acceleration at all, the colour pipeline runs 28 times faster than on
  the processor cores and the motion pipeline 19 times faster.
- A PyTorch backend, as the optional `torch` extra. It reaches no hardware the
  other backends miss, and is not how this project supports any vendor — it is
  for people who already have PyTorch set up, and for keeping results as tensors
  so magnification can sit inside a larger tensor computation. Both pipelines
  and the streaming path agree with the NumPy baseline to within one step of the
  8-bit output. On an Apple M2 Max it runs the colour clip in 653 ms and the
  motion clip in 2,320 ms, against 7,014 ms and 23,634 ms on the processor —
  slower than the three graphics backends, which is why it sits last in the
  selection order. Nothing imports PyTorch unless this backend is asked for.
- A backend interface and registry. Implementing about a dozen primitive
  operations gives a new device all four pipelines, and the choice of backend
  is always reported and never silently changed.
- The building blocks as public API on every backend: pyramids, the three
  temporal filters, colour conversion and gain, under the same names
  everywhere.
- `vidmag.cuda.DeviceArray`: a GPU array that knows its own shape and dtype and
  can hand its memory to PyTorch or CuPy without copying.
- A documentation site organised by task, with every example executed by the
  test suite.
- Continuous integration across operating systems and Python versions, a
  linter, and a type checker, all running on every commit. A graphics-processor
  test suite is written and every one of its commands has been run by hand on an
  RTX 3090, but the workflow itself has never executed: it needs a self-hosted
  runner, and none is registered. Treat the NVIDIA backend as tested by hand
  rather than automatically.

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
  Confirmed on three further architectures rather than assumed to generalise,
  each measured on the same hardware before and after. A Tesla P100 (Pascal,
  2016) runs the 16-bit motion pipeline in 82.8 ms against 139.7 ms, 1.7 times
  faster, with colour flat at 26.4 and 21.9 ms, so that pair is controlled. An
  H100 (Hopper, 2022) gives 13.8 ms against 34.5 ms, 2.5 times. A Tesla T4
  (Turing, 2018) gives 137.2 ms against 228.8 ms, 1.7 times. On the T4 and the
  H100 the colour control did not hold still — it moved 12% and 15% — which is
  the machine rather than the code, since the colour kernels are byte-identical
  across the H100 pair. On both, the motion change is far outside that movement,
  so the direction holds everywhere while the exact factor is trustworthy only
  on the RTX 3090 and the P100. An A100 could not be re-measured: the cluster
  partition holding those cards was down.
- Agreement between 16-bit and 32-bit output improved as a side effect, since
  the intermediate result is no longer rounded to half precision and read back:
  motion RMSE falls from 0.00232 to 0.00140 of full scale, and the largest
  single-level difference from 5 to 2. The colour pipeline is unchanged at
  0.00071 and 1 level.
- The OpenCL, Apple and Vulkan backends no longer evaluate the filter taps that
  cannot contribute when enlarging an image. Enlarging inserts a gap between
  every pair of samples, so only taps landing on a real sample carry data —
  three of five, or two — and which ones they are is fixed by whether the output
  row or column is odd. They were all being evaluated and then discarded by a
  test inside the loop. Measured on an Apple M2 Max, pyramid reconstruction runs
  about 8% faster on Vulkan and about 3% faster on Apple's graphics interface,
  with no measurable change on OpenCL. Pyramid construction is unaffected on all
  three, being dominated by the shrinking step, which this does not touch.
- One word names all three surfaces: `pip install vidmag`, `import vidmag`,
  and a `vidmag` command. The project was called `evm-cuda`, and briefly
  `evm-magnify`, while this branch was in progress; neither was ever published,
  so no install anywhere refers to them. The optional extras move with the name,
  so `pip install "vidmag[opencl]"` is the current spelling. The import root
  could not stay `evm` even if the distribution name had: `evm` on PyPI is the
  Extreme Value Machine, and it installs a top-level module spelled `EVM`, which
  collides with `import evm` on macOS and Windows because their filesystems
  ignore case.
- The package moved to a `src/` layout under one root package, with `vidmag.cpu`,
  `vidmag.io`, `vidmag.cuda` and the other backends as subpackages. A compatibility
  alias named `evm_cuda` existed briefly during that move and has been removed:
  the old name was only ever reachable inside this repository through a
  `PYTHONPATH` setting, the distribution was never published under it, so there
  is no installed copy anywhere for it to keep working.
- The licence now states its permitted uses explicitly — research, teaching,
  personal use, and inclusion in freely distributed open-source software —
  rather than leaving them to be inferred. The non-commercial restriction is
  unchanged.

## 0.1.0

*Never published to the package index; this section records what existed in the
repository before the work above. The first release on the index is 0.2.0.*

The original research implementation: a NumPy version of the method checked
against the original authors' published output, and a CUDA port of the same
four pipelines checked against that NumPy version.

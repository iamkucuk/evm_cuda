# Releasing

What is prepared, and what only the project owner can do.

## Prepared and automated

| Step | Where |
|---|---|
| Tests across operating systems and Python versions | `.github/workflows/ci.yml`, on every push |
| Linter, formatter, type checker | `.github/workflows/lint.yml`, on every push |
| Documentation site built and its examples executed | `.github/workflows/ci.yml`, on every push |
| Graphics-processor suite | `.github/workflows/gpu.yml`, manual and on release |
| Build, verify from the archive alone, publish | `.github/workflows/release.yml`, on a version tag |
| What the version number covers | `docs/stability.md` |
| What changed | `CHANGELOG.md` |
| How to cite it | `CITATION.cff` |

Tagging is what starts a release:

```bash
git tag v0.2.0
git push origin v0.2.0
```

The publishing job builds a source archive, installs it in a clean environment
from that archive alone, runs the tests there, and only then publishes. That
order is deliberate: installing from a checkout cannot catch a file missing
from the archive.

## Only the owner can do these

**Register the graphics-processor runner.** `.github/workflows/gpu.yml` has
never executed, because no self-hosted runner exists. Every command in it has
been run by hand on the machine, but the workflow itself is unproven until a
runner is registered. See `docs/dev/gpu-runner.md`.

**Set up trusted publishing on the package index.** The release workflow uses
it, so no token is stored anywhere. It has to be configured once, against this
repository, on the index's own website.

**Add repository topics.** These are how people find a project by subject.
Suggested: `eulerian-video-magnification`, `motion-magnification`,
`video-magnification`, `photoplethysmography`, `remote-ppg`,
`vibration-analysis`, `computer-vision`, `opencl`, `cuda`.

**Mint a citation identifier.** Connecting the repository to Zenodo gives each
release a permanent identifier, which is what makes it properly citable. A
journal paper is not an option here: the ones that would take this require a
licence approved by the Open Source Initiative, and the non-commercial
restriction rules that out.

**Announce it, if you want to.** Nothing has been posted anywhere. Places where
this would be on topic: communities around computer vision, remote vital-sign
measurement, and vibration analysis. Two things worth leading with, because
they are what is genuinely unusual here: it runs on hardware that is not
NVIDIA, and every result is checked against the original authors' published
output.

## Before tagging a release

- [ ] `CHANGELOG.md` describes what changed, including any defect that altered
      numerical output
- [ ] The version in `pyproject.toml` matches the tag
- [ ] The graphics-processor suite has been run and reports zero skips
- [ ] `bash scripts/dev/verify_install.sh` passes
- [ ] Any new public name is listed in `tests/test_public_api.py`

## Known gaps at the time of writing

Stated so nobody has to rediscover them.

- **No AMD or Intel graphics processor has run this.** The OpenCL kernels
  should work there, since the driver compiles them, but nobody has tried. The
  documentation says "expected" rather than "supported", and there is an issue
  template for contributed results.
- **Windows is untested** and deliberately absent from the test matrix.
- **The phase-based method is not compared against the authors' output**,
  because that output is not among the files this project can fetch. It is
  checked against constructed motion instead, and every page describing it says
  so.
- **The phase-based method runs on the processor only.**
- **Streaming is best on the processor**, not on a graphics processor, because
  magnifying one frame at a time is dominated by the cost of launching work.
  Measured numbers are in the streaming page.

# Contributing


## Documentation

The site is built with mkdocs-material from `mkdocs.yml` and the pages under
`docs/`. Build it the way both publishing paths do, which fails on a broken
internal link rather than publishing one:

```bash
pip install ".[docs]"
mkdocs build --strict
mkdocs serve            # live preview at http://127.0.0.1:8000
```

The same pages are published twice, from the same configuration:

| Where | Built by | When | What it is for |
|---|---|---|---|
| GitHub Pages | `.github/workflows/deploy-pages.yml` | every push to the default branch | the current documentation, and the address the project links to |
| Read the Docs | `.readthedocs.yaml` | every pull request and every tag | previewing a change before it merges, and keeping each released version's pages available after later ones supersede them |

Both install the package itself, not only the documentation tools, because the
API reference pages are generated from the code's own docstrings and are empty
without it.

Examples in the pages are executed by `tests/test_documentation.py`. An example
that stops working fails the test suite, which is deliberate: a documented
example that no longer runs is worse than no example.

## What this project is trying to be

An implementation of a published method that you can check. The NumPy version
is compared against the original authors' own rendered output; both graphics
implementations are compared, operation by operation, against that NumPy
version. Every change has to keep that chain intact, because a magnification
nobody has verified is a picture rather than a measurement.

## Getting set up

```bash
git clone https://github.com/iamkucuk/eulerian-video-magnification-cuda
cd eulerian-video-magnification-cuda
pip install -e ".[dev,lint]"
python -m pytest tests/ -q
```

The suite reports skips as well as passes, and the skips matter: on a machine
without a graphics processor, the entire hardware-comparison suite skips, and a
green run there says nothing about that hardware. Report both numbers when you
describe a test run.

For the portable backend add `.[opencl]`; for the documentation site add
`.[docs]`.

## Before you open a change

```bash
python -m pytest tests/ -q      # tests
ruff check .                    # the linter
ruff format --check .           # the formatter, which also checks the code
                                # blocks inside the documentation
mypy                            # types
mkdocs build --strict           # if you touched the documentation
```

All five are enforced automatically. Run the formatter check as well as the
linter: they are separate commands and pass independently, and it is easy to
satisfy one while failing the other.

## What is expected of a change

**A test that would have failed before it.** Not a test that exercises the new
code, one that fails without it. If you cannot write that, say so and explain
why.

**A test that cannot pass for the wrong reason.** This project has repeatedly
found tests that held because both sides were zero, or because the reference
returned its input unchanged. Where a comparison could become vacuous, assert
that it has not: check the reference actually did something.

**Only what the change needs.** No unrelated reformatting, no speculative
options, no "while I was in there".

**Numbers you measured.** "Should be faster" is not a result. Say what you ran,
on what, and what it printed.

## Adding support for new hardware

Implement the operations in `vidmag/backend/ops.py` for the device, then
`generic.bind(your_operations)` gives all four pipelines. Add your backend to
the conformance tests, which compare every operation against the NumPy
reference. There is no need to write a pipeline; a backend may override one for
speed, as the CUDA backend does, but correctness must come from the shared
implementation first.

## Things that need their own separate change

- **Loosening a tolerance** in `tests/cuda/conftest.py`. Include the
  measurement that justifies it.
- **Changing a preset's numbers.** Include what you compared against.
- **Anything touching the licence.**

## Reporting a result from hardware nobody here has

Particularly welcome: this project has no AMD or Intel graphics processor to
test on, and says so rather than claiming support it has not verified. If you
run the suite on one, please open an issue with the device name, the driver, and
the output of `python -m pytest tests/ -q`.

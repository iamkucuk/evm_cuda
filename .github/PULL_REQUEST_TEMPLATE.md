## What this changes, and why

<!-- What the change does, and the problem it solves. If it fixes an issue, link it. -->

## How it was verified

<!-- The commands you actually ran, with their output. Not "tests pass" — the
     numbers. This project reports skip counts alongside pass counts, because no
     single machine can run every backend: a machine without an NVIDIA card
     skips that whole suite, and a machine with one has no Apple or OpenCL
     driver. Say which machine you ran on. -->

- [ ] `python -m pytest tests/ -q -p no:randomly` — passed / skipped:
- [ ] `ruff check .` and `ruff format --check .`
- [ ] `mypy`
- [ ] Ran on (operating system, and graphics hardware if relevant):

## For performance changes

<!-- Delete if not applicable. -->

- [ ] Measured, with the before and after numbers and how they were taken
- [ ] Said which backends the change reached, and which it does not apply to
      and why (see rule 7 in `.claude/rules/development-practices.md`)

## For changes that alter numerical output

<!-- Delete if not applicable. -->

- [ ] Compared against the NumPy baseline, which is the correctness oracle
- [ ] No tolerance in `tests/cuda/conftest.py` was loosened. Loosening one is a
      separate pull request carrying the measurement that justifies it.

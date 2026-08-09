"""Frozen constants that define correctness against the MIT MATLAB reference.

Everything asserted here is compared to a *literal* value written into this
file, never to an imported expression. That is the whole point: a refactor
that rewrites ``BINOM5``'s derivation, retunes a tolerance, or "cleans up"
``DROP_LAST`` must fail here immediately and loudly, instead of silently
shifting the oracle the rest of the suite is measured against.

Covered:

* :data:`evm.cpu.magnify.DROP_LAST` and :data:`evm.cpu.magnify.EXAGGERATION_FACTOR`
  — the two magic numbers hardcoded in the four MATLAB amplification scripts.
* :data:`evm.cpu.pyramids.BINOM5` / :data:`evm.cpu.pyramids.BINOM5_SUM1` — the
  matlabPyrTools binom5 filter in both of its normalizations, element by
  element plus the two documented sum properties.
* ``TOL`` from ``tests/cuda/conftest.py`` — the per-stage CPU-vs-CUDA
  tolerance table. Loosening an entry is allowed only as a separate, reviewed
  commit that also updates the literal below.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from evm.cpu.magnify import DROP_LAST, EXAGGERATION_FACTOR
from evm.cpu.pyramids import BINOM5, BINOM5_SUM1

CUDA_CONFTEST = Path(__file__).resolve().parent / "cuda" / "conftest.py"

# Elementwise equality bar for the filter taps. rtol=0 so the check does not
# get looser as the values grow; atol=1e-15 is below the last bit of a float64
# in this magnitude range, so only a genuine change in the derivation trips it.
ATOL = 1e-15


def _read_tol_from_conftest() -> dict[str, float]:
    """Return the live ``TOL`` table by parsing ``tests/cuda/conftest.py``.

    Parsed rather than imported: ``tests/cuda/conftest.py`` is a pytest
    conftest, so importing it here would execute it a second time under a
    different module name (it attempts to import the compiled
    ``evm.cuda._evm_cuda`` extension and builds pytest markers from the
    result). ``ast`` reads the same live source
    with none of those side effects, and works no matter which directory
    pytest was invoked from. A ``TOL`` that stopped being a literal dict would
    raise here — deliberately, since this lock can only guard literals.
    """
    tree = ast.parse(CUDA_CONFTEST.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "TOL" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"no top-level TOL assignment found in {CUDA_CONFTEST}")


def test_drop_last_is_ten() -> None:
    """startIndex/endIndex in all four MATLAB scripts drop the last 10 frames."""
    assert DROP_LAST == 10


def test_exaggeration_factor_is_two() -> None:
    """The Figure-6 exaggeration factor is hardcoded to 2 in the reference."""
    assert EXAGGERATION_FACTOR == 2.0


def test_binom5_values() -> None:
    """binom5 = sqrt(2) * [1 4 6 4 1] / 16, as buildLpyr uses it."""
    np.testing.assert_allclose(
        BINOM5,
        [
            0.08838834764831845,
            0.3535533905932738,
            0.5303300858899107,
            0.3535533905932738,
            0.08838834764831845,
        ],
        rtol=0,
        atol=ATOL,
    )


def test_binom5_sum1_values() -> None:
    """The sum-normalized form blurDn renormalizes to: [1 4 6 4 1] / 16."""
    np.testing.assert_allclose(
        BINOM5_SUM1,
        [0.0625, 0.25, 0.375, 0.25, 0.0625],
        rtol=0,
        atol=ATOL,
    )


def test_binom5_sums_to_sqrt2() -> None:
    """L2-normalization property: the taps sum to sqrt(2) = 1.4142135623730951."""
    np.testing.assert_allclose(BINOM5.sum(), 1.4142135623730951, rtol=0, atol=ATOL)


def test_binom5_sum1_sums_to_one() -> None:
    np.testing.assert_allclose(BINOM5_SUM1.sum(), 1.0, rtol=0, atol=ATOL)


def test_cuda_tolerance_table() -> None:
    """The whole CPU-vs-CUDA tolerance table, keys and values.

    Compared as a whole dict, so an added, removed or renamed key fails too:
    the table is append-only by policy, and appending to it must be a
    deliberate edit here as well.
    """
    assert _read_tol_from_conftest() == {
        "color_cvt": 1e-6,
        "corr_dn": 1e-5,
        "up_conv": 1e-5,
        "lpyr_roundtrip": 1e-5,
        "blur_dn": 1e-5,
        "iir": 1e-5,
        "butter": 1e-5,
        "ideal": 1e-4,
        "amplify_render": 1e-6,
        "end_to_end_rmse": 1e-2,
    }

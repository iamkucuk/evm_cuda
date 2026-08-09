"""Deprecated alias for :mod:`evm.cuda`.

The CUDA wrapper used to be a separate top-level package reachable only with
``PYTHONPATH=cuda``. It is now ``evm.cuda``, installed with the rest of the
distribution. This shim exists so notebooks, scripts and snippets written
against the old name keep running; it forwards every attribute to
:mod:`evm.cuda`, so ``evm_cuda.have_cuda``, ``evm_cuda.require_cuda`` and
``from evm_cuda import batched`` behave exactly as before.

Migration is a one-line edit::

    import evm_cuda                 ->  from evm import cuda as evm_cuda
    from evm_cuda import benchmark  ->  from evm.cuda import benchmark
    from evm_cuda.batched import X  ->  from evm.cuda.batched import X

The last form — a dotted *module* import through the old name — is the one
thing this shim does not forward: ``evm_cuda.batched`` as an attribute works,
``import evm_cuda.batched`` does not. Aliasing submodules into ``sys.modules``
would import them a second time under a second name, giving the CUDA device
memory pool two sets of module-level state; a loud ``ModuleNotFoundError``
is the better failure.
"""

from __future__ import annotations

import warnings

from evm import cuda as _cuda

warnings.warn(
    "The top-level 'evm_cuda' package is deprecated and will be removed; "
    "import 'evm.cuda' instead (e.g. 'from evm import cuda as evm_cuda').",
    DeprecationWarning,
    stacklevel=2,
)


def __getattr__(name: str):
    return getattr(_cuda, name)


def __dir__() -> list[str]:
    return dir(_cuda)

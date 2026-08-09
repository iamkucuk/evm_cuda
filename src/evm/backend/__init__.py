"""Backend interface and registry — `docs/dev/PLAN.md` section 3c.

Two protocols and one registry, and nothing else:

* :class:`Ops` — the ten primitive pixel operations plus host transfer. Every
  backend implements these; that is the whole cost of adding one.
* :class:`Pipelines` — the four magnify cores. A generic default written once
  against :class:`Ops` serves any backend that does not override it; native
  CUDA overrides it because its speed comes from fusing stages.
* :func:`register` / :func:`select` / :func:`list_backends` — names to
  implementations, loaded lazily, with capability flags and loud failures.

Registration happens where the backends live, not here: this module imports no
implementation, so ``import evm`` on a CPU-only machine never reaches for the
CUDA extension. ``"cpu"`` and ``"cuda"`` are wired up by the facade during
integration; the ``"cuda"`` probe reports ``evm.cuda.runtime.import_error`` as
its reason, which is already the truthful explanation of why the extension is
missing.

Selecting a backend::

    from evm import backend

    name, impl = backend.select("auto")   # name is returned: never a silent choice
    for info in backend.list_backends():  # what exists, and why it is or is not usable
        print(info.name, info.capabilities, info.unavailable_reason)
"""

from __future__ import annotations

from .ops import Array, Ops
from .pipelines import Pipelines
from .registry import (
    PREFERENCE_ORDER,
    BackendError,
    BackendInfo,
    BackendUnavailableError,
    Capabilities,
    UnknownBackendError,
    list_backends,
    register,
    select,
)

__all__ = [
    "PREFERENCE_ORDER",
    "Array",
    "BackendError",
    "BackendInfo",
    "BackendUnavailableError",
    "Capabilities",
    "Ops",
    "Pipelines",
    "UnknownBackendError",
    "list_backends",
    "register",
    "select",
]

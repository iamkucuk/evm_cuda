"""The backend registry — names to implementations, with capability flags.

`docs/dev/PLAN.md` section 3c, step 3.5. Three properties matter more than the
code:

**Registration is data.** :func:`register` stores two callables and a capability
record; neither is called. The implementation module is imported the first time
somebody selects that backend, so ``import evm`` on a machine with no GPU never
touches the CUDA extension.

**Nothing fails quietly.** An unknown name lists what *is* registered, an
unavailable backend repeats the reason its probe gave (no driver, no device,
extra not installed), and ``"auto"`` with nothing usable reports every candidate
it tried and why each failed. Falling back from GPU to CPU by accident would cost
roughly 700x, so it never happens implicitly — ``"auto"`` reports its choice
through the returned name and an INFO log line, and an explicitly named backend
that is unavailable raises rather than substituting another.

**The order is fixed and documented**: :data:`PREFERENCE_ORDER`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

__all__ = [
    "PREFERENCE_ORDER",
    "BackendError",
    "BackendInfo",
    "BackendUnavailableError",
    "Capabilities",
    "UnknownBackendError",
    "list_backends",
    "register",
    "select",
]

_log = logging.getLogger(__name__)

#: The order ``select("auto")`` walks, from `docs/dev/PLAN.md` section 3b:
#: native CUDA first, then the portable native backends, then PyTorch if the
#: user installed it, then the CPU baseline. A backend registered under a name
#: outside this tuple is never auto-selected — it must be asked for by name.
PREFERENCE_ORDER: tuple[str, ...] = (
    "cuda",
    "metal",
    "vulkan",
    "opencl",
    "torch",
    "cpu",
)


@dataclass(frozen=True)
class Capabilities:
    """What a backend can do, readable without importing it."""

    #: NumPy dtype names the backend computes in, e.g. ``("float32", "float16")``.
    dtypes: tuple[str, ...]
    #: True if an FFT is available, so ``ideal_bandpass`` is the exact
    #: reference filter rather than a band-projection approximation.
    fft: bool
    #: True if the backend can run a causal, frame-at-a-time pipeline.
    streaming: bool


@dataclass(frozen=True)
class BackendInfo:
    """One row of :func:`list_backends`."""

    name: str
    capabilities: Capabilities
    #: ``None`` when usable; otherwise the probe's explanation.
    unavailable_reason: str | None

    @property
    def available(self) -> bool:
        return self.unavailable_reason is None


class BackendError(Exception):
    """Base for every backend-selection failure."""


class UnknownBackendError(BackendError, LookupError):
    """No backend is registered under that name."""


class BackendUnavailableError(BackendError, RuntimeError):
    """The backend exists but cannot run here, and says why."""


@dataclass(frozen=True)
class _Entry:
    name: str
    load: Callable[[], Any]
    probe: Callable[[], str | None]
    capabilities: Capabilities


_REGISTRY: dict[str, _Entry] = {}
_LOADED: dict[str, Any] = {}


def register(
    name: str,
    *,
    load: Callable[[], Any],
    probe: Callable[[], str | None],
    capabilities: Capabilities,
) -> None:
    """Record a backend without importing it.

    Args:
        name: selection name, e.g. ``"cuda"``. Names in
            :data:`PREFERENCE_ORDER` take part in ``"auto"`` selection.
        load: called at most once, on first selection; returns the object
            implementing :class:`~evm.backend.Ops` (and optionally
            :class:`~evm.backend.Pipelines`). Import the heavy module *inside*
            this callable, never at registration time.
        probe: called on every selection attempt; returns ``None`` if the
            backend can run here, otherwise a one-line reason ("extension not
            built", "no CUDA device", "install evm-cuda[torch]").
        capabilities: advertised without loading anything.

    Raises:
        ValueError: if the name is already registered.
    """
    if name in _REGISTRY:
        raise ValueError(
            f"backend {name!r} is already registered; "
            "each name may be registered exactly once"
        )
    _REGISTRY[name] = _Entry(
        name=name, load=load, probe=probe, capabilities=capabilities
    )


def list_backends() -> tuple[BackendInfo, ...]:
    """Every registered backend, in preference order, with its availability.

    Probes each backend, so the reasons are current. Does not load any
    implementation.
    """
    return tuple(
        BackendInfo(
            name=entry.name,
            capabilities=entry.capabilities,
            unavailable_reason=_probe(entry),
        )
        for entry in _ordered_entries()
    )


def select(name: str = "auto") -> tuple[str, Any]:
    """Resolve a backend name to ``(resolved_name, implementation)``.

    The resolved name is returned so the caller can show it — the choice is
    never silent. ``"auto"`` walks :data:`PREFERENCE_ORDER` and takes the first
    registered backend whose probe says it can run.

    Raises:
        UnknownBackendError: the name was never registered.
        BackendUnavailableError: the named backend cannot run here, or
            ``"auto"`` found nothing usable — the message carries every reason.
    """
    if name == "auto":
        return _select_auto()

    entry = _REGISTRY.get(name)
    if entry is None:
        known = ", ".join(sorted(_REGISTRY)) or "none"
        raise UnknownBackendError(
            f"unknown backend {name!r}; registered backends: {known}"
        )
    reason = _probe(entry)
    if reason is not None:
        raise BackendUnavailableError(
            f"backend {name!r} is registered but unavailable: {reason}"
        )
    _log.info("evm: using backend %r (requested explicitly)", name)
    return name, _load(entry)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _select_auto() -> tuple[str, Any]:
    if not _REGISTRY:
        raise BackendUnavailableError(
            "no backends are registered, so backend='auto' has nothing to "
            "choose from; call evm.backend.register() first"
        )

    tried: list[str] = []
    for candidate in PREFERENCE_ORDER:
        entry = _REGISTRY.get(candidate)
        if entry is None:
            continue
        reason = _probe(entry)
        if reason is None:
            _log.info(
                "evm: using backend %r (auto)%s",
                candidate,
                f"; skipped {'; '.join(tried)}" if tried else "",
            )
            return candidate, _load(entry)
        tried.append(f"{candidate}: {reason}")

    detail = "\n".join(f"  {line}" for line in tried)
    unlisted = sorted(set(_REGISTRY) - set(PREFERENCE_ORDER))
    extra = (
        f"\nregistered but outside the automatic order, ask for one by name: "
        f"{', '.join(unlisted)}"
        if unlisted
        else ""
    )
    raise BackendUnavailableError(
        "no usable backend found; tried, in preference order:\n"
        f"{detail or '  (none of the preference-order names are registered)'}"
        f"{extra}"
    )


def _ordered_entries() -> list[_Entry]:
    """Preference order first, then any other registered name, alphabetically."""
    ordered = [_REGISTRY[n] for n in PREFERENCE_ORDER if n in _REGISTRY]
    rest = sorted(set(_REGISTRY) - set(PREFERENCE_ORDER))
    return ordered + [_REGISTRY[n] for n in rest]


def _probe(entry: _Entry) -> str | None:
    reason = entry.probe()
    if reason is not None and not isinstance(reason, str):
        # A falsy non-None return would read as "available" and silently hand
        # the caller a backend that cannot run. Refuse it instead.
        raise TypeError(
            f"backend {entry.name!r}: probe must return None (available) or a "
            f"reason string; got {reason!r}"
        )
    return reason


def _load(entry: _Entry) -> Any:
    if entry.name not in _LOADED:
        _LOADED[entry.name] = entry.load()
    return _LOADED[entry.name]

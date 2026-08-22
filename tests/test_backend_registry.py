"""Tests for the backend registry (plan step 3.5, plan section 3c).

What is being pinned down here:

* registration is data-only — a backend's implementation module is imported the
  first time somebody *selects* it, never when it is registered. That is what
  keeps ``import vidmag`` from touching the CUDA extension on a machine with no GPU;
* every failure names its cause. An unknown backend lists the registered ones, an
  unavailable backend repeats the probe's reason verbatim, and ``"auto"`` with
  nothing usable reports every candidate it tried and why each one failed;
* ``"auto"`` walks :data:`vidmag.backend.PREFERENCE_ORDER`, not registration order,
  so a GPU backend is never skipped for a CPU one that happened to register first;
* the chosen name comes back to the caller, so the ~700x CPU/GPU cliff can never
  be crossed silently.

``import vidmag`` registers the two built-in backends (``vidmag.api`` does it, so that
``vidmag.backend`` itself still imports no implementation). These tests are about
the registry mechanism, not about those two, so the autouse fixture empties the
registry for the duration of each test and puts the real one back afterwards —
otherwise every ``_register("cpu")`` below would collide with the built-in of
the same name. ``tests/test_api.py`` is where the built-in registrations
themselves are checked.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from vidmag import backend
from vidmag.backend import registry


@pytest.fixture(autouse=True)
def _restore_registry():
    """Run each test against an empty registry, then put the real one back."""
    entries = dict(registry._REGISTRY)
    loaded = dict(registry._LOADED)
    registry._REGISTRY.clear()
    registry._LOADED.clear()
    yield
    registry._REGISTRY.clear()
    registry._REGISTRY.update(entries)
    registry._LOADED.clear()
    registry._LOADED.update(loaded)


CAPS = backend.Capabilities(dtypes=("float32",), fft=True, streaming=False)


def _register(name, *, obj="impl", reason=None, caps=CAPS, calls=None):
    """Register a fake backend; ``calls`` counts how often the loader ran."""

    def load():
        if calls is not None:
            calls.append(name)
        return obj

    backend.register(name, load=load, probe=lambda: reason, capabilities=caps)


# ---------------------------------------------------------------------------
# Registration and lookup
# ---------------------------------------------------------------------------


def test_register_then_select_by_name_returns_the_implementation():
    _register("cpu", obj="cpu-impl")
    assert backend.select("cpu") == ("cpu", "cpu-impl")


def test_capabilities_are_readable_without_loading_the_backend():
    calls: list[str] = []
    caps = backend.Capabilities(dtypes=("float32", "float16"), fft=True, streaming=True)
    _register("cuda", caps=caps, calls=calls)

    (info,) = backend.list_backends()
    assert info.name == "cuda"
    assert info.capabilities == caps
    assert info.capabilities.dtypes == ("float32", "float16")
    assert info.unavailable_reason is None and info.available
    assert calls == []  # listing must not import the implementation


def test_registering_the_same_name_twice_is_an_error():
    _register("cpu")
    with pytest.raises(ValueError, match="already registered"):
        _register("cpu")


# ---------------------------------------------------------------------------
# Laziness: no heavy import until something is actually selected
# ---------------------------------------------------------------------------


def test_backend_is_loaded_on_first_select_and_then_cached():
    calls: list[str] = []
    _register("cpu", obj="cpu-impl", calls=calls)
    assert calls == [], "registration must not load the backend"

    assert backend.select("cpu")[1] == "cpu-impl"
    assert calls == ["cpu"]

    assert backend.select("cpu")[1] == "cpu-impl"
    assert calls == ["cpu"], "the loaded backend must be cached, not re-imported"


def test_importing_the_registry_does_not_import_the_cuda_backend():
    """``import vidmag.backend`` must not reach for the compiled extension."""
    code = (
        "import sys; import vidmag.backend; "
        "print(sorted(m for m in sys.modules if m.startswith('vidmag.')))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    assert "vidmag.backend" in out
    assert "vidmag.cuda" not in out


# ---------------------------------------------------------------------------
# Errors name their cause — never a bare KeyError
# ---------------------------------------------------------------------------


def test_unknown_backend_names_the_registered_ones():
    _register("cpu")
    with pytest.raises(backend.UnknownBackendError) as exc:
        backend.select("vulkan")
    msg = str(exc.value)
    assert "vulkan" in msg and "cpu" in msg
    assert not isinstance(exc.value, KeyError)


def test_unavailable_backend_reports_why():
    _register("cuda", reason="no CUDA driver on this host")
    with pytest.raises(backend.BackendUnavailableError) as exc:
        backend.select("cuda")
    assert "no CUDA driver on this host" in str(exc.value)


def test_a_probe_that_does_not_return_a_string_or_none_fails_loudly():
    backend.register(
        "cuda", load=lambda: "impl", probe=lambda: False, capabilities=CAPS
    )
    with pytest.raises(TypeError, match="probe"):
        backend.select("cuda")


def test_auto_with_nothing_usable_lists_every_candidate_and_its_reason():
    _register("cuda", reason="extension not built")
    _register("cpu", reason="numpy missing")
    with pytest.raises(backend.BackendUnavailableError) as exc:
        backend.select("auto")
    msg = str(exc.value)
    assert "extension not built" in msg and "numpy missing" in msg
    assert "cuda" in msg and "cpu" in msg


def test_auto_on_an_empty_registry_says_so():
    with pytest.raises(backend.BackendUnavailableError, match="no backends"):
        backend.select("auto")


# ---------------------------------------------------------------------------
# Automatic selection: fixed order, always discoverable
# ---------------------------------------------------------------------------


def test_auto_follows_the_preference_order_not_the_registration_order():
    _register("cpu", obj="cpu-impl")
    _register("cuda", obj="cuda-impl")  # registered second, must still win
    assert backend.select("auto") == ("cuda", "cuda-impl")


def test_auto_skips_an_unavailable_backend_and_reports_which_one_it_chose():
    _register("cuda", reason="no device")
    _register("cpu", obj="cpu-impl")
    name, impl = backend.select("auto")
    assert (name, impl) == ("cpu", "cpu-impl")


def test_auto_never_considers_a_backend_outside_the_preference_order():
    """An unlisted name must be asked for explicitly, never auto-selected."""
    _register("experimental", obj="experimental-impl")
    with pytest.raises(backend.BackendUnavailableError):
        backend.select("auto")
    assert backend.select("experimental") == ("experimental", "experimental-impl")


def test_the_preference_order_puts_native_cuda_first_and_cpu_last():
    order = backend.PREFERENCE_ORDER
    assert order[0] == "cuda"
    assert order[-1] == "cpu"
    assert order.index("torch") > order.index("vulkan")


def test_the_selected_backend_is_logged_not_silent(caplog):
    _register("cpu", obj="cpu-impl")
    with caplog.at_level("INFO", logger="vidmag.backend.registry"):
        backend.select("auto")
    assert "cpu" in caplog.text


def test_list_backends_reports_unavailable_ones_with_their_reason():
    _register("cuda", reason="nvcc not found")
    _register("cpu")
    infos = {i.name: i for i in backend.list_backends()}
    assert infos["cuda"].unavailable_reason == "nvcc not found"
    assert not infos["cuda"].available
    assert infos["cpu"].available
    # Listed in preference order so callers can print a truthful table.
    assert [i.name for i in backend.list_backends()] == ["cuda", "cpu"]

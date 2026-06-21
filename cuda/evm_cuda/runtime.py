"""Runtime helpers for the EVM CUDA port.

Host-side glue: a clean CUDA-presence probe (used by `tests/cuda/` to skip
cleanly on hosts without nvcc) and the scipy-side Butterworth coefficient
helper that mirrors what `evm/filters.py` does. cuFFT plan lifecycle is
owned by the bindings themselves (`_evm_cuda.ideal_bandpass` creates and
destroys its own plans per call), so this module is intentionally small.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.signal import butter

from . import _have_cuda

# The original ImportError captured in __init__.py (None if the import
# succeeded). Exposed for tests and callers that want to surface why CUDA
# isn't available.
have_cuda: bool = _have_cuda
import_error = None
try:
    from .__init__ import _import_error  # type: ignore
    import_error = _import_error
except Exception:
    pass


def require_cuda() -> None:
    """Raise a clear error if the CUDA extension isn't available."""
    if not _have_cuda:
        raise RuntimeError(
            "evm_cuda._evm_cuda not importable; the extension was not built "
            "(no nvcc?) or no CUDA device is available. "
            f"Underlying error: {import_error!r}"
        )


# ---------------------------------------------------------------------------
# Butterworth coefficients (host-side, mirrors evm/filters.py:butter_bandpass)
# ---------------------------------------------------------------------------

def butter_bandpass_coeffs(
    fl: float, fh: float, sampling_rate: float, order: int = 1
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Return ((b0_high, b1_high, a1_high), (b0_low, b1_low, a1_low)) for the
    first-order Butterworth bandpass, matching scipy.signal.butter on the host.

    The kernel only needs the 6 scalar coefficients (3 for each lowpass);
    we compute them here so the kernel never has to call into scipy.
    """
    nyq = sampling_rate / 2.0
    high_b, high_a = butter(order, fh / nyq, btype="low")
    low_b, low_a = butter(order, fl / nyq, btype="low")
    if len(high_b) != 2 or len(low_b) != 2:
        raise ValueError(f"butter(order={order}) did not return 2 taps")
    h = (float(high_b[0]), float(high_b[1]), float(high_a[1]))
    l = (float(low_b[0]), float(low_b[1]), float(low_a[1]))
    return h, l


# ---------------------------------------------------------------------------
# Convenience: numpy helpers used by pipelines.py
# ---------------------------------------------------------------------------

def to_contiguous_f32(a: np.ndarray) -> np.ndarray:
    """Return a C-contiguous float32 view/copy of `a`."""
    return np.ascontiguousarray(a, dtype=np.float32)

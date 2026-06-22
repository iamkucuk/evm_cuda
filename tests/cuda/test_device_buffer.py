"""DeviceBuffer upload/download round-trip tests.

Guards against the latent Phase 1a-b bug where the DeviceBuffer(array)
constructor used py::array_t<char>::ensure() with forcecast, which CAST each
element to char (1 byte) instead of treating the array as raw bytes. That
truncated float32 uploads to all-zero for values that round to 0 as char
(e.g. 0.5 -> (char)0.5 == 0). The uint8 path was unaffected (uint8 == char).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
CUDA_DIR = ROOT / "cuda"
for p in (str(ROOT), str(CUDA_DIR), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from conftest import have_cuda, skip_no_cuda  # noqa: E402

if have_cuda:
    from evm_cuda.batched import DeviceBuffer  # noqa: E402


@skip_no_cuda
def test_device_buffer_f32_round_trip():
    """float32 array uploaded then downloaded must be bit-identical."""
    rng = np.random.default_rng(0)
    arr = rng.uniform(-1.0, 1.0, size=(4, 5, 3)).astype(np.float32)
    buf = DeviceBuffer.from_array(arr)
    got = buf.download_f32(arr.size).reshape(arr.shape)
    np.testing.assert_array_equal(got, arr)


@skip_no_cuda
def test_device_buffer_u8_round_trip():
    """uint8 array uploaded then downloaded must be bit-identical."""
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 256, size=(4, 5, 3), dtype=np.uint8)
    buf = DeviceBuffer.from_array(arr)
    got = buf.download_u8(arr.size).reshape(arr.shape)
    np.testing.assert_array_equal(got, arr)


@skip_no_cuda
def test_device_buffer_f32_nonzero_values():
    """Regression for the float->char truncation bug: values in (0,1) must
    survive the round-trip. Under the old broken constructor, 0.5 cast to char
    became 0, so the downloaded buffer was all-zero."""
    arr = np.full((2, 2, 3), 0.5, dtype=np.float32)
    buf = DeviceBuffer.from_array(arr)
    got = buf.download_f32(arr.size).reshape(arr.shape)
    assert got.min() > 0.4 and got.max() < 0.6, "float32 values truncated to 0"

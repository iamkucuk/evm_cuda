"""The public GPU array type: shape and dtype, and safe zero-copy export.

``DeviceBuffer`` (the internal type these tests' neighbours use) is a bag of
bytes: it knows how many, and nothing else. Every batched binding therefore
takes a raw integer address plus separately-passed dimensions, and nothing
checks that the two agree. ``DeviceArray`` is the type users get: it carries
its own shape and dtype, so an operation can reject a wrong-shaped input
instead of reading past the end of a buffer.

The lifetime test below is the one that matters most. Device memory here comes
from a pool that hands a freed block straight to the next allocation of the
same size. If an exported pointer outlived the Python object that owned it, the
consumer would silently read someone else's data. These tests pin that shut.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import have_cuda, skip_no_cuda

if have_cuda:
    from vidmag.cuda import _vidmag_cuda
    from vidmag.cuda.array import DeviceArray


# ---------------------------------------------------------------------------
# Shape and dtype
# ---------------------------------------------------------------------------


@skip_no_cuda
@pytest.mark.parametrize("dtype", ["float32", "float16", "uint8"])
def test_round_trip_preserves_values_shape_and_dtype(dtype):
    rng = np.random.default_rng(7)
    if dtype == "uint8":
        host = rng.integers(0, 256, (4, 5, 3), dtype=np.uint8)
    else:
        host = rng.random((4, 5, 3)).astype(dtype)

    dev = DeviceArray.from_numpy(host)
    assert dev.shape == host.shape
    assert dev.dtype == host.dtype
    assert dev.nbytes == host.nbytes
    assert np.array_equal(dev.numpy(), host)


@skip_no_cuda
def test_wrong_dtype_is_rejected_rather_than_reinterpreted():
    """float64 has no kernel here; silently truncating it would corrupt data."""
    with pytest.raises((TypeError, ValueError), match="dtype"):
        DeviceArray.from_numpy(np.zeros((2, 2), dtype=np.float64))


@skip_no_cuda
def test_non_contiguous_input_is_rejected_not_silently_copied_wrong():
    host = np.zeros((4, 6), dtype=np.float32)[:, ::2]  # strided view
    assert not host.flags["C_CONTIGUOUS"]
    with pytest.raises((ValueError, RuntimeError), match="contiguous"):
        DeviceArray.from_numpy(host)


# ---------------------------------------------------------------------------
# Zero-copy export
# ---------------------------------------------------------------------------


@skip_no_cuda
def test_dlpack_export_reports_a_cuda_device_and_the_same_address():
    host = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    dev = DeviceArray.from_numpy(host)

    device_type, device_id = dev.__dlpack_device__()
    assert device_type == 2, "kDLCUDA is 2 in the DLPack device enum"
    assert device_id >= 0

    capsule = dev.__dlpack__()
    assert capsule is not None
    # Re-importing our own capsule must land on the very same device memory,
    # which is what makes the handoff zero-copy rather than a hidden copy.
    back = DeviceArray.from_dlpack(capsule)
    assert back.ptr == dev.ptr
    assert back.shape == dev.shape
    assert back.dtype == dev.dtype
    assert np.array_equal(back.numpy(), host)


@skip_no_cuda
def test_the_pool_really_does_recycle_addresses():
    """Establishes the hazard the next test guards against.

    If this ever stops being true, the lifetime test below stops proving
    anything, because there would be no recycling to survive.
    """
    a = DeviceArray.from_numpy(np.arange(4096, dtype=np.float32))
    address = a.ptr
    del a  # nothing holds this block now
    b = DeviceArray.from_numpy(np.full(4096, -7.0, dtype=np.float32))
    assert b.ptr == address, (
        "the pool no longer reuses a freed block for the next same-sized "
        "allocation; the lifetime test below is no longer meaningful"
    )


@skip_no_cuda
def test_exported_memory_survives_its_python_owner():
    """The pool recycles freed blocks, so an export must hold the block alive.

    Without shared ownership this is a use-after-free: dropping the owner
    returns the block to the pool, the next same-sized allocation gets that
    exact address, and the consumer silently reads the new contents. The
    allocation below is deliberately the same size, because that is the case
    the pool reuses.
    """
    host = np.arange(4096, dtype=np.float32)
    dev = DeviceArray.from_numpy(host)
    borrowed = DeviceArray.from_dlpack(dev.__dlpack__())
    address = dev.ptr

    del dev  # the only Python reference to the original owner

    # Same-sized allocations are exactly what the pool would hand the old block
    # to. Fill each with a value that is not in the original data.
    decoys = [
        DeviceArray.from_numpy(np.full(4096, -7.0, dtype=np.float32)) for _ in range(8)
    ]

    assert np.array_equal(borrowed.numpy(), host), (
        "exported memory was recycled while still referenced"
    )
    assert borrowed.ptr == address
    del decoys


@skip_no_cuda
def test_dlpack_capsule_is_consumed_only_once():
    """A second import of the same capsule must fail loudly, not double-free."""
    dev = DeviceArray.from_numpy(np.ones((8, 8), dtype=np.float32))
    capsule = dev.__dlpack__()
    DeviceArray.from_dlpack(capsule)
    with pytest.raises((ValueError, RuntimeError)):
        DeviceArray.from_dlpack(capsule)


@skip_no_cuda
def test_pool_still_reuses_blocks_once_nothing_references_them():
    """Shared ownership must not turn the pool into a leak.

    If holding a block alive for an export also stopped it ever returning to
    the free list, memory use would grow without bound. Dropping every
    reference must put the block back.
    """
    _vidmag_cuda.free_device_pool()
    before_free, _ = _vidmag_cuda.gpu_mem_info()

    for _ in range(50):
        a = DeviceArray.from_numpy(np.zeros(65536, dtype=np.float32))
        del a

    after_free, _ = _vidmag_cuda.gpu_mem_info()
    # 50 sequential 256 KiB allocations must not consume 50 blocks' worth: the
    # pool should hand the same block back each time.
    assert before_free - after_free < 50 * 65536 * 4, "pool stopped reusing blocks"

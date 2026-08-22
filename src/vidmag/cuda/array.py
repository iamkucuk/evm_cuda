"""``DeviceArray`` — the GPU array users are given.

The internal ``DeviceBuffer`` is a bag of bytes: it knows how many there are
and nothing else. Every batched binding therefore takes a raw integer address
plus separately-passed dimensions, and nothing checks the two agree. That is
workable inside a pipeline that allocated the memory moments earlier; it is not
something to hand to a user, because a wrong shape reads past the end of a
buffer instead of raising.

``DeviceArray`` carries its own shape and dtype, so operations can check their
inputs, and it implements the DLPack protocol so the result can go straight
into PyTorch or CuPy without a copy back through host memory.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import _vidmag_cuda

__all__ = ["DeviceArray"]

# DLPack's dtype encoding: (code, bits). Only what the kernels here accept.
# float64 is deliberately absent: no kernel takes it, and quietly narrowing a
# user's float64 to float32 would lose precision without saying so.
_DTYPE_TO_DLPACK: dict[np.dtype, tuple[int, int]] = {
    np.dtype("float32"): (2, 32),  # kDLFloat
    np.dtype("float16"): (2, 16),
    np.dtype("uint8"): (1, 8),  # kDLUInt
}
_DLPACK_TO_DTYPE: dict[tuple[int, int], np.dtype] = {
    v: k for k, v in _DTYPE_TO_DLPACK.items()
}

_KDLCUDA = 2


class DeviceArray:
    """An array in GPU memory, with a shape and a dtype.

    Construct one with :meth:`from_numpy`, or receive one from an operation in
    :mod:`vidmag.cuda.ops`. Instances own a reference to their memory; the memory
    outlives the object for exactly as long as something else still refers to
    it, which is what makes :meth:`__dlpack__` safe.
    """

    __slots__ = ("_buf", "_shape", "_dtype", "_keepalive")

    def __init__(
        self, buf: Any, shape: tuple[int, ...], dtype: np.dtype, _keepalive: Any = None
    ) -> None:
        self._buf = buf
        self._shape = tuple(int(s) for s in shape)
        self._dtype = np.dtype(dtype)
        # Holds an imported DLPack tensor's owner, when this array came from
        # another library rather than from our own allocation.
        self._keepalive = _keepalive

    # -- construction -------------------------------------------------------

    @classmethod
    def from_numpy(cls, host: np.ndarray) -> "DeviceArray":
        """Copy a host array to the device.

        Rejects dtypes no kernel accepts and non-contiguous input, rather than
        reinterpreting or silently copying them into a different layout.
        """
        dtype = np.dtype(host.dtype)
        if dtype not in _DTYPE_TO_DLPACK:
            raise TypeError(
                f"DeviceArray: unsupported dtype {dtype}; this backend accepts "
                f"{', '.join(str(d) for d in _DTYPE_TO_DLPACK)}"
            )
        if not host.flags["C_CONTIGUOUS"]:
            raise ValueError(
                "DeviceArray: input must be C-contiguous; call "
                "numpy.ascontiguousarray first"
            )
        return cls(_vidmag_cuda.DeviceBuffer(host), host.shape, dtype)

    @classmethod
    def empty(cls, shape: tuple[int, ...], dtype: Any) -> "DeviceArray":
        """Allocate uninitialised device memory of the given shape and dtype."""
        dtype = np.dtype(dtype)
        if dtype not in _DTYPE_TO_DLPACK:
            raise TypeError(f"DeviceArray: unsupported dtype {dtype}")
        nbytes = int(np.prod(shape)) * dtype.itemsize if shape else 0
        return cls(_vidmag_cuda.DeviceBuffer(nbytes), shape, dtype)

    # -- description --------------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def size(self) -> int:
        return int(np.prod(self._shape)) if self._shape else 0

    @property
    def nbytes(self) -> int:
        return self.size * self._dtype.itemsize

    @property
    def ptr(self) -> int:
        """The device address, for passing to a kernel binding."""
        return self._buf.ptr

    @property
    def device(self) -> str:
        return "cuda"

    def __len__(self) -> int:
        if not self._shape:
            raise TypeError("len() of a 0-dimensional DeviceArray")
        return self._shape[0]

    def __repr__(self) -> str:
        return (
            f"DeviceArray(shape={self._shape}, dtype={self._dtype.name}, "
            f"device='cuda', ptr=0x{self.ptr:x})"
        )

    # -- transfer -----------------------------------------------------------

    def numpy(self) -> np.ndarray:
        """Copy back to host memory."""
        if self._dtype == np.uint8:
            flat = self._buf.download_u8(self.size)
        elif self._dtype == np.float32:
            flat = self._buf.download_f32(self.size)
        else:
            # float16 has no dedicated download; take the raw bytes as float32
            # of half the count and reinterpret, which is exact.
            raw = self._buf.download_f32((self.nbytes + 3) // 4)
            flat = raw.view(np.float16)[: self.size]
        return np.ascontiguousarray(flat).reshape(self._shape)

    # -- zero-copy interoperability ----------------------------------------

    def __dlpack_device__(self) -> tuple[int, int]:
        """Report the device, as (device_type, device_id)."""
        return (_KDLCUDA, _vidmag_cuda.current_device())

    def __dlpack__(self, stream: Any = None) -> Any:
        """Export as a DLPack capsule, sharing ownership of the memory.

        ``stream`` is accepted because the protocol passes it, and must be the
        legacy default stream: every kernel in this project is launched on
        stream 0, so there is no other stream whose ordering could be honoured.
        Anything else is refused rather than ignored.
        """
        if stream is not None and stream not in (0, -1, 1):
            raise ValueError(
                f"DeviceArray.__dlpack__: stream {stream!r} is not supported; "
                "this backend launches every kernel on the legacy default "
                "stream, so only that stream can be synchronised against"
            )
        code, bits = _DTYPE_TO_DLPACK[self._dtype]
        return _vidmag_cuda.dlpack_capsule(self._buf, list(self._shape), code, bits)

    @classmethod
    def from_dlpack(cls, source: Any) -> "DeviceArray":
        """Adopt an array from another library without copying.

        Accepts either an object implementing ``__dlpack__`` (a PyTorch or CuPy
        tensor) or a capsule directly.
        """
        capsule = source.__dlpack__() if hasattr(source, "__dlpack__") else source
        (address, shape, code, bits, device_type, _device_id, managed) = (
            _vidmag_cuda.dlpack_import(capsule)
        )

        if device_type != _KDLCUDA:
            _vidmag_cuda.dlpack_release(managed)
            raise ValueError(
                f"DeviceArray.from_dlpack: tensor is on device type "
                f"{device_type}, but this is a CUDA array type"
            )
        key = (code, bits)
        if key not in _DLPACK_TO_DTYPE:
            _vidmag_cuda.dlpack_release(managed)
            raise TypeError(
                f"DeviceArray.from_dlpack: unsupported dtype code={code} bits={bits}"
            )
        dtype = _DLPACK_TO_DTYPE[key]
        view = _vidmag_cuda.DeviceBufferView(
            address, int(np.prod(shape)) * dtype.itemsize
        )
        return cls(view, tuple(shape), dtype, _keepalive=_ImportedTensor(managed))


class _ImportedTensor:
    """Owns a DLPack tensor taken from another library.

    Its only job is to call the producer's deleter once, when the importing
    DeviceArray is collected. Without it the producer would never learn that
    the borrow ended.
    """

    __slots__ = ("_managed",)

    def __init__(self, managed: int) -> None:
        self._managed = managed

    def __del__(self) -> None:  # pragma: no cover - runs at collection time
        if self._managed:
            _vidmag_cuda.dlpack_release(self._managed)
            self._managed = 0

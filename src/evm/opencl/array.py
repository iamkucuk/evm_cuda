"""An array living in OpenCL device memory.

The same idea as :class:`evm.cuda.array.DeviceArray`: a buffer that knows its
own shape and dtype, so the operations built on it can check their inputs. It
is a separate type because the memory it holds belongs to a different runtime
and the two cannot be mixed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import runtime

__all__ = ["ClArray"]


class ClArray:
    """A shaped, typed buffer in OpenCL memory."""

    __slots__ = ("buffer", "_shape", "_dtype")

    def __init__(self, buffer: Any, shape: tuple[int, ...], dtype: Any) -> None:
        self.buffer = buffer
        self._shape = tuple(int(s) for s in shape)
        self._dtype = np.dtype(dtype)

    @classmethod
    def from_numpy(cls, host: np.ndarray) -> "ClArray":
        import pyopencl as cl
        host = np.ascontiguousarray(host)
        flags = cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR
        buf = cl.Buffer(runtime.context(), flags, hostbuf=host)
        return cls(buf, host.shape, host.dtype)

    @classmethod
    def empty(cls, shape: tuple[int, ...], dtype: Any) -> "ClArray":
        import pyopencl as cl
        dtype = np.dtype(dtype)
        nbytes = max(int(np.prod(shape)) * dtype.itemsize, 1)
        buf = cl.Buffer(runtime.context(), cl.mem_flags.READ_WRITE, nbytes)
        return cls(buf, shape, dtype)

    @classmethod
    def zeros(cls, shape: tuple[int, ...], dtype: Any) -> "ClArray":
        return cls.from_numpy(np.zeros(shape, dtype=dtype))

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
    def device(self) -> str:
        return "opencl"

    def __len__(self) -> int:
        if not self._shape:
            raise TypeError("len() of a 0-dimensional ClArray")
        return self._shape[0]

    def __repr__(self) -> str:
        return (f"ClArray(shape={self._shape}, dtype={self._dtype.name}, "
                f"device='opencl')")

    def numpy(self) -> np.ndarray:
        import pyopencl as cl
        out = np.empty(self._shape, dtype=self._dtype)
        cl.enqueue_copy(runtime.queue(), out, self.buffer)
        runtime.queue().finish()
        return out

    def reshape(self, shape: tuple[int, ...]) -> "ClArray":
        """A view of the same memory with a different shape.

        The buffer is shared, not copied, so writing through one view is
        visible through the other.
        """
        if int(np.prod(shape)) != self.size:
            raise ValueError(
                f"cannot reshape {self._shape} ({self.size} elements) "
                f"to {tuple(shape)}")
        return ClArray(self.buffer, shape, self._dtype)

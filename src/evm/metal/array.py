"""An array in Metal device memory.

The same idea as the arrays the other backends use: a buffer that knows its own
shape and dtype, so operations can check what they are given.

Apple hardware shares memory between the processor and the graphics processor,
so a buffer created in shared mode is readable from both without an explicit
copy. That makes transfers here cheaper than on a card with its own memory, and
it is why this type has no separate upload and download step.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import runtime

__all__ = ["MetalArray"]


class MetalArray:
    """A shaped, typed buffer in Metal memory."""

    __slots__ = ("buffer", "_shape", "_dtype")

    def __init__(self, buffer: Any, shape: tuple[int, ...], dtype: Any) -> None:
        self.buffer = buffer
        self._shape = tuple(int(s) for s in shape)
        self._dtype = np.dtype(dtype)

    @classmethod
    def from_numpy(cls, host: np.ndarray) -> "MetalArray":
        import Metal

        host = np.ascontiguousarray(host)
        buffer = runtime.device().newBufferWithBytes_length_options_(
            host.tobytes(), max(host.nbytes, 1), Metal.MTLResourceStorageModeShared
        )
        return cls(buffer, host.shape, host.dtype)

    @classmethod
    def empty(cls, shape: tuple[int, ...], dtype: Any) -> "MetalArray":
        import Metal

        dtype = np.dtype(dtype)
        nbytes = max(int(np.prod(shape)) * dtype.itemsize, 1)
        buffer = runtime.device().newBufferWithLength_options_(
            nbytes, Metal.MTLResourceStorageModeShared
        )
        return cls(buffer, shape, dtype)

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
    def device(self) -> str:
        return "metal"

    def __len__(self) -> int:
        if not self._shape:
            raise TypeError("len() of a 0-dimensional MetalArray")
        return self._shape[0]

    def __repr__(self) -> str:
        return (
            f"MetalArray(shape={self._shape}, dtype={self._dtype.name}, device='metal')"
        )

    def numpy(self) -> np.ndarray:
        """Read the contents back as a NumPy array.

        A copy, so the result stays valid after the buffer is reused. The
        underlying memory is shared with the processor, so this costs a memory
        copy rather than a transfer across a bus.
        """
        # Anything encoded but not yet submitted has to finish first; reading
        # is the point at which the batching in runtime.flush() is settled.
        runtime.flush()
        raw = self.buffer.contents().as_buffer(max(self.nbytes, 1))
        flat = np.frombuffer(raw, dtype=self._dtype, count=self.size)
        return flat.reshape(self._shape).copy()

    def copy(self) -> "MetalArray":
        return MetalArray.from_numpy(self.numpy())

    def reshape(self, shape: tuple[int, ...]) -> "MetalArray":
        """A view of the same memory with a different shape."""
        if int(np.prod(shape)) != self.size:
            raise ValueError(
                f"cannot reshape {self._shape} ({self.size} elements) to {tuple(shape)}"
            )
        return MetalArray(self.buffer, shape, self._dtype)

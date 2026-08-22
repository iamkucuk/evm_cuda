"""An array in Vulkan device memory."""

from __future__ import annotations

from typing import Any

import numpy as np

from . import runtime

__all__ = ["VkArray"]


class VkArray:
    """A shaped, typed buffer in Vulkan memory.

    Backed by memory the host can map, so reading and writing need no separate
    transfer step. That keeps this backend simple, at some cost in speed on
    hardware that has its own memory.
    """

    __slots__ = ("buffer", "memory", "_shape", "_dtype")

    def __init__(
        self, buffer: Any, memory: Any, shape: tuple[int, ...], dtype: Any
    ) -> None:
        self.buffer = buffer
        self.memory = memory
        self._shape = tuple(int(s) for s in shape)
        self._dtype = np.dtype(dtype)

    @classmethod
    def empty(cls, shape: tuple[int, ...], dtype: Any) -> "VkArray":
        dtype = np.dtype(dtype)
        nbytes = max(int(np.prod(shape)) * dtype.itemsize, 4)
        buffer, memory = runtime.context().allocate(nbytes)
        return cls(buffer, memory, shape, dtype)

    @classmethod
    def zeros(cls, shape: tuple[int, ...], dtype: Any) -> "VkArray":
        return cls.from_numpy(np.zeros(shape, dtype=dtype))

    @classmethod
    def from_numpy(cls, host: np.ndarray) -> "VkArray":
        host = np.ascontiguousarray(host)
        out = cls.empty(host.shape, host.dtype)
        out._write(host)
        return out

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
        return "vulkan"

    def __len__(self) -> int:
        if not self._shape:
            raise TypeError("len() of a 0-dimensional VkArray")
        return self._shape[0]

    def __repr__(self) -> str:
        return (
            f"VkArray(shape={self._shape}, dtype={self._dtype.name}, device='vulkan')"
        )

    def _mapped_bytes(self) -> int:
        return max(self.nbytes, 4)

    def _write(self, host: np.ndarray) -> None:
        ctx = runtime.context()
        pointer = ctx.vk.vkMapMemory(
            ctx.device, self.memory, 0, self._mapped_bytes(), 0
        )
        view = np.frombuffer(pointer, dtype=self._dtype, count=max(self.size, 1))
        view[: self.size] = host.reshape(-1)[: self.size]
        ctx.vk.vkUnmapMemory(ctx.device, self.memory)

    def numpy(self) -> np.ndarray:
        ctx = runtime.context()
        pointer = ctx.vk.vkMapMemory(
            ctx.device, self.memory, 0, self._mapped_bytes(), 0
        )
        flat = np.frombuffer(pointer, dtype=self._dtype, count=max(self.size, 1))[
            : self.size
        ].copy()
        ctx.vk.vkUnmapMemory(ctx.device, self.memory)
        return flat.reshape(self._shape)

    def copy(self) -> "VkArray":
        return VkArray.from_numpy(self.numpy())

    def reshape(self, shape: tuple[int, ...]) -> "VkArray":
        """A view of the same memory with a different shape."""
        if int(np.prod(shape)) != self.size:
            raise ValueError(
                f"cannot reshape {self._shape} ({self.size} elements) to {tuple(shape)}"
            )
        return VkArray(self.buffer, self.memory, shape, self._dtype)

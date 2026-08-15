"""The PyTorch backend: the same pipelines, computed with tensors.

This is not how the project reaches any particular vendor's hardware — the
OpenCL, Vulkan and Apple backends do that, without asking anyone to install a
machine-learning framework. What this adds is different in kind.

It runs where PyTorch already runs, which on a machine that has PyTorch set up
means no further driver work. On NVIDIA hardware it is currently the only way to
magnify a live stream at all — the hand-written CUDA backend implements the four
whole-clip pipelines but not the primitive operations a frame-at-a-time pipeline
needs — and it is fast at it: 107.6 frames per second on 720p on an RTX 3090,
against a 30 frames per second target. It keeps results as tensors, so magnification can
sit inside a larger tensor computation without a round trip through the host.
And because it is written from the same definitions as the NumPy baseline but
through an entirely separate library, agreement between the two is evidence
about the definitions rather than about one implementation.

PyTorch is an optional dependency: ``pip install evm-magnify[torch]``. Nothing
here imports it until this backend is asked for, so a machine without it is
unaffected.
"""

from __future__ import annotations

from .runtime import available, device_name, pick_device, unavailable_reason

__all__ = ["available", "unavailable_reason", "device_name", "pick_device", "TorchOps"]


def __getattr__(name: str):
    # Deferred so that importing this package does not require torch.
    if name == "TorchOps":
        from .ops import TorchOps

        return TorchOps
    raise AttributeError(name)

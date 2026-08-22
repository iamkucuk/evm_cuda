"""The Vulkan backend.

Vulkan is the interface new graphics hardware ships with, across vendors and
operating systems. This backend exists so that a device appearing after this
was written has a path needing no new code here.

On Apple hardware Vulkan runs through MoltenVK, a translation layer onto Metal.
Where both are available the Metal backend is preferred, being one layer
shorter.
"""

from __future__ import annotations

from .runtime import available, device_name, unavailable_reason

__all__ = ["available", "unavailable_reason", "device_name", "VulkanOps", "VkArray"]


def __getattr__(name: str):
    if name == "VulkanOps":
        from .ops import VulkanOps

        return VulkanOps
    if name == "VkArray":
        from .array import VkArray

        return VkArray
    raise AttributeError(name)

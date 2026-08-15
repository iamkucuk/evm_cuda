"""Setting up Vulkan, and dispatching compute work through it.

Vulkan is the interface new graphics hardware ships with, across vendors and
across operating systems, and it is what a device that appears after this was
written is most likely to support. That is why this backend exists even though
OpenCL reaches the same hardware today: OpenCL is deprecated on Apple and not
guaranteed anywhere new.

Vulkan asks for a great deal to be set up before anything runs — an instance, a
device, a queue, descriptor layouts, pipelines, command buffers, memory. All of
it is built once here and reused, so the cost is paid at first use rather than
per operation.

The shaders are compiled to SPIR-V ahead of time and shipped in the package, so
installing needs no shader compiler. ``shaders/build.py`` regenerates them.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

__all__ = ["available", "unavailable_reason", "device_name", "Context"]

_SHADERS = Path(__file__).parent / "shaders"


def _point_loader_at_a_translation_driver() -> None:
    """On macOS, tell the Vulkan loader where MoltenVK is.

    macOS has no Vulkan driver of its own; MoltenVK provides one by translating
    to Metal. The loader finds it through an environment variable naming the
    driver's manifest, which a user should not have to set by hand, so the
    usual install locations are checked here. Anything already set is left
    alone: an explicit choice beats a guess.
    """
    import os
    import platform

    if platform.system() != "Darwin" or os.environ.get("VK_ICD_FILENAMES"):
        return
    for candidate in (
        "/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json",
        "/usr/local/etc/vulkan/icd.d/MoltenVK_icd.json",
        "/opt/homebrew/share/vulkan/icd.d/MoltenVK_icd.json",
        "/usr/local/share/vulkan/icd.d/MoltenVK_icd.json",
    ):
        if os.path.exists(candidate):
            os.environ["VK_ICD_FILENAMES"] = candidate
            return


def _import_vulkan() -> Any:
    _point_loader_at_a_translation_driver()
    import vulkan

    return vulkan


def unavailable_reason() -> str | None:
    """Why this backend cannot run here, or ``None`` if it can.

    Names which of the three things is missing, because the fixes differ
    completely: the Python bindings are installed with this project's
    ``vulkan`` extra, the loader and driver come from the operating system or
    the graphics vendor, and on macOS the driver is a translation layer that
    has to be installed separately.
    """
    try:
        vk = _import_vulkan()
    except ImportError:
        return (
            "the Vulkan bindings are not installed; install this project's "
            "'vulkan' extra (pip install evm-magnify[vulkan])"
        )
    except Exception as exc:
        # Importing the bindings is not only a Python import: the module looks
        # for the Vulkan SDK on the way in, and raises OSError rather than
        # ImportError when it cannot find one. That escaped this function and
        # propagated out of evm.backend.list_backends(), so merely asking which
        # backends exist crashed on a machine with the bindings installed and
        # no SDK — a Mac without MoltenVK, which is the common case. Reporting
        # it as a reason is the whole job of this function.
        return (
            f"the Vulkan bindings are installed but could not be loaded "
            f"({type(exc).__name__}: {exc}). On macOS this needs MoltenVK: "
            f"brew install molten-vk vulkan-loader. Elsewhere it comes with "
            f"the graphics driver."
        )
    if not any(_SHADERS.glob("*.spv")):
        return (
            f"the compiled shaders are missing from {_SHADERS}; "
            f"run shaders/build.py to regenerate them"
        )
    try:
        instance = _make_instance(vk)
        devices = vk.vkEnumeratePhysicalDevices(instance)
    except Exception as exc:
        return (
            f"no Vulkan driver found ({type(exc).__name__}: {exc}). On "
            f"macOS this needs MoltenVK, which translates Vulkan to Metal: "
            f"brew install molten-vk vulkan-loader. Elsewhere it comes with "
            f"the graphics driver."
        )
    if not devices:
        return "a Vulkan driver is present but reports no devices"
    return None


def available() -> bool:
    return unavailable_reason() is None


def _make_instance(vk: Any) -> Any:
    """A Vulkan instance that can also see translation-layer drivers.

    The portability flag and extension are what allow a driver that implements
    Vulkan on top of something else — MoltenVK on Apple hardware — to be
    enumerated at all. Without them this reports no devices on macOS.
    """
    application = vk.VkApplicationInfo(
        pApplicationName="evm", apiVersion=vk.VK_MAKE_VERSION(1, 1, 0)
    )
    try:
        return vk.vkCreateInstance(
            vk.VkInstanceCreateInfo(
                pApplicationInfo=application,
                flags=0x00000001,  # enumerate portability drivers
                ppEnabledExtensionNames=["VK_KHR_portability_enumeration"],
            ),
            None,
        )
    except Exception:
        # A loader without the portability extension: try again plainly, which
        # is the normal case on hardware with a native Vulkan driver.
        return vk.vkCreateInstance(
            vk.VkInstanceCreateInfo(pApplicationInfo=application), None
        )


class Context:
    """Everything Vulkan needs set up before any work can run.

    Built once and reused. Holding it in one object is what keeps the rest of
    the backend readable: an operation says which shader and which buffers,
    not how to allocate a descriptor pool.
    """

    def __init__(self) -> None:
        vk = _import_vulkan()
        self.vk = vk
        self.instance = _make_instance(vk)

        self.physical = vk.vkEnumeratePhysicalDevices(self.instance)[0]
        properties = vk.vkGetPhysicalDeviceProperties(self.physical)
        self.name = str(properties.deviceName)
        self.max_push_constants = int(properties.limits.maxPushConstantsSize)

        families = vk.vkGetPhysicalDeviceQueueFamilyProperties(self.physical)
        self.family = next(
            i for i, f in enumerate(families) if f.queueFlags & vk.VK_QUEUE_COMPUTE_BIT
        )

        # The portability extension is required by drivers that implement
        # Vulkan on top of something else; asking for it where it does not
        # exist fails, so it is attempted and then dropped.
        queue_info = vk.VkDeviceQueueCreateInfo(
            queueFamilyIndex=self.family, pQueuePriorities=[1.0]
        )
        try:
            self.device = vk.vkCreateDevice(
                self.physical,
                vk.VkDeviceCreateInfo(
                    pQueueCreateInfos=[queue_info],
                    ppEnabledExtensionNames=["VK_KHR_portability_subset"],
                ),
                None,
            )
        except Exception:
            self.device = vk.vkCreateDevice(
                self.physical,
                vk.VkDeviceCreateInfo(pQueueCreateInfos=[queue_info]),
                None,
            )

        self.queue = vk.vkGetDeviceQueue(self.device, self.family, 0)
        self.memory_properties = vk.vkGetPhysicalDeviceMemoryProperties(self.physical)
        self.command_pool = vk.vkCreateCommandPool(
            self.device,
            vk.VkCommandPoolCreateInfo(
                queueFamilyIndex=self.family,
                flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            ),
            None,
        )
        self._pipelines: dict[str, tuple[Any, Any, Any]] = {}
        self._descriptor_pool: Any = None

    # -- memory -------------------------------------------------------------

    def allocate(self, nbytes: int) -> tuple[Any, Any]:
        """A buffer in memory both the host and the device can address.

        Host-visible memory keeps this backend simple: there is no separate
        staging buffer and no explicit transfer. On hardware with its own
        memory that is slower than a device-local buffer would be, and it is
        the first thing to change if this backend is ever tuned.
        """
        vk = self.vk
        nbytes = max(int(nbytes), 4)
        buffer = vk.vkCreateBuffer(
            self.device,
            vk.VkBufferCreateInfo(
                size=nbytes,
                usage=vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            ),
            None,
        )
        requirements = vk.vkGetBufferMemoryRequirements(self.device, buffer)
        wanted = (
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        )
        index = next(
            i
            for i in range(self.memory_properties.memoryTypeCount)
            if (requirements.memoryTypeBits & (1 << i))
            and (self.memory_properties.memoryTypes[i].propertyFlags & wanted) == wanted
        )
        memory = vk.vkAllocateMemory(
            self.device,
            vk.VkMemoryAllocateInfo(
                allocationSize=requirements.size, memoryTypeIndex=index
            ),
            None,
        )
        vk.vkBindBufferMemory(self.device, buffer, memory, 0)
        return buffer, memory

    # -- pipelines ----------------------------------------------------------

    def pipeline(self, name: str, bindings: int, push_bytes: int):
        """The compiled shader, its layout and its descriptor layout.

        Built on first use and cached, because creating a pipeline is far more
        expensive than running one.
        """
        if name in self._pipelines:
            return self._pipelines[name]

        vk = self.vk
        path = _SHADERS / f"{name}.spv"
        if not path.exists():
            raise RuntimeError(f"no compiled shader at {path}")
        code = path.read_bytes()

        module = vk.vkCreateShaderModule(
            self.device,
            vk.VkShaderModuleCreateInfo(codeSize=len(code), pCode=code),
            None,
        )

        layout_bindings = [
            vk.VkDescriptorSetLayoutBinding(
                binding=i,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            )
            for i in range(bindings)
        ]
        set_layout = vk.vkCreateDescriptorSetLayout(
            self.device,
            vk.VkDescriptorSetLayoutCreateInfo(pBindings=layout_bindings),
            None,
        )

        ranges = []
        if push_bytes:
            ranges.append(
                vk.VkPushConstantRange(
                    stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT, offset=0, size=push_bytes
                )
            )
        pipeline_layout = vk.vkCreatePipelineLayout(
            self.device,
            vk.VkPipelineLayoutCreateInfo(
                pSetLayouts=[set_layout], pPushConstantRanges=ranges
            ),
            None,
        )

        pipeline = vk.vkCreateComputePipelines(
            self.device,
            vk.VK_NULL_HANDLE,
            1,
            [
                vk.VkComputePipelineCreateInfo(
                    stage=vk.VkPipelineShaderStageCreateInfo(
                        stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                        module=module,
                        pName="main",
                    ),
                    layout=pipeline_layout,
                )
            ],
            None,
        )[0]

        self._pipelines[name] = (pipeline, pipeline_layout, set_layout)
        return self._pipelines[name]

    def descriptor_pool(self) -> Any:
        """One pool, grown generously, rather than one per dispatch."""
        if self._descriptor_pool is None:
            vk = self.vk
            self._descriptor_pool = vk.vkCreateDescriptorPool(
                self.device,
                vk.VkDescriptorPoolCreateInfo(
                    maxSets=4096,
                    flags=vk.VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
                    pPoolSizes=[
                        vk.VkDescriptorPoolSize(
                            type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            descriptorCount=16384,
                        )
                    ],
                ),
                None,
            )
        return self._descriptor_pool

    def reset_descriptors(self) -> None:
        """Reclaim every descriptor set at once.

        Cheaper than freeing them individually, and safe once the queue is idle,
        which is the only point this is called.
        """
        if self._descriptor_pool is not None:
            self.vk.vkResetDescriptorPool(self.device, self._descriptor_pool, 0)


@functools.lru_cache(maxsize=1)
def context() -> Context:
    return Context()


def device_name() -> str:
    return context().name

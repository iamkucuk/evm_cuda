# Building blocks

The primitives the pipelines are made of. Each has the same name and argument
order on every backend, so a chain written against one runs on another.

See [use the building blocks](../recipes/building-blocks.md) for worked
examples.

## On the processor

::: evm.cpu.ops

## The pipelines, derived from the primitives

::: evm.backend.generic
    options:
      members:
        - color_gdown_ideal_core
        - motion_lpyr_ideal_core
        - motion_lpyr_butter_core
        - motion_lpyr_iir_core
        - bind

## Video reading and writing

::: evm.io.video
    options:
      members:
        - load_video
        - save_video
        - VideoInfo
        - rgb_to_yiq
        - yiq_to_rgb

# Kernel design notes

The detailed notes on how the CUDA kernels are laid out — which kernel does
what, the tolerance allowed for each stage against the NumPy reference, and why
each choice of precision and memory layout was made — live beside the code they
describe, in
[`src/vidmag/cuda/DESIGN.md`](https://github.com/iamkucuk/eulerian-video-magnification-cuda/blob/main/src/vidmag/cuda/DESIGN.md).

They are kept there rather than copied here on purpose. They are read while
changing the kernels, they refer to the files around them, and a second copy in
this site would be a second thing to keep current — which in practice means one
of them would quietly stop being true.

## Other internal documents

| Document | What it covers |
|---|---|
| [`docs/dev/PLAN.md`](https://github.com/iamkucuk/eulerian-video-magnification-cuda/blob/main/docs/dev/PLAN.md) | The plan this restructure follows, its decisions and their reasons |
| [`docs/dev/packaging-notes.md`](https://github.com/iamkucuk/eulerian-video-magnification-cuda/blob/main/docs/dev/packaging-notes.md) | Findings recorded while making the project installable, including defects found |
| [`docs/dev/gpu-runner.md`](https://github.com/iamkucuk/eulerian-video-magnification-cuda/blob/main/docs/dev/gpu-runner.md) | How the graphics-processor test machine is set up |
| [`.claude/rules/development-practices.md`](https://github.com/iamkucuk/eulerian-video-magnification-cuda/blob/main/.claude/rules/development-practices.md) | The development rules this project is built under |
